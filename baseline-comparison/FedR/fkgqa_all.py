"""
Baseline 4: Adapted FedR (Horizontal Federated)

Relation-based aggregation with privacy masking. Before uploading,
each client masks relation embeddings for relations not present in
its local KG. Server aggregates only non-masked entries.

Differences from FedKGQA:
  - Phase 1: Masked relation aggregation (only present relations updated)
  - Phase 2: Same (FedAvg on QA models)
  - Rare relations get fewer updates → slower convergence

Reference: Zhang et al., "Efficient Federated Learning on Knowledge
Graphs via Privacy-Preserving Relation Embedding Aggregation",
EMNLP Findings 2022.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy

from kge_model_all import (
    ComplExModel, initialize_embeddings, compute_kg_loss
)
from qa_model_all import (
    ImprovedKGQAModel, compute_qa_loss, evaluate_qa
)


class FederatedKGQA:
    """FedR: masked relation aggregation."""

    def __init__(self, args, all_clients_data, global_nrelation):
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        self.kge_model_name = args.kge_model.lower()

        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim

        if self.kge_model_name == 'complex':
            rel_init = torch.zeros(global_nrelation, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
        elif self.kge_model_name == 'rotate':
            rel_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
        else:
            rel_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)

        self.global_relation_embedding = nn.Parameter(rel_init, requires_grad=True)

        self.client_entity_embeddings = []
        self.client_kg_models = []
        self.client_qa_models = []
        self.client_optimizers_kg = []
        self.client_optimizers_qa = []

        # Build relation presence masks per client
        self.client_relation_masks = []

        self.best_dev_metrics = {'avg_hits@5': 0.0}

        logging.info(f"Baseline: FedR (Masked relation aggregation)")
        logging.info(f"KGE Model: {self.kge_model_name.upper()}")

    def _build_relation_masks(self):
        """Build binary masks: which relations are present in each client."""
        for cid, cd in enumerate(self.all_clients_data):
            mask = torch.zeros(self.global_nrelation, dtype=torch.bool)
            # Find which global relation IDs are present in this client
            for rel_name in cd['relations']:
                if rel_name in cd.get('global_relation2id', {}):
                    global_rid = cd['global_relation2id'][rel_name]
                    mask[global_rid] = True
            self.client_relation_masks.append(mask.to(self.args.gpu))
            present = mask.sum().item()
            logging.info(f"  Client {cid}: {present}/{self.global_nrelation} relations present")

    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders, test_qa_loaders=None):
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders

        self.global_relation_embedding.data = self.global_relation_embedding.data.to(self.args.gpu)

        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim

            if self.kge_model_name in ['complex', 'rotate']:
                ent_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            else:
                ent_init = torch.zeros(nentity, self.args.hidden_dim, device=self.args.gpu).uniform_(-embedding_range, embedding_range)

            self.client_entity_embeddings.append(nn.Parameter(ent_init, requires_grad=True))

            kg_model = ComplExModel(self.args).to(self.args.gpu)
            self.client_kg_models.append(kg_model)

            qa_model = ImprovedKGQAModel(self.args, nentity, self.global_nrelation).to(self.args.gpu)
            self.client_qa_models.append(qa_model)

            logging.info(f"Client {i} setup: {nentity} entities")

        for i in range(self.num_clients):
            self.client_optimizers_kg.append(optim.Adam([self.client_entity_embeddings[i]], lr=self.args.lr))
            self.client_optimizers_qa.append(optim.Adam(self.client_qa_models[i].parameters(), lr=self.args.qa_lr))

        self.global_relation_optimizer = optim.Adam([self.global_relation_embedding], lr=self.args.lr)

        self._build_relation_masks()
        logging.info("All client optimizers initialized")

    def _masked_relation_aggregation(self, rel_grads, selected_clients):
        """
        FedR-style masked aggregation: only aggregate gradients for
        relations that are present in each client's local KG.

        For each relation r:
          - Count how many selected clients have r
          - Average gradients only from those clients
          - Relations with 0 clients get no update
        """
        device = self.global_relation_embedding.device
        agg_grad = torch.zeros_like(self.global_relation_embedding)
        count = torch.zeros(self.global_nrelation, 1, device=device)

        for grad, cid in zip(rel_grads, selected_clients):
            mask = self.client_relation_masks[cid]  # (nrelation,)
            # Only include gradient for present relations
            mask_expanded = mask.unsqueeze(1).float()  # (nrelation, 1)
            agg_grad += grad * mask_expanded
            count += mask_expanded

        # Average only by count of contributing clients (avoid /0)
        count = count.clamp(min=1)
        agg_grad = agg_grad / count

        return agg_grad

    def train_phase1_kg(self):
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 1: Federated KG Training (Masked relation aggregation)")
        logging.info("=" * 70)

        best_avg_loss = float('inf')
        patience_counter = 0

        for round_idx in range(self.args.kg_max_rounds):
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected = torch.randperm(self.num_clients)[:num_selected].tolist()

            rel_grads, losses = [], []
            for cid in selected:
                loss, rg = self._train_client_kg(cid)
                losses.append(loss)
                rel_grads.append(rg)

            # KEY DIFFERENCE: Masked relation aggregation (not simple FedAvg)
            if rel_grads:
                masked_grad = self._masked_relation_aggregation(rel_grads, selected)
                self.global_relation_optimizer.zero_grad()
                self.global_relation_embedding.grad = masked_grad
                self.global_relation_optimizer.step()

            avg_loss = sum(losses) / len(losses)
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[KG Round {round_idx+1}/{self.args.kg_max_rounds}] "
                             f"Clients: {selected}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.args.early_stop_patience:
                logging.info(f"Early stopping at round {round_idx+1}")
                break

        logging.info(f"Phase 1 completed! Best Loss: {best_avg_loss:.4f}")

    def _train_client_kg(self, client_id):
        kg_model = self.client_kg_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]

        kg_model.train()
        total_loss = 0
        for epoch in range(self.args.local_epoch):
            for pos, neg, _ in self.kg_dataloaders[client_id]:
                pos, neg = pos.to(self.args.gpu), neg.to(self.args.gpu)
                optimizer.zero_grad()
                self.global_relation_optimizer.zero_grad()
                loss = compute_kg_loss(kg_model, pos, neg, entity_emb,
                                       self.global_relation_embedding, self.args)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        rel_grad = self.global_relation_embedding.grad.clone() if self.global_relation_embedding.grad is not None else torch.zeros_like(self.global_relation_embedding)
        n = len(self.kg_dataloaders[client_id]) * self.args.local_epoch
        return total_loss / n if n > 0 else 0, rel_grad

    def train_phase2_qa(self):
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Federated QA Training (FedAvg)")
        logging.info("=" * 70)

        self.global_relation_embedding.requires_grad = False
        for emb in self.client_entity_embeddings:
            emb.requires_grad = False

        best_dev_score = 0.0
        patience_counter = 0

        for round_idx in range(self.args.qa_max_rounds):
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected = torch.randperm(self.num_clients)[:num_selected].tolist()

            local_models, losses = [], []
            for cid in selected:
                loss, state = self._train_client_qa(cid)
                losses.append(loss)
                local_models.append(state)

            self._fedavg_qa(local_models)

            avg_loss = sum(losses) / len(losses)
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[QA Round {round_idx+1}/{self.args.qa_max_rounds}] "
                             f"Clients: {selected}, Avg Loss: {avg_loss:.4f}")

            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self._evaluate_dev()
                logging.info(f"  Dev → H@3: {dev_metrics['avg_hits@3']:.4f}, "
                             f"H@5: {dev_metrics['avg_hits@5']:.4f}, "
                             f"H@10: {dev_metrics['avg_hits@10']:.4f}, "
                             f"MRR: {dev_metrics['avg_mrr']:.4f}")
                if dev_metrics['avg_hits@5'] > best_dev_score:
                    best_dev_score = dev_metrics['avg_hits@5']
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    logging.info(f"  ✓ New best! (H@5: {best_dev_score:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.args.early_stop_patience:
                    logging.info(f"Early stopping at round {round_idx+1}")
                    break

        if not os.path.exists(os.path.join(self.args.state_dir, 'best_models')):
            self.save_best_models()
        logging.info("Phase 2 completed!")

    def _train_client_qa(self, client_id):
        cd = self.all_clients_data[client_id]
        qa_model = self.client_qa_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id].to(self.args.gpu)
        relation_emb = self.global_relation_embedding.to(self.args.gpu)
        optimizer = self.client_optimizers_qa[client_id]

        qa_model.train()
        total_loss = 0
        for epoch in range(self.args.qa_local_epoch):
            for questions, answer_ids, hop_counts in self.train_qa_loaders[client_id]:
                entity_scores, relation_scores, topic_ids = qa_model(
                    questions, relation_emb, entity_emb,
                    cd['entity2id'], answer_ids)
                loss = compute_qa_loss(entity_scores, answer_ids, self.args)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        state = copy.deepcopy(qa_model.state_dict())
        n = len(self.train_qa_loaders[client_id]) * self.args.qa_local_epoch
        return total_loss / n if n > 0 else 0, state

    def _fedavg_qa(self, local_models):
        if not local_models:
            return
        global_state = {}
        for key in local_models[0].keys():
            global_state[key] = torch.stack([m[key].float() for m in local_models]).mean(dim=0)
        for cid in range(self.num_clients):
            self.client_qa_models[cid].load_state_dict(global_state, strict=False)

    def _evaluate_dev(self):
        all_metrics = []
        for cid in range(self.num_clients):
            cd = self.all_clients_data[cid]
            metrics = evaluate_qa(
                self.client_qa_models[cid], self.dev_qa_loaders[cid],
                self.global_relation_embedding.to(self.args.gpu),
                self.client_entity_embeddings[cid].to(self.args.gpu),
                cd['entity2id'], cd['id2entity'], self.args)
            all_metrics.append(metrics)
        return {
            'avg_hits@1': sum(m['hits@1'] for m in all_metrics) / len(all_metrics),
            'avg_hits@3': sum(m['hits@3'] for m in all_metrics) / len(all_metrics),
            'avg_hits@5': sum(m['hits@5'] for m in all_metrics) / len(all_metrics),
            'avg_hits@10': sum(m['hits@10'] for m in all_metrics) / len(all_metrics),
            'avg_mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics),
        }

    def save_best_models(self):
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.global_relation_embedding, os.path.join(save_dir, 'global_relation_embeddings.pt'))
        with open(os.path.join(save_dir, 'kge_model.txt'), 'w') as f: f.write(self.kge_model_name)
        lm = self.args.lm_model.lower() if hasattr(self.args, 'lm_model') else 'bert'
        with open(os.path.join(save_dir, 'lm_model.txt'), 'w') as f: f.write(lm)
        for cid in range(self.num_clients):
            cdir = os.path.join(save_dir, f'client_{cid}')
            os.makedirs(cdir, exist_ok=True)
            torch.save(self.client_entity_embeddings[cid], os.path.join(cdir, 'entity_embeddings.pt'))
            torch.save(self.client_qa_models[cid].state_dict(), os.path.join(cdir, 'qa_model.pt'))
        logging.info(f"Models saved to {save_dir}")

    def train(self):
        self.train_phase1_kg()
        self.train_phase2_qa()
        logging.info("\nTraining completed!")
        if 'avg_hits@5' in self.best_dev_metrics:
            logging.info(f"Best Dev → H@3: {self.best_dev_metrics.get('avg_hits@3',0):.4f}, "
                         f"H@5: {self.best_dev_metrics['avg_hits@5']:.4f}, "
                         f"H@10: {self.best_dev_metrics.get('avg_hits@10',0):.4f}, "
                         f"MRR: {self.best_dev_metrics['avg_mrr']:.4f}")
