"""
Baseline 1: Adapted EmbedKGQA (Horizontal Federated)

No federation during training. Each client trains KGE and QA
independently on its own data. At inference, server queries all
clients and merges top-k predictions by score.

Differences from FedKGQA:
  - Phase 1: No relation aggregation (each client has its own relations)
  - Phase 2: No QA model aggregation (each client has its own QA model)
  - Clients are completely isolated during training

Reference: Saxena et al., "Improving multi-hop question answering over
knowledge graphs using knowledge base embeddings", ACL 2020.
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
    """EmbedKGQA adapted to horizontal federated setting — no federation."""

    def __init__(self, args, all_clients_data, global_nrelation):
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        self.kge_model_name = args.kge_model.lower()

        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim

        # Each client gets its OWN relation embeddings (no sharing)
        self.client_relation_embeddings = []
        for _ in range(self.num_clients):
            if self.kge_model_name == 'complex':
                rel_init = torch.zeros(global_nrelation, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
            elif self.kge_model_name == 'rotate':
                rel_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
            else:
                rel_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
            self.client_relation_embeddings.append(nn.Parameter(rel_init, requires_grad=True))

        # Also keep a global_relation_embedding for compatibility with save/eval
        self.global_relation_embedding = self.client_relation_embeddings[0]

        self.client_entity_embeddings = []
        self.client_kg_models = []
        self.client_qa_models = []
        self.client_optimizers_kg = []
        self.client_optimizers_qa = []
        self.client_relation_optimizers = []

        self.best_dev_metrics = {'avg_hits@5': 0.0}

        logging.info(f"Baseline: EmbedKGQA (No Federation)")
        logging.info(f"KGE Model: {self.kge_model_name.upper()}")

    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders, test_qa_loaders=None):
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders

        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim

            # Move relation embeddings to device
            self.client_relation_embeddings[i].data = self.client_relation_embeddings[i].data.to(self.args.gpu)

            if self.kge_model_name in ['complex', 'rotate']:
                ent_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            else:
                ent_init = torch.zeros(nentity, self.args.hidden_dim, device=self.args.gpu).uniform_(-embedding_range, embedding_range)

            entity_embedding = nn.Parameter(ent_init, requires_grad=True)
            self.client_entity_embeddings.append(entity_embedding)

            kg_model = ComplExModel(self.args).to(self.args.gpu)
            self.client_kg_models.append(kg_model)

            qa_model = ImprovedKGQAModel(self.args, nentity, self.global_nrelation).to(self.args.gpu)
            self.client_qa_models.append(qa_model)

            logging.info(f"Client {i} setup: {nentity} entities")

        for i in range(self.num_clients):
            # KG optimizer: entity + relation (both local)
            kg_opt = optim.Adam([self.client_entity_embeddings[i], self.client_relation_embeddings[i]], lr=self.args.lr)
            self.client_optimizers_kg.append(kg_opt)
            qa_opt = optim.Adam(self.client_qa_models[i].parameters(), lr=self.args.qa_lr)
            self.client_optimizers_qa.append(qa_opt)

        logging.info("All client optimizers initialized")

    def train_phase1_kg(self):
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 1: Independent KG Training (No Federation)")
        logging.info("=" * 70)

        for round_idx in range(self.args.kg_max_rounds):
            losses = []
            for client_id in range(self.num_clients):
                loss = self._train_client_kg(client_id)
                losses.append(loss)

            # NO aggregation — each client keeps its own embeddings

            avg_loss = sum(losses) / len(losses)
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[KG Round {round_idx+1}/{self.args.kg_max_rounds}] Avg Loss: {avg_loss:.4f}")

        logging.info("Phase 1 completed (no aggregation)")

    def _train_client_kg(self, client_id):
        kg_model = self.client_kg_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id]
        relation_emb = self.client_relation_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]
        dataloader = self.kg_dataloaders[client_id]

        kg_model.train()
        total_loss = 0

        for epoch in range(self.args.local_epoch):
            for positive_sample, negative_sample, _ in dataloader:
                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)

                optimizer.zero_grad()
                loss = compute_kg_loss(kg_model, positive_sample, negative_sample,
                                       entity_emb, relation_emb, self.args)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss / (len(dataloader) * self.args.local_epoch) if len(dataloader) > 0 else 0

    def train_phase2_qa(self):
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Independent QA Training (No Federation)")
        logging.info("=" * 70)

        # Freeze embeddings
        for i in range(self.num_clients):
            self.client_entity_embeddings[i].requires_grad = False
            self.client_relation_embeddings[i].requires_grad = False

        best_dev_score = 0.0
        patience_counter = 0

        for round_idx in range(self.args.qa_max_rounds):
            losses = []
            for client_id in range(self.num_clients):
                loss = self._train_client_qa(client_id)
                losses.append(loss)

            # NO QA model aggregation — each client keeps its own

            avg_loss = sum(losses) / len(losses)
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[QA Round {round_idx+1}/{self.args.qa_max_rounds}] Avg Loss: {avg_loss:.4f}")

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
        logging.info("Phase 2 completed (no aggregation)")

    def _train_client_qa(self, client_id):
        client_data = self.all_clients_data[client_id]
        qa_model = self.client_qa_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id].to(self.args.gpu)
        relation_emb = self.client_relation_embeddings[client_id].to(self.args.gpu)
        optimizer = self.client_optimizers_qa[client_id]
        dataloader = self.train_qa_loaders[client_id]

        qa_model.train()
        total_loss = 0

        for epoch in range(self.args.qa_local_epoch):
            for questions, answer_ids, hop_counts in dataloader:
                entity_scores, relation_scores, topic_ids = qa_model(
                    questions, relation_emb, entity_emb,
                    client_data['entity2id'], answer_ids)
                loss = compute_qa_loss(entity_scores, answer_ids, self.args)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss / (len(dataloader) * self.args.qa_local_epoch) if len(dataloader) > 0 else 0

    def _evaluate_dev(self):
        all_metrics = []
        for client_id in range(self.num_clients):
            cd = self.all_clients_data[client_id]
            metrics = evaluate_qa(
                self.client_qa_models[client_id], self.dev_qa_loaders[client_id],
                self.client_relation_embeddings[client_id].to(self.args.gpu),
                self.client_entity_embeddings[client_id].to(self.args.gpu),
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
        # Save first client's relation as global (for eval compatibility)
        torch.save(self.client_relation_embeddings[0], os.path.join(save_dir, 'global_relation_embeddings.pt'))
        with open(os.path.join(save_dir, 'kge_model.txt'), 'w') as f: f.write(self.kge_model_name)
        lm = self.args.lm_model.lower() if hasattr(self.args, 'lm_model') else 'bert'
        with open(os.path.join(save_dir, 'lm_model.txt'), 'w') as f: f.write(lm)
        for cid in range(self.num_clients):
            cdir = os.path.join(save_dir, f'client_{cid}')
            os.makedirs(cdir, exist_ok=True)
            torch.save(self.client_entity_embeddings[cid], os.path.join(cdir, 'entity_embeddings.pt'))
            torch.save(self.client_qa_models[cid].state_dict(), os.path.join(cdir, 'qa_model.pt'))
            # Also save client-specific relation embeddings
            torch.save(self.client_relation_embeddings[cid], os.path.join(cdir, 'relation_embeddings.pt'))
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
