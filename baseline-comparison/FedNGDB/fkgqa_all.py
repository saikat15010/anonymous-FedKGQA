"""
Baseline 5: Adapted FedNGDB (Horizontal Federated)

No federated training. Each client trains KGE and QA independently.
Server coordinates multi-hop query answering at inference time by
routing sub-queries to clients based on entity ownership.

Differences from FedKGQA:
  - Phase 1: No relation aggregation (independent training)
  - Phase 2: No QA model aggregation (independent training)
  - Multi-hop: Yes, but via inference-time query routing (not
    training-time federation)
  - Each client has its own relation embeddings (no shared semantics)

Reference: Hu et al., "Learning Federated Neural Graph Databases for
Answering Complex Queries from Distributed Knowledge Graphs",
TMLR 2025.
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
    """FedNGDB: independent training + inference-time query routing."""

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

        self.global_relation_embedding = self.client_relation_embeddings[0]

        self.client_entity_embeddings = []
        self.client_kg_models = []
        self.client_qa_models = []
        self.client_optimizers_kg = []
        self.client_optimizers_qa = []
        self.client_relation_optimizers = []

        self.best_dev_metrics = {'avg_hits@5': 0.0}

        logging.info(f"Baseline: FedNGDB (Independent training + query routing)")
        logging.info(f"KGE Model: {self.kge_model_name.upper()}")

    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders, test_qa_loaders=None):
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders

        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim

            self.client_relation_embeddings[i].data = self.client_relation_embeddings[i].data.to(self.args.gpu)

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
            # Optimize entity + relation jointly (both local)
            kg_opt = optim.Adam([self.client_entity_embeddings[i],
                                 self.client_relation_embeddings[i]], lr=self.args.lr)
            self.client_optimizers_kg.append(kg_opt)
            self.client_optimizers_qa.append(
                optim.Adam(self.client_qa_models[i].parameters(), lr=self.args.qa_lr))

        logging.info("All client optimizers initialized")

    def train_phase1_kg(self):
        """Phase 1: Independent KGE training per client (no federation)"""
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 1: Independent KG Training (No Federation)")
        logging.info("=" * 70)

        for round_idx in range(self.args.kg_max_rounds):
            losses = []
            for client_id in range(self.num_clients):
                loss = self._train_client_kg(client_id)
                losses.append(loss)

            # NO aggregation

            avg_loss = sum(losses) / len(losses)
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[KG Round {round_idx+1}/{self.args.kg_max_rounds}] Avg Loss: {avg_loss:.4f}")

        logging.info("Phase 1 completed (no aggregation)")

    def _train_client_kg(self, client_id):
        kg_model = self.client_kg_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id]
        relation_emb = self.client_relation_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]

        kg_model.train()
        total_loss = 0
        for epoch in range(self.args.local_epoch):
            for pos, neg, _ in self.kg_dataloaders[client_id]:
                pos, neg = pos.to(self.args.gpu), neg.to(self.args.gpu)
                optimizer.zero_grad()
                loss = compute_kg_loss(kg_model, pos, neg, entity_emb,
                                       relation_emb, self.args)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        n = len(self.kg_dataloaders[client_id]) * self.args.local_epoch
        return total_loss / n if n > 0 else 0

    def train_phase2_qa(self):
        """Phase 2: Independent QA training per client (no federation)"""
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Independent QA Training (No Federation)")
        logging.info("=" * 70)

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

            # NO QA model aggregation

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
        cd = self.all_clients_data[client_id]
        qa_model = self.client_qa_models[client_id]
        entity_emb = self.client_entity_embeddings[client_id].to(self.args.gpu)
        relation_emb = self.client_relation_embeddings[client_id].to(self.args.gpu)
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

        n = len(self.train_qa_loaders[client_id]) * self.args.qa_local_epoch
        return total_loss / n if n > 0 else 0

    def _evaluate_dev(self):
        all_metrics = []
        for cid in range(self.num_clients):
            cd = self.all_clients_data[cid]
            metrics = evaluate_qa(
                self.client_qa_models[cid], self.dev_qa_loaders[cid],
                self.client_relation_embeddings[cid].to(self.args.gpu),
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
        torch.save(self.client_relation_embeddings[0], os.path.join(save_dir, 'global_relation_embeddings.pt'))
        with open(os.path.join(save_dir, 'kge_model.txt'), 'w') as f: f.write(self.kge_model_name)
        lm = self.args.lm_model.lower() if hasattr(self.args, 'lm_model') else 'bert'
        with open(os.path.join(save_dir, 'lm_model.txt'), 'w') as f: f.write(lm)
        for cid in range(self.num_clients):
            cdir = os.path.join(save_dir, f'client_{cid}')
            os.makedirs(cdir, exist_ok=True)
            torch.save(self.client_entity_embeddings[cid], os.path.join(cdir, 'entity_embeddings.pt'))
            torch.save(self.client_qa_models[cid].state_dict(), os.path.join(cdir, 'qa_model.pt'))
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
