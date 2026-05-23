"""
Federated KGQA Training System

Implements two-phase federated learning:
Phase 1: Federated KG Embedding Training (TransE/DistMult/RotatE/ComplEx with global relations)
Phase 2: Federated QA Model Training (RoBERTa with local QA pairs)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from typing import List, Dict

from kge_model_all import (
    get_kge_model,
    initialize_embeddings,
    compute_kg_loss,
    evaluate_kg
)
from qa_model_all import (
    ImprovedKGQAModel,
    compute_qa_loss,
    evaluate_qa
)


class FederatedKGQA:
    """
    Federated KGQA system with two-phase training
    Supports: TransE, DistMult, RotatE, ComplEx
    """
    
    def __init__(self, args, all_clients_data, global_nrelation):
        """
        Initialize federated KGQA system
        
        Args:
            args: Training arguments
            all_clients_data: List of client data dicts
            global_nrelation: Number of global relations
        """
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        self.kge_model_name = args.kge_model.lower()
        
        # Initialize global relation embeddings (shared across clients)
        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Different KGE models have different relation embedding sizes
        if self.kge_model_name == 'complex':
            # ComplEx: 2*hidden_dim (real + imaginary)
            global_relation_init = torch.zeros(global_nrelation, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
        elif self.kge_model_name == 'rotate':
            # RotatE: hidden_dim (phase angles)
            global_relation_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
        else:
            # TransE, DistMult: hidden_dim (real-valued)
            global_relation_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
        
        self.global_relation_embedding = nn.Parameter(global_relation_init, requires_grad=True)
        
        # Client-specific components
        self.client_entity_embeddings = []
        self.client_kg_models = []
        self.client_qa_models = []
        self.client_optimizers_kg = []
        self.client_optimizers_qa = []
        
        # Server components
        self.global_qa_model = None
        
        # Training state
        self.current_phase = None
        self.best_dev_metrics = {'avg_hits@5': 0.0}
        
        logging.info(f"Initialized Federated KGQA with {self.num_clients} clients")
        logging.info(f"KGE Model: {self.kge_model_name.upper()}")
        logging.info(f"Global relations: {global_nrelation}")
    
    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders, test_qa_loaders=None):
        """
        Setup client models and dataloaders
        
        Args:
            kg_dataloaders: List of KG training dataloaders
            train_qa_loaders: List of QA training dataloaders
            dev_qa_loaders: List of QA dev dataloaders
            test_qa_loaders: Optional list of QA test dataloaders
        """
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders
        self.test_qa_loaders = test_qa_loaders
        
        # Move global relation embedding to device
        self.global_relation_embedding.data = self.global_relation_embedding.data.to(self.args.gpu)
        
        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            
            # Initialize entity embeddings based on KGE model type
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim
            
            if self.kge_model_name in ['complex', 'rotate']:
                # Complex and RotatE: 2*hidden_dim for entities (real + imaginary)
                entity_embedding_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            else:
                # TransE, DistMult: hidden_dim (real-valued)
                entity_embedding_init = torch.zeros(nentity, self.args.hidden_dim, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            
            entity_embedding = nn.Parameter(entity_embedding_init, requires_grad=True)
            self.client_entity_embeddings.append(entity_embedding)
            
            # Initialize KG model
            kg_model = get_kge_model(self.kge_model_name, self.args)
            kg_model = kg_model.to(self.args.gpu)
            self.client_kg_models.append(kg_model)
            
            # Initialize QA model
            qa_model = ImprovedKGQAModel(self.args, nentity, self.global_nrelation)
            qa_model = qa_model.to(self.args.gpu)
            self.client_qa_models.append(qa_model)
            
            logging.info(f"Client {i} setup: {nentity} entities")
        
        # Initialize optimizers
        for i in range(self.num_clients):
            # KG optimizer (for entity embeddings only)
            kg_optimizer = optim.Adam([self.client_entity_embeddings[i]], lr=self.args.lr)
            self.client_optimizers_kg.append(kg_optimizer)
            
            # QA optimizer
            qa_optimizer = optim.Adam(self.client_qa_models[i].parameters(), lr=self.args.qa_lr)
            self.client_optimizers_qa.append(qa_optimizer)
        
        # Global relation optimizer
        self.global_relation_optimizer = optim.Adam([self.global_relation_embedding], lr=self.args.lr)
        
        logging.info("All client optimizers initialized successfully")
    
    def train_phase1_kg(self):
        """
        Phase 1: Federated KG Embedding Training
        
        - Clients train local entity embeddings
        - Global relation embeddings are shared and aggregated
        """
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 1: Federated KG Embedding Training ({self.kge_model_name.upper()})")
        logging.info("=" * 70)
        
        self.current_phase = 'kg'
        
        patience_counter = 0
        best_avg_loss = float('inf')
        
        for round_idx in range(self.args.kg_max_rounds):
            # Sample clients
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training on selected clients
            local_relation_grads = []
            local_losses = []
            
            for client_id in selected_clients:
                loss, rel_grad = self.train_client_kg(client_id)
                local_losses.append(loss)
                local_relation_grads.append(rel_grad)
            
            # Federated aggregation of relation gradients
            if local_relation_grads:
                avg_relation_grad = torch.stack(local_relation_grads).mean(dim=0)
                self.global_relation_optimizer.zero_grad()
                self.global_relation_embedding.grad = avg_relation_grad.clone()
                self.global_relation_optimizer.step()
            
            # Logging
            avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"[KG Round {round_idx + 1}/{self.args.kg_max_rounds}] "
                    f"Clients: {selected_clients}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Early stopping
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.args.early_stop_patience:
                logging.info(f"Early stopping at round {round_idx + 1}")
                break
        
        logging.info("Phase 1 (KG Training) completed!")
        logging.info(f"Best Average Loss: {best_avg_loss:.4f}")
    
    def train_client_kg(self, client_id):
        """
        Train KG embeddings for a single client
        
        Returns:
            (loss, relation_gradient)
        """
        client_data = self.all_clients_data[client_id]
        kg_model = self.client_kg_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]
        dataloader = self.kg_dataloaders[client_id]
        
        kg_model.train()
        total_loss = 0
        
        for epoch in range(self.args.local_epoch):
            for batch_idx, (positive_sample, negative_sample, _) in enumerate(dataloader):
                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)
                
                # Zero previous gradients
                optimizer.zero_grad()
                self.global_relation_optimizer.zero_grad()
                
                # Compute loss
                loss = compute_kg_loss(
                    kg_model,
                    positive_sample,
                    negative_sample,
                    entity_embedding,
                    self.global_relation_embedding,
                    self.args
                )
                
                # Backward pass
                loss.backward()
                
                # Update entity embeddings (local)
                optimizer.step()
                
                total_loss += loss.item()
        
        # Collect relation gradient for federated aggregation
        relation_grad = self.global_relation_embedding.grad.clone() if self.global_relation_embedding.grad is not None else torch.zeros_like(self.global_relation_embedding)
        
        avg_loss = total_loss / (len(dataloader) * self.args.local_epoch) if len(dataloader) > 0 else 0
        
        return avg_loss, relation_grad
    
    def train_phase2_qa(self):
        """
        Phase 2: Federated QA Model Training
        
        - Entity and relation embeddings are frozen
        - Only QA models are trained
        """
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Federated QA Model Training ({self.args.lm_model.upper()})")
        logging.info("=" * 70)
        
        self.current_phase = 'qa'
        
        # Freeze KG embeddings
        self.global_relation_embedding.requires_grad = False
        for entity_emb in self.client_entity_embeddings:
            entity_emb.requires_grad = False
        
        patience_counter = 0
        best_dev_score = 0.0
        
        for round_idx in range(self.args.qa_max_rounds):
            # Sample clients
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training
            local_qa_models = []
            local_losses = []
            
            for client_id in selected_clients:
                loss, model_state = self.train_client_qa(client_id)
                local_losses.append(loss)
                local_qa_models.append(model_state)
            
            # Federated averaging of QA models
            self.federated_averaging_qa(local_qa_models, selected_clients)
            
            # Logging
            avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"[QA Round {round_idx + 1}/{self.args.qa_max_rounds}] "
                    f"Clients: {selected_clients}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Evaluation on dev set
            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_dev()
                
                logging.info(
                    f"  Dev → Hits@3: {dev_metrics['avg_hits@3']:.4f}, "
                    f"Hits@5: {dev_metrics['avg_hits@5']:.4f}, "
                    f"Hits@10: {dev_metrics['avg_hits@10']:.4f}, "
                    f"MRR: {dev_metrics['avg_mrr']:.4f}"
                )
                
                # Save best model
                if dev_metrics['avg_hits@5'] > best_dev_score:
                    best_dev_score = dev_metrics['avg_hits@5']
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    logging.info(f"  ✓ New best model saved! (Hits@5: {best_dev_score:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.args.early_stop_patience:
                    logging.info(f"Early stopping at round {round_idx + 1}")
                    break
        
        # Save final models if no best model was saved
        if not os.path.exists(os.path.join(self.args.state_dir, 'best_models')):
            logging.info("Saving final models...")
            self.save_best_models()
        
        logging.info("Phase 2 (QA Training) completed!")
        if 'avg_hits@5' in self.best_dev_metrics:
            logging.info(
                f"Best Dev → Hits@3: {self.best_dev_metrics.get('avg_hits@3', 0):.4f}, "
                f"Hits@5: {self.best_dev_metrics['avg_hits@5']:.4f}, "
                f"Hits@10: {self.best_dev_metrics.get('avg_hits@10', 0):.4f}, "
                f"MRR: {self.best_dev_metrics['avg_mrr']:.4f}"
            )
    
    def train_client_qa(self, client_id):
        """
        Train QA model for a single client
        
        Returns:
            (loss, updated_model_state)
        """
        client_data = self.all_clients_data[client_id]
        qa_model = self.client_qa_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id].to(self.args.gpu)
        relation_embedding = self.global_relation_embedding.to(self.args.gpu)
        optimizer = self.client_optimizers_qa[client_id]
        dataloader = self.train_qa_loaders[client_id]
        
        qa_model.train()
        total_loss = 0
        
        for epoch in range(self.args.qa_local_epoch):
            for batch_idx, (questions, answer_ids, hop_counts) in enumerate(dataloader):
                # Forward pass
                entity_scores, relation_scores, topic_entity_ids = qa_model(
                    questions,
                    relation_embedding,
                    entity_embedding,
                    client_data['entity2id'],
                    answer_ids
                )
                
                # Compute loss
                loss = compute_qa_loss(entity_scores, answer_ids, self.args)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(dataloader) * self.args.qa_local_epoch) if len(dataloader) > 0 else 0
        
        # Return model state for aggregation
        model_state = copy.deepcopy(qa_model.state_dict())
        
        return avg_loss, model_state
    
    def federated_averaging_qa(self, local_models, selected_clients):
        """
        Federated averaging of QA models
        
        Args:
            local_models: List of model state dicts
            selected_clients: List of selected client IDs
        """
        if not local_models:
            return
        
        # Average model parameters
        global_state = {}
        for key in local_models[0].keys():
            global_state[key] = torch.stack([m[key].float() for m in local_models]).mean(dim=0)
        
        # Update all client models with averaged parameters
        for client_id in range(self.num_clients):
            self.client_qa_models[client_id].load_state_dict(global_state, strict=False)
    
    def evaluate_dev(self):
        """
        Evaluate all clients on their dev sets
        
        Returns:
            Aggregated metrics
        """
        all_metrics = []
        
        for client_id in range(self.num_clients):
            client_data = self.all_clients_data[client_id]
            qa_model = self.client_qa_models[client_id]
            entity_embedding = self.client_entity_embeddings[client_id].to(self.args.gpu)
            relation_embedding = self.global_relation_embedding.to(self.args.gpu)
            dev_loader = self.dev_qa_loaders[client_id]
            
            metrics = evaluate_qa(
                qa_model,
                dev_loader,
                relation_embedding,
                entity_embedding,
                client_data['entity2id'],
                client_data['id2entity'],
                self.args
            )
            
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {
            'avg_hits@1': sum(m['hits@1'] for m in all_metrics) / len(all_metrics),
            'avg_hits@3': sum(m['hits@3'] for m in all_metrics) / len(all_metrics),
            'avg_hits@5': sum(m['hits@5'] for m in all_metrics) / len(all_metrics),
            'avg_hits@10': sum(m['hits@10'] for m in all_metrics) / len(all_metrics),
            'avg_mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics),
            'per_client': all_metrics
        }
        
        return avg_metrics
    
    def save_best_models(self):
        """Save best models to disk"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save global relation embeddings
        torch.save(
            self.global_relation_embedding,
            os.path.join(save_dir, 'global_relation_embeddings.pt')
        )
        
        # Save KGE model name
        with open(os.path.join(save_dir, 'kge_model.txt'), 'w') as f:
            f.write(self.kge_model_name)
        
        # Save Language model name
        lm_model_name = self.args.lm_model.lower() if hasattr(self.args, 'lm_model') else 'roberta'
        with open(os.path.join(save_dir, 'lm_model.txt'), 'w') as f:
            f.write(lm_model_name)
        
        # Save client models
        for client_id in range(self.num_clients):
            client_save_dir = os.path.join(save_dir, f'client_{client_id}')
            os.makedirs(client_save_dir, exist_ok=True)
            
            # Save entity embeddings
            torch.save(
                self.client_entity_embeddings[client_id],
                os.path.join(client_save_dir, 'entity_embeddings.pt')
            )
            
            # Save QA model
            torch.save(
                self.client_qa_models[client_id].state_dict(),
                os.path.join(client_save_dir, 'qa_model.pt')
            )
        
        logging.info(f"Best models saved to {save_dir}")
    
    def load_best_models(self):
        """Load best models from disk"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        
        # Load global relation embeddings
        self.global_relation_embedding = torch.load(
            os.path.join(save_dir, 'global_relation_embeddings.pt')
        )
        
        # Load client models
        for client_id in range(self.num_clients):
            client_save_dir = os.path.join(save_dir, f'client_{client_id}')
            
            # Load entity embeddings
            self.client_entity_embeddings[client_id] = torch.load(
                os.path.join(client_save_dir, 'entity_embeddings.pt')
            )
            
            # Load QA model
            self.client_qa_models[client_id].load_state_dict(
                torch.load(os.path.join(client_save_dir, 'qa_model.pt'))
            )
        
        logging.info(f"Best models loaded from {save_dir}")
    
    def train(self):
        """
        Main training loop: run both phases sequentially
        """
        # Phase 1: KG Embedding Training
        self.train_phase1_kg()
        
        # Phase 2: QA Model Training
        self.train_phase2_qa()
        
        logging.info("\nTraining completed!")
        logging.info("Best Dev Metrics:")
        if 'avg_hits@5' in self.best_dev_metrics:
            logging.info(f"  Hits@3:  {self.best_dev_metrics.get('avg_hits@3', 0):.4f}")
            logging.info(f"  Hits@5:  {self.best_dev_metrics['avg_hits@5']:.4f}")
            logging.info(f"  Hits@10: {self.best_dev_metrics.get('avg_hits@10', 0):.4f}")
            logging.info(f"  MRR:     {self.best_dev_metrics['avg_mrr']:.4f}")
        
        logging.info(f"\nModels saved to: {os.path.join(self.args.state_dir, 'best_models')}")
