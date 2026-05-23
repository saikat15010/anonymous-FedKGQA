"""
Federated KGQA Training System for PathQuestion Dataset

Implements two-phase federated learning:
Phase 1: Federated KG Embedding Training (RotatE with global relations)
Phase 2: Federated QA Model Training (BERT/DistilBERT/RoBERTa with local QA pairs)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import copy
from typing import List, Dict

from kge_model_rotate import (
    RotatEModel,
    initialize_embeddings,
    compute_kg_loss,
    evaluate_kg
)


class FederatedKGQA:
    """
    Federated KGQA system with two-phase training
    Works with BERT, DistilBERT, or RoBERTa
    """
    
    def __init__(self, args, all_clients_data, global_nrelation, qa_model_module):
        """
        Initialize federated KGQA system
        
        Args:
            args: Training arguments
            all_clients_data: List of client data dicts
            global_nrelation: Number of global relations
            qa_model_module: QA model module (qa_model_bert, qa_model_distilbert, or qa_model_roberta)
        """
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        self.qa_model_module = qa_model_module
        
        # Initialize global relation embeddings (shared across clients)
        # RotatE: relations are phase rotations, size = hidden_dim
        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        global_relation_init = torch.zeros(global_nrelation, args.hidden_dim).uniform_(-embedding_range, embedding_range)
        self.global_relation_embedding = nn.Parameter(global_relation_init, requires_grad=True)
        
        # Client-specific components
        self.client_entity_embeddings = []
        self.client_kg_models = []
        self.client_qa_models = []
        self.client_optimizers_kg = []
        self.client_optimizers_qa = []
        
        # Training state
        self.current_phase = None
        self.best_dev_metrics = {
            'avg_hits@1': 0.0,
            'avg_hits@3': 0.0,
            'avg_hits@5': 0.0,
            'avg_hits@10': 0.0,
            'avg_mrr': 0.0
        }
        
        logging.info(f"Initialized Federated KGQA with {self.num_clients} clients")
        logging.info(f"Global relations: {global_nrelation}")
        logging.info(f"QA Model: {qa_model_module.__name__}")
    
    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders):
        """
        Setup client models and dataloaders
        """
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders
        
        # Move global relation embedding to device FIRST
        self.global_relation_embedding.data = self.global_relation_embedding.data.to(self.args.gpu)
        
        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            
            # Initialize entity embeddings (RotatE: 2*hidden_dim for complex embeddings)
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim
            entity_embedding_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            # RotatE does not require normalization
            entity_embedding = nn.Parameter(entity_embedding_init, requires_grad=True)
            self.client_entity_embeddings.append(entity_embedding)
            
            # Initialize KG model (RotatE)
            kg_model = RotatEModel(self.args)
            kg_model = kg_model.to(self.args.gpu)
            self.client_kg_models.append(kg_model)
            
            # Initialize QA model
            qa_model = self.qa_model_module.ImprovedKGQAModel(self.args, nentity, self.global_nrelation)
            qa_model = qa_model.to(self.args.gpu)
            self.client_qa_models.append(qa_model)
            
            logging.info(f"Client {i} setup: {nentity} entities")
            
            # Initialize optimizers for KG training (Phase 1)
            kg_params = [entity_embedding]
            kg_optimizer = optim.Adam(kg_params, lr=self.args.lr)
            self.client_optimizers_kg.append(kg_optimizer)
            
            # Initialize optimizers for QA training (Phase 2)
            qa_params = list(qa_model.parameters())
            qa_optimizer = optim.Adam(qa_params, lr=self.args.qa_lr)
            self.client_optimizers_qa.append(qa_optimizer)
        
        # Global relation optimizer (for Phase 1)
        self.global_relation_optimizer = optim.Adam([self.global_relation_embedding], lr=self.args.lr)
        
        logging.info("All client optimizers initialized successfully")
    
    def train_client_kg(self, client_id, dataloader):
        """Train KG embeddings on a single client"""
        client_model = self.client_kg_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]
        
        client_model.train()
        total_loss = 0
        num_batches = 0
        
        for positive_sample, negative_sample, _ in dataloader:
            positive_sample = positive_sample.to(self.args.gpu)
            negative_sample = negative_sample.to(self.args.gpu)
            
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_kg_loss(
                client_model,
                positive_sample,
                negative_sample,
                entity_embedding,
                self.global_relation_embedding,
                self.args
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train_phase1_kg(self):
        """Phase 1: Federated KG Embedding Training"""
        self.current_phase = 1
        
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 1: Federated KG Embedding Training (RotatE)")
        logging.info("=" * 70)
        
        best_avg_loss = float('inf')
        patience_counter = 0
        
        for round_idx in range(self.args.kg_max_rounds):
            # Sample clients (can be all or a fraction)
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training on selected clients
            client_losses = []
            relation_gradients = []
            
            for client_id in selected_clients:
                # Train for local_epoch epochs
                for _ in range(self.args.local_epoch):
                    loss = self.train_client_kg(client_id, self.kg_dataloaders[client_id])
                    client_losses.append(loss)
                
                # Collect relation gradient
                if self.global_relation_embedding.grad is not None:
                    relation_gradients.append(self.global_relation_embedding.grad.clone())
            
            # Aggregate relation gradients (FedAvg)
            if len(relation_gradients) > 0:
                avg_relation_grad = torch.stack(relation_gradients).mean(dim=0)
                self.global_relation_embedding.grad = avg_relation_grad
                self.global_relation_optimizer.step()
                self.global_relation_optimizer.zero_grad()
            
            # Log
            avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[KG Round {round_idx + 1}/{self.args.kg_max_rounds}] "
                           f"Clients: {selected_clients}, Avg Loss: {avg_loss:.4f}")
            
            # Early stopping based on loss
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
    
    def train_client_qa(self, client_id, dataloader):
        """Train QA model on a single client"""
        client_model = self.client_qa_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_qa[client_id]
        client_data = self.all_clients_data[client_id]
        
        client_model.train()
        total_loss = 0
        num_batches = 0
        
        for questions, answer_ids, _ in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            entity_scores, _, _ = client_model(
                questions,
                self.global_relation_embedding,
                entity_embedding,
                entity2id=client_data['entity2id'],
                answer_ids=answer_ids
            )
            
            # Compute loss
            loss = self.qa_model_module.compute_qa_loss(entity_scores, answer_ids, self.args)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate_dev_set(self):
        """Evaluate on development sets across all clients"""
        all_metrics = []
        
        for client_id in range(self.num_clients):
            client_model = self.client_qa_models[client_id]
            entity_embedding = self.client_entity_embeddings[client_id]
            dev_loader = self.dev_qa_loaders[client_id]
            client_data = self.all_clients_data[client_id]
            
            metrics = self.qa_model_module.evaluate_qa(
                client_model,
                dev_loader,
                self.global_relation_embedding,
                entity_embedding,
                client_data['entity2id'],
                client_data['id2entity'],
                self.args
            )
            
            all_metrics.append(metrics)
        
        # Average metrics across clients
        avg_metrics = {
            'avg_hits@1': sum(m['hits@1'] for m in all_metrics) / len(all_metrics),
            'avg_hits@3': sum(m['hits@3'] for m in all_metrics) / len(all_metrics),
            'avg_hits@5': sum(m['hits@5'] for m in all_metrics) / len(all_metrics),
            'avg_hits@10': sum(m['hits@10'] for m in all_metrics) / len(all_metrics),
            'avg_mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics)
        }
        
        return avg_metrics
    
    def aggregate_qa_models(self, selected_clients):
        """Aggregate QA model parameters using FedAvg"""
        # Get state dicts from selected clients
        client_states = [self.client_qa_models[i].state_dict() for i in selected_clients]
        
        # Average parameters
        avg_state = {}
        for key in client_states[0].keys():
            avg_state[key] = torch.stack([state[key].float() for state in client_states]).mean(dim=0)
        
        # Update all client models
        for client_id in range(self.num_clients):
            self.client_qa_models[client_id].load_state_dict(avg_state)
    
    def train_phase2_qa(self):
        """Phase 2: Federated QA Model Training"""
        self.current_phase = 2
        
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Federated QA Model Training ({self.qa_model_module.__name__.upper()})")
        logging.info("=" * 70)
        
        best_hits_at_5 = 0
        patience_counter = 0
        
        for round_idx in range(self.args.qa_max_rounds):
            # Sample clients
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training
            client_losses = []
            for client_id in selected_clients:
                for _ in range(self.args.qa_local_epoch):
                    loss = self.train_client_qa(client_id, self.train_qa_loaders[client_id])
                    client_losses.append(loss)
            
            # Aggregate models (FedAvg)
            self.aggregate_qa_models(selected_clients)
            
            # Log
            avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(f"[QA Round {round_idx + 1}/{self.args.qa_max_rounds}] "
                           f"Clients: {selected_clients}, Avg Loss: {avg_loss:.4f}")
            
            # Evaluate on dev set
            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_dev_set()
                
                logging.info(f"Dev Metrics: Hits@1={dev_metrics['avg_hits@1']:.4f}, "
                           f"Hits@3={dev_metrics['avg_hits@3']:.4f}, "
                           f"Hits@5={dev_metrics['avg_hits@5']:.4f}, "
                           f"Hits@10={dev_metrics['avg_hits@10']:.4f}, "
                           f"MRR={dev_metrics['avg_mrr']:.4f}")
                
                # Save best model
                if dev_metrics['avg_hits@5'] > best_hits_at_5:
                    best_hits_at_5 = dev_metrics['avg_hits@5']
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    patience_counter = 0
                    logging.info(f"New best Hits@5: {best_hits_at_5:.4f} - Model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= self.args.early_stop_patience:
                        logging.info(f"Early stopping at round {round_idx + 1}")
                        break
        
        logging.info("Phase 2 (QA Training) completed!")
        logging.info(f"Best Dev Hits@5: {best_hits_at_5:.4f}")
    
    def save_best_models(self):
        """Save best models to disk"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save global relation embeddings
        torch.save(
            self.global_relation_embedding.cpu(),
            os.path.join(save_dir, 'global_relation_embeddings.pt')
        )
        self.global_relation_embedding.data = self.global_relation_embedding.data.to(self.args.gpu)
        
        # Save client models
        for client_id in range(self.num_clients):
            client_save_dir = os.path.join(save_dir, f'client_{client_id}')
            os.makedirs(client_save_dir, exist_ok=True)
            
            # Save entity embeddings
            torch.save(
                self.client_entity_embeddings[client_id].cpu(),
                os.path.join(client_save_dir, 'entity_embeddings.pt')
            )
            self.client_entity_embeddings[client_id].data = self.client_entity_embeddings[client_id].data.to(self.args.gpu)
            
            # Save QA model
            torch.save(
                self.client_qa_models[client_id].state_dict(),
                os.path.join(client_save_dir, 'qa_model.pt')
            )
        
        logging.info(f"Models saved to {save_dir}")
    
    def train(self):
        """Main training loop"""
        # Phase 1: KG Training
        self.train_phase1_kg()
        
        # Phase 2: QA Training
        self.train_phase2_qa()
        
        logging.info("\n" + "=" * 70)
        logging.info("Training Complete!")
        logging.info("=" * 70)
        if self.best_dev_metrics.get('avg_hits@1', 0) > 0:
            # Metrics were computed
            logging.info(f"Best Dev Metrics:")
            logging.info(f"  Hits@1:  {self.best_dev_metrics['avg_hits@1']:.4f}")
            logging.info(f"  Hits@3:  {self.best_dev_metrics['avg_hits@3']:.4f}")
            logging.info(f"  Hits@5:  {self.best_dev_metrics['avg_hits@5']:.4f}")
            logging.info(f"  Hits@10: {self.best_dev_metrics['avg_hits@10']:.4f}")
            logging.info(f"  MRR:     {self.best_dev_metrics['avg_mrr']:.4f}")
        else:
            logging.info("Note: No evaluation was performed (increase qa_max_rounds or decrease check_per_round)")
            logging.info("Run evaluation script to get metrics on test set.")
