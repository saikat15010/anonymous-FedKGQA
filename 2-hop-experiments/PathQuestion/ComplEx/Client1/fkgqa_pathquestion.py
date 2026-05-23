"""
Federated KGQA Training System for PathQuestion Dataset

Implements two-phase federated learning:
Phase 1: Federated KG Embedding Training (ComplEx with global relations)
Phase 2: Federated QA Model Training (BERT/DistilBERT/RoBERTa with local QA pairs)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from typing import List, Dict

from kge_model_complex import (
    ComplExModel,
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
        # ComplEx: relations are complex embeddings, size = 2*hidden_dim
        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        global_relation_init = torch.zeros(global_nrelation, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
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
            
            # Initialize entity embeddings (ComplEx: 2*hidden_dim)
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim
            entity_embedding_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            entity_embedding = nn.Parameter(entity_embedding_init, requires_grad=True)
            self.client_entity_embeddings.append(entity_embedding)
            
            # Initialize KG model (ComplEx)
            kg_model = ComplExModel(self.args)
            kg_model = kg_model.to(self.args.gpu)
            self.client_kg_models.append(kg_model)
            
            # Initialize QA model
            qa_model = self.qa_model_module.ImprovedKGQAModel(self.args, nentity, self.global_nrelation)
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
        """Phase 1: Federated KG Embedding Training"""
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 1: Federated KG Embedding Training (ComplEx)")
        logging.info("=" * 70)
        
        self.current_phase = 'kg'
        
        patience_counter = 0
        best_avg_loss = float('inf')
        
        for round_idx in range(self.args.kg_max_rounds):
            # Sample clients
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training
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
        """Train KG embeddings for a single client"""
        kg_model = self.client_kg_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]
        dataloader = self.kg_dataloaders[client_id]
        
        kg_model.train()
        total_loss = 0
        relation_grads = []
        
        for epoch in range(self.args.local_epoch):
            for batch_idx, (positive_sample, negative_sample, sample_idx) in enumerate(dataloader):
                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)
                
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
                optimizer.zero_grad()
                self.global_relation_optimizer.zero_grad()
                loss.backward()
                
                # Collect relation gradient
                if self.global_relation_embedding.grad is not None:
                    relation_grads.append(self.global_relation_embedding.grad.clone())
                
                # Update entity embeddings
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(dataloader) * self.args.local_epoch) if len(dataloader) > 0 else 0
        avg_relation_grad = torch.stack(relation_grads).mean(dim=0) if relation_grads else torch.zeros_like(self.global_relation_embedding)
        
        return avg_loss, avg_relation_grad
    
    def train_phase2_qa(self):
        """Phase 2: Federated QA Model Training"""
        logging.info("\n" + "=" * 70)
        logging.info(f"PHASE 2: Federated QA Model Training ({self.qa_model_module.__name__.upper()})")
        logging.info("=" * 70)
        
        self.current_phase = 'qa'
        
        patience_counter = 0
        best_dev_score = 0.0
        
        for round_idx in range(self.args.qa_max_rounds):
            # Sample clients
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            # Local training
            local_models = []
            local_losses = []
            
            for client_id in selected_clients:
                loss, model_state = self.train_client_qa(client_id)
                local_losses.append(loss)
                local_models.append(model_state)
            
            # Federated averaging
            self.federated_averaging_qa(local_models, selected_clients)
            
            # Logging
            avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"[QA Round {round_idx + 1}/{self.args.qa_max_rounds}] "
                    f"Clients: {selected_clients}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Evaluation
            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_dev()
                
                logging.info(
                    f"[Dev Eval] Round {round_idx + 1} → "
                    f"Avg Hits@3: {dev_metrics['avg_hits@3']:.4f}, "
                    f"Avg Hits@5: {dev_metrics['avg_hits@5']:.4f}, "
                    f"Avg Hits@10: {dev_metrics['avg_hits@10']:.4f}, "
                    f"Avg MRR: {dev_metrics['avg_mrr']:.4f}"
                )
                
                # Save best model
                if dev_metrics['avg_hits@5'] > best_dev_score:
                    best_dev_score = dev_metrics['avg_hits@5']
                    patience_counter = 0
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    logging.info(f"✓ New best model saved! Hits@5: {best_dev_score:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.args.early_stop_patience:
                    logging.info(f"Early stopping at round {round_idx + 1}")
                    break
        
        logging.info("Phase 2 (QA Training) completed!")
        logging.info(f"Best Dev Hits@5: {best_dev_score:.4f}")
    
    def train_client_qa(self, client_id):
        """Train QA model for a single client"""
        client_data = self.all_clients_data[client_id]
        qa_model = self.client_qa_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_qa[client_id]
        dataloader = self.train_qa_loaders[client_id]
        
        qa_model.train()
        total_loss = 0
        
        # Freeze KG embeddings
        entity_embedding.requires_grad = False
        self.global_relation_embedding.requires_grad = False
        
        for epoch in range(self.args.qa_local_epoch):
            for batch_idx, (questions, answer_ids, hop_counts) in enumerate(dataloader):
                # Forward pass
                entity_scores, relation_scores, topic_entity_ids = qa_model(
                    questions,
                    self.global_relation_embedding,
                    entity_embedding,
                    entity2id=client_data['entity2id'],
                    answer_ids=answer_ids
                )
                
                # Compute loss
                loss = self.qa_model_module.compute_qa_loss(entity_scores, answer_ids, self.args)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Unfreeze for next phase
        entity_embedding.requires_grad = True
        self.global_relation_embedding.requires_grad = True
        
        avg_loss = total_loss / (len(dataloader) * self.args.qa_local_epoch) if len(dataloader) > 0 else 0
        model_state = copy.deepcopy(qa_model.state_dict())
        
        return avg_loss, model_state
    
    def federated_averaging_qa(self, local_models, selected_clients):
        """FedAvg for QA models"""
        if not local_models:
            return
        
        # Initialize global model state
        global_state = copy.deepcopy(local_models[0])
        
        # Average parameters
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            for model_state in local_models:
                global_state[key] += model_state[key]
            global_state[key] /= len(local_models)
        
        # Update all client models
        for client_id in range(self.num_clients):
            self.client_qa_models[client_id].load_state_dict(global_state)
    
    def evaluate_dev(self):
        """Evaluate on development set"""
        all_metrics = []
        
        for client_id in range(self.num_clients):
            client_data = self.all_clients_data[client_id]
            qa_model = self.client_qa_models[client_id]
            entity_embedding = self.client_entity_embeddings[client_id]
            dataloader = self.dev_qa_loaders[client_id]
            
            metrics = self.qa_model_module.evaluate_qa(
                qa_model,
                dataloader,
                self.global_relation_embedding,
                entity_embedding,
                client_data['entity2id'],
                client_data['id2entity'],
                self.args
            )
            
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            'avg_hits@1': sum(m['hits@1'] for m in all_metrics) / len(all_metrics),
            'avg_hits@3': sum(m['hits@3'] for m in all_metrics) / len(all_metrics),
            'avg_hits@5': sum(m['hits@5'] for m in all_metrics) / len(all_metrics),
            'avg_hits@10': sum(m['hits@10'] for m in all_metrics) / len(all_metrics),
            'avg_mrr': sum(m['mrr'] for m in all_metrics) / len(all_metrics),
            'client_metrics': all_metrics
        }
        
        return avg_metrics
    
    def save_best_models(self):
        """Save best models"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save global relation embeddings
        torch.save(
            {'global_relation_embedding': self.global_relation_embedding},
            os.path.join(save_dir, 'global_relation_embeddings.pt')
        )
        
        # Save each client's models
        for client_id in range(self.num_clients):
            client_save_dir = os.path.join(save_dir, f'client_{client_id}')
            os.makedirs(client_save_dir, exist_ok=True)
            
            torch.save(
                {'entity_embedding': self.client_entity_embeddings[client_id]},
                os.path.join(client_save_dir, 'entity_embeddings.pt')
            )
            
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