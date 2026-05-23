"""
Federated KGQA Training System with Differential Privacy

Implements DP through:
- Gradient clipping (L2 norm)
- Gaussian noise addition (calibrated by sigma)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from typing import List, Dict

from kge_model_updated import (
    ComplExModel,
    initialize_embeddings,
    compute_kg_loss,
    evaluate_kg
)
from qa_model_updated import (
    ImprovedKGQAModel,
    compute_qa_loss,
    evaluate_qa
)


class FederatedKGQA_DP:
    """
    Federated KGQA system with Differential Privacy
    """
    
    def __init__(self, args, all_clients_data, global_nrelation):
        """
        Initialize federated KGQA system with DP
        
        Args:
            args: Training arguments (including DP parameters)
            all_clients_data: List of client data dicts
            global_nrelation: Number of global relations
        """
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        
        # DP parameters
        self.dp_enabled = args.dp_enabled
        self.dp_sigma = args.dp_sigma
        self.dp_clipping_bound = args.dp_clipping_bound
        
        if self.dp_enabled:
            logging.info(f"Differential Privacy ENABLED")
            logging.info(f"  Sigma: {self.dp_sigma}")
            logging.info(f"  Clipping Bound: {self.dp_clipping_bound}")
        
        # Initialize global relation embeddings
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
        self.best_dev_metrics = {'avg_hits@5': 0.0}
        
        logging.info(f"Initialized Federated KGQA with {self.num_clients} clients")
        logging.info(f"Global relations: {global_nrelation}")
    
    def clip_gradient(self, gradient, clipping_bound):
        """
        Clip gradient to maximum L2 norm
        
        Args:
            gradient: Gradient tensor
            clipping_bound: Maximum L2 norm
        
        Returns:
            Clipped gradient
        """
        grad_norm = torch.norm(gradient, p=2)
        if grad_norm > clipping_bound:
            gradient = gradient * (clipping_bound / grad_norm)
        return gradient
    
    def add_gaussian_noise(self, gradient, sigma, clipping_bound):
        """
        Add Gaussian noise to gradient for DP
        
        Args:
            gradient: Gradient tensor
            sigma: Noise scale parameter
            clipping_bound: Clipping bound (for noise calibration)
        
        Returns:
            Noisy gradient
        """
        noise_std = sigma * clipping_bound
        noise = torch.randn_like(gradient) * noise_std
        return gradient + noise
    
    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders, test_qa_loaders=None):
        """Setup client models and dataloaders"""
        self.kg_dataloaders = kg_dataloaders
        self.train_qa_loaders = train_qa_loaders
        self.dev_qa_loaders = dev_qa_loaders
        self.test_qa_loaders = test_qa_loaders
        
        # Move global relation embedding to device
        self.global_relation_embedding.data = self.global_relation_embedding.data.to(self.args.gpu)
        
        for i, client_data in enumerate(self.all_clients_data):
            nentity = client_data['nentity']
            
            # Initialize entity embeddings
            embedding_range = (self.args.gamma + self.args.epsilon) / self.args.hidden_dim
            entity_embedding_init = torch.zeros(nentity, self.args.hidden_dim * 2, device=self.args.gpu).uniform_(-embedding_range, embedding_range)
            entity_embedding = nn.Parameter(entity_embedding_init, requires_grad=True)
            self.client_entity_embeddings.append(entity_embedding)
            
            # Initialize KG model
            kg_model = ComplExModel(self.args)
            kg_model = kg_model.to(self.args.gpu)
            self.client_kg_models.append(kg_model)
            
            # Initialize QA model
            qa_model = ImprovedKGQAModel(self.args, nentity, self.global_nrelation)
            qa_model = qa_model.to(self.args.gpu)
            self.client_qa_models.append(qa_model)
            
            logging.info(f"Client {i} setup: {nentity} entities")
        
        # Initialize optimizers
        for i in range(self.num_clients):
            kg_optimizer = optim.Adam([self.client_entity_embeddings[i]], lr=self.args.lr)
            self.client_optimizers_kg.append(kg_optimizer)
            
            qa_optimizer = optim.Adam(self.client_qa_models[i].parameters(), lr=self.args.qa_lr)
            self.client_optimizers_qa.append(qa_optimizer)
        
        # Global relation optimizer
        self.global_relation_optimizer = optim.Adam([self.global_relation_embedding], lr=self.args.lr)
        
        logging.info("All client optimizers initialized successfully")
    
    def train_phase1_kg(self):
        """Phase 1: Federated KG Embedding Training with DP"""
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 1: Federated KG Embedding Training (ComplEx)")
        if self.dp_enabled:
            logging.info(f"  DP: sigma={self.dp_sigma}, clipping_bound={self.dp_clipping_bound}")
        logging.info("=" * 70)
        
        self.current_phase = 'kg'
        
        patience_counter = 0
        best_avg_loss = float('inf')
        
        for round_idx in range(self.args.kg_max_rounds):
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            local_relation_grads = []
            local_losses = []
            
            for client_id in selected_clients:
                loss, rel_grad = self.train_client_kg(client_id)
                local_losses.append(loss)
                local_relation_grads.append(rel_grad)
            
            # Federated aggregation with DP
            if local_relation_grads:
                avg_relation_grad = torch.stack(local_relation_grads).mean(dim=0)
                
                # Apply DP if enabled
                if self.dp_enabled:
                    avg_relation_grad = self.clip_gradient(avg_relation_grad, self.dp_clipping_bound)
                    avg_relation_grad = self.add_gaussian_noise(avg_relation_grad, self.dp_sigma, self.dp_clipping_bound)
                
                self.global_relation_optimizer.zero_grad()
                self.global_relation_embedding.grad = avg_relation_grad.clone()
                self.global_relation_optimizer.step()
            
            avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"[KG Round {round_idx + 1}/{self.args.kg_max_rounds}] "
                    f"Clients: {selected_clients}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
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
        """Train KG embeddings for a single client with DP"""
        client_data = self.all_clients_data[client_id]
        kg_model = self.client_kg_models[client_id]
        entity_embedding = self.client_entity_embeddings[client_id]
        optimizer = self.client_optimizers_kg[client_id]
        dataloader = self.kg_dataloaders[client_id]
        
        kg_model.train()
        total_loss = 0
        
        # Accumulate relation gradients
        accumulated_relation_grad = torch.zeros_like(self.global_relation_embedding.data)
        num_batches = 0
        
        for epoch in range(self.args.local_epoch):
            for batch_idx, (positive_sample, negative_sample, sample_idx) in enumerate(dataloader):
                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)
                
                # Enable gradients for relation embeddings temporarily
                self.global_relation_embedding.requires_grad = True
                
                loss = compute_kg_loss(
                    kg_model,
                    positive_sample,
                    negative_sample,
                    entity_embedding,
                    self.global_relation_embedding,
                    self.args
                )
                
                optimizer.zero_grad()
                self.global_relation_optimizer.zero_grad()
                
                loss.backward()
                
                # Clip and accumulate relation gradient
                if self.global_relation_embedding.grad is not None:
                    rel_grad = self.global_relation_embedding.grad.clone()
                    if self.dp_enabled:
                        rel_grad = self.clip_gradient(rel_grad, self.dp_clipping_bound)
                    accumulated_relation_grad += rel_grad
                    num_batches += 1
                
                # Update entity embeddings only
                optimizer.step()
                
                total_loss += loss.item()
                
                # Clear relation gradients
                self.global_relation_embedding.grad = None
        
        # Average accumulated relation gradient
        if num_batches > 0:
            avg_relation_grad = accumulated_relation_grad / num_batches
        else:
            avg_relation_grad = accumulated_relation_grad
        
        avg_loss = total_loss / (len(dataloader) * self.args.local_epoch) if len(dataloader) > 0 else 0
        
        return avg_loss, avg_relation_grad
    
    def train_phase2_qa(self):
        """Phase 2: Federated QA Model Training with DP"""
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 2: Federated QA Model Training (RoBERTa)")
        if self.dp_enabled:
            logging.info(f"  DP: sigma={self.dp_sigma}, clipping_bound={self.dp_clipping_bound}")
        logging.info("=" * 70)
        
        self.current_phase = 'qa'
        
        patience_counter = 0
        
        for round_idx in range(self.args.qa_max_rounds):
            num_selected = max(1, int(self.args.fraction * self.num_clients))
            selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
            
            local_models = []
            local_losses = []
            
            for client_id in selected_clients:
                loss, model_state = self.train_client_qa(client_id)
                local_losses.append(loss)
                local_models.append(model_state)
            
            # Federated averaging with DP
            self.federated_averaging_qa_with_dp(local_models, selected_clients)
            
            avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"[QA Round {round_idx + 1}/{self.args.qa_max_rounds}] "
                    f"Clients: {selected_clients}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Evaluate on dev set
            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_dev()
                logging.info(
                    f"[Dev Evaluation] "
                    f"Hits@3: {dev_metrics['avg_hits@3']:.4f}, "
                    f"Hits@5: {dev_metrics['avg_hits@5']:.4f}, "
                    f"Hits@10: {dev_metrics['avg_hits@10']:.4f}, "
                    f"MRR: {dev_metrics['avg_mrr']:.4f}"
                )
                
                if dev_metrics['avg_hits@5'] > self.best_dev_metrics['avg_hits@5']:
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    logging.info("  -> New best model saved!")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.args.early_stop_patience:
                    logging.info(f"Early stopping at round {round_idx + 1}")
                    break
        
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
        """Train QA model for a single client"""
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
                entity_scores, relation_scores, topic_entity_ids = qa_model(
                    questions,
                    relation_embedding,
                    entity_embedding,
                    client_data['entity2id'],
                    answer_ids
                )
                
                loss = compute_qa_loss(entity_scores, answer_ids, self.args)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping for DP
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(qa_model.parameters(), self.dp_clipping_bound)
                
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(dataloader) * self.args.qa_local_epoch) if len(dataloader) > 0 else 0
        model_state = copy.deepcopy(qa_model.state_dict())
        
        return avg_loss, model_state
    
    def federated_averaging_qa_with_dp(self, local_models, selected_clients):
        """Federated averaging with DP noise"""
        if not local_models:
            return
        
        # Average model parameters
        global_state = {}
        for key in local_models[0].keys():
            stacked = torch.stack([m[key].float() for m in local_models])
            averaged = stacked.mean(dim=0)
            
            # Add DP noise if enabled
            if self.dp_enabled:
                noise_std = self.dp_sigma * self.dp_clipping_bound / len(local_models)
                noise = torch.randn_like(averaged) * noise_std
                averaged = averaged + noise
            
            global_state[key] = averaged
        
        # Update all client models
        for client_id in range(self.num_clients):
            self.client_qa_models[client_id].load_state_dict(global_state, strict=False)
    
    def evaluate_dev(self):
        """Evaluate all clients on dev sets"""
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
        
        torch.save(
            self.global_relation_embedding,
            os.path.join(save_dir, 'global_relation_embeddings.pt')
        )
        
        for client_id in range(self.num_clients):
            client_save_dir = os.path.join(save_dir, f'client_{client_id}')
            os.makedirs(client_save_dir, exist_ok=True)
            
            torch.save(
                self.client_entity_embeddings[client_id],
                os.path.join(client_save_dir, 'entity_embeddings.pt')
            )
            
            torch.save(
                self.client_qa_models[client_id].state_dict(),
                os.path.join(client_save_dir, 'qa_model.pt')
            )
        
        logging.info(f"Best models saved to {save_dir}")
    
    def train(self):
        """Main training loop"""
        self.train_phase1_kg()
        self.train_phase2_qa()
        
        logging.info("\nTraining completed!")
        logging.info("Best Dev Metrics:")
        if 'avg_hits@5' in self.best_dev_metrics:
            logging.info(f"  Hits@3:  {self.best_dev_metrics.get('avg_hits@3', 0):.4f}")
            logging.info(f"  Hits@5:  {self.best_dev_metrics['avg_hits@5']:.4f}")
            logging.info(f"  Hits@10: {self.best_dev_metrics.get('avg_hits@10', 0):.4f}")
            logging.info(f"  MRR:     {self.best_dev_metrics['avg_mrr']:.4f}")
        
        logging.info(f"\nModels saved to: {os.path.join(self.args.state_dir, 'best_models')}")
