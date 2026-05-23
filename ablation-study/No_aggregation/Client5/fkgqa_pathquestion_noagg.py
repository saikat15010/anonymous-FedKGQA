"""
Federated KGQA for PathQuestion Dataset
ABLATION STUDY: NO AGGREGATION OF RELATION EMBEDDINGS

Key differences from standard version:
- Each client maintains separate relation embeddings
- No FedAvg on relation embeddings
- Only entity embeddings are local to each client
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from collections import OrderedDict

from kge_model_complex import (
    ComplExModel,
    initialize_embeddings,
    compute_kg_loss,
    evaluate_kg
)


class FederatedKGQA:
    """
    Federated KGQA System - NO RELATION AGGREGATION
    
    Each client has:
    - Local entity embeddings (not shared)
    - Local relation embeddings (NOT aggregated)
    - Local QA model (aggregated)
    """
    
    def __init__(self, args, all_clients_data, global_nrelation, qa_model_module):
        self.args = args
        self.all_clients_data = all_clients_data
        self.num_clients = len(all_clients_data)
        self.global_nrelation = global_nrelation
        self.qa_model_module = qa_model_module
        self.device = args.gpu
        
        # Track best models
        self.best_dev_metric = 0.0
        self.best_round = 0
        self.patience_counter = 0
        
        logging.info(f"Initialized Federated KGQA - NO RELATION AGGREGATION")
        logging.info(f"Number of clients: {self.num_clients}")
        logging.info(f"Global relations: {self.global_nrelation}")
    
    def setup_clients(self, kg_dataloaders, train_qa_loaders, dev_qa_loaders):
        """Initialize all clients with their models and data"""
        self.clients = []
        
        for i in range(self.num_clients):
            client = FederatedClient(
                client_id=i,
                client_data=self.all_clients_data[i],
                global_nrelation=self.global_nrelation,
                args=self.args,
                kg_dataloader=kg_dataloaders[i],
                train_qa_loader=train_qa_loaders[i],
                dev_qa_loader=dev_qa_loaders[i],
                qa_model_module=self.qa_model_module,
                device=self.device
            )
            self.clients.append(client)
            
            logging.info(
                f"Client {i}: {client.nentity} entities, "
                f"{client.nrelation} local relations, "
                f"{len(client.client_data['triples'])} triples"
            )
    
    def train(self):
        """Main federated training loop"""
        
        # Phase 1: Train KG embeddings (ComplEx)
        logging.info("\n" + "="*70)
        logging.info("PHASE 1: Training Knowledge Graph Embeddings (ComplEx)")
        logging.info("NO RELATION AGGREGATION - Each client maintains separate embeddings")
        logging.info("="*70)
        
        for round_idx in range(self.args.kg_max_rounds):
            round_losses = []
            
            # Each client trains locally on KG
            for client in self.clients:
                loss = client.train_kg_local()
                round_losses.append(loss)
            
            avg_loss = np.mean(round_losses)
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"KG Round {round_idx + 1}/{self.args.kg_max_rounds} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # NO AGGREGATION - Skip relation embedding aggregation
            # Each client keeps its own relation embeddings
        
        logging.info("KG training completed - Each client has separate relation embeddings")
        
        # Phase 2: Train QA models
        logging.info("\n" + "="*70)
        logging.info("PHASE 2: Training Question Answering Models")
        logging.info("QA models ARE aggregated (only QA parameters, not KG embeddings)")
        logging.info("="*70)
        
        for round_idx in range(self.args.qa_max_rounds):
            round_losses = []
            
            # Local training
            for client in self.clients:
                loss = client.train_qa_local()
                round_losses.append(loss)
            
            # Aggregate QA models only (not relation embeddings)
            self.aggregate_qa_models()
            
            avg_loss = np.mean(round_losses)
            
            if (round_idx + 1) % self.args.log_per_round == 0:
                logging.info(
                    f"QA Round {round_idx + 1}/{self.args.qa_max_rounds} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Evaluation
            if (round_idx + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_on_dev()
                
                logging.info(
                    f"\n--- Dev Evaluation (Round {round_idx + 1}) ---"
                )
                logging.info(
                    f"Hits@5: {dev_metrics['hits@5']:.4f} | "
                    f"Hits@10: {dev_metrics['hits@10']:.4f} | "
                    f"MRR: {dev_metrics['mrr']:.4f}"
                )
                
                # Save best model
                current_metric = dev_metrics['hits@5']
                if current_metric > self.best_dev_metric:
                    self.best_dev_metric = current_metric
                    self.best_round = round_idx + 1
                    self.patience_counter = 0
                    self.save_best_models()
                    logging.info(f"*** New best model saved! (Hits@5: {current_metric:.4f}) ***")
                else:
                    self.patience_counter += 1
                    logging.info(
                        f"No improvement. Patience: {self.patience_counter}/"
                        f"{self.args.early_stop_patience}"
                    )
                
                # Early stopping
                if self.patience_counter >= self.args.early_stop_patience:
                    logging.info(
                        f"\nEarly stopping triggered at round {round_idx + 1}"
                    )
                    logging.info(
                        f"Best model from round {self.best_round} "
                        f"(Hits@5: {self.best_dev_metric:.4f})"
                    )
                    break
    
    def aggregate_qa_models(self):
        """
        Aggregate QA model parameters using FedAvg
        NO AGGREGATION for relation embeddings
        """
        # Collect QA model parameters
        client_qa_states = []
        client_weights = []
        
        for client in self.clients:
            client_qa_states.append(client.qa_model.state_dict())
            # Weight by number of training samples
            client_weights.append(len(client.client_data['train_qa']))
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # FedAvg for QA models
        global_qa_state = OrderedDict()
        
        for key in client_qa_states[0].keys():
            global_qa_state[key] = sum(
                client_weights[i] * client_qa_states[i][key]
                for i in range(self.num_clients)
            )
        
        # Update all clients with aggregated QA model
        for client in self.clients:
            client.qa_model.load_state_dict(global_qa_state)
    
    def evaluate_on_dev(self):
        """Evaluate all clients on their dev sets"""
        all_metrics = []
        
        for client in self.clients:
            metrics = client.evaluate_dev()
            all_metrics.append(metrics)
        
        # Average metrics across clients
        avg_metrics = {
            'hits@1': np.mean([m['hits@1'] for m in all_metrics]),
            'hits@3': np.mean([m['hits@3'] for m in all_metrics]),
            'hits@5': np.mean([m['hits@5'] for m in all_metrics]),
            'hits@10': np.mean([m['hits@10'] for m in all_metrics]),
            'mrr': np.mean([m['mrr'] for m in all_metrics])
        }
        
        return avg_metrics
    
    def save_best_models(self):
        """Save best models for all clients - NO GLOBAL RELATION EMBEDDINGS"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        
        logging.info(f"Saving best models to {save_dir}")
        
        # Save each client's models
        for client in self.clients:
            client_dir = os.path.join(save_dir, f'client_{client.client_id}')
            os.makedirs(client_dir, exist_ok=True)
            
            # Save entity embeddings (save the Parameter data directly)
            torch.save(
                client.entity_embedding,
                os.path.join(client_dir, 'entity_embeddings.pt')
            )
            
            # Save relation embeddings (client-specific, NO aggregation)
            torch.save(
                client.relation_embedding,
                os.path.join(client_dir, 'relation_embeddings.pt')
            )
            
            # Save QA model
            torch.save(
                client.qa_model.state_dict(),
                os.path.join(client_dir, 'qa_model.pt')
            )
        
        logging.info(f"Saved models for {self.num_clients} clients (no global relation embeddings)")


class FederatedClient:
    """Individual client in federated KGQA - with separate relation embeddings"""
    
    def __init__(self, client_id, client_data, global_nrelation, args, 
                 kg_dataloader, train_qa_loader, dev_qa_loader, 
                 qa_model_module, device):
        self.client_id = client_id
        self.client_data = client_data
        self.args = args
        self.device = device
        self.qa_model_module = qa_model_module  # Store the QA model module
        
        self.nentity = client_data['nentity']
        self.nrelation = client_data['nrelation']  # Local relation count
        self.global_nrelation = global_nrelation
        
        self.kg_dataloader = kg_dataloader
        self.train_qa_loader = train_qa_loader
        self.dev_qa_loader = dev_qa_loader
        
        # Initialize ComplEx model
        self.kg_model = ComplExModel(args).to(device)
        
        # Initialize embeddings
        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Entity embeddings (local)
        entity_emb_temp, _ = initialize_embeddings(
            self.nentity,
            self.nrelation,
            args.hidden_dim,
            embedding_range
        )
        # Move to device before creating Parameter to maintain leaf status
        self.entity_embedding = nn.Parameter(entity_emb_temp.data.to(device))
        
        # Relation embeddings (LOCAL - NOT shared/aggregated)
        # Use GLOBAL relation count for consistent dimensionality
        _, relation_emb_temp = initialize_embeddings(
            self.nentity,
            self.global_nrelation,
            args.hidden_dim,
            embedding_range
        )
        # Move to device before creating Parameter to maintain leaf status
        self.relation_embedding = nn.Parameter(relation_emb_temp.data.to(device))
        
        # KG optimizer
        self.kg_optimizer = optim.Adam([
            {'params': self.entity_embedding, 'lr': args.lr},
            {'params': self.relation_embedding, 'lr': args.lr}
        ])
        
        # QA model
        self.qa_model = qa_model_module.ImprovedKGQAModel(
            args, self.nentity, self.global_nrelation
        ).to(device)
        
        # QA optimizer
        self.qa_optimizer = optim.Adam(
            self.qa_model.parameters(),
            lr=args.qa_lr
        )
    
    def train_kg_local(self):
        """Train KG embeddings locally"""
        self.kg_model.train()
        total_loss = 0
        num_batches = 0
        
        for _ in range(self.args.local_epoch):
            for positive_sample, negative_sample, _ in self.kg_dataloader:
                positive_sample = positive_sample.to(self.device)
                negative_sample = negative_sample.to(self.device)
                
                self.kg_optimizer.zero_grad()
                
                loss = compute_kg_loss(
                    self.kg_model,
                    positive_sample,
                    negative_sample,
                    self.entity_embedding,
                    self.relation_embedding,
                    self.args
                )
                
                loss.backward()
                self.kg_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_qa_local(self):
        """Train QA model locally"""
        self.qa_model.train()
        total_loss = 0
        num_batches = 0
        
        for _ in range(self.args.qa_local_epoch):
            for questions, answer_ids, hop_counts in self.train_qa_loader:
                self.qa_optimizer.zero_grad()
                
                # Forward pass
                entity_scores, relation_scores, topic_entity_ids = self.qa_model(
                    questions,
                    self.relation_embedding,
                    self.entity_embedding,
                    entity2id=self.client_data['entity2id'],
                    answer_ids=answer_ids
                )
                
                # Compute loss
                loss = self.qa_model_module.compute_qa_loss(
                    entity_scores,
                    answer_ids,
                    self.args
                )
                
                loss.backward()
                self.qa_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate_dev(self):
        """Evaluate on dev set"""
        metrics = self.qa_model_module.evaluate_qa(
            self.qa_model,
            self.dev_qa_loader,
            self.relation_embedding,
            self.entity_embedding,
            self.client_data['entity2id'],
            self.client_data['id2entity'],
            self.args
        )
        return metrics