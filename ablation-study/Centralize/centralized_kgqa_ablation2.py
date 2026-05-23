"""
Centralized KGQA Training System - Ablation Study 2

Single model trained on all data (no federated learning).
Baseline for comparison with federated approaches.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from kge_model_complex import (
    ComplExModel,
    compute_kg_loss
)


class CentralizedKGQA:
    """
    Centralized KGQA system - Single model on all data
    No federation, no clients - just one big model
    """
    
    def __init__(self, args, kb_data, nrelation, qa_model_module):
        """
        Initialize centralized KGQA system
        
        Args:
            args: Training arguments
            kb_data: KB triples and entity/relation mappings
            nrelation: Number of relations
            qa_model_module: QA model module (qa_model_roberta)
        """
        self.args = args
        self.kb_data = kb_data
        self.nentity = kb_data['nentity']
        self.nrelation = nrelation
        self.qa_model_module = qa_model_module
        
        # Initialize embeddings
        embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Entity embeddings (ComplEx: 2*hidden_dim)
        entity_embedding_init = torch.zeros(self.nentity, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
        self.entity_embedding = nn.Parameter(entity_embedding_init, requires_grad=True)
        self.entity_embedding.data = self.entity_embedding.data.to(args.gpu)
        
        # Relation embeddings (ComplEx: 2*hidden_dim)
        relation_embedding_init = torch.zeros(nrelation, args.hidden_dim * 2).uniform_(-embedding_range, embedding_range)
        self.relation_embedding = nn.Parameter(relation_embedding_init, requires_grad=True)
        self.relation_embedding.data = self.relation_embedding.data.to(args.gpu)
        
        # Initialize models
        self.kg_model = ComplExModel(args).to(args.gpu)
        self.qa_model = qa_model_module.ImprovedKGQAModel(args, self.nentity, nrelation).to(args.gpu)
        
        # Training state
        self.best_dev_metrics = {
            'hits@1': 0.0,
            'hits@3': 0.0,
            'hits@5': 0.0,
            'hits@10': 0.0,
            'mrr': 0.0
        }
        
        logging.info(f"Initialized Centralized KGQA")
        logging.info(f"Entities: {self.nentity}, Relations: {nrelation}")
        logging.info(f"QA Model: {qa_model_module.__name__}")
    
    def setup_dataloaders(self, kg_dataloader, train_qa_loader, dev_qa_loader):
        """Setup dataloaders"""
        self.kg_dataloader = kg_dataloader
        self.train_qa_loader = train_qa_loader
        self.dev_qa_loader = dev_qa_loader
        
        # Optimizers
        self.kg_optimizer = optim.Adam(
            [self.entity_embedding, self.relation_embedding],
            lr=self.args.lr
        )
        self.qa_optimizer = optim.Adam(
            self.qa_model.parameters(),
            lr=self.args.qa_lr
        )
        
        logging.info("Dataloaders and optimizers initialized")
    
    def train_kg_epoch(self):
        """Train KG embeddings for one epoch"""
        self.kg_model.train()
        total_loss = 0
        num_batches = 0
        
        for positive_sample, negative_sample, _ in self.kg_dataloader:
            positive_sample = positive_sample.to(self.args.gpu)
            negative_sample = negative_sample.to(self.args.gpu)
            
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
    
    def train_phase1_kg(self):
        """Phase 1: KG Embedding Training"""
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 1: KG Embedding Training (ComplEx)")
        logging.info("=" * 70)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.args.kg_max_rounds):
            avg_loss = self.train_kg_epoch()
            
            if (epoch + 1) % self.args.log_per_round == 0:
                logging.info(f"[KG Epoch {epoch + 1}/{self.args.kg_max_rounds}] Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.args.early_stop_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logging.info(f"Phase 1 completed! Best Loss: {best_loss:.4f}")
    
    def train_qa_epoch(self):
        """Train QA model for one epoch"""
        self.qa_model.train()
        total_loss = 0
        num_batches = 0
        
        for questions, answer_ids, _ in self.train_qa_loader:
            self.qa_optimizer.zero_grad()
            
            entity_scores, _, _ = self.qa_model(
                questions,
                self.relation_embedding,
                self.entity_embedding,
                entity2id=self.kb_data['entity2id'],
                answer_ids=answer_ids
            )
            
            loss = self.qa_model_module.compute_qa_loss(entity_scores, answer_ids, self.args)
            
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
            self.kb_data['entity2id'],
            self.kb_data['id2entity'],
            self.args
        )
        return metrics
    
    def train_phase2_qa(self):
        """Phase 2: QA Model Training"""
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 2: QA Model Training (RoBERTa)")
        logging.info("=" * 70)
        
        best_hits_at_5 = 0
        patience_counter = 0
        
        for epoch in range(self.args.qa_max_rounds):
            avg_loss = self.train_qa_epoch()
            
            if (epoch + 1) % self.args.log_per_round == 0:
                logging.info(f"[QA Epoch {epoch + 1}/{self.args.qa_max_rounds}] Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % self.args.check_per_round == 0:
                dev_metrics = self.evaluate_dev()
                
                logging.info(f"Dev Metrics: Hits@1={dev_metrics['hits@1']:.4f}, "
                           f"Hits@3={dev_metrics['hits@3']:.4f}, "
                           f"Hits@5={dev_metrics['hits@5']:.4f}, "
                           f"Hits@10={dev_metrics['hits@10']:.4f}, "
                           f"MRR={dev_metrics['mrr']:.4f}")
                
                if dev_metrics['hits@5'] > best_hits_at_5:
                    best_hits_at_5 = dev_metrics['hits@5']
                    self.best_dev_metrics = dev_metrics
                    self.save_best_models()
                    patience_counter = 0
                    logging.info(f"New best Hits@5: {best_hits_at_5:.4f} - Model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= self.args.early_stop_patience:
                        logging.info(f"Early stopping at epoch {epoch + 1}")
                        break
        
        logging.info(f"Phase 2 completed! Best Hits@5: {best_hits_at_5:.4f}")
        
        # Save final model if no evaluation happened
        if best_hits_at_5 == 0:
            logging.info("No evaluation performed - saving final model")
            self.save_best_models()
    
    def save_best_models(self):
        """Save best models"""
        save_dir = os.path.join(self.args.state_dir, 'best_models')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.entity_embedding.cpu(), os.path.join(save_dir, 'entity_embeddings.pt'))
        self.entity_embedding.data = self.entity_embedding.data.to(self.args.gpu)
        
        torch.save(self.relation_embedding.cpu(), os.path.join(save_dir, 'relation_embeddings.pt'))
        self.relation_embedding.data = self.relation_embedding.data.to(self.args.gpu)
        
        torch.save(self.qa_model.state_dict(), os.path.join(save_dir, 'qa_model.pt'))
        
        logging.info(f"Models saved to {save_dir}")
    
    def train(self):
        """Main training loop"""
        self.train_phase1_kg()
        self.train_phase2_qa()
        
        logging.info("\n" + "=" * 70)
        logging.info("Training Complete!")
        logging.info("=" * 70)
        if self.best_dev_metrics.get('hits@1', 0) > 0:
            logging.info(f"Best Dev Metrics:")
            logging.info(f"  Hits@1:  {self.best_dev_metrics['hits@1']:.4f}")
            logging.info(f"  Hits@3:  {self.best_dev_metrics['hits@3']:.4f}")
            logging.info(f"  Hits@5:  {self.best_dev_metrics['hits@5']:.4f}")
            logging.info(f"  Hits@10: {self.best_dev_metrics['hits@10']:.4f}")
            logging.info(f"  MRR:     {self.best_dev_metrics['mrr']:.4f}")
        else:
            logging.info("Note: No evaluation was performed")