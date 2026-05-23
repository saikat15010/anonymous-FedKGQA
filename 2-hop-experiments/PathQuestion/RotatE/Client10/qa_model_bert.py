import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from transformers import BertTokenizer, BertModel


class BertQuestionEncoder(nn.Module):
    """Question encoder using BERT"""
    def __init__(self, args, hidden_dim):
        super(BertQuestionEncoder, self).__init__()
        self.args = args
        
        torch.set_num_threads(1)
        
        # Use BERT
        model_name = 'bert-base-uncased'
        
        # Use slow tokenizer to avoid threading issues
        self.tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Load model
        self.encoder = BertModel.from_pretrained(model_name)
        
        # BERT hidden size is 768
        self.lm_hidden_size = 768
        
        # Multi-layer projection for better representation
        self.projection = nn.Sequential(
            nn.Linear(self.lm_hidden_size, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, questions):
        """
        Encode questions to fixed-size vectors
        
        Args:
            questions: List of question strings
        
        Returns:
            question_embeddings: [batch_size, hidden_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        outputs = self.encoder(**inputs)
        
        # Use [CLS] token (first token in BERT)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Project to KG space
        question_embeddings = self.projection(cls_embeddings)  # [batch_size, hidden_dim]
        
        return question_embeddings


class ImprovedKGQAModel(nn.Module):
    """
    Improved KGQA model with:
    1. BERT question encoder
    2. Better relation prediction
    3. Topic entity extraction
    4. Cross-attention between question and KG
    5. Updated for ComplEx embeddings (both entity and relation have 2*hidden_dim)
    """
    def __init__(self, args, nentity, nrelation):
        super(ImprovedKGQAModel, self).__init__()
        self.args = args
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = args.hidden_dim
        
        # Question encoder
        self.question_encoder = BertQuestionEncoder(args, args.hidden_dim)
        
        # Relation predictor with attention
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.relation_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        
        # Topic entity predictor (for identifying starting entity)
        self.topic_scorer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
        # Answer scorer with fusion
        self.answer_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_dim, 1)
        )
    
    def extract_topic_entity_from_question(self, question):
        """
        Extract topic entity from PathQuestion question
        Returns first underscored entity name as string
        """
        words = question.split()
        for word in words:
            # If word contains underscore, likely an entity
            if '_' in word and not word.startswith('_'):
                # Clean punctuation
                entity = word.strip('?.,!;:\'"')
                return entity
        return None
    
    def forward(self, questions, relation_embeddings, entity_embeddings, 
                entity2id=None, answer_ids=None):
        """
        Forward pass for question answering
        
        Args:
            questions: List of question strings
            relation_embeddings: [nrelation, 2*hidden_dim] for ComplEx (real + imaginary)
            entity_embeddings: [nentity, 2*hidden_dim] for ComplEx (real + imaginary)
            entity2id: Dict mapping entity names to IDs (for topic entity)
            answer_ids: List of answer entity ID lists (for training)
        
        Returns:
            entity_scores: [batch_size, nentity] - scores for each entity
            relation_scores: [batch_size, nrelation] - predicted relation scores
            topic_entity_ids: [batch_size] - predicted topic entity IDs
        """
        batch_size = len(questions)
        device = relation_embeddings.device
        
        # Encode questions
        question_emb = self.question_encoder(questions)  # [batch_size, hidden_dim]
        
        # For ComplEx, use only real part of embeddings for initial scoring
        # Entity embeddings: [nentity, 2*hidden_dim] (real + imaginary)
        # Relation embeddings: [nrelation, 2*hidden_dim] (real + imaginary)
        entity_emb_real = entity_embeddings[:, :self.hidden_dim]  # [nentity, hidden_dim]
        relation_emb_real = relation_embeddings[:, :self.hidden_dim]  # [nrelation, hidden_dim]
        
        # Predict relation using attention
        question_emb_expanded = question_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        relation_emb_expanded = relation_emb_real.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, nrelation, hidden_dim]
        
        attn_out, _ = self.relation_attention(
            question_emb_expanded,
            relation_emb_expanded,
            relation_emb_expanded
        )  # [batch_size, 1, hidden_dim]
        
        rel_query = self.relation_predictor(attn_out.squeeze(1))  # [batch_size, hidden_dim]
        relation_scores = torch.matmul(rel_query, relation_emb_real.t())  # [batch_size, nrelation]
        
        # Extract topic entities from questions
        topic_entity_ids = []
        for question in questions:
            topic_entity_name = self.extract_topic_entity_from_question(question)
            if topic_entity_name and entity2id and topic_entity_name in entity2id:
                topic_entity_ids.append(entity2id[topic_entity_name])
            else:
                topic_entity_ids.append(0)  # Default to first entity if not found
        
        topic_entity_ids = torch.LongTensor(topic_entity_ids).to(device)
        
        # Get topic entity embeddings
        topic_emb = entity_emb_real[topic_entity_ids]  # [batch_size, hidden_dim]
        
        # Compute answer scores with fusion
        # Combine question embedding and topic entity information
        combined_query = torch.cat([question_emb, topic_emb], dim=1)  # [batch_size, hidden_dim*2]
        
        # Score all entities
        entity_scores_list = []
        for i in range(batch_size):
            # Expand combined query for all entities
            query_expanded = combined_query[i:i+1].expand(self.nentity, -1)  # [nentity, hidden_dim*2]
            
            # Concatenate with entity embeddings
            entity_pairs = torch.cat([
                query_expanded[:, :self.hidden_dim],
                entity_emb_real
            ], dim=1)  # [nentity, hidden_dim*2]
            
            # Score
            scores = self.answer_fusion(entity_pairs).squeeze(1)  # [nentity]
            entity_scores_list.append(scores)
        
        entity_scores = torch.stack(entity_scores_list, dim=0)  # [batch_size, nentity]
        
        return entity_scores, relation_scores, topic_entity_ids
    
    def predict_answers(self, questions, relation_embeddings, entity_embeddings, 
                       entity2id=None, top_k=10):
        """
        Predict top-k answers for questions
        
        Args:
            questions: List of question strings
            relation_embeddings: [nrelation, 2*hidden_dim] for ComplEx
            entity_embeddings: [nentity, 2*hidden_dim] for ComplEx
            entity2id: Dict mapping entity names to IDs
            top_k: Number of top answers
        
        Returns:
            top_k_answers: [batch_size, top_k]
            top_k_scores: [batch_size, top_k]
        """
        with torch.no_grad():
            entity_scores, _, _ = self.forward(
                questions, relation_embeddings, entity_embeddings, entity2id
            )
            top_k_scores, top_k_answers = torch.topk(entity_scores, k=min(top_k, entity_scores.size(1)), dim=1)
        
        return top_k_answers, top_k_scores


def compute_qa_loss(entity_scores, answer_ids, args):
    """
    Cross-entropy loss with label smoothing
    
    Args:
        entity_scores: [batch_size, nentity] - raw scores for each entity
        answer_ids: List of lists of answer entity IDs
        args: Arguments
    
    Returns:
        loss: scalar
    """
    batch_size = entity_scores.size(0)
    nentity = entity_scores.size(1)
    device = entity_scores.device
    
    # Label smoothing parameter
    smoothing = getattr(args, 'label_smoothing', 0.1)
    
    total_loss = 0.0
    valid_samples = 0
    
    for i, answers in enumerate(answer_ids):
        if len(answers) == 0:
            continue
        
        valid_answers = [ans for ans in answers if ans < nentity]
        if len(valid_answers) == 0:
            continue
        
        # Create soft labels with label smoothing
        labels = torch.zeros(nentity, device=device)
        
        # Distribute (1 - smoothing) probability among correct answers
        correct_prob = (1.0 - smoothing) / len(valid_answers)
        for ans in valid_answers:
            labels[ans] = correct_prob
        
        # Distribute smoothing probability uniformly across all entities
        labels += smoothing / nentity
        
        # Compute log probabilities
        log_probs = F.log_softmax(entity_scores[i], dim=0)
        
        # Cross-entropy loss
        loss = -(labels * log_probs).sum()
        
        total_loss += loss
        valid_samples += 1
    
    if valid_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / valid_samples


def evaluate_qa(model, qa_dataloader, relation_embeddings, entity_embeddings, 
                entity2id, id2entity, args):
    """
    Evaluate QA performance with detailed metrics
    
    Returns:
        results: dict with metrics (hits@1, hits@3, hits@5, hits@10, mrr)
    """
    model.eval()
    
    total_hits_1 = 0
    total_hits_3 = 0
    total_hits_5 = 0
    total_hits_10 = 0
    total_mrr = 0
    total_count = 0
    
    # Track by hop count
    hop_metrics = {1: {'count': 0, 'hits@1': 0, 'mrr': 0},
                   2: {'count': 0, 'hits@1': 0, 'mrr': 0}}
    
    with torch.no_grad():
        for questions, answer_ids_batch, hop_counts in qa_dataloader:
            batch_size = len(questions)
            
            # Get predictions
            top_k_answers, top_k_scores = model.predict_answers(
                questions, relation_embeddings, entity_embeddings, 
                entity2id, top_k=10
            )
            
            # Evaluate each question
            for i in range(batch_size):
                answer_ids = answer_ids_batch[i]
                hop_count = hop_counts[i]
                
                if len(answer_ids) == 0:
                    continue
                
                predictions = top_k_answers[i].cpu().numpy()
                
                # Check if any correct answer is in top-k
                found_rank = None
                for rank, pred_id in enumerate(predictions):
                    if pred_id in answer_ids:
                        found_rank = rank
                        break
                
                if found_rank is not None:
                    # Hits@k
                    if found_rank < 1:
                        total_hits_1 += 1
                        hop_metrics[hop_count]['hits@1'] += 1
                    if found_rank < 3:
                        total_hits_3 += 1
                    if found_rank < 5:
                        total_hits_5 += 1
                    if found_rank < 10:
                        total_hits_10 += 1
                    
                    # MRR
                    mrr_score = 1.0 / (found_rank + 1)
                    total_mrr += mrr_score
                    hop_metrics[hop_count]['mrr'] += mrr_score
                
                total_count += 1
                hop_metrics[hop_count]['count'] += 1
    
    # Compute hop-specific metrics
    for hop in [1, 2]:
        if hop_metrics[hop]['count'] > 0:
            hop_metrics[hop]['hits@1'] /= hop_metrics[hop]['count']
            hop_metrics[hop]['mrr'] /= hop_metrics[hop]['count']
    
    results = {
        'hits@1': total_hits_1 / total_count if total_count > 0 else 0,
        'hits@3': total_hits_3 / total_count if total_count > 0 else 0,
        'hits@5': total_hits_5 / total_count if total_count > 0 else 0,
        'hits@10': total_hits_10 / total_count if total_count > 0 else 0,
        'mrr': total_mrr / total_count if total_count > 0 else 0,
        'count': total_count,
        'hop_metrics': hop_metrics
    }
    
    model.train()
    return results
