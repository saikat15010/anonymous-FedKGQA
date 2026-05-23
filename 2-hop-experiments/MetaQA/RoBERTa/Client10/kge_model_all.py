import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransEModel(nn.Module):
    """
    TransE model for knowledge graph embedding
    
    TransE models relations as translations in embedding space.
    Score = -||h + r - t||
    """
    def __init__(self, args):
        super(TransEModel, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        # Embedding range for initialization
        self.embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Gamma parameter for margin-based scoring
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )
        
        # Norm for distance (L1 or L2)
        self.norm = args.transe_norm if hasattr(args, 'transe_norm') else 1
    
    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        """
        Forward pass for TransE
        
        Args:
            sample: When neg=False: [batch_size, 3] tensor (h, r, t)
                   When neg=True: tuple of (positive_sample, negative_sample)
            relation_embedding: [nrelation, hidden_dim]
            entity_embedding: [nentity, hidden_dim]
            neg: Whether using negative sampling
        
        Returns:
            score: Scores for the samples
        """
        if not neg:
            # Positive samples only
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:
            # Negative sampling
            head_part, tail_part = sample
            batch_size = head_part.shape[0]

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part is None:
                tail = entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
        
        # TransE scoring: -||h + r - t||
        score = self.transe_score(head, relation, tail)
        
        return score
    
    def transe_score(self, head, relation, tail):
        """
        TransE scoring function: score = gamma - ||h + r - t||
        
        Args:
            head: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail: [batch_size, num_samples, hidden_dim]
        
        Returns:
            score: [batch_size, num_samples]
        """
        # h + r - t
        score = head + relation - tail
        
        # Distance (L1 or L2 norm)
        score = torch.norm(score, p=self.norm, dim=2)
        
        # Convert to similarity (higher is better)
        score = self.gamma.item() - score
        
        return score
    
    def regularization_loss(self, entity_embedding, relation_embedding):
        """L2 regularization on embeddings"""
        return (entity_embedding.norm(p=2) ** 2 + relation_embedding.norm(p=2) ** 2)


class DistMultModel(nn.Module):
    """
    DistMult model for knowledge graph embedding
    
    DistMult uses bilinear scoring with diagonal relation matrices.
    Score = <h, r, t> (element-wise product then sum)
    """
    def __init__(self, args):
        super(DistMultModel, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        # Embedding range for initialization
        self.embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Gamma parameter (not used in scoring but kept for compatibility)
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )
    
    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        """
        Forward pass for DistMult
        
        Args:
            sample: When neg=False: [batch_size, 3] tensor (h, r, t)
                   When neg=True: tuple of (positive_sample, negative_sample)
            relation_embedding: [nrelation, hidden_dim]
            entity_embedding: [nentity, hidden_dim]
            neg: Whether using negative sampling
        
        Returns:
            score: Scores for the samples
        """
        if not neg:
            # Positive samples only
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:
            # Negative sampling
            head_part, tail_part = sample
            batch_size = head_part.shape[0]

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part is None:
                tail = entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
        
        # DistMult scoring
        score = self.distmult_score(head, relation, tail)
        
        return score
    
    def distmult_score(self, head, relation, tail):
        """
        DistMult scoring function: score = <h, r, t>
        
        Args:
            head: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail: [batch_size, num_samples, hidden_dim]
        
        Returns:
            score: [batch_size, num_samples]
        """
        # Element-wise product: h * r * t
        score = head * relation * tail
        
        # Sum over dimensions
        score = score.sum(dim=2)
        
        return score
    
    def regularization_loss(self, entity_embedding, relation_embedding):
        """L2 regularization on embeddings"""
        return (entity_embedding.norm(p=2) ** 2 + relation_embedding.norm(p=2) ** 2)


def get_kge_model(model_name, args):
    """
    Factory function to get KGE model by name
    
    Args:
        model_name: 'transe' or 'distmult'
        args: Arguments
    
    Returns:
        model: KGE model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'transe':
        return TransEModel(args)
    elif model_name == 'distmult':
        return DistMultModel(args)
    else:
        raise ValueError(f"Unknown KGE model: {model_name}. Choose from: transe, distmult")


def initialize_embeddings(nentity, nrelation, hidden_dim, embedding_range, model_name='distmult'):
    """
    Initialize entity and relation embeddings based on model type
    
    Args:
        nentity: Number of entities
        nrelation: Number of relations  
        hidden_dim: Embedding dimension
        embedding_range: Range for uniform initialization
        model_name: 'transe' or 'distmult'
    
    Returns:
        entity_embedding: Entity embeddings
        relation_embedding: Relation embeddings
    """
    model_name = model_name.lower()
    
    if model_name == 'transe':
        # TransE: real-valued embeddings
        entity_embedding = nn.Parameter(
            torch.zeros(nentity, hidden_dim).uniform_(-embedding_range, embedding_range)
        )
        relation_embedding = nn.Parameter(
            torch.zeros(nrelation, hidden_dim).uniform_(-embedding_range, embedding_range)
        )
        
    elif model_name == 'distmult':
        # DistMult: real-valued embeddings
        entity_embedding = nn.Parameter(
            torch.zeros(nentity, hidden_dim).uniform_(-embedding_range, embedding_range)
        )
        relation_embedding = nn.Parameter(
            torch.zeros(nrelation, hidden_dim).uniform_(-embedding_range, embedding_range)
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: transe, distmult")
    
    return entity_embedding, relation_embedding


def compute_kg_loss(model, positive_sample, negative_sample, 
                    entity_embedding, relation_embedding, args):
    """
    Compute knowledge graph embedding loss with self-adversarial negative sampling
    
    Args:
        model: KGE model (TransE or DistMult)
        positive_sample: [batch_size, 3] (h, r, t)
        negative_sample: [batch_size, num_neg] (negative tail entities)
        entity_embedding: Entity embeddings
        relation_embedding: Relation embeddings
        args: Arguments containing adversarial_temperature, etc.
    
    Returns:
        loss: Scalar loss value
    """
    # Positive scores - pass just positive_sample, not tuple
    positive_score = model(positive_sample, relation_embedding, entity_embedding, neg=False)
    positive_score = positive_score.squeeze(1)
    
    # Negative scores - pass tuple
    negative_score = model((positive_sample, negative_sample), relation_embedding, entity_embedding, neg=True)
    
    # Self-adversarial negative sampling
    if args.adversarial_temperature > 0:
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-negative_score)).sum(dim=1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)
    
    # Positive loss
    positive_loss = F.logsigmoid(positive_score).squeeze()
    
    # Total loss
    loss = -(positive_loss + negative_score).mean()
    
    # Add L2 regularization
    if hasattr(args, 'reg_lambda') and args.reg_lambda > 0:
        reg_loss = model.regularization_loss(entity_embedding, relation_embedding)
        loss = loss + args.reg_lambda * reg_loss
    
    return loss


def evaluate_kg(model, test_triples, entity_embedding, relation_embedding, 
                nentity, device, batch_size=16):
    """
    Evaluate knowledge graph embeddings on test triples
    
    Args:
        model: KGE model
        test_triples: List of (h, r, t) triples
        entity_embedding: Entity embeddings
        relation_embedding: Relation embeddings
        nentity: Number of entities
        device: torch device
        batch_size: Evaluation batch size
    
    Returns:
        metrics: Dict with MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    
    ranks = []
    
    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size]
            batch_tensor = torch.LongTensor(batch).to(device)
            
            # Score against all entities - pass just the batch tensor
            scores = model(batch_tensor, relation_embedding, entity_embedding, neg=False)
            scores = scores.squeeze(1)  # [batch_size, nentity]
            
            # Get rank of true tail
            for j, (h, r, t) in enumerate(batch):
                # Sort scores in descending order
                _, sorted_indices = torch.sort(scores[j], descending=True)
                
                # Find rank of true entity (1-indexed)
                rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
    
    ranks = np.array(ranks)
    
    metrics = {
        'mrr': np.mean(1.0 / ranks),
        'hits@3': np.mean(ranks <= 3),
        'hits@5': np.mean(ranks <= 5),
        'hits@10': np.mean(ranks <= 10)
    }
    
    model.train()
    return metrics