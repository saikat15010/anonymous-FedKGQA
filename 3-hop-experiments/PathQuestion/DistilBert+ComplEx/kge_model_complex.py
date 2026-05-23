import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplExModel(nn.Module):
    """
    ComplEx model for knowledge graph embedding
    
    ComplEx uses complex-valued embeddings to model asymmetric relations.
    Score = Re(<h, r, conj(t)>) where <> is complex inner product
    """
    def __init__(self, args):
        super(ComplExModel, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        # Embedding range for initialization
        self.embedding_range = (args.gamma + args.epsilon) / args.hidden_dim
        
        # Gamma parameter for margin-based ranking
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )
    
    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        """
        Forward pass for ComplEx
        
        Args:
            sample: When neg=False: [batch_size, 3] tensor (h, r, t)
                   When neg=True: tuple of (positive_sample, negative_sample)
            relation_embedding: [nrelation, 2*hidden_dim] (real + imaginary)
            entity_embedding: [nentity, 2*hidden_dim] (real + imaginary)
            neg: Whether using negative sampling
        
        Returns:
            score: Scores for the samples
        """
        if not neg:
            # Positive samples only - sample is a tensor [batch_size, 3]
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
            # Negative sampling - sample is a tuple (head_part, tail_part)
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
                # All entities as negatives
                tail = entity_embedding.unsqueeze(0)
            else:
                # Specific negative samples
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
        
        # ComplEx scoring
        score = self.complex_score(head, relation, tail)
        
        return score
    
    def complex_score(self, head, relation, tail):
        """
        ComplEx scoring function: Re(<h, r, conj(t)>)
        
        Args:
            head: [batch_size, 1, 2*hidden_dim]
            relation: [batch_size, 1, 2*hidden_dim]
            tail: [batch_size, num_samples, 2*hidden_dim]
        
        Returns:
            score: [batch_size, num_samples]
        """
        # Split into real and imaginary parts
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        # ComplEx score: Re(<h, r, conj(t)>)
        # = Re(h) * Re(r) * Re(t) + Re(h) * Im(r) * Im(t) 
        #   + Im(h) * Re(r) * Im(t) - Im(h) * Im(r) * Re(t)
        
        score = (
            (re_head * re_relation * re_tail).sum(dim=2) +
            (re_head * im_relation * im_tail).sum(dim=2) +
            (im_head * re_relation * im_tail).sum(dim=2) -
            (im_head * im_relation * re_tail).sum(dim=2)
        )
        
        return score
    
    def regularization_loss(self, entity_embedding, relation_embedding):
        """
        L2 regularization on embeddings
        
        Args:
            entity_embedding: [nentity, 2*hidden_dim]
            relation_embedding: [nrelation, 2*hidden_dim]
        
        Returns:
            reg_loss: Scalar regularization loss
        """
        return (entity_embedding.norm(p=2) ** 2 + relation_embedding.norm(p=2) ** 2)


def initialize_embeddings(nentity, nrelation, hidden_dim, embedding_range):
    """
    Initialize entity and relation embeddings for ComplEx
    
    ComplEx:
    - Entities: 2*hidden_dim (real + imaginary parts)
    - Relations: 2*hidden_dim (real + imaginary parts)
    
    Args:
        nentity: Number of entities
        nrelation: Number of relations  
        hidden_dim: Embedding dimension
        embedding_range: Range for uniform initialization
    
    Returns:
        entity_embedding: [nentity, 2*hidden_dim]
        relation_embedding: [nrelation, 2*hidden_dim]
    """
    # Initialize entity embeddings with uniform distribution
    entity_embedding = nn.Parameter(
        torch.zeros(nentity, hidden_dim * 2).uniform_(-embedding_range, embedding_range)
    )
    
    # Initialize relation embeddings with uniform distribution
    relation_embedding = nn.Parameter(
        torch.zeros(nrelation, hidden_dim * 2).uniform_(-embedding_range, embedding_range)
    )
    
    return entity_embedding, relation_embedding


def compute_kg_loss(model, positive_sample, negative_sample, 
                    entity_embedding, relation_embedding, args):
    """
    Compute knowledge graph embedding loss with self-adversarial negative sampling
    
    Args:
        model: ComplExModel
        positive_sample: [batch_size, 3] (h, r, t)
        negative_sample: [batch_size, num_neg] (negative tail entities)
        entity_embedding: [nentity, 2*hidden_dim]
        relation_embedding: [nrelation, 2*hidden_dim]
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
        model: ComplExModel
        test_triples: List of (h, r, t) triples
        entity_embedding: [nentity, 2*hidden_dim]
        relation_embedding: [nrelation, 2*hidden_dim]
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
                # Sort scores in descending order (higher is better)
                _, sorted_indices = torch.sort(scores[j], descending=True)
                
                # Find rank of true entity (1-indexed)
                rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
    
    ranks = np.array(ranks)
    
    metrics = {
        'mrr': np.mean(1.0 / ranks),
        'hits@1': np.mean(ranks <= 1),
        'hits@3': np.mean(ranks <= 3),
        'hits@10': np.mean(ranks <= 10)
    }
    
    model.train()
    return metrics
