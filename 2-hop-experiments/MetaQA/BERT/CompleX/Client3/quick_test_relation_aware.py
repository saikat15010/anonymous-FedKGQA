"""
Quick Test Script for Relation-Aware Filtering

Tests the new post-ranking type filtering approach on a small subset.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import argparse
import logging
import json
import random
from collections import defaultdict

from qa_dataloader_updated import (
    load_all_metaqa_clients,
    get_global_relation_mapping
)
from qa_model_updated import ImprovedKGQAModel
from server_inference_relation_aware import RelationAwareFederatedKGQAServer


class QuickTestDataset:
    """Quick test dataset from qa_test.txt"""
    
    def __init__(self, test_file, num_samples=100, random_sample=False, seed=42, only_2hop=False):
        self.qa_pairs = []
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split('|')
                    hop_count = self.detect_hop_count(question)
                    
                    # Filter by hop count if requested
                    if only_2hop and hop_count != 2:
                        continue
                    
                    self.qa_pairs.append({
                        'question': question,
                        'answers': answers,
                        'hop_count': hop_count
                    })
        
        # Sample subset
        if random_sample:
            random.seed(seed)
            self.qa_pairs = random.sample(self.qa_pairs, min(num_samples, len(self.qa_pairs)))
        else:
            self.qa_pairs = self.qa_pairs[:num_samples]
        
        logging.info(f"Loaded {len(self.qa_pairs)} test questions")
        if only_2hop:
            logging.info(f"  (Filtered to 2-hop questions only)")
    
    def detect_hop_count(self, question):
        """Detect if 1-hop or 2-hop"""
        question_lower = question.lower()
        
        two_hop_indicators = [
            'starred by', 'acted by', 'directed by', 'written by',
            'same actor', 'same director', 'same writer',
            'also directed', 'also wrote', 'also starred',
            'co-direct', 'co-wrote', 'appeared in the same'
        ]
        
        for indicator in two_hop_indicators:
            if indicator in question_lower:
                return 2
        
        return 1
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]


def load_trained_models(args, all_clients_data, global_nrelation, device):
    """Load trained models"""
    model_dir = os.path.join(args.state_dir, 'best_models')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logging.info(f"Loading models from {model_dir}")
    
    # Load global relation embeddings
    global_relation_embedding = torch.load(
        os.path.join(model_dir, 'global_relation_embeddings.pt'),
        map_location=device
    )
    
    if isinstance(global_relation_embedding, dict):
        global_relation_embedding = list(global_relation_embedding.values())[0]
    
    global_relation_embedding = global_relation_embedding.to(device)
    
    client_models = []
    client_embeddings = []
    
    for client_id in range(len(all_clients_data)):
        client_data = all_clients_data[client_id]
        client_save_dir = os.path.join(model_dir, f'client_{client_id}')
        
        # Load entity embeddings
        entity_emb_state = torch.load(
            os.path.join(client_save_dir, 'entity_embeddings.pt'),
            map_location=device
        )
        
        if isinstance(entity_emb_state, dict):
            entity_embedding = list(entity_emb_state.values())[0]
        else:
            entity_embedding = entity_emb_state
        
        entity_embedding = entity_embedding.to(device)
        
        # Initialize QA model with specified encoder type
        encoder_type = getattr(args, 'encoder_type', 'roberta')
        qa_model = ImprovedKGQAModel(args, client_data['nentity'], global_nrelation, encoder_type=encoder_type)
        qa_model.load_state_dict(
            torch.load(os.path.join(client_save_dir, 'qa_model.pt'), map_location=device)
        )
        
        qa_model = qa_model.to(device)
        qa_model.eval()
        
        client_models.append(qa_model)
        client_embeddings.append({
            'entity': entity_embedding,
            'relation': global_relation_embedding
        })
        
        logging.info(f"Loaded client {client_id} model")
    
    return client_models, client_embeddings, global_relation_embedding


def run_quick_test(args):
    """Run quick test on small subset"""
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    
    logging.info("=" * 70)
    logging.info("QUICK TEST - Relation-Aware Federated KGQA")
    logging.info("=" * 70)
    
    # Setup device
    if args.gpu == '-1' or not torch.cuda.is_available():
        device = torch.device("cpu")
        logging.info("Using CPU")
    else:
        device = torch.device(f'cuda:{args.gpu}')
        logging.info(f"Using GPU: {device}")
    
    args.gpu = device
    
    # Load client data
    logging.info("\nLoading client data...")
    all_clients_data = load_all_metaqa_clients(
        args.client_data_path,
        num_clients=args.num_clients
    )
    
    # Get global relation mapping
    global_relation2id, global_id2relation, global_nrelation = get_global_relation_mapping(all_clients_data)
    
    # Load test dataset (small subset)
    logging.info(f"\nLoading test subset ({args.num_samples} questions)...")
    test_dataset = QuickTestDataset(
        args.test_file,
        num_samples=args.num_samples,
        random_sample=args.random_sample,
        seed=args.seed,
        only_2hop=args.only_2hop
    )
    
    # Analyze test set
    hop_counts = defaultdict(int)
    for qa_pair in test_dataset.qa_pairs:
        hop_counts[qa_pair['hop_count']] += 1
    
    logging.info("Test subset distribution:")
    for hop, count in sorted(hop_counts.items()):
        logging.info(f"  {hop}-hop: {count} questions")
    
    # Load trained models
    logging.info("\nLoading trained models...")
    client_models, client_embeddings, global_relation_embedding = load_trained_models(
        args, all_clients_data, global_nrelation, device
    )
    
    # Initialize server with relation-aware filtering
    logging.info("\nInitializing server with relation-aware filtering...")
    server = RelationAwareFederatedKGQAServer(args.num_clients, all_clients_data)
    server.client_models = client_models
    server.client_embeddings = client_embeddings
    
    # Run evaluation
    logging.info("\n" + "=" * 70)
    logging.info("Starting Quick Test Evaluation")
    logging.info("=" * 70)
    
    metrics = server.evaluate_on_dataset(
        test_dataset.qa_pairs,
        output_file=args.output_file,
        beam_width=args.beam_width
    )
    
    # Show example outputs
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUTS")
    print("=" * 70)
    
    for idx in range(min(args.num_examples, len(test_dataset))):
        qa_pair = test_dataset[idx]
        question = qa_pair['question']
        ground_truth = qa_pair['answers']
        
        try:
            predicted, metadata = server.answer_question(question, top_k=5, beam_width=args.beam_width)
        except Exception as e:
            predicted = []
            metadata = {'error': str(e)}
        
        # Check if correct (Hit@10)
        correct = any(pred in ground_truth for pred in predicted[:10]) if predicted else False
        
        print(f"\nExample {idx+1}:")
        print(f"Question: {question}")
        print(f"Ground Truth: {', '.join(ground_truth[:3])}{'...' if len(ground_truth) > 3 else ''}")
        print(f"Predicted: {', '.join(predicted) if predicted else 'NO ANSWER'}")
        print(f"Correct: {'✓ YES' if correct else '✗ NO'}")
        
        if metadata.get('relation_aware_filtering') == 'enabled' and 'intermediate_results' in metadata:
            print("\n  Relation-aware filtering applied:")
            for result in metadata['intermediate_results']:
                print(f"    Sub-query: {result['sub_query']}")
                print(f"      Relation: {result['relation']}")
                print(f"      Answers: {', '.join(result['answers'][:3])}")
        
        print("-" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test for Relation-Aware KGQA')
    
    # Paths
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to qa_test.txt file')
    parser.add_argument('--client_data_path', type=str, required=True,
                       help='Path to client data (federated_clients/)')
    parser.add_argument('--state_dir', type=str, required=True,
                       help='Directory containing saved models')
    
    # Test parameters
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of questions to test (default: 100)')
    parser.add_argument('--random_sample', action='store_true',
                       help='Sample randomly instead of taking first N')
    parser.add_argument('--only_2hop', action='store_true',
                       help='Evaluate only on 2-hop questions (filter out 1-hop)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to display')
    
    # Model parameters (must match training)
    parser.add_argument('--num_clients', type=int, default=3,
                       help='Number of clients')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--gamma', type=float, default=12.0,
                       help='Gamma for ComplEx')
    parser.add_argument('--epsilon', type=float, default=2.0,
                       help='Epsilon for initialization')
    parser.add_argument('--encoder_type', type=str, default='roberta',
                       choices=['bert', 'distilbert', 'roberta'],
                       help='Question encoder type (must match training): bert, distilbert, or roberta')
    
    parser.add_argument('--beam_width', type=int, default=3,
                       help='Beam width for multi-hop reasoning (1=single path, 3=top-3 beam, 5=top-5 beam)')
    
    # Output
    parser.add_argument('--output_file', type=str, default='quick_test_relation_aware_results.json',
                       help='Output file for results')
    
    # System
    parser.add_argument('--gpu', type=str, default='-1',
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    run_quick_test(args)