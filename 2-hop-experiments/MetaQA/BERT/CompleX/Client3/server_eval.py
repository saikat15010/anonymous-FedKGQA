"""
Server-Side Evaluation Script

Evaluates trained federated KGQA models on server test set
with support for complex multi-hop question answering.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import argparse
import logging
import json

from qa_dataloader_updated import (
    load_all_metaqa_clients,
    load_server_test_set,
    get_global_relation_mapping
)
from qa_model_updated import ImprovedKGQAModel
from server_inference import FederatedKGQAServer


def init_logger(log_file='server_eval.log'):
    """Initialize logger"""
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_trained_models(args, all_clients_data, global_nrelation, device):
    """
    Load trained models from checkpoint
    
    Returns:
        (client_models, client_embeddings, global_relation_embedding)
    """
    model_dir = os.path.join(args.state_dir, 'best_models')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logging.info(f"Loading models from {model_dir}")
    
    # Load global relation embeddings
    global_relation_embedding = torch.load(
        os.path.join(model_dir, 'global_relation_embeddings.pt'),
        map_location=device
    )
    
    # Convert to tensor if it's a state dict
    if isinstance(global_relation_embedding, dict):
        # Get the parameter value
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
        
        # Convert to tensor if state dict
        if isinstance(entity_emb_state, dict):
            entity_embedding = list(entity_emb_state.values())[0]
        else:
            entity_embedding = entity_emb_state
        
        entity_embedding = entity_embedding.to(device)
        
        # Initialize QA model with specified encoder type
        encoder_type = getattr(args, 'encoder_type', 'roberta')
        qa_model = ImprovedKGQAModel(
            args,
            client_data['nentity'],
            global_nrelation,
            encoder_type=encoder_type
        )
        
        # Load QA model weights
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


def main(args):
    """Main evaluation function"""
    
    init_logger()
    
    logging.info("=" * 70)
    logging.info("Federated KGQA Server-Side Evaluation")
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
    
    for i, client_data in enumerate(all_clients_data):
        logging.info(
            f"Client {i}: {client_data['nentity']} entities, "
            f"{len(client_data['triples'])} triples"
        )
    
    # Get global relation mapping
    global_relation2id, global_id2relation, global_nrelation = get_global_relation_mapping(all_clients_data)
    logging.info(f"Global relations: {global_nrelation}")
    
    # Load server test set
    logging.info(f"\nLoading server test set from {args.server_test_path}")
    test_dataset = load_server_test_set(args.server_test_path)
    logging.info(f"Test questions: {len(test_dataset)}")
    
    # Analyze test set by hop count
    hop_counts = {1: 0, 2: 0, 3: 0}
    for qa_pair in test_dataset.qa_pairs:
        hop_counts[qa_pair['hop_count']] += 1
    
    logging.info("Test set distribution:")
    for hop, count in hop_counts.items():
        if count > 0:
            logging.info(f"  {hop}-hop: {count} questions")
    
    # Load trained models
    logging.info("\nLoading trained models...")
    client_models, client_embeddings, global_relation_embedding = load_trained_models(
        args, all_clients_data, global_nrelation, device
    )
    
    # Initialize server
    logging.info("\nInitializing federated server...")
    server = FederatedKGQAServer(args.num_clients, all_clients_data)
    server.client_models = client_models
    server.client_embeddings = client_embeddings
    
    # Evaluate on test set
    logging.info("\nStarting evaluation on server test set...")
    metrics = server.evaluate_test_set(test_dataset, top_k=10)
    
    # Print results
    server.print_metrics(metrics)
    
    # Save results to JSON
    results_file = os.path.join(args.state_dir, 'server_test_results.json')
    
    # Convert to serializable format
    serializable_metrics = {
        'total': metrics['total'],
        'hits@1': float(metrics.get('hits@1', 0)),
        'hits@3': float(metrics.get('hits@3', 0)),
        'hits@10': float(metrics.get('hits@10', 0)),
        'mrr': float(metrics.get('mrr', 0)),
        'no_answer': metrics['no_answer'],
        'entity_not_found': metrics['entity_not_found'],
        'by_hop': {
            str(hop): {
                'total': m['total'],
                'hits@1': float(m['hits@1']),
                'mrr': float(m['mrr'])
            }
            for hop, m in metrics['by_hop'].items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logging.info(f"\nResults saved to: {results_file}")
    
    logging.info("\n" + "=" * 70)
    logging.info("Evaluation completed!")
    logging.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server-side evaluation for Federated KGQA')
    
    # Paths
    parser.add_argument('--client_data_path', type=str, required=True,
                       help='Path to client data (federated_clients/)')
    parser.add_argument('--server_test_path', type=str, required=True,
                       help='Path to server test directory (federated_server/)')
    parser.add_argument('--state_dir', type=str, required=True,
                       help='Directory containing saved models (./state/)')
    
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
    
    # System
    parser.add_argument('--gpu', type=str, default='-1',
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    main(args)
