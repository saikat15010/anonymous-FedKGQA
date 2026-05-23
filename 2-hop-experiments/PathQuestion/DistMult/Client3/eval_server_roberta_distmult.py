"""
Server-side Evaluation Script for RoBERTa-DistMult on PathQuestion
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import argparse
import logging
import json
from collections import defaultdict

from qa_dataloader_pathquestion import (
    load_all_pathquestion_clients,
    get_global_relation_mapping,
    ServerTestDataset
)
import qa_model_roberta


def load_trained_models(args, all_clients_data, global_nrelation, device):
    """Load trained models from checkpoint"""
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
        
        # Initialize QA model
        qa_model = qa_model_roberta.ImprovedKGQAModel(args, client_data['nentity'], global_nrelation)
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


def evaluate_test_set(args, all_clients_data, client_models, client_embeddings, test_dataset):
    """Evaluate on test set"""
    
    # Build entity routing
    entity_to_clients = {}
    for client_id, client_data in enumerate(all_clients_data):
        for entity in client_data['entities']:
            if entity not in entity_to_clients:
                entity_to_clients[entity] = []
            entity_to_clients[entity].append(client_id)
    
    total_hits_1 = 0
    total_hits_3 = 0
    total_hits_5 = 0
    total_hits_10 = 0
    total_mrr = 0
    total_count = 0
    
    hop_metrics = {1: {'count': 0, 'hits@1': 0, 'mrr': 0},
                   2: {'count': 0, 'hits@1': 0, 'mrr': 0}}
    
    results_list = []
    
    for idx, qa_pair in enumerate(test_dataset):
        question = qa_pair['question']
        ground_truth = qa_pair['answers']
        hop_count = qa_pair['hop_count']
        topic_entity = qa_pair['topic_entity']
        
        # Route to appropriate client(s)
        if topic_entity and topic_entity in entity_to_clients:
            client_ids = entity_to_clients[topic_entity]
        else:
            # Try all clients if topic entity not found
            client_ids = list(range(len(all_clients_data)))
        
        # Query clients and aggregate answers
        all_predictions = []
        
        for client_id in client_ids:
            client_data = all_clients_data[client_id]
            qa_model = client_models[client_id]
            entity_embedding = client_embeddings[client_id]['entity']
            relation_embedding = client_embeddings[client_id]['relation']
            
            with torch.no_grad():
                top_k_ids, top_k_scores = qa_model.predict_answers(
                    [question],
                    relation_embedding,
                    entity_embedding,
                    entity2id=client_data['entity2id'],
                    top_k=10
                )
                
                # Convert IDs to entity names
                for pred_id, score in zip(top_k_ids[0].cpu().numpy(), top_k_scores[0].cpu().numpy()):
                    entity_name = client_data['id2entity'].get(pred_id, f"entity_{pred_id}")
                    all_predictions.append((entity_name, float(score)))
        
        # Sort by score and deduplicate
        all_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)
        seen = set()
        unique_predictions = []
        for entity, score in all_predictions:
            if entity not in seen:
                seen.add(entity)
                unique_predictions.append(entity)
            if len(unique_predictions) >= 10:
                break
        
        # Evaluate
        found_rank = None
        for rank, pred_entity in enumerate(unique_predictions):
            if pred_entity in ground_truth:
                found_rank = rank
                break
        
        if found_rank is not None:
            if found_rank < 1:
                total_hits_1 += 1
                hop_metrics[hop_count]['hits@1'] += 1
            if found_rank < 3:
                total_hits_3 += 1
            if found_rank < 5:
                total_hits_5 += 1
            if found_rank < 10:
                total_hits_10 += 1
            
            mrr_score = 1.0 / (found_rank + 1)
            total_mrr += mrr_score
            hop_metrics[hop_count]['mrr'] += mrr_score
        
        total_count += 1
        hop_metrics[hop_count]['count'] += 1
        
        # Store result
        results_list.append({
            'question': question,
            'ground_truth': ground_truth,
            'predictions': unique_predictions[:10],
            'correct': found_rank is not None and found_rank < 10,
            'rank': found_rank + 1 if found_rank is not None else None
        })
        
        if (idx + 1) % 100 == 0:
            logging.info(f"Evaluated {idx + 1}/{len(test_dataset)} questions")
    
    # Compute metrics
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
        'total_questions': total_count,
        'hop_metrics': hop_metrics,
        'detailed_results': results_list
    }
    
    return results


def main(args):
    """Main evaluation function"""
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    
    logging.info("=" * 70)
    logging.info("RoBERTa-DistMult Server Evaluation on PathQuestion")
    logging.info("=" * 70)
    
    # Setup device
    if args.gpu == '-1' or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f'cuda:{args.gpu}')
    
    args.gpu = device
    logging.info(f"Using device: {device}")
    
    # Load client data
    logging.info("\nLoading client data...")
    all_clients_data = load_all_pathquestion_clients(
        args.client_data_path,
        num_clients=args.num_clients
    )
    
    global_relation2id, global_id2relation, global_nrelation = get_global_relation_mapping(all_clients_data)
    
    # Load test dataset
    logging.info(f"\nLoading test dataset from {args.test_file}")
    test_dataset = ServerTestDataset(args.test_file)
    logging.info(f"Loaded {len(test_dataset)} test questions")
    
    # Load trained models
    logging.info("\nLoading trained models...")
    client_models, client_embeddings, global_relation_embedding = load_trained_models(
        args, all_clients_data, global_nrelation, device
    )
    
    # Evaluate
    logging.info("\n" + "=" * 70)
    logging.info("Starting Evaluation")
    logging.info("=" * 70)
    
    results = evaluate_test_set(args, all_clients_data, client_models, client_embeddings, test_dataset)
    
    # Print results
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 70)
    logging.info(f"Total Questions: {results['total_questions']}")
    logging.info(f"\nOverall Metrics:")
    logging.info(f"  Hits@3:  {results['hits@3']:.4f}")
    logging.info(f"  Hits@5:  {results['hits@5']:.4f}")
    logging.info(f"  Hits@10: {results['hits@10']:.4f}")
    logging.info(f"  MRR:     {results['mrr']:.4f}")
    logging.info(f"  Hits@1:  {results['hits@1']:.4f}")
    
    for hop in [1, 2]:
        if results['hop_metrics'][hop]['count'] > 0:
            logging.info(f"\n{hop}-hop Questions ({results['hop_metrics'][hop]['count']} questions):")
            logging.info(f"  Hits@1: {results['hop_metrics'][hop]['hits@1']:.4f}")
            logging.info(f"  MRR:    {results['hop_metrics'][hop]['mrr']:.4f}")
    
    # Save results
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RoBERTa-DistMult on PathQuestion')
    
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to qa_test.txt')
    parser.add_argument('--client_data_path', type=str, required=True,
                       help='Path to federated_clients/')
    parser.add_argument('--state_dir', type=str, required=True,
                       help='Directory containing saved models')
    parser.add_argument('--num_clients', type=int, required=True,
                       help='Number of clients')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=12.0)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--output_file', type=str, default='eval_results_roberta.json')
    parser.add_argument('--gpu', type=str, default='-1')
    
    args = parser.parse_args()
    main(args)
