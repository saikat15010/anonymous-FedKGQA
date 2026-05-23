"""
Evaluation Script for Ablation Study 2: Centralized Training
Evaluates centralized model on test set
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
torch.set_num_threads(1)

import sys
import numpy as np
import argparse
import logging
import json
from collections import defaultdict
from torch.utils.data import DataLoader

from qa_dataloader_centralized_client1 import (
    load_centralized_data_from_clients,
    CentralizedQADataset
)
from torch.utils.data import DataLoader
import qa_model_roberta


def init_logger():
    """Initialize logger"""
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console_handler)


def evaluate_ablation2(args):
    """Evaluate Ablation 2 (Centralized) model on test set"""
    
    logging.info("=" * 70)
    logging.info("ABLATION STUDY 2 EVALUATION: CENTRALIZED TRAINING")
    logging.info("=" * 70)
    
    # Load centralized data from Client1
    logging.info("\nLoading centralized data from Client1...")
    kb_data, qa_data = load_centralized_data_from_clients(args.data_path, args.num_clients)
    
    logging.info(f"Entities: {kb_data['nentity']}")
    logging.info(f"Relations: {kb_data['nrelation']}")
    logging.info(f"Test questions: {len(qa_data['test'])}")
    
    # Create test dataloader
    test_loader = DataLoader(
        qa_data['test'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CentralizedQADataset.collate_fn
    )
    
    # Load saved model
    logging.info("\nLoading saved model...")
    best_models_dir = os.path.join(args.state_dir, 'best_models')
    
    # Load embeddings
    entity_embedding = torch.load(
        os.path.join(best_models_dir, 'entity_embeddings.pt'),
        map_location=args.gpu
    )
    
    relation_embedding = torch.load(
        os.path.join(best_models_dir, 'relation_embeddings.pt'),
        map_location=args.gpu
    )
    
    # Load QA model
    qa_model = qa_model_roberta.ImprovedKGQAModel(
        args,
        kb_data['nentity'],
        kb_data['nrelation']
    ).to(args.gpu)
    
    qa_model.load_state_dict(torch.load(
        os.path.join(best_models_dir, 'qa_model.pt'),
        map_location=args.gpu
    ))
    qa_model.eval()
    
    logging.info("Model loaded successfully")
    
    # Evaluate
    logging.info("\n" + "=" * 70)
    logging.info("Evaluating on Test Set")
    logging.info("=" * 70)
    
    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    reciprocal_ranks = []
    
    results_by_hop = defaultdict(lambda: {'correct_at_1': 0, 'rr': [], 'total': 0})
    detailed_results = []
    
    question_count = 0
    
    with torch.no_grad():
        for batch_idx, (questions, answer_ids_list, hop_counts) in enumerate(test_loader):
            # Get entity scores
            entity_scores, _, _ = qa_model(
                questions,
                relation_embedding,
                entity_embedding,
                entity2id=kb_data['entity2id'],
                answer_ids=None
            )
            
            # Process each question in batch
            for i, (question, answer_ids, hop_count) in enumerate(zip(questions, answer_ids_list, hop_counts)):
                if len(answer_ids) == 0:
                    logging.warning(f"No valid answer for question: {question}")
                    continue
                
                answer_id = answer_ids[0]  # Use first answer
                scores = entity_scores[i].cpu().numpy()
                
                # Get ranking
                sorted_indices = np.argsort(-scores)
                rank = np.where(sorted_indices == answer_id)[0][0] + 1
                
                # Update metrics
                if rank == 1:
                    correct_at_1 += 1
                    results_by_hop[hop_count]['correct_at_1'] += 1
                if rank <= 3:
                    correct_at_3 += 1
                if rank <= 5:
                    correct_at_5 += 1
                if rank <= 10:
                    correct_at_10 += 1
                
                rr = 1.0 / rank
                reciprocal_ranks.append(rr)
                results_by_hop[hop_count]['rr'].append(rr)
                results_by_hop[hop_count]['total'] += 1
                
                # Store detailed result
                answer_entity = kb_data['id2entity'][answer_id]
                detailed_results.append({
                    'question': question,
                    'answer': answer_entity,
                    'rank': int(rank),
                    'hop_count': hop_count
                })
                
                question_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Processed {question_count} questions")
    
    # Calculate metrics
    total = len(reciprocal_ranks)
    hits_at_1 = correct_at_1 / total
    hits_at_3 = correct_at_3 / total
    hits_at_5 = correct_at_5 / total
    hits_at_10 = correct_at_10 / total
    mrr = np.mean(reciprocal_ranks)
    
    # Print results
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION RESULTS - ABLATION 2 (CENTRALIZED)")
    logging.info("=" * 70)
    logging.info(f"Total Questions: {total}")
    logging.info("")
    logging.info("Overall Metrics:")
    logging.info(f"  Hits@1:  {hits_at_1:.4f}")
    logging.info(f"  Hits@3:  {hits_at_3:.4f}")
    logging.info(f"  Hits@5:  {hits_at_5:.4f}")
    logging.info(f"  Hits@10: {hits_at_10:.4f}")
    logging.info(f"  MRR:     {mrr:.4f}")
    
    # Results by hop count
    for hop_count in sorted(results_by_hop.keys()):
        hop_data = results_by_hop[hop_count]
        hop_total = hop_data['total']
        hop_hits_at_1 = hop_data['correct_at_1'] / hop_total
        hop_mrr = np.mean(hop_data['rr'])
        
        logging.info(f"\n{hop_count}-hop Questions ({hop_total} questions):")
        logging.info(f"  Hits@1: {hop_hits_at_1:.4f}")
        logging.info(f"  MRR:    {hop_mrr:.4f}")
    
    # Save results to file
    if args.output_file:
        results = {
            'overall': {
                'hits@1': float(hits_at_1),
                'hits@3': float(hits_at_3),
                'hits@5': float(hits_at_5),
                'hits@10': float(hits_at_10),
                'mrr': float(mrr),
                'total_questions': total
            },
            'by_hop': {},
            'detailed': detailed_results
        }
        
        for hop_count, hop_data in results_by_hop.items():
            results['by_hop'][f'{hop_count}-hop'] = {
                'hits@1': float(hop_data['correct_at_1'] / hop_data['total']),
                'mrr': float(np.mean(hop_data['rr'])),
                'total': hop_data['total']
            }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\nDetailed results saved to: {args.output_file}")
    
    logging.info("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Ablation 2 (Centralized using Client1) on Test Set')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Client1 directory (contains federated_clients and federated_server)')
    parser.add_argument('--num_clients', type=int, required=True,
                       help='Number of clients in federated_clients')
    parser.add_argument('--state_dir', type=str, required=True,
                       help='Directory with saved models')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='eval_results_ablation2.json',
                       help='Output file for detailed results')
    parser.add_argument('--gpu', type=str, default='-1',
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.gpu == '-1' or not torch.cuda.is_available():
        args.gpu = torch.device("cpu")
    else:
        args.gpu = torch.device(f'cuda:{args.gpu}')
    
    # Other args needed by QA model
    args.gamma = 12.0
    args.epsilon = 2.0
    args.num_neg_qa = 128
    
    init_logger()
    evaluate_ablation2(args)