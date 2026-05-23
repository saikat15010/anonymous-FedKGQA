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

from qa_dataloader_centralized_client1 import (
    load_centralized_data_from_clients,
    create_centralized_dataloaders
)
from centralized_kgqa_ablation2 import CentralizedKGQA
import qa_model_roberta


def init_dir(args):
    """Initialize directories"""
    for dir_path in [args.state_dir, args.log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def init_logger(args):
    """Initialize logger"""
    log_file = os.path.join(args.log_dir, args.name + '.log')
    
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    file_handler = logging.FileHandler(log_file, mode='a+')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(args):
    """Main function"""
    
    init_dir(args)
    init_logger(args)
    
    args_dict = vars(args).copy()
    args_dict['gpu'] = str(args.gpu)
    args_str = json.dumps(args_dict, indent=2)
    logging.info("=" * 70)
    logging.info("ABLATION STUDY 2: CENTRALIZED TRAINING")
    logging.info("ComplEx + RoBERTa - Centralized (No Federation)")
    logging.info("=" * 70)
    logging.info("Arguments:")
    logging.info(args_str)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu != torch.device("cpu"):
        torch.cuda.manual_seed(args.seed)
    
    logging.info("\n" + "=" * 70)
    logging.info("Loading Centralized Data from Client1 Federated Structure")
    logging.info("=" * 70)
    
    # Load data from Client1 (combines all clients, test from server)
    kb_data, qa_data = load_centralized_data_from_clients(args.data_path, args.num_clients)
    
    logging.info(f"\nDataset Statistics:")
    logging.info(f"  Entities: {kb_data['nentity']}")
    logging.info(f"  Relations: {kb_data['nrelation']}")
    logging.info(f"  Triples: {len(kb_data['triples'])}")
    logging.info(f"  Train QA: {len(qa_data['train'])}")
    logging.info(f"  Dev QA: {len(qa_data['dev'])}")
    logging.info(f"  Test QA: {len(qa_data['test'])}")
    
    # Create dataloaders
    logging.info("\nCreating dataloaders...")
    kg_dataloader, train_qa_loader, dev_qa_loader, test_qa_loader = create_centralized_dataloaders(
        kb_data, qa_data, args
    )
    
    # Initialize system
    logging.info("\n" + "=" * 70)
    logging.info("Initializing Centralized KGQA System")
    logging.info("=" * 70)
    logging.info(f"Model: ComplEx for KG, RoBERTa for QA")
    logging.info(f"Embedding dimension: {args.hidden_dim}")
    logging.info(f"Device: {args.gpu}")
    logging.info("No federation - single centralized model")
    
    centralized_kgqa = CentralizedKGQA(args, kb_data, kb_data['nrelation'], qa_model_roberta)
    centralized_kgqa.setup_dataloaders(kg_dataloader, train_qa_loader, dev_qa_loader)
    
    # Train
    logging.info("\n" + "=" * 70)
    logging.info("Starting Training")
    logging.info("=" * 70)
    centralized_kgqa.train()
    
    logging.info("\n" + "=" * 70)
    logging.info("Training completed successfully!")
    logging.info("=" * 70)
    logging.info(f"\nModels saved to: {args.state_dir}/best_models/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study 2: Centralized KGQA using Client1 Data')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Client1 directory (contains federated_clients and federated_server)')
    parser.add_argument('--num_clients', type=int, required=True,
                       help='Number of clients in federated_clients to combine')
    parser.add_argument('--name', type=str, default='ablation2_centralized',
                       help='Experiment name')
    parser.add_argument('--state_dir', type=str, default='./state_ablation2',
                       help='Directory to save model states')
    parser.add_argument('--log_dir', type=str, default='./log_ablation2',
                       help='Directory for logs')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--gamma', type=float, default=12.0,
                       help='Margin for ComplEx')
    parser.add_argument('--epsilon', type=float, default=2.0,
                       help='Epsilon for embedding initialization')
    
    # Training parameters - KG
    parser.add_argument('--kg_max_rounds', type=int, default=50,
                       help='Maximum epochs for KG training')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for KG training')
    parser.add_argument('--num_neg', type=int, default=256,
                       help='Number of negative samples for KG training')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate for KG training')
    parser.add_argument('--adversarial_temperature', type=float, default=1.0,
                       help='Temperature for self-adversarial negative sampling')
    parser.add_argument('--reg_lambda', type=float, default=0.0,
                       help='L2 regularization lambda')
    
    # Training parameters - QA
    parser.add_argument('--qa_max_rounds', type=int, default=50,
                       help='Maximum epochs for QA training')
    parser.add_argument('--qa_batch_size', type=int, default=16,
                       help='Batch size for QA training')
    parser.add_argument('--qa_lr', type=float, default=1e-5,
                       help='Learning rate for QA training')
    parser.add_argument('--num_neg_qa', type=int, default=128,
                       help='Number of negative samples for QA ranking loss')
    
    # Other
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--log_per_round', type=int, default=1,
                       help='Log every N rounds/epochs')
    parser.add_argument('--check_per_round', type=int, default=5,
                       help='Evaluate every N rounds/epochs')
    parser.add_argument('--gpu', type=str, default='-1',
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.gpu == '-1' or not torch.cuda.is_available():
        args.gpu = torch.device("cpu")
    else:
        args.gpu = torch.device(f'cuda:{args.gpu}')
    
    main(args)