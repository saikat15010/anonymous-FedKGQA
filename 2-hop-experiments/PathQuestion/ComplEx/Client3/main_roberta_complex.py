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

from qa_dataloader_pathquestion import (
    load_all_pathquestion_clients,
    get_global_relation_mapping,
    create_kg_dataloaders,
    create_qa_dataloaders
)
from fkgqa_pathquestion import FederatedKGQA
import qa_model_roberta


def init_dir(args):
    """Initialize directories"""
    for dir_path in [args.state_dir, args.log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def init_logger(args):
    """Initialize logger with immediate console output"""
    log_file = os.path.join(args.log_dir, args.name + '.log')
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a+')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(args):
    """Main function to run Federated KGQA training"""
    
    # Initialize directories and logging
    init_dir(args)
    init_logger(args)
    
    # Log arguments
    args_dict = vars(args).copy()
    args_dict['gpu'] = str(args.gpu)
    args_str = json.dumps(args_dict, indent=2)
    logging.info("=" * 70)
    logging.info("Federated KGQA Training - RoBERTa + ComplEx")
    logging.info("=" * 70)
    logging.info("Arguments:")
    logging.info(args_str)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu != torch.device("cpu"):
        torch.cuda.manual_seed(args.seed)
    
    logging.info("\n" + "=" * 70)
    logging.info("Loading PathQuestion Dataset")
    logging.info("=" * 70)
    
    # Load all client data
    all_clients_data = load_all_pathquestion_clients(
        args.data_path,
        num_clients=args.num_clients
    )
    
    logging.info(f"Loaded {len(all_clients_data)} clients")
    for i, client_data in enumerate(all_clients_data):
        logging.info(
            f"Client {i}: {client_data['nentity']} entities, "
            f"{client_data['nrelation']} local relations, "
            f"{len(client_data['triples'])} triples"
        )
        logging.info(
            f"          Train: {len(client_data['train_qa'])} QA, "
            f"Dev: {len(client_data['dev_qa'])} QA"
        )
    
    # Create global relation mapping
    logging.info("\nCreating global relation mapping...")
    global_relation2id, global_id2relation, global_nrelation = get_global_relation_mapping(all_clients_data)
    logging.info(f"Global relations: {global_nrelation}")
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    kg_dataloaders = create_kg_dataloaders(all_clients_data, args)
    train_qa_loaders, dev_qa_loaders = create_qa_dataloaders(all_clients_data, args)
    
    # Initialize Federated KGQA system
    logging.info("\n" + "=" * 70)
    logging.info("Initializing Federated KGQA System")
    logging.info("=" * 70)
    logging.info(f"Model: ComplEx for KG, RoBERTa for QA")
    logging.info(f"Embedding dimension: {args.hidden_dim}")
    logging.info(f"Device: {args.gpu}")
    
    fkgqa = FederatedKGQA(args, all_clients_data, global_nrelation, qa_model_roberta)
    fkgqa.setup_clients(kg_dataloaders, train_qa_loaders, dev_qa_loaders)
    
    # Train
    logging.info("\n" + "=" * 70)
    logging.info("Starting Training")
    logging.info("=" * 70)
    fkgqa.train()
    
    logging.info("\n" + "=" * 70)
    logging.info("Training completed successfully!")
    logging.info("=" * 70)
    logging.info(f"\nModels saved to: {args.state_dir}/best_models/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated KGQA on PathQuestion with RoBERTa + ComplEx')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to federated dataset (contains client_0, client_1, ...)')
    parser.add_argument('--num_clients', type=int, required=True,
                       help='Number of clients (3, 5, 10, etc.)')
    
    # Output directories
    parser.add_argument('--name', type=str, default='fkgqa_roberta_complex',
                       help='Experiment name')
    parser.add_argument('--state_dir', type=str, default='./state',
                       help='Directory to save model states')
    parser.add_argument('--log_dir', type=str, default='./log',
                       help='Directory for logs')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Embedding dimension (ComplEx: entities=2*hidden_dim, relations=2*hidden_dim)')
    parser.add_argument('--gamma', type=float, default=12.0,
                       help='Margin for ComplEx')
    parser.add_argument('--epsilon', type=float, default=2.0,
                       help='Epsilon for embedding initialization')
    
    # Training parameters - Phase 1 (KG)
    parser.add_argument('--kg_max_rounds', type=int, default=100,
                       help='Maximum rounds for KG training')
    parser.add_argument('--local_epoch', type=int, default=3,
                       help='Local epochs per round for KG training')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for KG training')
    parser.add_argument('--num_neg', type=int, default=256,
                       help='Number of negative samples for KG training')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate for KG training')
    parser.add_argument('--adversarial_temperature', type=float, default=1.0,
                       help='Temperature for self-adversarial negative sampling')
    parser.add_argument('--reg_lambda', type=float, default=0.0,
                       help='L2 regularization lambda (0 to disable)')
    
    # Training parameters - Phase 2 (QA)
    parser.add_argument('--qa_max_rounds', type=int, default=200,
                       help='Maximum rounds for QA training')
    parser.add_argument('--qa_local_epoch', type=int, default=2,
                       help='Local epochs per round for QA training')
    parser.add_argument('--qa_batch_size', type=int, default=16,
                       help='Batch size for QA training (smaller for RoBERTa)')
    parser.add_argument('--qa_lr', type=float, default=1e-5,
                       help='Learning rate for QA training')
    parser.add_argument('--num_neg_qa', type=int, default=128,
                       help='Number of negative samples for QA ranking loss')
    
    # Federated learning parameters
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of clients to sample per round')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Logging
    parser.add_argument('--log_per_round', type=int, default=1,
                       help='Log every N rounds')
    parser.add_argument('--check_per_round', type=int, default=5,
                       help='Evaluate every N rounds')
    
    # System parameters
    parser.add_argument('--gpu', type=str, default='-1',
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.gpu == '-1' or not torch.cuda.is_available():
        args.gpu = torch.device("cpu")
        logging.info("Using CPU")
    else:
        args.gpu = torch.device(f'cuda:{args.gpu}')
        logging.info(f"Using GPU: {args.gpu}")
    
    main(args)
