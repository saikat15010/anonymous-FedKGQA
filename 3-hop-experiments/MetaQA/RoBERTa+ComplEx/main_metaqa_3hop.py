import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch; torch.set_num_threads(1)
import sys, numpy as np, argparse, logging, json

# KEY CHANGE: use 3-hop dataloader instead of qa_dataloader_updated
from qa_dataloader_metaqa_3hop import (
    load_all_metaqa_clients, get_global_relation_mapping,
    create_kg_dataloaders, create_qa_dataloaders)
from fkgqa_all import FederatedKGQA


def init_dir(args):
    for d in [args.state_dir, args.log_dir]: os.makedirs(d, exist_ok=True)

def init_logger(args):
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(args.log_dir, args.name+'.log'), mode='a+')
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger = logging.getLogger(); logger.setLevel(logging.INFO); logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(ch)

def main(args):
    init_dir(args); init_logger(args)
    logging.info("="*70)
    logging.info(f"FedKGQA — {args.kge_model.upper()} + {args.lm_model.upper()} | MetaQA 3-hop | Client {args.num_clients}")
    logging.info("="*70)
    logging.info(json.dumps({k: str(v) for k,v in vars(args).items()}, indent=2))
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu != torch.device("cpu"):
        torch.cuda.manual_seed(args.seed)

    clients = load_all_metaqa_clients(args.data_path, args.num_clients)
    logging.info(f"Loaded {len(clients)} clients")
    for i,c in enumerate(clients):
        logging.info(f"  Client {i}: {c['nentity']} ent, {c['nrelation']} rels, "
                     f"{len(c['triples'])} triples, {len(c['train_qa'])} train QA")
    _,_,gnr = get_global_relation_mapping(clients)
    logging.info(f"Global relations: {gnr}")

    kg_dl = create_kg_dataloaders(clients, args)
    tqa, dqa = create_qa_dataloaders(clients, args)

    fkgqa = FederatedKGQA(args, clients, gnr)
    fkgqa.setup_clients(kg_dl, tqa, dqa)
    fkgqa.train()
    logging.info("Training completed!")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--num_clients', type=int, default=3)
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--state_dir', type=str, default='./state')
    p.add_argument('--log_dir', type=str, default='./log')
    p.add_argument('--kge_model', type=str, default='transe',
                   choices=['transe', 'distmult', 'rotate', 'complex'])
    p.add_argument('--lm_model', type=str, default='bert',
                   choices=['roberta', 'bert', 'distilbert'])
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--gamma', type=float, default=12.0)
    p.add_argument('--epsilon', type=float, default=2.0)
    p.add_argument('--transe_norm', type=int, default=1)
    p.add_argument('--kg_max_rounds', type=int, default=10)
    p.add_argument('--local_epoch', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--num_neg', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--adversarial_temperature', type=float, default=1.0)
    p.add_argument('--reg_lambda', type=float, default=0.0)
    p.add_argument('--qa_max_rounds', type=int, default=5)
    p.add_argument('--qa_local_epoch', type=int, default=2)
    p.add_argument('--qa_batch_size', type=int, default=8)
    p.add_argument('--qa_lr', type=float, default=5e-6)
    p.add_argument('--num_neg_qa', type=int, default=256)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--early_stop_patience', type=int, default=5)
    p.add_argument('--log_per_round', type=int, default=1)
    p.add_argument('--check_per_round', type=int, default=2)
    p.add_argument('--gpu', type=str, default='-1')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.name is None:
        args.name = f'fkgqa_{args.kge_model}_{args.lm_model}_3hop'
    args.gpu = torch.device("cpu") if args.gpu=='-1' or not torch.cuda.is_available() \
        else torch.device(f'cuda:{args.gpu}')
    main(args)
