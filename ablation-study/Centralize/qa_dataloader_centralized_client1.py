"""
Centralized Data Loader using Client1 Federated Data

Combines all client data into one centralized dataset
Uses federated_server test set for evaluation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


def load_kb_from_clients(client_data_path, num_clients):
    """
    Load and combine KB from all clients
    
    Args:
        client_data_path: Path to federated_clients directory
        num_clients: Number of clients
    
    Returns:
        Combined KB data with unified entity/relation mappings
    """
    all_triples = []
    all_entities = set()
    all_relations = set()
    
    # Load from all clients
    for client_id in range(num_clients):
        client_dir = os.path.join(client_data_path, f'client_{client_id}')
        kb_file = os.path.join(client_dir, 'kb.txt')
        
        if not os.path.exists(kb_file):
            raise FileNotFoundError(f"KB file not found: {kb_file}")
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    all_entities.add(head)
                    all_entities.add(tail)
                    all_relations.add(relation)
                    all_triples.append((head, relation, tail))
    
    # Create unified mappings
    entity2id = {e: i for i, e in enumerate(sorted(all_entities))}
    relation2id = {r: i for i, r in enumerate(sorted(all_relations))}
    id2entity = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}
    
    # Convert triples to IDs
    triples_id = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in all_triples]
    
    print(f"Combined KB from {num_clients} clients:")
    print(f"  Entities: {len(entity2id)}")
    print(f"  Relations: {len(relation2id)}")
    print(f"  Triples: {len(triples_id)}")
    
    return {
        'triples': triples_id,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'id2entity': id2entity,
        'id2relation': id2relation,
        'nentity': len(entity2id),
        'nrelation': len(relation2id)
    }


def load_qa_from_clients(client_data_path, num_clients, entity2id, relation2id):
    """
    Load and combine QA data from all clients
    
    Args:
        client_data_path: Path to federated_clients directory
        num_clients: Number of clients
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
    
    Returns:
        train_qa_list, dev_qa_list (test comes from server)
    """
    all_train_qa = []
    all_dev_qa = []
    
    # Load from all clients
    for client_id in range(num_clients):
        client_dir = os.path.join(client_data_path, f'client_{client_id}')
        
        # Load train QA
        train_file = os.path.join(client_dir, 'qa_train.txt')
        if os.path.exists(train_file):
            train_qa = load_qa_file(train_file, entity2id)
            all_train_qa.extend(train_qa)
        
        # Load dev QA
        dev_file = os.path.join(client_dir, 'qa_dev.txt')
        if os.path.exists(dev_file):
            dev_qa = load_qa_file(dev_file, entity2id)
            all_dev_qa.extend(dev_qa)
    
    print(f"Combined QA from {num_clients} clients:")
    print(f"  Train: {len(all_train_qa)}")
    print(f"  Dev: {len(all_dev_qa)}")
    
    return all_train_qa, all_dev_qa


def load_server_test_qa(server_path, entity2id):
    """
    Load test QA from federated_server
    
    Args:
        server_path: Path to federated_server directory
        entity2id: Entity to ID mapping
    
    Returns:
        test_qa_list
    """
    test_file = os.path.join(server_path, 'qa_test.txt')
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_qa = load_qa_file(test_file, entity2id)
    
    print(f"Loaded server test set:")
    print(f"  Test: {len(test_qa)}")
    
    return test_qa


def load_qa_file(qa_file, entity2id):
    """Load QA pairs from file"""
    qa_pairs = []
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                question = parts[0]
                answer = parts[1]
                
                # Detect hop count
                hop_count = detect_hop_count(question)
                
                qa_pairs.append({
                    'question': question,
                    'answers': [answer],
                    'hop_count': hop_count
                })
    
    return qa_pairs


def detect_hop_count(question):
    """Detect hop count from question"""
    question_lower = question.lower()
    possessive_count = question_lower.count("'s")
    of_count = question_lower.count(" of ")
    
    if possessive_count >= 2 or of_count >= 2:
        return 2
    
    two_hop_indicators = [
        "parent", "child", "spouse", "couple", "darling",
        "father", "mother", "son", "daughter", "husband", "wife"
    ]
    
    for indicator in two_hop_indicators:
        if indicator in question_lower and (possessive_count > 0 or of_count > 0):
            return 2
    
    return 1


class CentralizedQADataset(Dataset):
    """QA dataset for centralized training"""
    
    def __init__(self, qa_data, entity2id, relation2id):
        self.qa_pairs = qa_data
        self.entity2id = entity2id
        self.relation2id = relation2id
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        question = qa_pair['question']
        answers = qa_pair['answers']
        hop_count = qa_pair['hop_count']
        
        # Convert answers to IDs
        answer_ids = []
        for ans in answers:
            if ans in self.entity2id:
                answer_ids.append(self.entity2id[ans])
        
        return question, answer_ids, hop_count
    
    @staticmethod
    def collate_fn(batch):
        questions = [item[0] for item in batch]
        answers = [item[1] for item in batch]
        hop_counts = [item[2] for item in batch]
        return questions, answers, hop_counts


class CentralizedKGDataset(Dataset):
    """KG dataset for centralized training with negative sampling"""
    
    def __init__(self, triples, nentity, num_neg):
        self.triples = np.array(triples)
        self.nentity = nentity
        self.num_neg = num_neg
        self.num_triples = len(triples)
    
    def __len__(self):
        return self.num_triples
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        
        # Negative sampling - corrupt tail
        negative_sample = []
        while len(negative_sample) < self.num_neg:
            neg_tail = np.random.randint(0, self.nentity)
            if neg_tail != tail:
                negative_sample.append(neg_tail)
        
        return positive_sample, np.array(negative_sample), 0
    
    @staticmethod
    def collate_fn(batch):
        positive_samples = torch.LongTensor(np.array([item[0] for item in batch]))
        negative_samples = torch.LongTensor(np.array([item[1] for item in batch]))
        mode = 0
        return positive_samples, negative_samples, mode


def load_centralized_data_from_clients(base_path, num_clients):
    """
    Load centralized data from Client1 federated structure
    
    Args:
        base_path: Path to Client1 directory (contains federated_clients and federated_server)
        num_clients: Number of clients in federated_clients
    
    Returns:
        kb_data: Combined KB from all clients
        qa_data: Combined QA with train/dev from clients, test from server
    """
    client_data_path = os.path.join(base_path, 'federated_clients')
    server_path = os.path.join(base_path, 'federated_server')
    
    # Check paths exist
    if not os.path.exists(client_data_path):
        raise FileNotFoundError(f"Client data path not found: {client_data_path}")
    if not os.path.exists(server_path):
        raise FileNotFoundError(f"Server path not found: {server_path}")
    
    # Load KB from all clients
    print("\nLoading KB from all clients...")
    kb_data = load_kb_from_clients(client_data_path, num_clients)
    
    # Load QA from all clients
    print("\nLoading QA from all clients...")
    train_qa_list, dev_qa_list = load_qa_from_clients(
        client_data_path, 
        num_clients,
        kb_data['entity2id'],
        kb_data['relation2id']
    )
    
    # Load test QA from server
    print("\nLoading test QA from server...")
    test_qa_list = load_server_test_qa(server_path, kb_data['entity2id'])
    
    # Create datasets
    train_qa_dataset = CentralizedQADataset(train_qa_list, kb_data['entity2id'], kb_data['relation2id'])
    dev_qa_dataset = CentralizedQADataset(dev_qa_list, kb_data['entity2id'], kb_data['relation2id'])
    test_qa_dataset = CentralizedQADataset(test_qa_list, kb_data['entity2id'], kb_data['relation2id'])
    
    qa_data = {
        'train': train_qa_dataset,
        'dev': dev_qa_dataset,
        'test': test_qa_dataset
    }
    
    return kb_data, qa_data


def create_centralized_dataloaders(kb_data, qa_data, args):
    """
    Create dataloaders for centralized training
    
    Args:
        kb_data: KB data dict
        qa_data: QA data dict with train/dev/test datasets
        args: Training arguments
    
    Returns:
        kg_dataloader, train_qa_loader, dev_qa_loader, test_qa_loader
    """
    # KG dataloader
    kg_dataset = CentralizedKGDataset(kb_data['triples'], kb_data['nentity'], args.num_neg)
    kg_dataloader = DataLoader(
        kg_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CentralizedKGDataset.collate_fn
    )
    
    # QA dataloaders
    train_qa_loader = DataLoader(
        qa_data['train'],
        batch_size=args.qa_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CentralizedQADataset.collate_fn
    )
    
    dev_qa_loader = DataLoader(
        qa_data['dev'],
        batch_size=args.qa_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CentralizedQADataset.collate_fn
    )
    
    test_qa_loader = DataLoader(
        qa_data['test'],
        batch_size=args.qa_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CentralizedQADataset.collate_fn
    )
    
    print(f"\nDataLoaders created:")
    print(f"  KG batches: {len(kg_dataloader)}")
    print(f"  Train QA batches: {len(train_qa_loader)}")
    print(f"  Dev QA batches: {len(dev_qa_loader)}")
    print(f"  Test QA batches: {len(test_qa_loader)}")
    
    return kg_dataloader, train_qa_loader, dev_qa_loader, test_qa_loader