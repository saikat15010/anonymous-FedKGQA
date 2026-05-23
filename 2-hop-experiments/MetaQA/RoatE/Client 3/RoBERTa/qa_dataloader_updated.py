import numpy as np
import torch
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
import re
import os


class QADataset(Dataset):
    """Dataset for Question Answering pairs with hop detection"""
    def __init__(self, qa_file, entity2id, relation2id):
        self.qa_pairs = []
        self.entity2id = entity2id
        self.relation2id = relation2id
        
        # Load QA pairs
        with open(qa_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: question\tanswer1|answer2|...
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split('|')
                    
                    # Detect hop count from question structure
                    hop_count = self.detect_hop_count(question)
                    
                    self.qa_pairs.append({
                        'question': question,
                        'answers': answers,
                        'hop_count': hop_count
                    })
    
    def detect_hop_count(self, question):
        """
        Detect hop count from question structure
        
        All MetaQA 2-hop questions have compound structure with indicators
        """
        question_lower = question.lower()
        
        # 2-hop indicators - any of these make it 2-hop
        two_hop_indicators = [
            # Actor -> Movies -> Property patterns
            'starred by', 'acted by', 'acted together', 'appeared in', 'co-starred',
            'same actor', 'same movie', 'same director', 'same screenwriter', 'same writer',
            
            # Director -> Movies -> Property patterns  
            'directed by', 'films directed', 'movies directed', 'also directed',
            'co-directed', 'same director',
            
            # Writer -> Movies -> Property patterns
            'written by', 'wrote', 'films written', 'movies written', 'co-wrote',
            'screenwriter', 'scriptwriter', 'same screenwriter', 'same writer',
            
            # Movie -> Person -> Movies patterns
            'director of', 'writer of', 'actor of', 'actor in',
            'also directed', 'also wrote', 'also starred', 'also appears',
            
            # Compound property queries
            'were in which', 'were released', 'were written',
            'starred who', 'starred which', 'directed who',
        ]
        
        for indicator in two_hop_indicators:
            if indicator in question_lower:
                return 2
        
        return 1
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        question = qa_pair['question']
        answers = qa_pair['answers']
        hop_count = qa_pair['hop_count']
        
        # Convert answers to entity IDs
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


class ServerTestDataset(Dataset):
    """
    Dataset for server-side test set
    Does NOT require entity2id mapping since entities may be across clients
    """
    def __init__(self, test_file):
        self.qa_pairs = []
        
        # Load QA pairs
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: question\tanswer1|answer2|...
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split('|')
                    
                    # Detect hop count
                    hop_count = self.detect_hop_count(question)
                    
                    # Extract topic entity
                    topic_entity = self.extract_topic_entity(question)
                    
                    self.qa_pairs.append({
                        'question': question,
                        'answers': answers,  # Keep as strings
                        'hop_count': hop_count,
                        'topic_entity': topic_entity
                    })
    
    def detect_hop_count(self, question):
        """Detect hop count from question structure"""
        question_lower = question.lower()
        
        # 2-hop indicators
        two_hop_indicators = [
            'starred by', 'acted by', 'acted together', 'appeared in', 'co-starred',
            'same actor', 'same movie', 'same director', 'same screenwriter', 'same writer',
            'directed by', 'films directed', 'movies directed', 'also directed', 'co-directed',
            'written by', 'wrote', 'films written', 'movies written', 'co-wrote',
            'screenwriter', 'scriptwriter',
            'director of', 'writer of', 'actor of', 'actor in',
            'also directed', 'also wrote', 'also starred', 'also appears',
            'were in which', 'were released', 'were written',
            'starred who', 'starred which', 'directed who',
        ]
        
        for indicator in two_hop_indicators:
            if indicator in question_lower:
                return 2
        
        return 1
    
    def extract_topic_entity(self, question):
        """Extract topic entity from question (entity in brackets)"""
        pattern = re.compile(r'\[(.*?)\]')
        match = pattern.search(question)
        return match.group(1) if match else None
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]


def load_kb_file(kb_file):
    """Load knowledge base triples from kb.txt file"""
    triples = []
    entities = set()
    relations = set()
    
    with open(kb_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: head|relation|tail
            parts = line.split('|')
            if len(parts) == 3:
                head, relation, tail = parts
                entities.add(head)
                entities.add(tail)
                relations.add(relation)
                triples.append((head, relation, tail))
    
    # Create mappings
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    id2entity = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}
    
    # Convert triples to IDs
    triples_id = [(entity2id[h], relation2id[r], entity2id[t]) 
                  for h, r, t in triples]
    
    return triples_id, entity2id, relation2id, id2entity, id2relation, list(entities), list(relations)


def load_metaqa_client(client_path):
    """Load all data for a single MetaQA client"""
    # Load KB
    kb_file = os.path.join(client_path, 'kb.txt')
    triples, entity2id, relation2id, id2entity, id2relation, entities, relations = load_kb_file(kb_file)
    
    # Load QA datasets
    qa_train_file = os.path.join(client_path, 'qa_train.txt')
    qa_dev_file = os.path.join(client_path, 'qa_dev.txt')
    
    # Create QA datasets (no test set at client level)
    train_qa_dataset = QADataset(qa_train_file, entity2id, relation2id)
    dev_qa_dataset = QADataset(qa_dev_file, entity2id, relation2id)
    
    return {
        'triples': triples,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'id2entity': id2entity,
        'id2relation': id2relation,
        'entities': entities,
        'relations': relations,
        'train_qa': train_qa_dataset,
        'dev_qa': dev_qa_dataset,
        'nentity': len(entity2id),
        'nrelation': len(relation2id)
    }


def load_all_metaqa_clients(base_path, num_clients=3):
    """Load data for all MetaQA clients"""
    all_clients_data = []
    
    for i in range(num_clients):
        client_path = os.path.join(base_path, f'client_{i}')
        client_data = load_metaqa_client(client_path)
        client_data['client_id'] = i
        all_clients_data.append(client_data)
    
    return all_clients_data


def load_server_test_set(server_path):
    """Load server-side test set"""
    test_file = os.path.join(server_path, 'qa_test.txt')
    test_dataset = ServerTestDataset(test_file)
    return test_dataset


def get_global_relation_mapping(all_clients_data):
    """Create global relation mapping across all clients (simulating PSU)"""
    all_relations = set()
    
    for client_data in all_clients_data:
        all_relations.update(client_data['relations'])
    
    # Global relation to ID mapping
    global_relation2id = {r: i for i, r in enumerate(sorted(all_relations))}
    global_id2relation = {i: r for r, i in global_relation2id.items()}
    
    # Update each client's relation IDs to match global mapping
    for client_data in all_clients_data:
        # Create mapping from local to global relation IDs
        local_to_global = {}
        for rel in client_data['relations']:
            local_id = client_data['relation2id'][rel]
            global_id = global_relation2id[rel]
            local_to_global[local_id] = global_id
        
        client_data['local_to_global_rel'] = local_to_global
        client_data['global_nrelation'] = len(global_relation2id)
        client_data['global_relation2id'] = global_relation2id
        client_data['global_id2relation'] = global_id2relation
        
        # Remap triples to use global relation IDs
        triples_global = []
        for h, r, t in client_data['triples']:
            triples_global.append((h, local_to_global[r], t))
        client_data['triples_global'] = triples_global
    
    return global_relation2id, global_id2relation, len(global_relation2id)


class KGTrainDataset(Dataset):
    """Dataset for KG training with ComplEx"""
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        
        # Build (h,r) -> t mapping for filtering
        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0
        
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.isin(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        
        return positive_sample, negative_sample, idx
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


def create_kg_dataloaders(all_clients_data, args):
    """Create KG training dataloaders for all clients"""
    kg_dataloaders = []
    
    for client_data in all_clients_data:
        triples = client_data['triples_global']
        nentity = client_data['nentity']
        
        kg_dataset = KGTrainDataset(triples, nentity, args.num_neg)
        kg_dataloader = DataLoader(
            kg_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=KGTrainDataset.collate_fn
        )
        kg_dataloaders.append(kg_dataloader)
    
    return kg_dataloaders


def create_qa_dataloaders(all_clients_data, args):
    """Create QA dataloaders for all clients (train and dev only)"""
    train_qa_loaders = []
    dev_qa_loaders = []
    
    for client_data in all_clients_data:
        train_loader = DataLoader(
            client_data['train_qa'],
            batch_size=args.qa_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=QADataset.collate_fn
        )
        
        dev_loader = DataLoader(
            client_data['dev_qa'],
            batch_size=args.qa_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=QADataset.collate_fn
        )
        
        train_qa_loaders.append(train_loader)
        dev_qa_loaders.append(dev_loader)
    
    return train_qa_loaders, dev_qa_loaders