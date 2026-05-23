"""
QA Dataloader for MetaQA 3-hop (Horizontal Federated)

Same as 2-hop version. The dataloader doesn't need structural changes
since MetaQA uses the same format for all hop levels:
  question\\tanswer1|answer2|...
  KB: head|relation|tail
"""

import numpy as np
import torch
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
import re
import os


class QADataset(Dataset):
    def __init__(self, qa_file, entity2id, relation2id):
        self.qa_pairs = []
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(qa_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split('|')
                    hop_count = self.detect_hop_count(question)
                    self.qa_pairs.append({
                        'question': question,
                        'answers': answers,
                        'hop_count': hop_count
                    })

    def detect_hop_count(self, question):
        # All questions in this dataset are 3-hop
        return 2  # Use 2 for compatibility with metrics dict keys

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        answer_ids = [self.entity2id[a] for a in qa['answers'] if a in self.entity2id]
        return qa['question'], answer_ids, qa['hop_count']

    @staticmethod
    def collate_fn(batch):
        return ([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch])


class ServerTestDataset(Dataset):
    def __init__(self, test_file):
        self.qa_pairs = []
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split('|')
                    topic = self.extract_topic_entity(question)
                    self.qa_pairs.append({
                        'question': question,
                        'answers': answers,
                        'hop_count': 2,
                        'topic_entity': topic
                    })

    def extract_topic_entity(self, question):
        match = re.compile(r'\[(.*?)\]').search(question)
        return match.group(1) if match else None

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        return self.qa_pairs[idx]


def load_kb_file(kb_file):
    triples, entities, relations = [], set(), set()
    with open(kb_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 3:
                h, r, t = parts
                entities.add(h); entities.add(t); relations.add(r)
                triples.append((h, r, t))
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    id2entity = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}
    triples_id = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]
    return triples_id, entity2id, relation2id, id2entity, id2relation, list(entities), list(relations)


def load_metaqa_client(client_path):
    kb_file = os.path.join(client_path, 'kb.txt')
    triples, entity2id, relation2id, id2entity, id2relation, entities, relations = load_kb_file(kb_file)
    train_qa = QADataset(os.path.join(client_path, 'qa_train.txt'), entity2id, relation2id)
    dev_qa = QADataset(os.path.join(client_path, 'qa_dev.txt'), entity2id, relation2id)
    return {
        'triples': triples, 'entity2id': entity2id, 'relation2id': relation2id,
        'id2entity': id2entity, 'id2relation': id2relation,
        'entities': entities, 'relations': relations,
        'train_qa': train_qa, 'dev_qa': dev_qa,
        'nentity': len(entity2id), 'nrelation': len(relation2id)
    }


def load_all_metaqa_clients(base_path, num_clients=3):
    all_clients = []
    for i in range(num_clients):
        data = load_metaqa_client(os.path.join(base_path, f'client_{i}'))
        data['client_id'] = i
        all_clients.append(data)
    return all_clients


def load_server_test_set(server_path):
    test_file = os.path.join(server_path, 'qa_test.txt')
    return ServerTestDataset(test_file)


def get_global_relation_mapping(all_clients_data):
    all_relations = set()
    for cd in all_clients_data:
        all_relations.update(cd['relations'])
    global_r2id = {r: i for i, r in enumerate(sorted(all_relations))}
    global_id2r = {i: r for r, i in global_r2id.items()}
    for cd in all_clients_data:
        local_to_global = {cd['relation2id'][r]: global_r2id[r] for r in cd['relations']}
        cd['local_to_global_rel'] = local_to_global
        cd['global_nrelation'] = len(global_r2id)
        cd['global_relation2id'] = global_r2id
        cd['global_id2relation'] = global_id2r
        cd['triples_global'] = [(h, local_to_global[r], t) for h, r, t in cd['triples']]
    return global_r2id, global_id2r, len(global_r2id)


class KGTrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for k in self.hr2t:
            self.hr2t[k] = np.array(list(self.hr2t[k]))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        neg_list, neg_size = [], 0
        while neg_size < self.negative_sample_size:
            neg = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.isin(neg, self.hr2t[(h, r)], assume_unique=True, invert=True)
            neg = neg[mask]; neg_list.append(neg); neg_size += neg.size
        neg = np.concatenate(neg_list)[:self.negative_sample_size]
        return torch.LongTensor([h, r, t]), torch.from_numpy(neg), idx

    @staticmethod
    def collate_fn(data):
        return (torch.stack([d[0] for d in data]),
                torch.stack([d[1] for d in data]),
                torch.tensor([d[2] for d in data]))


def create_kg_dataloaders(all_clients_data, args):
    loaders = []
    for cd in all_clients_data:
        ds = KGTrainDataset(cd['triples_global'], cd['nentity'], args.num_neg)
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=KGTrainDataset.collate_fn))
    return loaders


def create_qa_dataloaders(all_clients_data, args):
    train_loaders, dev_loaders = [], []
    for cd in all_clients_data:
        train_loaders.append(DataLoader(cd['train_qa'], batch_size=args.qa_batch_size,
                                        shuffle=True, num_workers=0, collate_fn=QADataset.collate_fn))
        dev_loaders.append(DataLoader(cd['dev_qa'], batch_size=args.qa_batch_size,
                                      shuffle=False, num_workers=0, collate_fn=QADataset.collate_fn))
    return train_loaders, dev_loaders
