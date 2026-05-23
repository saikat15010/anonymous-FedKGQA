"""Server-side Evaluation for MetaQA 3-hop"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
import torch, argparse, logging, json

from qa_dataloader_metaqa_3hop import (
    load_all_metaqa_clients, get_global_relation_mapping, ServerTestDataset)
from qa_model_all import ImprovedKGQAModel


def load_models(args, clients, global_nrel, device):
    model_dir = os.path.join(args.state_dir, 'best_models')

    # Auto-detect KGE and LM models
    kge_file = os.path.join(model_dir, 'kge_model.txt')
    lm_file = os.path.join(model_dir, 'lm_model.txt')
    if os.path.exists(kge_file):
        with open(kge_file) as f: args.kge_model = f.read().strip()
    if os.path.exists(lm_file):
        with open(lm_file) as f: args.lm_model = f.read().strip()

    glob_rel = torch.load(os.path.join(model_dir, 'global_relation_embeddings.pt'), map_location=device)
    if isinstance(glob_rel, dict): glob_rel = list(glob_rel.values())[0]
    glob_rel = glob_rel.to(device)

    models, embeddings = [], []
    for cid, cd in enumerate(clients):
        cdir = os.path.join(model_dir, f'client_{cid}')
        ent = torch.load(os.path.join(cdir, 'entity_embeddings.pt'), map_location=device)
        if isinstance(ent, dict): ent = list(ent.values())[0]
        ent = ent.to(device)
        qa = ImprovedKGQAModel(args, cd['nentity'], global_nrel)
        qa.load_state_dict(torch.load(os.path.join(cdir, 'qa_model.pt'), map_location=device))
        qa = qa.to(device); qa.eval()
        models.append(qa)
        embeddings.append({'entity': ent, 'relation': glob_rel})
    return models, embeddings


def evaluate(clients, models, embeddings, test_ds):
    e2c = {}
    for cid, cd in enumerate(clients):
        for e in cd['entities']:
            e2c.setdefault(e, []).append(cid)

    hits = {k: 0 for k in [1,3,5,10]}
    mrr, total = 0, 0

    for qa in test_ds:
        topic = qa['topic_entity']
        cids = e2c.get(topic, list(range(len(clients)))) if topic else list(range(len(clients)))
        preds = []
        for cid in cids:
            with torch.no_grad():
                ids, scores = models[cid].predict_answers(
                    [qa['question']], embeddings[cid]['relation'],
                    embeddings[cid]['entity'], clients[cid]['entity2id'], top_k=10)
                for pid, s in zip(ids[0].cpu().numpy(), scores[0].cpu().numpy()):
                    preds.append((clients[cid]['id2entity'].get(pid, ''), float(s)))
        preds.sort(key=lambda x: x[1], reverse=True)
        seen, unique = set(), []
        for e, s in preds:
            if e not in seen: seen.add(e); unique.append(e)
            if len(unique) >= 10: break
        rank = next((i for i, e in enumerate(unique) if e in qa['answers']), None)
        if rank is not None:
            for k in [1,3,5,10]:
                if rank < k: hits[k] += 1
            mrr += 1.0 / (rank + 1)
        total += 1

    n = max(total, 1)
    return {f'hits@{k}': hits[k]/n for k in [1,3,5,10]} | {'mrr': mrr/n, 'total': total}


def main(args):
    logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.INFO)
    device = torch.device("cpu") if args.gpu == '-1' else torch.device(f'cuda:{args.gpu}')
    args.gpu = device

    clients = load_all_metaqa_clients(args.client_data_path, args.num_clients)
    _, _, gnr = get_global_relation_mapping(clients)
    test_ds = ServerTestDataset(args.test_file)
    logging.info(f"Test questions: {len(test_ds)}")

    models, embs = load_models(args, clients, gnr, device)
    results = evaluate(clients, models, embs, test_ds)

    logging.info(f"MRR: {results['mrr']:.4f}  H@1: {results['hits@1']:.4f}  "
                 f"H@3: {results['hits@3']:.4f}  H@5: {results['hits@5']:.4f}  "
                 f"H@10: {results['hits@10']:.4f}  N: {results['total']}")
    with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test_file', type=str, required=True)
    p.add_argument('--client_data_path', type=str, required=True)
    p.add_argument('--state_dir', type=str, required=True)
    p.add_argument('--num_clients', type=int, default=3)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--gamma', type=float, default=12.0)
    p.add_argument('--epsilon', type=float, default=2.0)
    p.add_argument('--kge_model', type=str, default='transe')
    p.add_argument('--lm_model', type=str, default='bert')
    p.add_argument('--output_file', type=str, default='eval_3hop.json')
    p.add_argument('--gpu', type=str, default='-1')
    main(p.parse_args())
