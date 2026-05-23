import os
import random
import re
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
NUM_CLIENTS = 3
SEED = 42
OUTPUT_DIR = "federated_clients"
SERVER_DIR = "federated_server"
KB_FILE = "Freebase13.txt"
QA_FILE = "PQ-2H.txt"  # PathQuestion dataset (2-hop questions)

# Relations where tail is a literal value (not a semantic entity)
# Based on Freebase13, these relations have literal/categorical values
LITERAL_RELATIONS = {
    'gender', 'profession', 'religion', 'ethnicity', 
    'cause_of_death', 'nationality'
}

random.seed(SEED)

# -----------------------------
# STEP 1: LOAD KB AND ENTITIES
# -----------------------------
def load_kb(kb_path):
    """Load knowledge base triples and extract semantic entities."""
    triples = []
    semantic_entities = set()  # Entities (excluding literals from specific relations)
    
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) != 3:
                continue
                
            h, r, t = parts
            triples.append((h, r, t))
            
            # HEAD is always semantic
            semantic_entities.add(h)
            
            # Only add TAIL if it's not a literal relation
            if r not in LITERAL_RELATIONS:
                semantic_entities.add(t)
    
    return triples, list(semantic_entities), semantic_entities

# -----------------------------
# STEP 2: PARTITION ENTITIES
# -----------------------------
def partition_entities(entities, num_clients):
    """Partition entities into disjoint sets."""
    random.shuffle(entities)
    partitions = [set() for _ in range(num_clients)]
    
    for i, e in enumerate(entities):
        partitions[i % num_clients].add(e)
    
    return partitions

# -----------------------------
# STEP 3: ASSIGN TRIPLES (OR RULE)
# -----------------------------
def assign_triples(triples, entity_sets, semantic_entities):
    """Assign triples to clients using OR rule (filtered for semantic tail entities).
    
    Triples are assigned to a client if:
    - HEAD is in the client's entity set, OR
    - TAIL is semantic (not a literal) AND in the client's entity set
    
    This prevents literal values (gender, profession, etc.) from causing replication.
    """
    client_triples = [set() for _ in range(len(entity_sets))]
    
    for h, r, t in triples:
        for i, Ei in enumerate(entity_sets):
            # Always check head
            if h in Ei:
                client_triples[i].add((h, r, t))
            # Only check tail if it's a semantic entity (not a literal)
            elif t in semantic_entities and t in Ei:
                client_triples[i].add((h, r, t))
    
    return client_triples

# -----------------------------
# STEP 4: LOAD QA FILE
# -----------------------------
def load_qa(qa_path):
    """Load question-answer pairs from PathQuestion file.
    
    Format: question\tanswer(candidates)\tpath
    Example: what is the gender of father of yixin_prince_gong ?\tmale(male/)\tyixin_prince_gong#parents#daoguang_emperor#gender#male#<end>#male
    """
    qa_pairs = []
    
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) != 3:
                continue
                
            question, answer_with_candidates, path = parts
            
            # Extract answer (before parentheses)
            answer = answer_with_candidates.split("(")[0] if "(" in answer_with_candidates else answer_with_candidates
            
            qa_pairs.append((question, answer, path))
    
    return qa_pairs

def extract_topic_entity(path):
    """Extract the topic entity from the reasoning path.
    
    The path format is: entity#relation#entity#relation#...#<end>#answer
    The first entity is the topic entity.
    
    Example: yixin_prince_gong#parents#daoguang_emperor#gender#male#<end>#male
    Returns: yixin_prince_gong
    """
    parts = path.split("#")
    if len(parts) > 0:
        return parts[0]
    return None

# -----------------------------
# STEP 5: SPLIT QA INTO TRAIN/DEV/TEST
# -----------------------------
def split_qa_data(qa_pairs, train_ratio=0.7, dev_ratio=0.15):
    """Split QA pairs into train/dev/test sets.
    
    Args:
        qa_pairs: List of (question, answer, path) tuples
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        
    Returns:
        train_qa, dev_qa, test_qa
    """
    random.shuffle(qa_pairs)
    n = len(qa_pairs)
    
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    
    train_qa = qa_pairs[:train_end]
    dev_qa = qa_pairs[train_end:dev_end]
    test_qa = qa_pairs[dev_end:]
    
    return train_qa, dev_qa, test_qa

# -----------------------------
# STEP 6: ASSIGN QUESTIONS (TRAIN/DEV ONLY)
# -----------------------------
def assign_questions(qa_pairs, entity_sets):
    """Assign questions to clients based on topic entity from path."""
    client_qa = [list() for _ in range(len(entity_sets))]
    unassigned = []
    
    for question, answer, path in qa_pairs:
        topic = extract_topic_entity(path)
        if topic is None:
            unassigned.append((question, answer, path))
            continue
        
        assigned = False
        for i, Ei in enumerate(entity_sets):
            if topic in Ei:
                client_qa[i].append((question, answer, path))
                assigned = True
                break
        
        if not assigned:
            unassigned.append((question, answer, path))
    
    return client_qa, unassigned

# -----------------------------
# STEP 7: SAVE CLIENT OUTPUT
# -----------------------------
def save_client_output(client_triples, client_qas, split_name):
    """Save partitioned data to client directories."""
    for i in range(NUM_CLIENTS):
        client_dir = os.path.join(OUTPUT_DIR, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Save KB (only once during train split)
        if split_name == "train":
            kb_path = os.path.join(client_dir, "kb.txt")
            with open(kb_path, "w", encoding="utf-8") as f:
                for h, r, t in sorted(client_triples[i]):
                    f.write(f"{h}\t{r}\t{t}\n")
        
        # Save QA pairs (only train and dev, NOT test)
        qa_path = os.path.join(client_dir, f"qa_{split_name}.txt")
        with open(qa_path, "w", encoding="utf-8") as f:
            for question, answer, path in client_qas[i]:
                f.write(f"{question}\t{answer}\t{path}\n")

# -----------------------------
# STEP 8: SAVE SERVER TEST SET
# -----------------------------
def save_server_test(qa_pairs):
    """Save test set to server directory (no partitioning)."""
    os.makedirs(SERVER_DIR, exist_ok=True)
    test_path = os.path.join(SERVER_DIR, "qa_test.txt")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for question, answer, path in qa_pairs:
            f.write(f"{question}\t{answer}\t{path}\n")
    
    print(f"\n  Server test set: {len(qa_pairs)} questions")
    print(f"  Saved to: {test_path}")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    """Run the complete partitioning pipeline."""
    print("="*70)
    print("PathQuestion Federated Partitioning (Server-Client Architecture)")
    print("="*70)
    
    # Step 1: Load KB
    print("\nStep 1: Loading Freebase13 KB...")
    triples, entities, semantic_entities = load_kb(KB_FILE)
    print(f"  Loaded {len(triples)} triples")
    print(f"  Found {len(semantic_entities)} semantic entities (excluding literals from {LITERAL_RELATIONS})")
    print(f"  Total unique values: {len(set(h for h,r,t in triples) | set(t for h,r,t in triples))}")
    
    # Step 2: Partition entities
    print("\nStep 2: Partitioning entities...")
    entity_sets = partition_entities(entities, NUM_CLIENTS)
    print("  Entity distribution:")
    for i in range(NUM_CLIENTS):
        print(f"    Client {i}: {len(entity_sets[i])} entities")
    
    # Step 3: Assign triples
    print("\nStep 3: Assigning triples to clients...")
    client_triples = assign_triples(triples, entity_sets, semantic_entities)
    total_assigned = sum(len(ct) for ct in client_triples)
    overlap = (total_assigned - len(triples)) / len(triples) * 100
    print("  Triple distribution:")
    for i in range(NUM_CLIENTS):
        print(f"    Client {i}: {len(client_triples[i])} triples")
    print(f"  Overlap: {overlap:.2f}% ({total_assigned} assigned from {len(triples)} original)")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SERVER_DIR, exist_ok=True)
    
    # Step 4: Load and split QA data
    print(f"\nStep 4: Loading and splitting PathQuestion data...")
    
    if not os.path.exists(QA_FILE):
        print(f"  Error: {QA_FILE} not found!")
        return
    
    all_qa = load_qa(QA_FILE)
    print(f"  Loaded {len(all_qa)} question-answer pairs")
    
    train_qa, dev_qa, test_qa = split_qa_data(all_qa, train_ratio=0.7, dev_ratio=0.15)
    print(f"  Split into:")
    print(f"    Train: {len(train_qa)} questions (70%)")
    print(f"    Dev:   {len(dev_qa)} questions (15%)")
    print(f"    Test:  {len(test_qa)} questions (15%)")
    
    # Step 5: Process TRAIN split (distribute to clients)
    print(f"\nStep 5: Processing train split (CLIENT distribution)...")
    client_train_qas, train_unassigned = assign_questions(train_qa, entity_sets)
    save_client_output(client_triples, client_train_qas, "train")
    
    print(f"  Question distribution:")
    assigned = sum(len(cq) for cq in client_train_qas)
    for i in range(NUM_CLIENTS):
        print(f"    Client {i}: {len(client_train_qas[i])} questions")
    print(f"  Total assigned: {assigned}/{len(train_qa)}")
    if train_unassigned:
        print(f"  Unassigned (topic entity not in any partition): {len(train_unassigned)}")
    
    # Step 6: Process DEV split (distribute to clients)
    print(f"\nStep 6: Processing dev split (CLIENT distribution)...")
    client_dev_qas, dev_unassigned = assign_questions(dev_qa, entity_sets)
    save_client_output(client_triples, client_dev_qas, "dev")
    
    print(f"  Question distribution:")
    assigned = sum(len(cq) for cq in client_dev_qas)
    for i in range(NUM_CLIENTS):
        print(f"    Client {i}: {len(client_dev_qas[i])} questions")
    print(f"  Total assigned: {assigned}/{len(dev_qa)}")
    if dev_unassigned:
        print(f"  Unassigned (topic entity not in any partition): {len(dev_unassigned)}")
    
    # Step 7: Process TEST split (keep at server)
    print(f"\nStep 7: Processing test split (SERVER only)...")
    save_server_test(test_qa)
    
    print("\n" + "="*70)
    print("Partitioning complete!")
    print("="*70)
    print(f"\nClient directories: {OUTPUT_DIR}/client_{{0..{NUM_CLIENTS-1}}}/")
    print("  Each client contains:")
    print("    - kb.txt: Local knowledge base triples")
    print("    - qa_train.txt: Training questions (question, answer, path)")
    print("    - qa_dev.txt: Development questions (question, answer, path)")
    print(f"\nServer directory: {SERVER_DIR}/")
    print("  Server contains:")
    print("    - qa_test.txt: Test questions (for final evaluation)")
    print("\nArchitecture:")
    print("  - Clients: Train locally on their KG partition and QA pairs")
    print("  - Server: Orchestrates test queries to clients and aggregates answers")
    print("\nDataset Info:")
    print(f"  - Knowledge Base: Freebase13 ({len(triples)} triples)")
    print(f"  - Questions: PathQuestion 2-hop ({len(all_qa)} total)")
    print(f"  - Literal relations (excluded from partitioning): {LITERAL_RELATIONS}")

if __name__ == "__main__":
    main()
