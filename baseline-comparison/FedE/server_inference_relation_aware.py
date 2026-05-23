"""
Relation-Aware Server Inference with Post-Ranking Type Filtering

This approach:
1. Lets the model rank ALL entities (preserves learned patterns)
2. THEN filters results by relation-expected types
3. More robust to corrupt individual triples

Key difference from previous approaches:
- Previous: Pre-filter candidates → rank subset
- This: Rank all → post-filter by learned types
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import logging

from question_decomposer import QuestionDecomposer
from entity_type_index import EntityTypeIndex


class RelationAwareFederatedKGQAServer:
    """
    Federated KGQA Server with relation-aware post-ranking type filtering
    """
    
    def __init__(self, num_clients: int, all_clients_data: List[Dict]):
        """
        Initialize server with entity type index
        
        Args:
            num_clients: Number of federated clients
            all_clients_data: List of client data dicts
        """
        self.num_clients = num_clients
        self.all_clients_data = all_clients_data
        
        # Question decomposer with relation extraction
        self.decomposer = QuestionDecomposer()
        
        # Build entity-to-client routing table
        self.entity_to_clients = self.build_entity_routing()
        
        # Build entity type index from ALL client KBs
        logging.info("Building entity type index from client KBs...")
        self.type_index = EntityTypeIndex()
        for client_id, client_data in enumerate(all_clients_data):
            # Load types from this client's KB
            if 'triples' in client_data:
                logging.info(f"Loading types from client {client_id} KB...")
                self._load_types_from_triples(
                    client_data['triples'],
                    client_data['id2entity'],
                    client_data['id2relation']
                )
        
        # Update stats
        self.type_index.stats['total_entities'] = len(self.type_index.entity_types)
        self.type_index.stats['typed_entities'] = sum(1 for types in self.type_index.entity_types.values() if len(types) > 0)
        self.type_index.stats['relations_mapped'] = len(self.type_index.relation_object_types)
        
        self.type_index.print_stats()
        
        # Client models (will be set externally)
        self.client_models = []
        self.client_embeddings = []
        
        logging.info(f"Server initialized with {num_clients} clients")
        logging.info(f"Entity routing table: {len(self.entity_to_clients)} entities")
        logging.info("Relation-aware type filtering: ENABLED")
    
    def _load_types_from_triples(self, triples: List[Tuple], id2entity: Dict, id2relation: Dict):
        """
        Load entity types from client's triples
        
        Args:
            triples: List of (subject_id, relation_id, object_id) tuples (as integers)
            id2entity: Mapping from entity ID to entity name
            id2relation: Mapping from relation ID to relation name
        """
        triples_processed = 0
        for subj_id, rel_id, obj_id in triples:
            # Convert IDs to names
            subj = id2entity.get(subj_id, str(subj_id))
            rel = id2relation.get(rel_id, str(rel_id))
            obj = id2entity.get(obj_id, str(obj_id))
            
            # Infer object type
            obj_type = self.type_index._infer_type_from_relation(rel)
            if obj_type:
                self.type_index.entity_types[obj].add(obj_type)
                self.type_index.relation_object_types[rel].add(obj_type)
            
            # Infer subject type
            subj_type = self.type_index._infer_subject_type_from_relation(rel)
            if subj_type:
                self.type_index.entity_types[subj].add(subj_type)
            
            triples_processed += 1
        
        logging.info(f"  Processed {triples_processed} triples from client KB")
    
    def build_entity_routing(self) -> Dict[str, List[int]]:
        """Build routing table: entity -> list of client IDs that have it"""
        entity_to_clients = {}
        
        for client_id, client_data in enumerate(self.all_clients_data):
            for entity in client_data['entities']:
                if entity not in entity_to_clients:
                    entity_to_clients[entity] = []
                entity_to_clients[entity].append(client_id)
        
        return entity_to_clients
    
    def query_client_with_filtering(self, client_id: int, question: str, 
                                    relation: str = None, top_k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Query a specific client with POST-RANKING type filtering
        
        This is the KEY improvement: 
        1. Model ranks ALL entities (preserves learned patterns)
        2. THEN we filter by relation type constraints
        
        Args:
            client_id: ID of client to query
            question: Question string
            relation: Relation type (for filtering)
            top_k: Number of top answers to return
        
        Returns:
            (answer_entities, scores)
        """
        if client_id >= len(self.client_models):
            return [], []
        
        client_model = self.client_models[client_id]
        client_data = self.all_clients_data[client_id]
        
        entity_embeddings = self.client_embeddings[client_id]['entity']
        relation_embeddings = self.client_embeddings[client_id]['relation']
        
        # Get predictions from client model - rank ALL entities
        with torch.no_grad():
            retrieval_k = min(top_k * 10, len(client_data['entities']))  # Get 10x more for filtering
            
            # Call predict_answers with the actual QAModel interface
            top_k_ids, top_k_scores = client_model.predict_answers(
                [question],
                relation_embeddings,
                entity_embeddings,
                client_data['entity2id'],
                top_k=retrieval_k
            )
        
        # Convert IDs to entity names
        top_k_ids = top_k_ids[0].cpu().numpy()
        top_k_scores = top_k_scores[0].cpu().numpy()
        
        ranked_entities = []
        ranked_scores = []
        
        for entity_id, score in zip(top_k_ids, top_k_scores):
            entity_id = int(entity_id)
            if entity_id in client_data['id2entity']:
                entity_name = client_data['id2entity'][entity_id]
                ranked_entities.append(entity_name)
                ranked_scores.append(float(score))
        
        # POST-RANKING TYPE FILTERING (if relation is provided)
        if relation:
            logging.debug(f"Before filtering: {ranked_entities[:5]}")
            filtered_entities = self.type_index.filter_by_relation_type(relation, ranked_entities)
            logging.debug(f"After filtering: {filtered_entities[:5]}")
            
            # Keep only scores for entities that passed filtering
            filtered_scores = []
            for entity in filtered_entities:
                if entity in ranked_entities:
                    idx = ranked_entities.index(entity)
                    filtered_scores.append(ranked_scores[idx])
            
            ranked_entities = filtered_entities[:top_k]
            ranked_scores = filtered_scores[:top_k]
        else:
            ranked_entities = ranked_entities[:top_k]
            ranked_scores = ranked_scores[:top_k]
        
        return ranked_entities, ranked_scores
    
    def answer_simple_question(self, question: str, relation: str = None, 
                              top_k: int = 10) -> Tuple[List[str], Dict]:
        """
        Answer a simple (1-hop) question with relation-aware filtering
        
        Args:
            question: Question text
            relation: Relation type (extracted from question)
            top_k: Number of answers to return
        
        Returns:
            (answers, metadata)
        """
        # Extract topic entity
        topic_entity = self.decomposer.extract_topic_entity(question)
        
        if not topic_entity:
            return [], {'error': 'Could not extract topic entity'}
        
        # Route to clients that have this entity
        if topic_entity not in self.entity_to_clients:
            return [], {'error': 'Entity not found', 'entity': topic_entity}
        
        client_ids = self.entity_to_clients[topic_entity]
        
        # Query all relevant clients with POST-RANKING filtering
        all_answers = {}
        
        for client_id in client_ids:
            answers, scores = self.query_client_with_filtering(
                client_id, question, relation=relation, top_k=top_k
            )
            
            for answer, score in zip(answers, scores):
                if answer not in all_answers:
                    all_answers[answer] = 0.0
                all_answers[answer] += score
        
        # Sort by aggregated scores
        sorted_answers = sorted(all_answers.items(), key=lambda x: x[1], reverse=True)
        final_answers = [ans for ans, _ in sorted_answers[:top_k]]
        
        metadata = {
            'topic_entity': topic_entity,
            'routed_to': client_ids,
            'question_type': '1-hop',
            'relation': relation,
            'type_filtered': relation is not None
        }
        
        return final_answers, metadata
    
    def _extract_relation_from_question(self, question: str) -> Optional[str]:
        """
        Extract relation type from question text to determine expected answer type
        
        Returns a pseudo-relation that indicates expected answer type
        """
        question_lower = question.lower()
        
        # Questions asking for FILMS/MOVIES
        if any(pattern in question_lower for pattern in ['what movies', 'what films', 'which movies', 'which films']):
            if any(kw in question_lower for kw in ['star', 'act', 'appear']):
                return 'person_to_film'  # Expects films
            if any(kw in question_lower for kw in ['direct']):
                return 'person_to_film'  # Expects films
            if any(kw in question_lower for kw in ['writ', 'wrote']):
                return 'person_to_film'  # Expects films
            return 'expects_film'
        
        # Questions asking for PEOPLE
        if any(pattern in question_lower for pattern in ['who', 'which person']):
            if any(kw in question_lower for kw in ['star', 'act', 'appear', 'co-star']):
                return 'film_to_person'  # Expects people
            if any(kw in question_lower for kw in ['direct']):
                return 'film_to_person'  # Expects people  
            if any(kw in question_lower for kw in ['writ', 'wrote', 'script']):
                return 'film_to_person'  # Expects people
            return 'expects_person'
        
        # Questions asking for GENRES
        if any(pattern in question_lower for pattern in ['what genre', 'what type', 'which genre']):
            return 'expects_genre'
        
        # Questions asking for LANGUAGES
        if any(pattern in question_lower for pattern in ['what language', 'which language', 'primary language']):
            return 'expects_language'
        
        # Questions asking for YEARS
        if any(pattern in question_lower for pattern in ['what year', 'when', 'release']):
            return 'expects_year'
        
        return None
    
    def answer_question(self, question: str, top_k: int = 10, beam_width: int = 3) -> Tuple[List[str], Dict]:
        """
        Answer a question (simple or complex) with relation-aware filtering
        
        Args:
            question: Question text
            top_k: Number of answers
            beam_width: Number of paths to explore in multi-hop (1=single path, 3=top-3 beam)
        
        Returns:
            (answers, metadata)
        """
        # Decompose question using actual QuestionDecomposer
        sub_queries_raw = self.decomposer.decompose(question)
        
        # Check if it's a simple question
        if len(sub_queries_raw) == 1 and sub_queries_raw[0].get('query_type') == 'simple':
            # Simple 1-hop question
            relation = self._extract_relation_from_question(question)
            return self.answer_simple_question(question, relation=relation, top_k=top_k)
        
        # Multi-hop question - execute sub-queries with BEAM SEARCH IN ALL HOPS
        logging.info(f"Decomposed into {len(sub_queries_raw)} sub-queries with relation-aware filtering (beam_width={beam_width})")
        
        # Initialize beam paths
        # Each path is: {'answers': [entity or list], 'score': float, 'history': []}
        # For intermediate hops: 'answers' contains ONE entity (the intermediate answer for this path)
        # For final hop: 'answers' contains multiple final answers
        beam_paths = [{'answers': ['[START]'], 'score': 1.0, 'history': []}]
        
        for hop_idx, sub_query_info in enumerate(sub_queries_raw):
            sub_query_template = sub_query_info.get('query', '')
            relation = self._extract_relation_from_question(sub_query_template)
            
            logging.info(f"Hop {hop_idx+1}: {sub_query_template}")
            logging.info(f"  -> Relation: {relation}")
            logging.info(f"  -> Expanding {len(beam_paths)} paths...")
            
            # Expand each path in the beam
            new_beam_paths = []
            
            for path in beam_paths:
                # Get the last answer from this path to use as entity
                if hop_idx == 0:
                    # First hop - extract entity from original question
                    current_entity = self.decomposer.extract_topic_entity(question)
                    sub_query = sub_query_template.replace('[PLACEHOLDER]', f'[{current_entity}]') if '[PLACEHOLDER]' in sub_query_template else sub_query_template
                else:
                    # Subsequent hops - use the single answer from this path
                    prev_answer = path['answers'][0] if path['answers'] and path['answers'][0] != '[START]' else None
                    
                    if not prev_answer:
                        continue
                    
                    # Replace placeholder with this answer
                    sub_query = sub_query_template.replace('[PLACEHOLDER]', f'[{prev_answer}]')
                    
                    # Execute sub-query
                    answers, sub_metadata = self.answer_simple_question(sub_query, relation=relation, top_k=top_k)
                    
                    if answers:
                        # Create new path
                        new_path = {
                            'answers': answers,
                            'score': path['score'] * (1.0 / (1.0 + hop_idx)),  # Decay score by hop
                            'history': path['history'] + [{
                                'hop': hop_idx + 1,
                                'query': sub_query,
                                'relation': relation,
                                'entity': prev_answer,
                                'answers': answers[:3]  # Store top-3 for debugging
                            }]
                        }
                        new_beam_paths.append(new_path)
                    
                    # Don't process first hop inside this loop
                    continue
                
                # First hop - execute query and create SEPARATE paths for each answer
                if hop_idx == 0:
                    answers, sub_metadata = self.answer_simple_question(sub_query, relation=relation, top_k=top_k)
                    
                    if answers:
                        # Create SEPARATE path for EACH intermediate answer (beam search in Hop 1)
                        for answer in answers[:beam_width]:  # Only keep top beam_width intermediate answers
                            new_path = {
                                'answers': [answer],  # Each path has ONE intermediate answer
                                'score': 1.0,
                                'history': [{
                                    'hop': 1,
                                    'query': sub_query,
                                    'relation': relation,
                                    'entity': current_entity,
                                    'intermediate_answer': answer,  # Track which intermediate answer this path follows
                                    'all_answers': answers[:3]  # Store top-3 for debugging
                                }]
                            }
                            new_beam_paths.append(new_path)
            
            # Keep top beam_width paths
            if new_beam_paths:
                # Sort by score and keep top beam_width
                new_beam_paths.sort(key=lambda x: x['score'], reverse=True)
                beam_paths = new_beam_paths[:beam_width * 3]  # Keep 3x beam width for diversity
                
                logging.info(f"  -> Generated {len(new_beam_paths)} new paths, keeping top {len(beam_paths)}")
            else:
                logging.warning(f"  -> No paths generated at hop {hop_idx+1}")
                beam_paths = []
                break
        
        # Aggregate final answers from all paths
        if beam_paths:
            # Collect all final answers with their scores
            final_answer_scores = {}
            
            for path in beam_paths:
                for answer in path['answers'][:top_k]:
                    if answer not in final_answer_scores:
                        final_answer_scores[answer] = 0.0
                    final_answer_scores[answer] += path['score']
            
            # Sort by aggregated scores
            sorted_answers = sorted(final_answer_scores.items(), key=lambda x: x[1], reverse=True)
            current_answers = [ans for ans, score in sorted_answers[:top_k]]
            
            logging.info(f"Final aggregated answers: {current_answers[:5]}")
            
            metadata = {
                'question_type': f'{len(sub_queries_raw)}-hop',
                'beam_width': beam_width,
                'paths_explored': len(beam_paths),
                'best_path': beam_paths[0]['history'] if beam_paths else [],
                'original_question': question,
                'relation_aware_filtering': 'enabled',
                'multi_path': True
            }
        else:
            current_answers = []
            metadata = {
                'question_type': f'{len(sub_queries_raw)}-hop',
                'beam_width': beam_width,
                'error': 'No valid paths found',
                'original_question': question
            }
        
        return current_answers, metadata
    
    def evaluate_on_dataset(self, test_data: List[Dict], output_file: str = None, beam_width: int = 3):
        """
        Evaluate server on test dataset
        
        Args:
            test_data: List of {'question': str, 'answers': List[str]} dicts
            output_file: Optional file to save results
        """
        from tqdm import tqdm
        
        metrics = {
            'total': 0,
            'correct_hits@3': 0,
            'correct_hits@5': 0,
            'correct_hits@10': 0,
            'mrr': 0.0,
            'no_answer': 0,
            'entity_not_found': 0,
            'by_hop': {
                1: {'total': 0, 'hits@3': 0, 'hits@5': 0, 'hits@10': 0, 'mrr': 0.0},
                2: {'total': 0, 'hits@3': 0, 'hits@5': 0, 'hits@10': 0, 'mrr': 0.0}
            }
        }
        
        logging.info("=" * 70)
        logging.info(f"Starting Evaluation (WITH RELATION-AWARE FILTERING, beam_width={beam_width})")
        logging.info("=" * 70)
        
        for qa_pair in tqdm(test_data, desc="Evaluating"):
            question = qa_pair['question']
            ground_truth = set(qa_pair['answers'])
            
            # Determine hop count
            hop_count = 2 if any(kw in question.lower() for kw in 
                               ['starred by', 'directed by', 'written by', 'same actor', 'same director']) else 1
            
            # Get predictions
            try:
                predicted_answers, metadata = self.answer_question(question, top_k=10, beam_width=beam_width)
            except Exception as e:
                logging.error(f"Error on question: {question}")
                logging.error(f"Error: {e}")
                predicted_answers = []
                metadata = {'error': str(e)}
            
            # Evaluate
            if not predicted_answers:
                metrics['no_answer'] += 1
                if metadata.get('error') == 'Entity not found':
                    metrics['entity_not_found'] += 1
            else:
                # Find rank of first correct answer
                rank = None
                for i, pred in enumerate(predicted_answers[:10]):
                    if pred in ground_truth:
                        rank = i
                        break
                
                if rank is not None:
                    if rank < 3:
                        metrics['correct_hits@3'] += 1
                        metrics['by_hop'][hop_count]['hits@3'] += 1
                    if rank < 5:
                        metrics['correct_hits@5'] += 1
                        metrics['by_hop'][hop_count]['hits@5'] += 1
                    if rank < 10:
                        metrics['correct_hits@10'] += 1
                        metrics['by_hop'][hop_count]['hits@10'] += 1
                    
                    mrr_score = 1.0 / (rank + 1)
                    metrics['mrr'] += mrr_score
                    metrics['by_hop'][hop_count]['mrr'] += mrr_score
            
            metrics['total'] += 1
            metrics['by_hop'][hop_count]['total'] += 1
        
        # Compute final metrics
        if metrics['total'] > 0:
            metrics['hits@3'] = metrics['correct_hits@3'] / metrics['total']
            metrics['hits@5'] = metrics['correct_hits@5'] / metrics['total']
            metrics['hits@10'] = metrics['correct_hits@10'] / metrics['total']
            metrics['mrr'] = metrics['mrr'] / metrics['total']
        
        for hop in [1, 2]:
            if metrics['by_hop'][hop]['total'] > 0:
                metrics['by_hop'][hop]['hits@3'] /= metrics['by_hop'][hop]['total']
                metrics['by_hop'][hop]['hits@5'] /= metrics['by_hop'][hop]['total']
                metrics['by_hop'][hop]['hits@10'] /= metrics['by_hop'][hop]['total']
                metrics['by_hop'][hop]['mrr'] /= metrics['by_hop'][hop]['total']
        
        # Print results
        self.print_metrics(metrics)
        
        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"\nResults saved to: {output_file}")
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (RELATION-AWARE FILTERING)")
        print("=" * 70)
        print(f"Total Questions:        {metrics['total']}")
        print(f"Questions w/o Answer:   {metrics['no_answer']}")
        print(f"Entity Not Found:       {metrics['entity_not_found']}")
        print("-" * 70)
        print(f"Hits@3:                 {metrics.get('hits@3', 0):.4f} ({metrics['correct_hits@3']}/{metrics['total']})")
        print(f"Hits@5:                 {metrics.get('hits@5', 0):.4f} ({metrics['correct_hits@5']}/{metrics['total']})")
        print(f"Hits@10:                {metrics.get('hits@10', 0):.4f} ({metrics['correct_hits@10']}/{metrics['total']})")
        print(f"MRR:                    {metrics.get('mrr', 0):.4f}")
        print("-" * 70)
        print("Results by Hop Count:")
        for hop in [1, 2]:
            hop_metrics = metrics['by_hop'][hop]
            if hop_metrics['total'] > 0:
                print(f"  {hop}-hop ({hop_metrics['total']} questions):")
                print(f"    Hits@3:  {hop_metrics['hits@3']:.4f}")
                print(f"    Hits@5:  {hop_metrics['hits@5']:.4f}")
                print(f"    Hits@10: {hop_metrics['hits@10']:.4f}")
                print(f"    MRR:     {hop_metrics['mrr']:.4f}")
        print("=" * 70)
        print("=" * 70)