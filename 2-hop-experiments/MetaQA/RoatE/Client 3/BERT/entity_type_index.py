"""
Entity Type Index - Learns entity types from KB relations

This learns entity types by looking at ALL relations an entity appears in,
making it robust even when individual triples are corrupt.

Example:
    Even if one triple says: (John Krasinski, starred_in, Nick Frost) [WRONG]
    Nick Frost appears in many other relations:
        (Nick Frost, directed_by, some_movie) → "person"
        (Someone, has_actor, Nick Frost) → "person"
    So Nick Frost is correctly typed as "person" overall.
"""

from collections import defaultdict
from typing import Set, List, Optional
import logging


class EntityTypeIndex:
    """
    Lightweight semantic constraint index for inference-time filtering.
    Learns entity types from relation patterns in the KB.
    """

    def __init__(self):
        """Initialize empty type index"""
        self.entity_types = defaultdict(set)  # entity -> set of types
        self.relation_object_types = defaultdict(set)  # relation -> expected object types
        self.stats = {
            'total_entities': 0,
            'typed_entities': 0,
            'relations_mapped': 0
        }

    def load_kb(self, kb_file: str):
        """
        Load KB and infer entity types from relation patterns.
        
        KB format (tab-separated):
            subject<TAB>relation<TAB>object
        
        Args:
            kb_file: Path to KB file
        """
        logging.info(f"Loading KB for type inference: {kb_file}")
        
        triples_processed = 0
        
        with open(kb_file, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split("\t")
                
                if len(parts) != 3:
                    continue
                
                subject, relation, obj = parts
                triples_processed += 1
                
                # Infer object type from relation
                obj_type = self._infer_type_from_relation(relation)
                
                if obj_type is not None:
                    self.entity_types[obj].add(obj_type)
                    self.relation_object_types[relation].add(obj_type)
                
                # Also infer subject type for some relations
                subj_type = self._infer_subject_type_from_relation(relation)
                if subj_type is not None:
                    self.entity_types[subject].add(subj_type)
        
        # Update stats
        self.stats['total_entities'] = len(self.entity_types)
        self.stats['typed_entities'] = sum(1 for types in self.entity_types.values() if len(types) > 0)
        self.stats['relations_mapped'] = len(self.relation_object_types)
        
        logging.info(f"Processed {triples_processed} triples")
        logging.info(f"Indexed {self.stats['typed_entities']} typed entities")
        logging.info(f"Mapped {self.stats['relations_mapped']} relations to types")

    def _infer_type_from_relation(self, relation: str) -> Optional[str]:
        """
        Infer the expected OBJECT type from a relation name.
        
        Based on actual MetaQA KB relations:
        - directed_by: film → director (person)
        - starred_actors: film → actor (person)
        - written_by: film → writer (person)
        - has_genre: film → genre
        - in_language: film → language
        - release_year: film → year
        
        Args:
            relation: Relation name
        
        Returns:
            Expected object type, or None if unknown
        """
        r = relation.lower()
        
        # Exact relation name matching (for MetaQA KB)
        if r == 'directed_by':
            return 'person'
        
        if r == 'starred_actors':
            return 'person'
        
        if r == 'written_by':
            return 'person'
        
        if r == 'has_genre':
            return 'genre'
        
        if r == 'in_language':
            return 'language'
        
        if r == 'release_year':
            return 'year'
        
        # Fallback patterns for similar relation names
        if 'director' in r or 'directed' in r:
            return 'person'
        
        if 'actor' in r or 'starred' in r or 'star' in r:
            return 'person'
        
        if 'writer' in r or 'written' in r or 'wrote' in r:
            return 'person'
        
        if 'genre' in r:
            return 'genre'
        
        if 'language' in r:
            return 'language'
        
        if 'year' in r or 'release' in r:
            return 'year'
        
        return None
    
    def _infer_subject_type_from_relation(self, relation: str) -> Optional[str]:
        """
        Infer the expected SUBJECT type from a relation name.
        
        In MetaQA KB, ALL relations have FILM as subject:
        - (film, directed_by, director)
        - (film, starred_actors, actor)
        - (film, written_by, writer)
        - (film, has_genre, genre)
        - (film, in_language, language)
        - (film, release_year, year)
        
        Args:
            relation: Relation name
        
        Returns:
            Expected subject type, or None if unknown
        """
        r = relation.lower()
        
        # In MetaQA KB, all these relations have FILM as subject
        kb_relations = [
            'directed_by', 'starred_actors', 'written_by',
            'has_genre', 'in_language', 'release_year',
            'has_imdb_rating', 'has_imdb_votes', 'has_tags'
        ]
        
        if r in kb_relations:
            return 'film'
        
        # Fallback patterns
        if any(kw in r for kw in ['directed_by', 'written_by', 'has_genre', 'has_', 'in_language', 'release_year']):
            return 'film'
        
        return None

    def filter_by_expected_type(self, expected_type_relation: str, candidates: List[str]) -> List[str]:
        """
        Filter candidates by expected answer type (direct type checking)
        
        This bypasses the relation lookup and directly checks entity types
        
        Args:
            expected_type_relation: Pseudo-relation indicating expected type
            candidates: List of candidate entities
        
        Returns:
            Filtered list of candidates
        """
        # Map pseudo-relations to expected types
        type_mapping = {
            'person_to_film': {'film'},
            'film_to_person': {'person'},
            'expects_film': {'film'},
            'expects_person': {'person'},
            'expects_genre': {'genre'},
            'expects_language': {'language'},
            'expects_year': {'year'}
        }
        
        expected_types = type_mapping.get(expected_type_relation, None)
        
        if expected_types is None:
            # No filtering if we don't know the expected type
            return candidates
        
        logging.debug(f"Filtering for expected types: {expected_types}")
        
        filtered = []
        removed_count = 0
        
        for entity in candidates:
            entity_types = self.entity_types.get(entity, set())
            
            # Keep entity if it has at least one matching type
            if len(entity_types.intersection(expected_types)) > 0:
                filtered.append(entity)
            else:
                removed_count += 1
                logging.debug(f"  Removed '{entity}' (has types: {entity_types}, expects: {expected_types})")
        
        # Safety fallback - if filtering removes everything, return originals
        if len(filtered) == 0:
            logging.warning(f"Type filtering removed ALL candidates (expected: {expected_types}) - returning originals")
            return candidates
        
        logging.debug(f"Type filtering: kept {len(filtered)}/{len(candidates)} candidates (removed {removed_count})")
        
        return filtered
    
    def filter_by_relation_type(self, relation: str, candidates: List[str]) -> List[str]:
        """
        Filter ranked candidates using relation semantic constraints.
        
        This is the KEY function - it takes the model's ranked predictions
        and removes entities that don't match the expected type for this relation.
        
        Args:
            relation: The relation being queried (or pseudo-relation)
            candidates: List of candidate entities (in ranked order)
        
        Returns:
            Filtered list of candidates (preserving order)
        """
        # Check if it's a pseudo-relation (direct type specification)
        if relation and (relation.startswith('expects_') or '_to_' in relation):
            return self.filter_by_expected_type(relation, candidates)
        
        # Original relation-based filtering (for KB relations)
        # Get expected types for this relation
        allowed_types = self.relation_object_types.get(relation, None)
        
        # If no constraint available - keep original ranking
        if allowed_types is None or len(allowed_types) == 0:
            logging.debug(f"No type constraints for relation '{relation}' - keeping all candidates")
            return candidates
        
        logging.debug(f"Filtering candidates for relation '{relation}' (expects: {allowed_types})")
        
        filtered = []
        removed_count = 0
        
        for entity in candidates:
            entity_types = self.entity_types.get(entity, set())
            
            # Keep entity if it has at least one matching type
            if len(entity_types.intersection(allowed_types)) > 0:
                filtered.append(entity)
            else:
                removed_count += 1
                logging.debug(f"  Removed '{entity}' (has types: {entity_types}, expects: {allowed_types})")
        
        # Safety fallback - if filtering removes everything, return originals
        if len(filtered) == 0:
            logging.warning(f"Type filtering removed ALL candidates for relation '{relation}' - returning originals")
            return candidates
        
        logging.debug(f"Type filtering: kept {len(filtered)}/{len(candidates)} candidates (removed {removed_count})")
        
        return filtered
    
    def get_entity_types(self, entity: str) -> Set[str]:
        """Get the inferred types for an entity"""
        return self.entity_types.get(entity, set())
    
    def get_relation_types(self, relation: str) -> Set[str]:
        """Get the expected object types for a relation"""
        return self.relation_object_types.get(relation, set())
    
    def print_stats(self):
        """Print statistics about the type index"""
        print("=" * 70)
        print("Entity Type Index Statistics")
        print("=" * 70)
        print(f"Total entities indexed: {self.stats['total_entities']}")
        print(f"Entities with types: {self.stats['typed_entities']}")
        print(f"Relations mapped: {self.stats['relations_mapped']}")
        print()
        print("Sample entity types:")
        for entity, types in list(self.entity_types.items())[:10]:
            print(f"  {entity}: {types}")
        print("=" * 70)