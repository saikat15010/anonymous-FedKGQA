"""
FULLY CORRECTED Question Decomposer for MetaQA 2-Hop Questions

Fixes:
1. Added fill_placeholder method
2. Proper dict keys: depends_on, is_intermediate, is_final
3. Comprehensive 12-category pattern matching
"""

import re
from typing import List, Dict, Optional


class QuestionDecomposer:
    """
    Decompose complex 2-hop questions into sequential sub-queries
    
    MetaQA 2-hop structure: Entity -> Relation1 -> Intermediate -> Relation2 -> Answer
    """
    
    def __init__(self):
        self.entity_pattern = re.compile(r'\[(.*?)\]')
    
    def extract_topic_entity(self, question: str) -> Optional[str]:
        """Extract the entity in square brackets"""
        match = self.entity_pattern.search(question)
        return match.group(1) if match else None
    
    def fill_placeholder(self, query_template: str, entity: str) -> str:
        """Fill placeholder in query template with actual entity"""
        return query_template.replace('[PLACEHOLDER]', f'[{entity}]')
    
    def detect_question_type(self, question: str) -> str:
        """
        Detect if question is 1-hop or 2-hop
        
        2-hop questions have compound structure with two relations
        """
        question_lower = question.lower()
        
        # 2-hop indicators
        two_hop_patterns = [
            # Actor patterns
            'starred by', 'acted by', 'acted in', 'acted together', 'appeared in',
            'co-starred', 'co-star', 'same actor', 'actor of', 'actor in',
            
            # Director patterns  
            'directed by', 'director of', 'films directed', 'movies directed',
            'also directed', 'co-directed', 'same director', 'is listed as director',
            
            # Writer patterns
            'written by', 'writer of', 'wrote', 'films written', 'movies written',
            'co-wrote', 'screenwriter', 'scriptwriter', 'same screenwriter', 'same writer',
            
            # Compound queries
            'were in which', 'were released', 'were written', 'were directed',
            'starred who', 'starred which', 'directed who', 'also directed which',
            'also wrote', 'also starred', 'also appears',
            
            # Property queries on related entities
            'same movie', 'share the same',
        ]
        
        for pattern in two_hop_patterns:
            if pattern in question_lower:
                return '2-hop'
        
        return '1-hop'
    
    def decompose(self, question: str) -> List[Dict]:
        """
        Decompose a question into sub-queries
        
        For 2-hop: Creates two sequential sub-queries
        For 1-hop: Returns the original question
        """
        question_type = self.detect_question_type(question)
        
        if question_type == '1-hop':
            topic_entity = self.extract_topic_entity(question)
            return [{
                'query': question,
                'query_type': 'simple',
                'topic_entity': topic_entity,
                'original_question': question,
                'is_intermediate': False,
                'is_final': True
            }]
        else:
            return self.decompose_2hop(question)
    
    def decompose_2hop(self, question: str) -> List[Dict]:
        """
        Decompose 2-hop questions using comprehensive pattern matching
        
        Returns list of 2 sub-queries with proper dict keys
        """
        question_lower = question.lower()
        topic_entity = self.extract_topic_entity(question)
        
        if not topic_entity:
            # Cannot decompose without topic entity
            return [{
                'query': question,
                'query_type': 'simple',
                'topic_entity': None,
                'original_question': question,
                'is_intermediate': False,
                'is_final': True
            }]
        
        # ========== CATEGORY 1: Actor/Person -> Movies -> Director ==========
        # "who directed the movies starred by [X]"
        # "the movies acted by [X] were directed by who"
        # "who is listed as director of [X] starred movies"
        if any(pattern in question_lower for pattern in ['starred by', 'acted by', 'films acted', 'movies acted']):
            if any(pattern in question_lower for pattern in ['directed', 'director']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] star in?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who directed [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 2: Actor/Person -> Movies -> Writer ==========
        # "who wrote the movies starred by [X]"
        # "the films acted by [X] were written by who"
        if any(pattern in question_lower for pattern in ['starred by', 'acted by', 'films acted']):
            if any(pattern in question_lower for pattern in ['written', 'wrote', 'writer', 'screenwriter', 'scriptwriter']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] star in?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who wrote [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 3: Director -> Movies -> Actors ==========
        # "who acted in the films directed by [X]"
        # "the movies directed by [X] starred who"
        if any(pattern in question_lower for pattern in ['directed by', 'films directed', 'movies directed']):
            if any(pattern in question_lower for pattern in ['starred', 'actor', 'acted', 'who starred']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] direct?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who starred in [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 4: Director -> Movies -> Writers ==========
        # "who wrote the movies directed by [X]"
        # "the films directed by [X] were written by who"
        if any(pattern in question_lower for pattern in ['directed by', 'films directed', 'movies directed']):
            if any(pattern in question_lower for pattern in ['written', 'wrote', 'writer', 'screenwriter']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] direct?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who wrote [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 5: Writer -> Movies -> Actors ==========
        # "who acted in the films written by [X]"
        # "the movies written by [X] starred who"
        if any(pattern in question_lower for pattern in ['written by', 'films written', 'movies written']):
            if any(pattern in question_lower for pattern in ['starred', 'actor', 'acted']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] write?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who starred in [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 6: Writer -> Movies -> Directors ==========
        # "who directed the films written by [X]"
        # "who is listed as director of [X] written movies"
        if any(pattern in question_lower for pattern in ['written by', 'films written', 'movies written']):
            if any(pattern in question_lower for pattern in ['directed', 'director']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] write?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "Who directed [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 7: Person -> Movies -> Genres ==========
        # "what are the genres of the films starred by [X]"
        # "the movies directed by [X] were in which genres"
        if any(pattern in question_lower for pattern in ['genre', 'type']):
            if any(pattern in question_lower for pattern in ['starred by', 'acted by']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] star in?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "What genre is [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
            elif any(pattern in question_lower for pattern in ['directed by', 'films directed']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] direct?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "What genre is [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
            elif any(pattern in question_lower for pattern in ['written by', 'films written']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] write?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "What genre is [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 8: Person -> Movies -> Release Years ==========
        # "when were the movies starred by [X] released"
        # "what were the release years the films directed by [X]"
        if any(pattern in question_lower for pattern in ['release', 'when were', 'when did']):
            if any(pattern in question_lower for pattern in ['starred by', 'acted by']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] star in?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "When was [PLACEHOLDER] released?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
            elif any(pattern in question_lower for pattern in ['directed by', 'films directed']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] direct?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "When was [PLACEHOLDER] released?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
            elif any(pattern in question_lower for pattern in ['written by', 'films written']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] write?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "When was [PLACEHOLDER] released?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 9: Person -> Movies -> Languages ==========
        # "what are the main languages in [X] directed films"
        if 'language' in question_lower:
            if any(pattern in question_lower for pattern in ['directed by', 'films directed', 'movies directed']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] direct?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "What language is [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
            elif any(pattern in question_lower for pattern in ['starred by', 'acted by']):
                return [
                    {
                        'query': f"What movies did [{topic_entity}] star in?",
                        'query_type': 'simple',
                        'topic_entity': topic_entity,
                        'original_question': question,
                        'is_intermediate': True,
                        'is_final': False
                    },
                    {
                        'query': "What language is [PLACEHOLDER]?",
                        'query_type': 'simple',
                        'topic_entity': None,
                        'original_question': question,
                        'depends_on': 0,
                        'is_intermediate': False,
                        'is_final': True
                    }
                ]
        
        # ========== CATEGORY 10: Movie -> Person -> Their Other Movies ==========
        # "the director of [X] also directed which movies"
        # "the actor in [X] also appears in which films"
        if 'also directed' in question_lower or ('director of' in question_lower and ('also' in question_lower or 'which' in question_lower)):
            return [
                {
                    'query': f"Who directed [{topic_entity}]?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "What movies did [PLACEHOLDER] direct?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        if 'also wrote' in question_lower or ('writer of' in question_lower and ('also' in question_lower or 'which' in question_lower)) or ('scriptwriter of' in question_lower and 'also' in question_lower):
            return [
                {
                    'query': f"Who wrote [{topic_entity}]?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "What movies did [PLACEHOLDER] write?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        if any(pattern in question_lower for pattern in ['actor of', 'actor in']) and any(pattern in question_lower for pattern in ['also starred', 'also appears', 'also acted']):
            return [
                {
                    'query': f"Who starred in [{topic_entity}]?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "What movies did [PLACEHOLDER] star in?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        # ========== CATEGORY 11: Same Actor/Director/Writer Patterns ==========
        # "which movies have the same actor of [X]"
        # "who appeared in the same movie with [X]"
        # "who acted together with [X]"
        if any(pattern in question_lower for pattern in ['same actor', 'same movie', 'appeared', 'acted together', 'co-star', 'starred together']):
            return [
                {
                    'query': f"What movies did [{topic_entity}] star in?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "Who starred in [PLACEHOLDER]?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        if 'same director' in question_lower or 'share the same director' in question_lower:
            return [
                {
                    'query': f"Who directed [{topic_entity}]?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "What movies did [PLACEHOLDER] direct?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        if 'same screenwriter' in question_lower or 'same writer' in question_lower or 'share the same screenwriter' in question_lower:
            return [
                {
                    'query': f"Who wrote [{topic_entity}]?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "What movies did [PLACEHOLDER] write?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        # ========== CATEGORY 12: Co-director/Co-writer Patterns ==========
        # "who are movie co-directors of [X]"
        # "who co-wrote films with [X]"
        if 'co-direct' in question_lower:
            return [
                {
                    'query': f"What movies did [{topic_entity}] direct?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "Who directed [PLACEHOLDER]?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        if any(pattern in question_lower for pattern in ['co-wrote', 'co-write', 'wrote together', 'wrote films together']):
            return [
                {
                    'query': f"What movies did [{topic_entity}] write?",
                    'query_type': 'simple',
                    'topic_entity': topic_entity,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False
                },
                {
                    'query': "Who wrote [PLACEHOLDER]?",
                    'query_type': 'simple',
                    'topic_entity': None,
                    'original_question': question,
                    'depends_on': 0,
                    'is_intermediate': False,
                    'is_final': True
                }
            ]
        
        # ========== FALLBACK: Return as simple query ==========
        return [{
            'query': question,
            'query_type': 'simple',
            'topic_entity': topic_entity,
            'original_question': question,
            'is_intermediate': False,
            'is_final': True
        }]