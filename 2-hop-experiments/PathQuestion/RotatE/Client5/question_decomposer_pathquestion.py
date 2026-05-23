"""
Question Decomposer for PathQuestion 2-Hop Questions

PathQuestion format: Freebase relations (parents, spouse, children, gender, nationality, etc.)
Example: "what is the gender of father of yixin_prince_gong ?"
Path: yixin_prince_gong#parents#daoguang_emperor#gender#male
"""

import re
from typing import List, Dict, Optional


class QuestionDecomposer:
    """
    Decompose complex 2-hop questions into sequential sub-queries
    
    PathQuestion 2-hop structure: Entity -> Relation1 -> Intermediate -> Relation2 -> Answer
    """
    
    def __init__(self):
        # PathQuestion entities are underscored (e.g., john_doe)
        self.entity_pattern = re.compile(r'\b([a-z_]+_[a-z_]+(?:_[a-z_]+)*)\b')
    
    def extract_topic_entity(self, question: str) -> Optional[str]:
        """Extract first underscored entity from question"""
        words = question.split()
        for word in words:
            if '_' in word and not word.startswith('_'):
                # Clean punctuation
                entity = word.strip('?.,!;:\'"')
                return entity
        return None
    
    def fill_placeholder(self, query_template: str, entity: str) -> str:
        """Fill placeholder in query template with actual entity"""
        return query_template.replace('[PLACEHOLDER]', entity)
    
    def detect_question_type(self, question: str) -> str:
        """
        Detect if question is 1-hop or 2-hop
        
        2-hop questions have possessive chains or compound "of" constructions
        """
        question_lower = question.lower()
        
        # Count possessive and "of" patterns
        possessive_count = question_lower.count("'s")
        of_count = question_lower.count(" of ")
        
        # 2-hop if multiple possessives or "of"s
        if possessive_count >= 2 or of_count >= 2:
            return '2-hop'
        
        # 2-hop indicators for PathQuestion
        two_hop_patterns = [
            # Family relation chains
            "parent", "child", "offspring", "heir",
            "father", "mother", "son", "daughter",
            "mom", "dad",
            
            # Marriage relation chains
            "spouse", "couple", "husband", "wife", "darling",
            
            # Combined patterns
            "of.*'s", "'s.*of",  # Possessive + of
        ]
        
        for pattern in two_hop_patterns:
            if re.search(pattern, question_lower):
                # Check if it's actually a 2-hop question
                if possessive_count > 0 or of_count > 0:
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
                'is_final': True,
                'depends_on': None
            }]
        else:
            return self.decompose_2hop(question)
    
    def decompose_2hop(self, question: str) -> List[Dict]:
        """
        Decompose 2-hop PathQuestion questions
        
        Returns list of 2 sub-queries
        """
        question_lower = question.lower()
        topic_entity = self.extract_topic_entity(question)
        
        # Category 1: "what is X of Y's Z" -> nationality of spouse
        # Example: "which nationality is frederica_of_mecklenburg-strelitz 's couple ?"
        pattern1 = r"what|which|who"
        pattern1_match = re.search(
            r"(what|which|who)\s+(?:is\s+)?(?:the\s+)?(\w+)\s+(?:of\s+)?(?:\w+\s+)?'s\s+(\w+)",
            question_lower
        )
        
        if pattern1_match:
            target_property = pattern1_match.group(2)  # nationality, gender, etc.
            intermediate_relation = pattern1_match.group(3)  # couple, spouse, parent, etc.
            
            # Map intermediate relation to Freebase relation
            relation_map = {
                'couple': 'spouse',
                'darling': 'spouse',
                'husband': 'spouse',
                'wife': 'spouse',
                'parent': 'parents',
                'father': 'parents',
                'mother': 'parents',
                'mom': 'parents',
                'dad': 'parents',
                'child': 'children',
                'son': 'children',
                'daughter': 'children',
                'offspring': 'children',
                'heir': 'children',
            }
            
            first_relation = relation_map.get(intermediate_relation, intermediate_relation)
            
            return [
                {
                    'query': f"who is the {intermediate_relation} of {topic_entity}?",
                    'query_type': 'relation_query',
                    'topic_entity': topic_entity,
                    'relation': first_relation,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False,
                    'depends_on': None
                },
                {
                    'query': f"what is the {target_property} of [PLACEHOLDER]?",
                    'query_type': 'relation_query',
                    'topic_entity': '[PLACEHOLDER]',
                    'relation': target_property,
                    'original_question': question,
                    'is_intermediate': False,
                    'is_final': True,
                    'depends_on': 0
                }
            ]
        
        # Category 2: "X of Y of Z" pattern
        # Example: "what is the child of parent of shah_shuja ?"
        pattern2_match = re.search(
            r"(what|which|who)\s+(?:is\s+)?(?:the\s+)?(\w+)\s+of\s+(\w+)\s+of\s+",
            question_lower
        )
        
        if pattern2_match:
            final_property = pattern2_match.group(2)  # child, parent, etc.
            intermediate_relation = pattern2_match.group(3)  # parent, child, etc.
            
            relation_map = {
                'parent': 'parents',
                'father': 'parents',
                'mother': 'parents',
                'mom': 'parents',
                'dad': 'parents',
                'child': 'children',
                'son': 'children',
                'daughter': 'children',
                'offspring': 'children',
                'heir': 'children',
            }
            
            first_relation = relation_map.get(intermediate_relation, intermediate_relation)
            second_relation = relation_map.get(final_property, final_property)
            
            return [
                {
                    'query': f"who is the {intermediate_relation} of {topic_entity}?",
                    'query_type': 'relation_query',
                    'topic_entity': topic_entity,
                    'relation': first_relation,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False,
                    'depends_on': None
                },
                {
                    'query': f"who is the {final_property} of [PLACEHOLDER]?",
                    'query_type': 'relation_query',
                    'topic_entity': '[PLACEHOLDER]',
                    'relation': second_relation,
                    'original_question': question,
                    'is_intermediate': False,
                    'is_final': True,
                    'depends_on': 0
                }
            ]
        
        # Category 3: possessive pattern "X's Y's Z"
        # Example: "what is the gender of yixin_prince_gong 's father ?"
        pattern3_match = re.search(
            r"(what|which|who)\s+(?:is\s+)?(?:the\s+)?(\w+)\s+of\s+\w+\s*'s\s+(\w+)",
            question_lower
        )
        
        if pattern3_match:
            final_property = pattern3_match.group(2)  # gender, nationality, etc.
            intermediate_relation = pattern3_match.group(3)  # father, mother, etc.
            
            relation_map = {
                'parent': 'parents',
                'father': 'parents',
                'mother': 'parents',
                'mom': 'parents',
                'dad': 'parents',
                'child': 'children',
                'son': 'children',
                'daughter': 'children',
                'offspring': 'children',
                'heir': 'children',
                'spouse': 'spouse',
                'couple': 'spouse',
                'husband': 'spouse',
                'wife': 'spouse',
            }
            
            first_relation = relation_map.get(intermediate_relation, intermediate_relation)
            
            return [
                {
                    'query': f"who is the {intermediate_relation} of {topic_entity}?",
                    'query_type': 'relation_query',
                    'topic_entity': topic_entity,
                    'relation': first_relation,
                    'original_question': question,
                    'is_intermediate': True,
                    'is_final': False,
                    'depends_on': None
                },
                {
                    'query': f"what is the {final_property} of [PLACEHOLDER]?",
                    'query_type': 'relation_query',
                    'topic_entity': '[PLACEHOLDER]',
                    'relation': final_property,
                    'original_question': question,
                    'is_intermediate': False,
                    'is_final': True,
                    'depends_on': 0
                }
            ]
        
        # Fallback: treat as 1-hop if patterns don't match
        return [{
            'query': question,
            'query_type': 'simple',
            'topic_entity': topic_entity,
            'original_question': question,
            'is_intermediate': False,
            'is_final': True,
            'depends_on': None
        }]
    
    def get_decomposition_summary(self, decomposed_queries: List[Dict]) -> str:
        """Generate human-readable summary of decomposition"""
        if len(decomposed_queries) == 1:
            return f"1-hop question: {decomposed_queries[0]['query']}"
        
        summary = "2-hop question decomposition:\n"
        for i, sub_query in enumerate(decomposed_queries):
            summary += f"  Step {i+1}: {sub_query['query']}\n"
            if sub_query.get('depends_on') is not None:
                summary += f"    (depends on Step {sub_query['depends_on'] + 1})\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    decomposer = QuestionDecomposer()
    
    test_questions = [
        "which nationality is frederica_of_mecklenburg-strelitz 's couple ?",
        "what is the gender of father of yixin_prince_gong ?",
        "what is the child of parent of shah_shuja ?",
        "what is the nationality of claudius ?",  # 1-hop
    ]
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        decomposed = decomposer.decompose(q)
        print(decomposer.get_decomposition_summary(decomposed))
        print("Decomposed queries:")
        for i, sub_q in enumerate(decomposed):
            print(f"  {i+1}. {sub_q}")
