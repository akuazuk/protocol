"""Clinical knowledge layer for protocol rules and consult fact checking."""
from clinical_knowledge.consult_facts import extract_consult_facts_heuristic
from clinical_knowledge.loader import clinical_knowledge_status
from clinical_knowledge.protocol_match import match_protocol_cards
from clinical_knowledge.rule_checker import run_rule_checker

__all__ = [
    "clinical_knowledge_status",
    "extract_consult_facts_heuristic",
    "match_protocol_cards",
    "run_rule_checker",
]
