"""
Fact Verification Utilities Module

This module provides essential utilities for fact verification systems including
claim processing, evidence handling, and knowledge base integration capabilities
for comprehensive fact-checking pipeline support.

Key components:
- ClaimProcessor: Advanced claim preprocessing with NER, coreference resolution, and normalization
- EvidenceUtils: Evidence formatting, deduplication, and multimodal handling utilities
- KnowledgeBaseConnector: Integration with external knowledge bases (Wikidata, DBpedia, Wikipedia)

Example Usage:
    >>> from fact_verification.utils import ClaimProcessor, EvidenceUtils, KnowledgeBaseConnector
    >>> 
    >>> # Process claims with NER and normalization
    >>> processor = ClaimProcessor()
    >>> processed_claims = processor.process_claims(["The Earth is flat", "COVID vaccines are safe"])
    >>> 
    >>> # Handle and format evidence
    >>> evidence_utils = EvidenceUtils()
    >>> formatted_evidence = evidence_utils.prepare_evidence_batch(evidence_list)
    >>> 
    >>> # Query external knowledge bases
    >>> kb_connector = KnowledgeBaseConnector()
    >>> entity_info = kb_connector.query_wikidata("Albert Einstein")
"""

from .claim_processing import ClaimProcessor
from .evidence_utils import EvidenceUtils
from .knowledge_bases import KnowledgeBaseConnector

__all__ = [
    "ClaimProcessor",
    "EvidenceUtils", 
    "KnowledgeBaseConnector"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
