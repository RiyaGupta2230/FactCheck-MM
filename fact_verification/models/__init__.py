"""
Fact Verification Models Module

This module provides comprehensive fact verification capabilities including claim detection,
evidence retrieval, stance detection, fact verification, and end-to-end pipeline integration
for robust automated fact-checking systems.

Key components:
- ClaimDetector: Identifies factual claims in text with span extraction
- EvidenceRetriever: Retrieves relevant evidence using dense/sparse retrieval 
- StanceDetector: Determines stance relationship between claims and evidence
- FactVerifier: Verifies claims against evidence with interpretable attention
- FactCheckPipeline: Complete end-to-end fact-checking pipeline

Example Usage:
    >>> from fact_verification.models import FactCheckPipeline, FactVerifier, EvidenceRetriever
    >>> 
    >>> # Create complete pipeline
    >>> pipeline = FactCheckPipeline()
    >>> result = pipeline.verify_claim("The Earth is flat.")
    >>> print(f"Verdict: {result['label']}, Confidence: {result['confidence']:.3f}")
    >>>
    >>> # Use individual components
    >>> verifier = FactVerifier()
    >>> retriever = EvidenceRetriever(mode='hybrid')
    >>> 
    >>> claim = "COVID-19 vaccines are effective"
    >>> evidence = retriever.retrieve(claim, top_k=5)
    >>> verdict = verifier.verify(claim, evidence)
"""

from .claim_detector import ClaimDetector
from .evidence_retriever import EvidenceRetriever
from .fact_verifier import FactVerifier
from .stance_detector import StanceDetector
from .end_to_end_model import FactCheckPipeline

__all__ = [
    "ClaimDetector",
    "EvidenceRetriever", 
    "FactVerifier",
    "StanceDetector",
    "FactCheckPipeline"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
