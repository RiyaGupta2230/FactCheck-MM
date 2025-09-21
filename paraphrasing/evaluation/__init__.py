"""
Paraphrasing Evaluation Module

This module provides comprehensive evaluation capabilities for paraphrasing models
including automatic metrics, semantic similarity analysis, human evaluation workflows,
and quality assessment using trained quality scorers.

Key components:
- GenerationMetrics: BLEU, ROUGE, METEOR, BERTScore, and diversity metrics
- SemanticSimilarity: Sentence-transformer-based semantic similarity analysis  
- HumanEvalHooks: Tools for preparing and processing human evaluations
- QualityAssessor: Automated quality scoring using trained models

Example Usage:
    >>> from paraphrasing.evaluation import GenerationMetrics, SemanticSimilarity
    >>> 
    >>> # Compute generation metrics
    >>> metrics = GenerationMetrics()
    >>> metrics.update(references=['The weather is nice'], predictions=['It is a nice day'])
    >>> scores = metrics.compute()
    >>> print(f"BLEU: {scores['bleu']:.3f}, ROUGE-L: {scores['rouge_l']:.3f}")
    >>>
    >>> # Semantic similarity analysis
    >>> similarity = SemanticSimilarity()
    >>> sim_scores = similarity.compute_similarity(['Hello world'], ['Hi there'])
    >>> print(f"Similarity: {sim_scores[0]:.3f}")
"""

from .generation_metrics import GenerationMetrics
from .semantic_similarity import SemanticSimilarity
from .human_eval import HumanEvalHooks
from .quality_assessment import QualityAssessor

__all__ = [
    "GenerationMetrics",
    "SemanticSimilarity", 
    "HumanEvalHooks",
    "QualityAssessor"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
