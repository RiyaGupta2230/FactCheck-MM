"""
Fact Verification Evaluation Module

This module provides comprehensive evaluation capabilities for fact verification systems,
including metrics computation, evidence quality assessment, pipeline evaluation, and
systematic error analysis for robust fact-checking performance measurement.

Key components:
- FactCheckMetrics: Core metrics for fact verification (Precision@k, Recall@k, MRR, F1)
- EvidenceEvaluator: Evidence retrieval quality assessment (relevance, diversity, NDCG)
- PipelineEvaluator: End-to-end pipeline evaluation with domain-specific breakdowns
- ErrorAnalyzer: Systematic failure analysis and error categorization

Example Usage:
    >>> from fact_verification.evaluation import PipelineEvaluator, FactCheckMetrics
    >>> 
    >>> # Evaluate complete pipeline
    >>> evaluator = PipelineEvaluator(pipeline, test_dataset)
    >>> results = evaluator.evaluate()
    >>> print(f"Pipeline Accuracy: {results['accuracy']:.3f}")
    >>> 
    >>> # Compute detailed metrics
    >>> metrics = FactCheckMetrics()
    >>> scores = metrics.compute_all_metrics(predictions, labels, confidences)
    >>> 
    >>> # Domain-specific evaluation
    >>> domain_results = evaluator.evaluate_by_domain()
    >>> for domain, scores in domain_results.items():
    ...     print(f"{domain}: F1={scores['f1']:.3f}")
"""

from .fact_check_metrics import FactCheckMetrics
from .evidence_eval import EvidenceEvaluator
from .pipeline_evaluation import PipelineEvaluator
from .error_analysis import ErrorAnalyzer

__all__ = [
    "FactCheckMetrics",
    "EvidenceEvaluator", 
    "PipelineEvaluator",
    "ErrorAnalyzer"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
