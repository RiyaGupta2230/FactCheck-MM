#!/usr/bin/env python3
"""
Fact Check Metrics Computation

Implements comprehensive metrics for fact verification evaluation including
Precision@k, Recall@k, MRR, F1 scores, and accuracy with support for both
retrieval and verification components.

Example Usage:
    >>> from fact_verification.evaluation import FactCheckMetrics
    >>> 
    >>> metrics = FactCheckMetrics()
    >>> 
    >>> # Compute retrieval metrics
    >>> retrieval_scores = metrics.compute_retrieval_metrics(
    ...     retrieved_items, ground_truth, k_values=[1, 5, 10]
    ... )
    >>> 
    >>> # Compute classification metrics
    >>> classification_scores = metrics.compute_classification_metrics(
    ...     predictions, labels, class_names=['SUPPORTS', 'REFUTES', 'NEI']
    ... )
    >>> 
    >>> # Generate comprehensive report
    >>> report = metrics.generate_report(all_results)
    >>> print(report)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import json
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.metrics import BaseMetrics
from shared.utils.logging_utils import get_logger

# Optional imports
try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, classification_report,
        confusion_matrix, average_precision_score, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class MetricResult:
    """Container for metric computation results."""
    
    name: str
    value: float
    description: str
    higher_is_better: bool = True
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def __str__(self) -> str:
        ci_str = ""
        if self.confidence_interval:
            ci_str = f" (95% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}])"
        return f"{self.name}: {self.value:.4f}{ci_str}"


class FactCheckMetrics(BaseMetrics):
    """
    Comprehensive metrics computation for fact verification systems.
    
    Supports both retrieval metrics (Precision@k, Recall@k, MRR) and 
    classification metrics (Accuracy, F1, Precision, Recall) with 
    detailed analysis capabilities.
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize fact check metrics calculator.
        
        Args:
            logger: Optional logger instance
        """
        super().__init__()
        self.logger = logger or get_logger("FactCheckMetrics")
        
        # Default class names for fact verification
        self.default_class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        
        self.logger.info("Initialized FactCheckMetrics")
    
    def compute_precision_at_k(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Precision@k for retrieval evaluation.
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with Precision@k scores
        """
        if len(retrieved_items) != len(ground_truth):
            raise ValueError("Number of retrieved and ground truth lists must match")
        
        precision_scores = {}
        
        for k in k_values:
            total_precision = 0.0
            valid_queries = 0
            
            for retrieved, truth in zip(retrieved_items, ground_truth):
                if not truth:  # Skip queries with no ground truth
                    continue
                
                # Get top-k retrieved items
                top_k_retrieved = retrieved[:k]
                
                # Count relevant items in top-k
                relevant_count = sum(1 for item in top_k_retrieved if item in truth)
                
                # Precision@k for this query
                precision_at_k = relevant_count / min(k, len(top_k_retrieved)) if top_k_retrieved else 0.0
                
                total_precision += precision_at_k
                valid_queries += 1
            
            # Average precision across all queries
            precision_scores[f'precision@{k}'] = total_precision / valid_queries if valid_queries > 0 else 0.0
        
        return precision_scores
    
    def compute_recall_at_k(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@k for retrieval evaluation.
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with Recall@k scores
        """
        if len(retrieved_items) != len(ground_truth):
            raise ValueError("Number of retrieved and ground truth lists must match")
        
        recall_scores = {}
        
        for k in k_values:
            total_recall = 0.0
            valid_queries = 0
            
            for retrieved, truth in zip(retrieved_items, ground_truth):
                if not truth:  # Skip queries with no ground truth
                    continue
                
                # Get top-k retrieved items
                top_k_retrieved = retrieved[:k]
                
                # Count relevant items in top-k
                relevant_count = sum(1 for item in top_k_retrieved if item in truth)
                
                # Recall@k for this query
                recall_at_k = relevant_count / len(truth)
                
                total_recall += recall_at_k
                valid_queries += 1
            
            # Average recall across all queries
            recall_scores[f'recall@{k}'] = total_recall / valid_queries if valid_queries > 0 else 0.0
        
        return recall_scores
    
    def compute_mrr(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            
        Returns:
            MRR score
        """
        if len(retrieved_items) != len(ground_truth):
            raise ValueError("Number of retrieved and ground truth lists must match")
        
        reciprocal_ranks = []
        
        for retrieved, truth in zip(retrieved_items, ground_truth):
            if not truth:  # Skip queries with no ground truth
                continue
            
            # Find the rank of the first relevant item
            first_relevant_rank = None
            
            for rank, item in enumerate(retrieved, 1):
                if item in truth:
                    first_relevant_rank = rank
                    break
            
            # Add reciprocal rank (0 if no relevant item found)
            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_ndcg_at_k(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]],
        relevance_scores: Optional[List[List[float]]] = None,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@k).
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            relevance_scores: Optional relevance scores for retrieved items
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with NDCG@k scores
        """
        if len(retrieved_items) != len(ground_truth):
            raise ValueError("Number of retrieved and ground truth lists must match")
        
        def dcg_at_k(relevances: List[float], k: int) -> float:
            """Compute DCG@k."""
            relevances = relevances[:k]
            dcg = 0.0
            for i, rel in enumerate(relevances):
                dcg += (2**rel - 1) / np.log2(i + 2)
            return dcg
        
        ndcg_scores = {}
        
        for k in k_values:
            total_ndcg = 0.0
            valid_queries = 0
            
            for i, (retrieved, truth) in enumerate(zip(retrieved_items, ground_truth)):
                if not truth:  # Skip queries with no ground truth
                    continue
                
                # Get relevance scores for retrieved items
                if relevance_scores and i < len(relevance_scores):
                    rel_scores = relevance_scores[i][:k]
                else:
                    # Binary relevance: 1 if in ground truth, 0 otherwise
                    rel_scores = [1.0 if item in truth else 0.0 for item in retrieved[:k]]
                
                # Compute DCG@k for retrieved items
                dcg = dcg_at_k(rel_scores, k)
                
                # Compute ideal DCG@k (perfect ranking)
                ideal_relevances = [1.0] * min(len(truth), k) + [0.0] * max(0, k - len(truth))
                ideal_dcg = dcg_at_k(ideal_relevances, k)
                
                # Compute NDCG@k
                if ideal_dcg > 0:
                    ndcg = dcg / ideal_dcg
                else:
                    ndcg = 0.0
                
                total_ndcg += ndcg
                valid_queries += 1
            
            ndcg_scores[f'ndcg@{k}'] = total_ndcg / valid_queries if valid_queries > 0 else 0.0
        
        return ndcg_scores
    
    def compute_map(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]]
    ) -> float:
        """
        Compute Mean Average Precision (MAP).
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            
        Returns:
            MAP score
        """
        if len(retrieved_items) != len(ground_truth):
            raise ValueError("Number of retrieved and ground truth lists must match")
        
        average_precisions = []
        
        for retrieved, truth in zip(retrieved_items, ground_truth):
            if not truth:  # Skip queries with no ground truth
                continue
            
            # Compute average precision for this query
            precisions = []
            relevant_count = 0
            
            for rank, item in enumerate(retrieved, 1):
                if item in truth:
                    relevant_count += 1
                    precision_at_rank = relevant_count / rank
                    precisions.append(precision_at_rank)
            
            # Average precision for this query
            avg_precision = np.mean(precisions) if precisions else 0.0
            average_precisions.append(avg_precision)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def compute_retrieval_metrics(
        self,
        retrieved_items: List[List[Any]],
        ground_truth: List[List[Any]],
        relevance_scores: Optional[List[List[float]]] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute comprehensive retrieval metrics.
        
        Args:
            retrieved_items: List of retrieved item lists for each query
            ground_truth: List of ground truth item lists for each query
            relevance_scores: Optional relevance scores for retrieved items
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with all retrieval metrics
        """
        metrics = {}
        
        # Precision@k and Recall@k
        metrics.update(self.compute_precision_at_k(retrieved_items, ground_truth, k_values))
        metrics.update(self.compute_recall_at_k(retrieved_items, ground_truth, k_values))
        
        # MRR
        metrics['mrr'] = self.compute_mrr(retrieved_items, ground_truth)
        
        # MAP
        metrics['map'] = self.compute_map(retrieved_items, ground_truth)
        
        # NDCG@k
        metrics.update(self.compute_ndcg_at_k(retrieved_items, ground_truth, relevance_scores, k_values))
        
        return metrics
    
    def compute_classification_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        class_names: Optional[List[str]] = None,
        average: str = 'macro'
    ) -> Dict[str, Any]:
        """
        Compute classification metrics for fact verification.
        
        Args:
            predictions: Predicted class labels
            labels: Ground truth class labels
            class_names: Optional class names for detailed reporting
            average: Averaging strategy ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary with classification metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classification metrics")
        
        if len(predictions) != len(labels):
            raise ValueError("Number of predictions and labels must match")
        
        class_names = class_names or self.default_class_names
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            f'precision_{average}': precision,
            f'recall_{average}': recall,
            f'f1_{average}': f1
        }
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
                metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
                metrics[f'support_{class_name.lower()}'] = support_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def compute_confidence_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        confidences: List[float],
        k_values: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """
        Compute confidence-based metrics.
        
        Args:
            predictions: Predicted class labels
            labels: Ground truth class labels
            confidences: Prediction confidence scores
            k_values: Percentage values for confidence-based evaluation
            
        Returns:
            Dictionary with confidence metrics
        """
        if len(predictions) != len(labels) or len(predictions) != len(confidences):
            raise ValueError("Predictions, labels, and confidences must have same length")
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        
        metrics = {}
        
        # Accuracy at top-k% most confident predictions
        for k_percent in k_values:
            k_count = max(1, int(len(predictions) * k_percent / 100))
            top_k_indices = sorted_indices[:k_count]
            
            top_k_predictions = [predictions[i] for i in top_k_indices]
            top_k_labels = [labels[i] for i in top_k_indices]
            
            if SKLEARN_AVAILABLE:
                accuracy_at_k = accuracy_score(top_k_labels, top_k_predictions)
            else:
                accuracy_at_k = sum(p == l for p, l in zip(top_k_predictions, top_k_labels)) / len(top_k_predictions)
            
            metrics[f'accuracy@top{k_percent}%'] = accuracy_at_k
        
        # Calibration metrics
        metrics.update(self._compute_calibration_metrics(predictions, labels, confidences))
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        confidences: List[float],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics like ECE (Expected Calibration Error)."""
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        confidences = np.array(confidences)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predictions)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                # Accuracy and confidence for this bin
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Contribution to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': ece,
            'max_calibration_error': max([
                abs(confidences[in_bin].mean() - (predictions[in_bin] == labels[in_bin]).mean())
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers)
                for in_bin in [(confidences > bin_lower) & (confidences <= bin_upper)]
                if in_bin.sum() > 0
            ]) if len(confidences) > 0 else 0.0
        }
    
    def compute_all_metrics(
        self,
        predictions: Optional[List[int]] = None,
        labels: Optional[List[int]] = None,
        confidences: Optional[List[float]] = None,
        retrieved_items: Optional[List[List[Any]]] = None,
        ground_truth: Optional[List[List[Any]]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics based on provided data.
        
        Args:
            predictions: Predicted class labels (for classification)
            labels: Ground truth class labels (for classification)
            confidences: Prediction confidence scores
            retrieved_items: Retrieved items for each query (for retrieval)
            ground_truth: Ground truth items for each query (for retrieval)
            class_names: Class names for detailed reporting
            
        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}
        
        # Classification metrics
        if predictions is not None and labels is not None:
            all_metrics.update(self.compute_classification_metrics(predictions, labels, class_names))
            
            # Confidence-based metrics
            if confidences is not None:
                all_metrics.update(self.compute_confidence_metrics(predictions, labels, confidences))
        
        # Retrieval metrics
        if retrieved_items is not None and ground_truth is not None:
            all_metrics.update(self.compute_retrieval_metrics(retrieved_items, ground_truth))
        
        return all_metrics
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        title: str = "Fact Check Evaluation Report"
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            metrics: Dictionary of computed metrics
            title: Report title
            
        Returns:
            Formatted report string
        """
        report_lines = [
            f"\n{'='*60}",
            f"{title:^60}",
            f"{'='*60}\n"
        ]
        
        # Group metrics by type
        classification_metrics = {}
        retrieval_metrics = {}
        confidence_metrics = {}
        
        for key, value in metrics.items():
            if any(keyword in key for keyword in ['accuracy', 'precision', 'recall', 'f1']):
                if 'top' in key or 'calibration' in key:
                    confidence_metrics[key] = value
                else:
                    classification_metrics[key] = value
            elif any(keyword in key for keyword in ['@', 'mrr', 'map', 'ndcg']):
                retrieval_metrics[key] = value
            else:
                classification_metrics[key] = value
        
        # Classification metrics section
        if classification_metrics:
            report_lines.append("CLASSIFICATION METRICS")
            report_lines.append("-" * 30)
            
            # Main metrics first
            for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
                if metric in classification_metrics:
                    value = classification_metrics[metric]
                    if isinstance(value, (int, float)):
                        report_lines.append(f"{metric.replace('_', ' ').title():<20}: {value:.4f}")
            
            # Per-class metrics
            per_class_metrics = {k: v for k, v in classification_metrics.items() 
                               if any(class_name.lower() in k for class_name in self.default_class_names)}
            
            if per_class_metrics:
                report_lines.append("\nPer-Class Metrics:")
                for class_name in self.default_class_names:
                    class_key = class_name.lower()
                    precision_key = f'precision_{class_key}'
                    recall_key = f'recall_{class_key}'
                    f1_key = f'f1_{class_key}'
                    
                    if all(key in per_class_metrics for key in [precision_key, recall_key, f1_key]):
                        report_lines.append(
                            f"  {class_name:<15}: P={per_class_metrics[precision_key]:.3f}, "
                            f"R={per_class_metrics[recall_key]:.3f}, "
                            f"F1={per_class_metrics[f1_key]:.3f}"
                        )
            
            report_lines.append("")
        
        # Retrieval metrics section
        if retrieval_metrics:
            report_lines.append("RETRIEVAL METRICS")
            report_lines.append("-" * 30)
            
            # Precision@k and Recall@k
            precision_metrics = {k: v for k, v in retrieval_metrics.items() if k.startswith('precision@')}
            recall_metrics = {k: v for k, v in retrieval_metrics.items() if k.startswith('recall@')}
            
            if precision_metrics:
                report_lines.append("Precision@k:")
                for k in sorted(precision_metrics.keys(), key=lambda x: int(x.split('@')[1])):
                    report_lines.append(f"  {k:<15}: {precision_metrics[k]:.4f}")
            
            if recall_metrics:
                report_lines.append("Recall@k:")
                for k in sorted(recall_metrics.keys(), key=lambda x: int(x.split('@')[1])):
                    report_lines.append(f"  {k:<15}: {recall_metrics[k]:.4f}")
            
            # Other retrieval metrics
            for metric in ['mrr', 'map']:
                if metric in retrieval_metrics:
                    report_lines.append(f"{metric.upper():<15}: {retrieval_metrics[metric]:.4f}")
            
            # NDCG@k
            ndcg_metrics = {k: v for k, v in retrieval_metrics.items() if k.startswith('ndcg@')}
            if ndcg_metrics:
                report_lines.append("NDCG@k:")
                for k in sorted(ndcg_metrics.keys(), key=lambda x: int(x.split('@')[1])):
                    report_lines.append(f"  {k:<15}: {ndcg_metrics[k]:.4f}")
            
            report_lines.append("")
        
        # Confidence metrics section
        if confidence_metrics:
            report_lines.append("CONFIDENCE METRICS")
            report_lines.append("-" * 30)
            
            for key, value in confidence_metrics.items():
                if isinstance(value, (int, float)):
                    display_key = key.replace('_', ' ').replace('@', ' @ ').title()
                    report_lines.append(f"{display_key:<25}: {value:.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        filepath: str,
        format: str = 'json'
    ):
        """
        Save metrics to file.
        
        Args:
            metrics: Dictionary of computed metrics
            filepath: Output file path
            format: Output format ('json' or 'csv')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        elif format == 'csv':
            # Flatten metrics for CSV format
            flattened_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    flattened_metrics[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, np.integer, np.floating)):
                            flattened_metrics[f"{key}_{sub_key}"] = sub_value
            
            df = pd.DataFrame([flattened_metrics])
            df.to_csv(filepath, index=False)
        
        self.logger.info(f"Metrics saved to {filepath}")


def main():
    """Example usage of FactCheckMetrics."""
    
    # Initialize metrics calculator
    metrics_calc = FactCheckMetrics()
    
    print("=== FactCheckMetrics Example ===")
    
    # Example classification data
    predictions = [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]
    labels = [0, 1, 1, 0, 2, 1, 2, 0, 1, 2]
    confidences = [0.9, 0.8, 0.7, 0.95, 0.6, 0.85, 0.9, 0.88, 0.75, 0.8]
    
    # Example retrieval data
    retrieved_items = [
        ['doc1', 'doc2', 'doc3'],
        ['doc4', 'doc5', 'doc6'],
        ['doc7', 'doc8', 'doc9']
    ]
    ground_truth = [
        ['doc1', 'doc10'],
        ['doc4', 'doc11'],
        ['doc12', 'doc8']
    ]
    
    print("Computing classification metrics...")
    classification_metrics = metrics_calc.compute_classification_metrics(
        predictions, labels, ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
    )
    
    print("Computing retrieval metrics...")
    retrieval_metrics = metrics_calc.compute_retrieval_metrics(
        retrieved_items, ground_truth
    )
    
    print("Computing confidence metrics...")
    confidence_metrics = metrics_calc.compute_confidence_metrics(
        predictions, labels, confidences
    )
    
    # Combine all metrics
    all_metrics = {**classification_metrics, **retrieval_metrics, **confidence_metrics}
    
    # Generate and display report
    report = metrics_calc.generate_report(all_metrics, "Example Evaluation Report")
    print(report)
    
    # Save metrics
    output_dir = Path("fact_verification/evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_calc.save_metrics(all_metrics, output_dir / "example_metrics.json")
    metrics_calc.save_metrics(all_metrics, output_dir / "example_metrics.csv", format='csv')
    
    print(f"\nMetrics saved to {output_dir}")


if __name__ == "__main__":
    main()
