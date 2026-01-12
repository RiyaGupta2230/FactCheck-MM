#!/usr/bin/env python3
"""
End-to-End Pipeline Evaluation

Comprehensive evaluation of complete fact-checking pipelines including
domain-specific analysis, performance breakdown, and detailed reporting
with support for both individual component and full pipeline assessment.

Example Usage:
    >>> from fact_verification.evaluation import PipelineEvaluator
    >>> from fact_verification.models import FactCheckPipeline
    >>> 
    >>> # Load pipeline and test data
    >>> pipeline = FactCheckPipeline.from_pretrained("checkpoints/best_pipeline")
    >>> test_dataset = UnifiedFactDataset('test')
    >>> 
    >>> # Initialize evaluator
    >>> evaluator = PipelineEvaluator(pipeline, test_dataset)
    >>> 
    >>> # Run comprehensive evaluation
    >>> results = evaluator.evaluate()
    >>> print(f"Pipeline Accuracy: {results['accuracy']:.3f}")
    >>> 
    >>> # Domain-specific evaluation
    >>> domain_results = evaluator.evaluate_by_domain()
    >>> for domain, metrics in domain_results.items():
    ...     print(f"{domain}: F1={metrics['f1']:.3f}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import re

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
from fact_verification.models import FactCheckPipeline
from shared.utils.logging_utils import get_logger
from shared.datasets.data_loaders import ChunkedDataLoader
from .fact_check_metrics import FactCheckMetrics
from .evidence_eval import EvidenceEvaluator

# Optional imports
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PipelineEvaluator:
    """
    Comprehensive evaluation system for end-to-end fact-checking pipelines.
    
    Provides detailed performance analysis including accuracy, precision/recall,
    domain-specific breakdowns, component-wise analysis, and comprehensive
    reporting with both human-readable and machine-readable outputs.
    """
    
    def __init__(
        self,
        pipeline: FactCheckPipeline,
        test_dataset: Union[torch.utils.data.Dataset, DataLoader],
        output_dir: str = "fact_verification/evaluation/results",
        logger: Optional[Any] = None,
        chunk_size: int = 32
    ):
        """
        Initialize pipeline evaluator.
        
        Args:
            pipeline: FactCheckPipeline instance to evaluate
            test_dataset: Test dataset or DataLoader
            output_dir: Directory for saving evaluation results
            logger: Optional logger instance
            chunk_size: Batch size for evaluation (for memory efficiency)
        """
        self.pipeline = pipeline
        self.test_dataset = test_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger("PipelineEvaluator")
        self.chunk_size = chunk_size
        
        # Initialize metrics calculators
        self.fact_check_metrics = FactCheckMetrics()
        self.evidence_evaluator = EvidenceEvaluator()
        
        # Create data loader if dataset provided
        if not isinstance(test_dataset, DataLoader):
            self.data_loader = DataLoader(
                test_dataset,
                batch_size=chunk_size,
                shuffle=False,
                num_workers=2
            )
        else:
            self.data_loader = test_dataset
        
        # Domain classification setup
        self.domain_classifier = self._setup_domain_classifier()
        
        # Evaluation state
        self.evaluation_results = {}
        self.detailed_results = []
        
        self.logger.info(f"Initialized PipelineEvaluator with {len(self.test_dataset)} test samples")
    
    def _setup_domain_classifier(self) -> Optional[Any]:
        """Setup domain classifier for domain-specific evaluation."""
        
        # Simple keyword-based domain classification
        self.domain_keywords = {
            'politics': [
                'election', 'vote', 'president', 'congress', 'senate', 'republican', 
                'democrat', 'policy', 'government', 'politician', 'campaign', 'ballot'
            ],
            'health': [
                'vaccine', 'medicine', 'doctor', 'hospital', 'treatment', 'disease',
                'virus', 'covid', 'flu', 'health', 'medical', 'patient', 'drug'
            ],
            'science': [
                'research', 'study', 'scientist', 'experiment', 'climate', 'global warming',
                'temperature', 'carbon', 'emission', 'environment', 'earth', 'space'
            ],
            'technology': [
                'computer', 'software', 'internet', 'ai', 'artificial intelligence',
                'robot', 'technology', 'digital', 'algorithm', 'data', 'tech'
            ],
            'economy': [
                'economy', 'market', 'stock', 'inflation', 'unemployment', 'gdp',
                'recession', 'financial', 'bank', 'investment', 'trade', 'money'
            ]
        }
        
        return True  # Indicate that domain classification is available
    
    def _classify_domain(self, text: str) -> str:
        """
        Classify text into domain categories.
        
        Args:
            text: Input text to classify
            
        Returns:
            Domain label
        """
        if not self.domain_classifier:
            return 'general'
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no keywords found
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def evaluate(
        self,
        include_evidence_analysis: bool = True,
        include_timing: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive pipeline evaluation.
        
        Args:
            include_evidence_analysis: Whether to include evidence quality analysis
            include_timing: Whether to include timing analysis
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting comprehensive pipeline evaluation")
        
        # Initialize collection variables
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_claims = []
        all_retrieved_evidence = []
        all_ground_truth_evidence = []
        all_processing_times = []
        all_verdicts = []
        
        total_samples = 0
        successful_evaluations = 0
        
        # Set pipeline to evaluation mode
        self.pipeline.eval()
        
        # Process dataset in chunks
        for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Evaluating Pipeline")):
            batch_size = len(batch['claim']) if 'claim' in batch else len(batch.get('claim_text', []))
            total_samples += batch_size
            
            # Extract batch data
            if 'claim_text' in batch:
                claims = batch['claim_text']
            elif 'claim' in batch:
                claims = batch['claim']
            else:
                claims = [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)]
            
            if 'label' in batch:
                if torch.is_tensor(batch['label']):
                    labels = batch['label'].cpu().numpy().tolist()
                else:
                    labels = batch['label']
            else:
                labels = [2] * batch_size  # Default to NOT_ENOUGH_INFO
            
            # Get ground truth evidence if available
            if 'evidence_text' in batch:
                ground_truth_evidence = batch['evidence_text']
            else:
                ground_truth_evidence = [[] for _ in range(batch_size)]
            
            # Process each claim in the batch
            for i, claim in enumerate(claims):
                try:
                    start_time = time.time()
                    
                    # Run pipeline evaluation
                    result = self.pipeline.check_fact(
                        claim,
                        include_evidence_details=include_evidence_analysis,
                        include_stance_details=False
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Extract results
                    verdict = result.get('verdict', 'NOT_ENOUGH_INFO')
                    confidence = result.get('confidence', 0.0)
                    evidence = result.get('evidence', [])
                    
                    # Map verdict to label ID
                    verdict_to_id = {
                        'SUPPORTS': 0,
                        'REFUTES': 1,
                        'NOT_ENOUGH_INFO': 2
                    }
                    prediction = verdict_to_id.get(verdict, 2)
                    
                    # Store results
                    all_predictions.append(prediction)
                    all_labels.append(labels[i])
                    all_confidences.append(confidence)
                    all_claims.append(claim)
                    all_verdicts.append(verdict)
                    all_processing_times.append(processing_time)
                    
                    # Evidence analysis
                    if include_evidence_analysis and evidence:
                        evidence_texts = [ev.get('text', str(ev)) for ev in evidence]
                        all_retrieved_evidence.append(evidence_texts)
                    else:
                        all_retrieved_evidence.append([])
                    
                    all_ground_truth_evidence.append(ground_truth_evidence[i] if i < len(ground_truth_evidence) else [])
                    
                    # Store detailed result
                    detailed_result = {
                        'claim': claim,
                        'ground_truth_label': labels[i],
                        'predicted_label': prediction,
                        'verdict': verdict,
                        'confidence': confidence,
                        'processing_time': processing_time,
                        'evidence_count': len(evidence) if evidence else 0,
                        'success': True
                    }
                    
                    self.detailed_results.append(detailed_result)
                    successful_evaluations += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate claim {i} in batch {batch_idx}: {e}")
                    
                    # Add failure record
                    all_predictions.append(2)  # Default prediction
                    all_labels.append(labels[i])
                    all_confidences.append(0.0)
                    all_claims.append(claim)
                    all_verdicts.append('ERROR')
                    all_processing_times.append(0.0)
                    all_retrieved_evidence.append([])
                    all_ground_truth_evidence.append(ground_truth_evidence[i] if i < len(ground_truth_evidence) else [])
                    
                    detailed_result = {
                        'claim': claim,
                        'ground_truth_label': labels[i],
                        'predicted_label': 2,
                        'verdict': 'ERROR',
                        'confidence': 0.0,
                        'processing_time': 0.0,
                        'evidence_count': 0,
                        'success': False,
                        'error': str(e)
                    }
                    
                    self.detailed_results.append(detailed_result)
        
        # Compute comprehensive metrics
        evaluation_results = self._compute_comprehensive_metrics(
            all_predictions, all_labels, all_confidences,
            all_retrieved_evidence, all_ground_truth_evidence,
            all_processing_times, include_evidence_analysis, include_timing
        )
        
        # Add summary statistics
        evaluation_results.update({
            'total_samples': total_samples,
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / total_samples if total_samples > 0 else 0.0,
            'evaluation_timestamp': datetime.now().isoformat()
        })
        
        # Store results
        self.evaluation_results = evaluation_results
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_results)
        
        self.logger.info(f"Pipeline evaluation completed: {successful_evaluations}/{total_samples} successful")
        return evaluation_results
    
    def evaluate_by_domain(
        self,
        include_evidence_analysis: bool = False,
        save_results: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate pipeline performance by domain.
        
        Args:
            include_evidence_analysis: Whether to include evidence analysis
            save_results: Whether to save domain-specific results
            
        Returns:
            Dictionary with domain-specific evaluation results
        """
        self.logger.info("Starting domain-specific pipeline evaluation")
        
        # Group detailed results by domain
        domain_results = defaultdict(lambda: {
            'predictions': [],
            'labels': [],
            'confidences': [],
            'claims': [],
            'processing_times': [],
            'evidence_lists': [],
            'ground_truth_lists': []
        })
        
        # If no detailed results available, run evaluation first
        if not self.detailed_results:
            self.evaluate(include_evidence_analysis=include_evidence_analysis, save_results=False)
        
        # Classify each result by domain
        for result in self.detailed_results:
            if not result['success']:
                continue
            
            claim = result['claim']
            domain = self._classify_domain(claim)
            
            domain_results[domain]['predictions'].append(result['predicted_label'])
            domain_results[domain]['labels'].append(result['ground_truth_label'])
            domain_results[domain]['confidences'].append(result['confidence'])
            domain_results[domain]['claims'].append(claim)
            domain_results[domain]['processing_times'].append(result['processing_time'])
            
            # Find corresponding evidence (simplified lookup)
            evidence_idx = len(domain_results[domain]['predictions']) - 1
            if evidence_idx < len(self.detailed_results):
                domain_results[domain]['evidence_lists'].append([])  # Simplified
                domain_results[domain]['ground_truth_lists'].append([])
        
        # Compute metrics for each domain
        domain_metrics = {}
        
        for domain, data in domain_results.items():
            if not data['predictions']:
                continue
            
            # Compute domain-specific metrics
            domain_metrics[domain] = self._compute_comprehensive_metrics(
                data['predictions'], data['labels'], data['confidences'],
                data['evidence_lists'], data['ground_truth_lists'],
                data['processing_times'], include_evidence_analysis, True
            )
            
            # Add domain-specific information
            domain_metrics[domain].update({
                'domain': domain,
                'sample_count': len(data['predictions']),
                'sample_percentage': len(data['predictions']) / len(self.detailed_results) * 100
            })
        
        # Save domain results if requested
        if save_results:
            self._save_domain_results(domain_metrics)
        
        self.logger.info(f"Domain evaluation completed for {len(domain_metrics)} domains")
        return domain_metrics
    
    def _compute_comprehensive_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        confidences: List[float],
        retrieved_evidence: List[List[str]],
        ground_truth_evidence: List[List[str]],
        processing_times: List[float],
        include_evidence_analysis: bool = True,
        include_timing: bool = True
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        metrics = {}
        
        # Classification metrics
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        classification_metrics = self.fact_check_metrics.compute_classification_metrics(
            predictions, labels, class_names
        )
        metrics.update(classification_metrics)
        
        # Confidence-based metrics
        if confidences:
            confidence_metrics = self.fact_check_metrics.compute_confidence_metrics(
                predictions, labels, confidences
            )
            metrics.update(confidence_metrics)
        
        # Evidence analysis
        if include_evidence_analysis and retrieved_evidence and any(retrieved_evidence):
            # Create dummy claims for evidence evaluation
            claims = [f"claim_{i}" for i in range(len(retrieved_evidence))]
            
            try:
                evidence_quality = self.evidence_evaluator.evaluate_evidence_quality(
                    claims, retrieved_evidence, ground_truth_evidence
                )
                
                metrics.update({
                    'evidence_relevance': evidence_quality.relevance_score,
                    'evidence_diversity': evidence_quality.diversity_score,
                    'evidence_coverage': evidence_quality.coverage_score,
                    'evidence_redundancy': evidence_quality.redundancy_score
                })
                
                # Add ranking quality metrics
                metrics.update({f'evidence_{k}': v for k, v in evidence_quality.ranking_quality.items()})
                
            except Exception as e:
                self.logger.warning(f"Evidence analysis failed: {e}")
                metrics.update({
                    'evidence_relevance': 0.0,
                    'evidence_diversity': 0.0,
                    'evidence_coverage': 0.0,
                    'evidence_redundancy': 0.0
                })
        
        # Timing analysis
        if include_timing and processing_times:
            metrics.update({
                'avg_processing_time': np.mean(processing_times),
                'median_processing_time': np.median(processing_times),
                'min_processing_time': np.min(processing_times),
                'max_processing_time': np.max(processing_times),
                'std_processing_time': np.std(processing_times),
                'total_processing_time': np.sum(processing_times)
            })
        
        return metrics
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results as JSON
        results_file = self.output_dir / f"pipeline_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save detailed results as CSV
        if self.detailed_results:
            detailed_file = self.output_dir / f"detailed_results_{timestamp}.csv"
            df = pd.DataFrame(self.detailed_results)
            df.to_csv(detailed_file, index=False)
        
        # Save summary report
        report_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_evaluation_report(results))
        
        self.logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def _save_domain_results(self, domain_metrics: Dict[str, Dict[str, Any]]):
        """Save domain-specific evaluation results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save domain metrics as JSON
        domain_file = self.output_dir / f"domain_evaluation_{timestamp}.json"
        with open(domain_file, 'w') as f:
            serializable_metrics = self._make_json_serializable(domain_metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save domain comparison as CSV
        domain_comparison = []
        for domain, metrics in domain_metrics.items():
            row = {
                'domain': domain,
                'sample_count': metrics.get('sample_count', 0),
                'accuracy': metrics.get('accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'precision_macro': metrics.get('precision_macro', 0),
                'recall_macro': metrics.get('recall_macro', 0),
                'avg_processing_time': metrics.get('avg_processing_time', 0)
            }
            
            # Add evidence metrics if available
            if 'evidence_relevance' in metrics:
                row.update({
                    'evidence_relevance': metrics['evidence_relevance'],
                    'evidence_diversity': metrics['evidence_diversity'],
                    'evidence_coverage': metrics['evidence_coverage']
                })
            
            domain_comparison.append(row)
        
        comparison_file = self.output_dir / f"domain_comparison_{timestamp}.csv"
        df = pd.DataFrame(domain_comparison)
        df.to_csv(comparison_file, index=False)
        
        self.logger.info(f"Domain evaluation results saved to {self.output_dir}")
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        title: str = "Pipeline Evaluation Report"
    ) -> str:
        """Generate human-readable evaluation report."""
        
        report_lines = [
            f"\n{'='*80}",
            f"{title:^80}",
            f"{'='*80}\n"
        ]
        
        # Summary section
        report_lines.extend([
            "EVALUATION SUMMARY",
            "-" * 50,
            f"{'Total Samples':<25}: {results.get('total_samples', 0):,}",
            f"{'Successful Evaluations':<25}: {results.get('successful_evaluations', 0):,}",
            f"{'Success Rate':<25}: {results.get('success_rate', 0):.1%}",
            f"{'Evaluation Time':<25}: {results.get('evaluation_timestamp', 'N/A')}",
            ""
        ])
        
        # Performance metrics
        report_lines.extend([
            "PERFORMANCE METRICS",
            "-" * 50,
            f"{'Accuracy':<25}: {results.get('accuracy', 0):.4f}",
            f"{'Macro F1':<25}: {results.get('f1_macro', 0):.4f}",
            f"{'Macro Precision':<25}: {results.get('precision_macro', 0):.4f}",
            f"{'Macro Recall':<25}: {results.get('recall_macro', 0):.4f}",
            ""
        ])
        
        # Per-class metrics
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        report_lines.extend([
            "PER-CLASS METRICS",
            "-" * 50
        ])
        
        for class_name in class_names:
            class_key = class_name.lower()
            precision = results.get(f'precision_{class_key}', 0)
            recall = results.get(f'recall_{class_key}', 0)
            f1 = results.get(f'f1_{class_key}', 0)
            
            report_lines.append(
                f"{class_name:<15}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}"
            )
        
        report_lines.append("")
        
        # Confidence metrics
        if any(key.startswith('accuracy@') for key in results.keys()):
            report_lines.extend([
                "CONFIDENCE-BASED METRICS",
                "-" * 50
            ])
            
            for key, value in results.items():
                if key.startswith('accuracy@'):
                    display_key = key.replace('@', ' @ ').replace('top', 'Top ').replace('%', '%')
                    report_lines.append(f"{display_key:<25}: {value:.4f}")
            
            report_lines.append("")
        
        # Evidence quality metrics
        if 'evidence_relevance' in results:
            report_lines.extend([
                "EVIDENCE QUALITY METRICS",
                "-" * 50,
                f"{'Relevance':<25}: {results.get('evidence_relevance', 0):.4f}",
                f"{'Diversity':<25}: {results.get('evidence_diversity', 0):.4f}",
                f"{'Coverage':<25}: {results.get('evidence_coverage', 0):.4f}",
                f"{'Redundancy':<25}: {results.get('evidence_redundancy', 0):.4f}",
                ""
            ])
        
        # Timing metrics
        if 'avg_processing_time' in results:
            report_lines.extend([
                "TIMING ANALYSIS",
                "-" * 50,
                f"{'Average Time':<25}: {results.get('avg_processing_time', 0):.3f}s",
                f"{'Median Time':<25}: {results.get('median_processing_time', 0):.3f}s",
                f"{'Min Time':<25}: {results.get('min_processing_time', 0):.3f}s",
                f"{'Max Time':<25}: {results.get('max_processing_time', 0):.3f}s",
                f"{'Total Time':<25}: {results.get('total_processing_time', 0):.1f}s",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and capabilities."""
        
        summary = {
            'pipeline_components': {
                'claim_detection': self.pipeline.config.enable_claim_detection,
                'evidence_retrieval': self.pipeline.config.enable_evidence_retrieval,
                'stance_detection': self.pipeline.config.enable_stance_detection,
                'fact_verification': self.pipeline.config.enable_fact_verification
            },
            'pipeline_config': {
                'max_evidence_per_claim': self.pipeline.config.max_evidence_per_claim,
                'evidence_aggregation_strategy': self.pipeline.config.evidence_aggregation_strategy,
                'confidence_threshold': self.pipeline.config.confidence_threshold
            },
            'evaluation_setup': {
                'test_dataset_size': len(self.test_dataset),
                'chunk_size': self.chunk_size,
                'output_directory': str(self.output_dir)
            }
        }
        
        return summary


def main():
    """Example usage of PipelineEvaluator."""
    
    print("=== PipelineEvaluator Example ===")
    
    # This example demonstrates the interface - actual usage requires trained models
    try:
        from fact_verification.models import FactCheckPipeline, FactCheckPipelineConfig
        from fact_verification.data import UnifiedFactDataset
        
        # Create a simple pipeline for demonstration
        config = FactCheckPipelineConfig(
            enable_evidence_retrieval=True,
            enable_fact_verification=True,
            max_evidence_per_claim=3
        )
        
        pipeline = FactCheckPipeline(config)
        
        # Create a small test dataset
        print("Creating test dataset...")
        test_dataset = UnifiedFactDataset('test', use_both_datasets=False)
        
        # Limit to small subset for example
        from torch.utils.data import Subset
        test_subset = Subset(test_dataset, range(min(10, len(test_dataset))))
        
        # Initialize evaluator
        evaluator = PipelineEvaluator(
            pipeline=pipeline,
            test_dataset=test_subset,
            output_dir="fact_verification/evaluation/results/example"
        )
        
        print("Pipeline Summary:")
        summary = evaluator.get_pipeline_summary()
        for category, details in summary.items():
            print(f"\n{category.upper()}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        print("\nRunning evaluation...")
        results = evaluator.evaluate(
            include_evidence_analysis=False,  # Skip for faster example
            include_timing=True,
            save_results=True
        )
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"F1 Macro: {results.get('f1_macro', 0):.3f}")
        print(f"Success Rate: {results.get('success_rate', 0):.1%}")
        
        # Generate and display report
        report = evaluator.generate_evaluation_report(results)
        print(report)
        
        print("\nRunning domain-specific evaluation...")
        domain_results = evaluator.evaluate_by_domain(save_results=True)
        
        print("Domain Results:")
        for domain, metrics in domain_results.items():
            print(f"  {domain}: Accuracy={metrics.get('accuracy', 0):.3f}, "
                  f"Samples={metrics.get('sample_count', 0)}")
        
        print(f"\nResults saved to: {evaluator.output_dir}")
        
    except ImportError as e:
        print(f"Required modules not available: {e}")
        print("This example requires trained models and datasets to run properly.")
    
    except Exception as e:
        print(f"Example failed: {e}")
        print("This is expected without proper trained models and datasets.")


if __name__ == "__main__":
    main()
# Alias for backward compatibility with main.py
FactVerificationEvaluator = PipelineEvaluator 
