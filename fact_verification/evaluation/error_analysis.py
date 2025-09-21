#!/usr/bin/env python3
"""
Error Analysis for Fact Verification Systems

Systematic analysis of failure cases in fact verification pipelines including
categorization of error types, visualization of failure patterns, and
detailed inspection capabilities for model improvement.

Example Usage:
    >>> from fact_verification.evaluation import ErrorAnalyzer
    >>> 
    >>> # Initialize analyzer with evaluation results
    >>> analyzer = ErrorAnalyzer(predictions, labels, claims, evidence_lists)
    >>> 
    >>> # Analyze error patterns
    >>> error_analysis = analyzer.analyze_errors()
    >>> print(f"Total errors: {error_analysis['total_errors']}")
    >>> 
    >>> # Generate error report with visualizations
    >>> analyzer.generate_error_report(save_visualizations=True)
    >>> 
    >>> # Export top errors for manual inspection
    >>> analyzer.export_top_errors(n=50, output_file="top_errors.csv")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict, Counter
import re

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger
from .fact_check_metrics import FactCheckMetrics

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from shared.utils.visualization import create_confusion_matrix, create_error_histogram
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Optional imports for text analysis
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ErrorCategory:
    """Enumeration of error categories for fact verification."""
    
    WRONG_EVIDENCE_RETRIEVED = "wrong_evidence_retrieved"
    CORRECT_EVIDENCE_WRONG_LABEL = "correct_evidence_wrong_label"
    BOTH_EVIDENCE_AND_LABEL_WRONG = "both_evidence_and_label_wrong"
    NO_EVIDENCE_FOUND = "no_evidence_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS_CLAIM = "ambiguous_claim"
    PROCESSING_ERROR = "processing_error"


class ErrorAnalyzer:
    """
    Comprehensive error analysis system for fact verification pipelines.
    
    Analyzes failure cases across multiple dimensions including evidence retrieval
    quality, verification accuracy, error patterns, and provides detailed 
    categorization with visualization and export capabilities.
    """
    
    def __init__(
        self,
        predictions: Optional[List[int]] = None,
        labels: Optional[List[int]] = None,
        claims: Optional[List[str]] = None,
        retrieved_evidence: Optional[List[List[str]]] = None,
        ground_truth_evidence: Optional[List[List[str]]] = None,
        confidences: Optional[List[float]] = None,
        pipeline_results: Optional[List[Dict[str, Any]]] = None,
        class_names: Optional[List[str]] = None,
        logger: Optional[Any] = None
    ):
        """
        Initialize error analyzer.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            claims: Original claim texts
            retrieved_evidence: Retrieved evidence for each claim
            ground_truth_evidence: Ground truth evidence for each claim
            confidences: Prediction confidence scores
            pipeline_results: Complete pipeline results (alternative input)
            class_names: Names of prediction classes
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("ErrorAnalyzer")
        
        # Initialize from pipeline results if provided
        if pipeline_results:
            self._initialize_from_pipeline_results(pipeline_results)
        else:
            self.predictions = predictions or []
            self.labels = labels or []
            self.claims = claims or []
            self.retrieved_evidence = retrieved_evidence or []
            self.ground_truth_evidence = ground_truth_evidence or []
            self.confidences = confidences or []
        
        self.class_names = class_names or ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        
        # Derived data
        self.errors = []
        self.error_categories = defaultdict(list)
        self.error_analysis_results = {}
        
        # Validation
        self._validate_inputs()
        
        # Initialize metrics calculator
        self.metrics_calc = FactCheckMetrics()
        
        self.logger.info(f"Initialized ErrorAnalyzer with {len(self.predictions)} samples")
    
    def _initialize_from_pipeline_results(self, pipeline_results: List[Dict[str, Any]]):
        """Initialize analyzer from pipeline evaluation results."""
        
        self.predictions = []
        self.labels = []
        self.claims = []
        self.retrieved_evidence = []
        self.ground_truth_evidence = []
        self.confidences = []
        
        for result in pipeline_results:
            if result.get('success', True):
                self.predictions.append(result.get('predicted_label', 2))
                self.labels.append(result.get('ground_truth_label', 2))
                self.claims.append(result.get('claim', ''))
                self.confidences.append(result.get('confidence', 0.0))
                
                # Extract evidence (simplified)
                evidence = result.get('evidence', [])
                if isinstance(evidence, list) and evidence:
                    evidence_texts = [str(ev) for ev in evidence]
                else:
                    evidence_texts = []
                
                self.retrieved_evidence.append(evidence_texts)
                self.ground_truth_evidence.append([])  # Not available in pipeline results
    
    def _validate_inputs(self):
        """Validate input data consistency."""
        
        data_lengths = [
            len(self.predictions),
            len(self.labels),
            len(self.claims)
        ]
        
        if len(set(data_lengths)) > 1:
            self.logger.warning(f"Inconsistent data lengths: {data_lengths}")
            
            # Truncate to minimum length
            min_length = min(data_lengths)
            self.predictions = self.predictions[:min_length]
            self.labels = self.labels[:min_length]
            self.claims = self.claims[:min_length]
            
            # Adjust optional data
            if self.retrieved_evidence:
                self.retrieved_evidence = self.retrieved_evidence[:min_length]
            if self.ground_truth_evidence:
                self.ground_truth_evidence = self.ground_truth_evidence[:min_length]
            if self.confidences:
                self.confidences = self.confidences[:min_length]
    
    def analyze_errors(
        self,
        include_evidence_analysis: bool = True,
        include_text_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive error analysis.
        
        Args:
            include_evidence_analysis: Whether to analyze evidence quality
            include_text_analysis: Whether to analyze text patterns
            
        Returns:
            Dictionary with error analysis results
        """
        self.logger.info("Starting comprehensive error analysis")
        
        # Identify errors
        self._identify_errors()
        
        # Categorize errors
        self._categorize_errors(include_evidence_analysis)
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns()
        
        # Text analysis of errors
        text_analysis = {}
        if include_text_analysis and SKLEARN_AVAILABLE:
            text_analysis = self._analyze_error_text_patterns()
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence_patterns()
        
        # Compile results
        self.error_analysis_results = {
            'total_samples': len(self.predictions),
            'total_errors': len(self.errors),
            'error_rate': len(self.errors) / len(self.predictions) if self.predictions else 0.0,
            'error_categories': dict(self.error_categories),
            'error_patterns': error_patterns,
            'confidence_analysis': confidence_analysis,
            'text_analysis': text_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Error analysis completed: {len(self.errors)} errors found")
        return self.error_analysis_results
    
    def _identify_errors(self):
        """Identify incorrect predictions."""
        
        self.errors = []
        
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if pred != label:
                error_info = {
                    'index': i,
                    'claim': self.claims[i] if i < len(self.claims) else '',
                    'predicted_label': pred,
                    'true_label': label,
                    'predicted_class': self.class_names[pred] if pred < len(self.class_names) else f'class_{pred}',
                    'true_class': self.class_names[label] if label < len(self.class_names) else f'class_{label}',
                    'confidence': self.confidences[i] if i < len(self.confidences) else 0.0,
                    'retrieved_evidence': self.retrieved_evidence[i] if i < len(self.retrieved_evidence) else [],
                    'ground_truth_evidence': self.ground_truth_evidence[i] if i < len(self.ground_truth_evidence) else []
                }
                
                self.errors.append(error_info)
    
    def _categorize_errors(self, include_evidence_analysis: bool = True):
        """Categorize errors into different types."""
        
        self.error_categories = defaultdict(list)
        
        for error in self.errors:
            categories = []
            
            # Basic categorization
            if not error['retrieved_evidence']:
                categories.append(ErrorCategory.NO_EVIDENCE_FOUND)
            elif len(error['retrieved_evidence']) < 2:
                categories.append(ErrorCategory.INSUFFICIENT_EVIDENCE)
            
            # Evidence quality analysis
            if include_evidence_analysis and error['ground_truth_evidence']:
                evidence_quality = self._assess_evidence_quality(
                    error['retrieved_evidence'],
                    error['ground_truth_evidence']
                )
                
                if evidence_quality['relevance_score'] < 0.3:
                    categories.append(ErrorCategory.WRONG_EVIDENCE_RETRIEVED)
                elif evidence_quality['relevance_score'] >= 0.7:
                    categories.append(ErrorCategory.CORRECT_EVIDENCE_WRONG_LABEL)
                else:
                    categories.append(ErrorCategory.BOTH_EVIDENCE_AND_LABEL_WRONG)
            else:
                # Fallback categorization without ground truth
                if not error['retrieved_evidence']:
                    categories.append(ErrorCategory.NO_EVIDENCE_FOUND)
                else:
                    categories.append(ErrorCategory.CORRECT_EVIDENCE_WRONG_LABEL)
            
            # Claim complexity analysis
            if self._is_ambiguous_claim(error['claim']):
                categories.append(ErrorCategory.AMBIGUOUS_CLAIM)
            
            # Assign primary category
            primary_category = categories[0] if categories else ErrorCategory.PROCESSING_ERROR
            error['category'] = primary_category
            error['all_categories'] = categories
            
            # Add to category groups
            for category in categories:
                self.error_categories[category].append(error)
    
    def _assess_evidence_quality(
        self,
        retrieved: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Assess quality of retrieved evidence against ground truth."""
        
        if not ground_truth or not retrieved:
            return {'relevance_score': 0.0, 'coverage_score': 0.0}
        
        # Simple overlap-based assessment
        retrieved_words = set(' '.join(retrieved).lower().split())
        ground_truth_words = set(' '.join(ground_truth).lower().split())
        
        if not retrieved_words or not ground_truth_words:
            return {'relevance_score': 0.0, 'coverage_score': 0.0}
        
        # Compute overlap-based scores
        intersection = len(retrieved_words & ground_truth_words)
        union = len(retrieved_words | ground_truth_words)
        
        relevance_score = intersection / len(retrieved_words) if retrieved_words else 0.0
        coverage_score = intersection / len(ground_truth_words) if ground_truth_words else 0.0
        
        return {
            'relevance_score': relevance_score,
            'coverage_score': coverage_score,
            'jaccard_similarity': intersection / union if union > 0 else 0.0
        }
    
    def _is_ambiguous_claim(self, claim: str) -> bool:
        """Determine if a claim is ambiguous based on heuristics."""
        
        # Simple heuristics for ambiguity detection
        ambiguous_indicators = [
            'might', 'could', 'possibly', 'perhaps', 'maybe',
            'some people say', 'it is said', 'allegedly',
            'unclear', 'ambiguous', 'uncertain'
        ]
        
        claim_lower = claim.lower()
        return any(indicator in claim_lower for indicator in ambiguous_indicators)
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in error occurrences."""
        
        patterns = {}
        
        # Error distribution by true class
        true_class_errors = defaultdict(int)
        for error in self.errors:
            true_class_errors[error['true_class']] += 1
        
        patterns['errors_by_true_class'] = dict(true_class_errors)
        
        # Error distribution by predicted class
        pred_class_errors = defaultdict(int)
        for error in self.errors:
            pred_class_errors[error['predicted_class']] += 1
        
        patterns['errors_by_predicted_class'] = dict(pred_class_errors)
        
        # Confusion patterns
        confusion_pairs = defaultdict(int)
        for error in self.errors:
            pair = f"{error['true_class']} -> {error['predicted_class']}"
            confusion_pairs[pair] += 1
        
        patterns['confusion_pairs'] = dict(confusion_pairs)
        
        # Category distribution
        category_counts = {category: len(errors) for category, errors in self.error_categories.items()}
        patterns['error_category_distribution'] = category_counts
        
        # Evidence length patterns
        if self.retrieved_evidence:
            evidence_lengths = [len(evidence) for error in self.errors for evidence in [error['retrieved_evidence']]]
            if evidence_lengths:
                patterns['evidence_length_stats'] = {
                    'mean': np.mean(evidence_lengths),
                    'median': np.median(evidence_lengths),
                    'std': np.std(evidence_lengths),
                    'min': np.min(evidence_lengths),
                    'max': np.max(evidence_lengths)
                }
        
        return patterns
    
    def _analyze_confidence_patterns(self) -> Dict[str, Any]:
        """Analyze confidence patterns in errors."""
        
        if not self.confidences:
            return {}
        
        error_confidences = [error['confidence'] for error in self.errors]
        correct_confidences = [
            self.confidences[i] for i, (pred, label) in enumerate(zip(self.predictions, self.labels))
            if pred == label and i < len(self.confidences)
        ]
        
        analysis = {}
        
        if error_confidences:
            analysis['error_confidence_stats'] = {
                'mean': np.mean(error_confidences),
                'median': np.median(error_confidences),
                'std': np.std(error_confidences),
                'min': np.min(error_confidences),
                'max': np.max(error_confidences)
            }
        
        if correct_confidences:
            analysis['correct_confidence_stats'] = {
                'mean': np.mean(correct_confidences),
                'median': np.median(correct_confidences),
                'std': np.std(correct_confidences),
                'min': np.min(correct_confidences),
                'max': np.max(correct_confidences)
            }
        
        # High-confidence errors (overconfident mistakes)
        if error_confidences:
            high_confidence_threshold = 0.8
            high_conf_errors = [conf for conf in error_confidences if conf >= high_confidence_threshold]
            analysis['high_confidence_errors'] = {
                'count': len(high_conf_errors),
                'percentage': len(high_conf_errors) / len(error_confidences) * 100
            }
        
        return analysis
    
    def _analyze_error_text_patterns(self) -> Dict[str, Any]:
        """Analyze text patterns in error cases using NLP techniques."""
        
        if not SKLEARN_AVAILABLE or not self.errors:
            return {}
        
        error_claims = [error['claim'] for error in self.errors if error['claim']]
        
        if not error_claims:
            return {}
        
        analysis = {}
        
        try:
            # TF-IDF analysis of error claims
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(error_claims)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms in error claims
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_terms_indices = tfidf_scores.argsort()[-20:][::-1]
            
            analysis['top_error_terms'] = [
                {
                    'term': feature_names[i],
                    'score': float(tfidf_scores[i])
                }
                for i in top_terms_indices
            ]
            
            # Cluster error claims to find patterns
            if len(error_claims) >= 5:  # Need minimum samples for clustering
                n_clusters = min(5, len(error_claims) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Analyze clusters
                cluster_analysis = {}
                for cluster_id in range(n_clusters):
                    cluster_claims = [
                        error_claims[i] for i, label in enumerate(cluster_labels)
                        if label == cluster_id
                    ]
                    
                    cluster_analysis[f'cluster_{cluster_id}'] = {
                        'size': len(cluster_claims),
                        'sample_claims': cluster_claims[:3]  # Show first 3 as examples
                    }
                
                analysis['error_clusters'] = cluster_analysis
        
        except Exception as e:
            self.logger.warning(f"Text pattern analysis failed: {e}")
        
        return analysis
    
    def generate_error_report(
        self,
        output_dir: str = "fact_verification/evaluation/results",
        save_visualizations: bool = True,
        include_detailed_errors: bool = True
    ) -> str:
        """
        Generate comprehensive error analysis report.
        
        Args:
            output_dir: Directory to save reports and visualizations
            save_visualizations: Whether to create and save visualizations
            include_detailed_errors: Whether to include detailed error listings
            
        Returns:
            Path to the generated report file
        """
        if not self.error_analysis_results:
            self.analyze_errors()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate human-readable report
        report_content = self._generate_text_report(include_detailed_errors)
        
        # Save text report
        report_file = output_path / f"error_analysis_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save JSON report
        json_file = output_path / f"error_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self._make_json_serializable(self.error_analysis_results), f, indent=2)
        
        # Generate visualizations
        if save_visualizations and VISUALIZATION_AVAILABLE:
            self._create_error_visualizations(output_path, timestamp)
        
        self.logger.info(f"Error analysis report saved to {report_file}")
        return str(report_file)
    
    def _generate_text_report(self, include_detailed_errors: bool = True) -> str:
        """Generate human-readable error analysis report."""
        
        results = self.error_analysis_results
        
        report_lines = [
            f"\n{'='*80}",
            f"{'ERROR ANALYSIS REPORT':^80}",
            f"{'='*80}\n"
        ]
        
        # Summary section
        report_lines.extend([
            "SUMMARY",
            "-" * 40,
            f"{'Total Samples':<25}: {results['total_samples']:,}",
            f"{'Total Errors':<25}: {results['total_errors']:,}",
            f"{'Error Rate':<25}: {results['error_rate']:.1%}",
            f"{'Analysis Time':<25}: {results['analysis_timestamp']}",
            ""
        ])
        
        # Error category distribution
        if 'error_category_distribution' in results.get('error_patterns', {}):
            report_lines.extend([
                "ERROR CATEGORY DISTRIBUTION",
                "-" * 40
            ])
            
            category_dist = results['error_patterns']['error_category_distribution']
            for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = count / results['total_errors'] * 100 if results['total_errors'] > 0 else 0
                display_category = category.replace('_', ' ').title()
                report_lines.append(f"{display_category:<30}: {count:>5} ({percentage:>5.1f}%)")
            
            report_lines.append("")
        
        # Confusion patterns
        if 'confusion_pairs' in results.get('error_patterns', {}):
            report_lines.extend([
                "TOP CONFUSION PATTERNS",
                "-" * 40
            ])
            
            confusion_pairs = results['error_patterns']['confusion_pairs']
            sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            
            for pair, count in sorted_pairs[:10]:  # Top 10 confusion patterns
                percentage = count / results['total_errors'] * 100 if results['total_errors'] > 0 else 0
                report_lines.append(f"{pair:<35}: {count:>5} ({percentage:>5.1f}%)")
            
            report_lines.append("")
        
        # Confidence analysis
        if 'confidence_analysis' in results:
            conf_analysis = results['confidence_analysis']
            
            report_lines.extend([
                "CONFIDENCE ANALYSIS",
                "-" * 40
            ])
            
            if 'error_confidence_stats' in conf_analysis:
                error_stats = conf_analysis['error_confidence_stats']
                report_lines.extend([
                    "Error Confidence Statistics:",
                    f"  {'Mean':<15}: {error_stats['mean']:.3f}",
                    f"  {'Median':<15}: {error_stats['median']:.3f}",
                    f"  {'Std Dev':<15}: {error_stats['std']:.3f}",
                ])
            
            if 'correct_confidence_stats' in conf_analysis:
                correct_stats = conf_analysis['correct_confidence_stats']
                report_lines.extend([
                    "Correct Prediction Confidence Statistics:",
                    f"  {'Mean':<15}: {correct_stats['mean']:.3f}",
                    f"  {'Median':<15}: {correct_stats['median']:.3f}",
                    f"  {'Std Dev':<15}: {correct_stats['std']:.3f}",
                ])
            
            if 'high_confidence_errors' in conf_analysis:
                high_conf = conf_analysis['high_confidence_errors']
                report_lines.extend([
                    f"High-Confidence Errors (≥0.8): {high_conf['count']} ({high_conf['percentage']:.1f}%)"
                ])
            
            report_lines.append("")
        
        # Text analysis patterns
        if 'text_analysis' in results and 'top_error_terms' in results['text_analysis']:
            report_lines.extend([
                "TOP TERMS IN ERROR CASES",
                "-" * 40
            ])
            
            top_terms = results['text_analysis']['top_error_terms'][:15]
            for term_info in top_terms:
                report_lines.append(f"{'  ' + term_info['term']:<25}: {term_info['score']:.3f}")
            
            report_lines.append("")
        
        # Detailed error examples
        if include_detailed_errors and self.errors:
            report_lines.extend([
                "SAMPLE ERROR CASES",
                "-" * 40
            ])
            
            # Show top errors by confidence (overconfident mistakes)
            sorted_errors = sorted(self.errors, key=lambda x: x['confidence'], reverse=True)
            
            for i, error in enumerate(sorted_errors[:5]):
                report_lines.extend([
                    f"Error {i+1}:",
                    f"  Claim: {error['claim'][:80]}..." if len(error['claim']) > 80 else f"  Claim: {error['claim']}",
                    f"  True: {error['true_class']} | Predicted: {error['predicted_class']}",
                    f"  Confidence: {error['confidence']:.3f}",
                    f"  Category: {error.get('category', 'Unknown').replace('_', ' ').title()}",
                    f"  Evidence Count: {len(error['retrieved_evidence'])}",
                    ""
                ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _create_error_visualizations(self, output_path: Path, timestamp: str):
        """Create and save error analysis visualizations."""
        
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # Confusion matrix
            confusion_matrix_path = output_path / f"confusion_matrix_{timestamp}.png"
            create_confusion_matrix(
                self.labels, self.predictions, self.class_names,
                title="Fact Verification Confusion Matrix",
                save_path=str(confusion_matrix_path)
            )
            
            # Error distribution histogram
            if self.error_analysis_results.get('error_patterns', {}).get('error_category_distribution'):
                error_dist = self.error_analysis_results['error_patterns']['error_category_distribution']
                
                categories = list(error_dist.keys())
                counts = list(error_dist.values())
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(range(len(categories)), counts)
                plt.xlabel('Error Category')
                plt.ylabel('Number of Errors')
                plt.title('Error Category Distribution')
                plt.xticks(range(len(categories)), [cat.replace('_', ' ').title() for cat in categories], rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(count), ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_path / f"error_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Confidence distribution comparison
            if self.confidences:
                error_confidences = [error['confidence'] for error in self.errors]
                correct_confidences = [
                    self.confidences[i] for i, (pred, label) in enumerate(zip(self.predictions, self.labels))
                    if pred == label and i < len(self.confidences)
                ]
                
                plt.figure(figsize=(10, 6))
                plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
                plt.hist(error_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path / f"confidence_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Error visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
    
    def export_top_errors(
        self,
        n: int = 50,
        sort_by: str = 'confidence',
        output_file: Optional[str] = None,
        output_dir: str = "fact_verification/evaluation/results"
    ) -> str:
        """
        Export top N errors for manual inspection.
        
        Args:
            n: Number of top errors to export
            sort_by: Sorting criterion ('confidence', 'index', 'random')
            output_file: Specific output filename
            output_dir: Output directory
            
        Returns:
            Path to exported CSV file
        """
        if not self.errors:
            self.analyze_errors()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sort errors
        if sort_by == 'confidence':
            sorted_errors = sorted(self.errors, key=lambda x: x['confidence'], reverse=True)
        elif sort_by == 'random':
            import random
            sorted_errors = random.sample(self.errors, min(n, len(self.errors)))
        else:
            sorted_errors = self.errors
        
        top_errors = sorted_errors[:n]
        
        # Prepare data for CSV export
        export_data = []
        for error in top_errors:
            row = {
                'index': error['index'],
                'claim': error['claim'],
                'true_class': error['true_class'],
                'predicted_class': error['predicted_class'],
                'confidence': error['confidence'],
                'category': error.get('category', 'Unknown'),
                'evidence_count': len(error['retrieved_evidence']),
                'retrieved_evidence': ' | '.join(error['retrieved_evidence'][:3]),  # First 3 pieces
                'ground_truth_evidence': ' | '.join(error['ground_truth_evidence'][:3])
            }
            export_data.append(row)
        
        # Save to CSV
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"top_{n}_errors_{timestamp}.csv"
        
        csv_path = output_path / output_file
        df = pd.DataFrame(export_data)
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Exported {len(top_errors)} top errors to {csv_path}")
        return str(csv_path)
    
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
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get concise error analysis summary."""
        
        if not self.error_analysis_results:
            self.analyze_errors()
        
        results = self.error_analysis_results
        
        summary = {
            'total_errors': results['total_errors'],
            'error_rate': results['error_rate'],
            'top_error_category': None,
            'most_confused_pair': None,
            'avg_error_confidence': None
        }
        
        # Top error category
        if 'error_patterns' in results and 'error_category_distribution' in results['error_patterns']:
            category_dist = results['error_patterns']['error_category_distribution']
            if category_dist:
                top_category = max(category_dist, key=category_dist.get)
                summary['top_error_category'] = {
                    'category': top_category.replace('_', ' ').title(),
                    'count': category_dist[top_category]
                }
        
        # Most confused class pair
        if 'error_patterns' in results and 'confusion_pairs' in results['error_patterns']:
            confusion_pairs = results['error_patterns']['confusion_pairs']
            if confusion_pairs:
                top_pair = max(confusion_pairs, key=confusion_pairs.get)
                summary['most_confused_pair'] = {
                    'pair': top_pair,
                    'count': confusion_pairs[top_pair]
                }
        
        # Average error confidence
        if 'confidence_analysis' in results and 'error_confidence_stats' in results['confidence_analysis']:
            summary['avg_error_confidence'] = results['confidence_analysis']['error_confidence_stats']['mean']
        
        return summary


def main():
    """Example usage of ErrorAnalyzer."""
    
    print("=== ErrorAnalyzer Example ===")
    
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate predictions with some errors
    labels = np.random.randint(0, 3, n_samples)
    predictions = labels.copy()
    
    # Introduce errors (about 20%)
    error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    for idx in error_indices:
        # Randomly change prediction to different class
        predictions[idx] = np.random.choice([i for i in range(3) if i != labels[idx]])
    
    # Generate example claims
    example_claims = [
        f"This is claim number {i} about some factual statement"
        for i in range(n_samples)
    ]
    
    # Generate confidence scores (errors tend to have lower confidence)
    confidences = np.random.beta(2, 2, n_samples)
    for idx in error_indices:
        confidences[idx] *= 0.7  # Lower confidence for errors
    
    # Generate evidence lists
    retrieved_evidence = [
        [f"Evidence piece {j} for claim {i}" for j in range(np.random.randint(1, 4))]
        for i in range(n_samples)
    ]
    
    ground_truth_evidence = [
        [f"Ground truth evidence {j} for claim {i}" for j in range(np.random.randint(1, 3))]
        for i in range(n_samples)
    ]
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(
        predictions=predictions.tolist(),
        labels=labels.tolist(),
        claims=example_claims,
        retrieved_evidence=retrieved_evidence,
        ground_truth_evidence=ground_truth_evidence,
        confidences=confidences.tolist()
    )
    
    print(f"Initialized with {n_samples} samples")
    print(f"True error rate: {len(error_indices)}/{n_samples} = {len(error_indices)/n_samples:.1%}")
    
    # Run error analysis
    print("\nRunning error analysis...")
    error_results = analyzer.analyze_errors()
    
    print(f"Detected {error_results['total_errors']} errors ({error_results['error_rate']:.1%})")
    
    # Get error summary
    summary = analyzer.get_error_summary()
    print("\nError Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Generate error report
    print("\nGenerating error report...")
    report_file = analyzer.generate_error_report(
        output_dir="fact_verification/evaluation/results/example",
        save_visualizations=True
    )
    
    print(f"Error report saved to: {report_file}")
    
    # Export top errors
    csv_file = analyzer.export_top_errors(
        n=10,
        sort_by='confidence',
        output_dir="fact_verification/evaluation/results/example"
    )
    
    print(f"Top errors exported to: {csv_file}")
    
    # Show sample from analysis
    if 'error_patterns' in error_results:
        patterns = error_results['error_patterns']
        if 'confusion_pairs' in patterns:
            print("\nTop confusion patterns:")
            for pair, count in list(patterns['confusion_pairs'].items())[:3]:
                print(f"  {pair}: {count}")


if __name__ == "__main__":
    main()
