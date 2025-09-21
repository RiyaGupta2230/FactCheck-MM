#!/usr/bin/env python3
"""
Evidence Quality Evaluation

Evaluates the quality of retrieved evidence for fact verification including
relevance assessment, redundancy analysis, diversity measurement, and
ranking quality metrics (NDCG@k, MAP).

Example Usage:
    >>> from fact_verification.evaluation import EvidenceEvaluator
    >>> 
    >>> evaluator = EvidenceEvaluator()
    >>> 
    >>> # Evaluate evidence quality
    >>> results = evaluator.evaluate_evidence_quality(
    ...     claims, retrieved_evidence, ground_truth_evidence
    ... )
    >>> 
    >>> # Generate detailed report
    >>> report = evaluator.generate_evidence_report(results)
    >>> print(report)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np
import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import pandas as pd
import re
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger
from .fact_check_metrics import FactCheckMetrics

# Optional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
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
class EvidenceQualityMetrics:
    """Container for evidence quality evaluation results."""
    
    relevance_score: float
    redundancy_score: float
    diversity_score: float
    coverage_score: float
    ranking_quality: Dict[str, float]
    source_distribution: Dict[str, int]
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'relevance_score': self.relevance_score,
            'redundancy_score': self.redundancy_score,
            'diversity_score': self.diversity_score,
            'coverage_score': self.coverage_score,
            'ranking_quality': self.ranking_quality,
            'source_distribution': self.source_distribution,
            'detailed_analysis': self.detailed_analysis
        }


class EvidenceEvaluator:
    """
    Comprehensive evidence quality evaluator for fact verification systems.
    
    Evaluates retrieved evidence across multiple dimensions including relevance,
    redundancy, diversity, coverage, and ranking quality with detailed analysis
    and reporting capabilities.
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize evidence evaluator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("EvidenceEvaluator")
        self.metrics_calc = FactCheckMetrics()
        
        # Initialize text similarity components
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        self.logger.info("Initialized EvidenceEvaluator")
    
    def evaluate_relevance(
        self,
        claims: List[str],
        retrieved_evidence: List[List[str]],
        ground_truth_evidence: List[List[str]],
        relevance_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Evaluate relevance of retrieved evidence to claims.
        
        Args:
            claims: List of claim texts
            retrieved_evidence: List of retrieved evidence for each claim
            ground_truth_evidence: List of ground truth evidence for each claim
            relevance_threshold: Threshold for considering evidence relevant
            
        Returns:
            Dictionary with relevance metrics
        """
        if len(claims) != len(retrieved_evidence) or len(claims) != len(ground_truth_evidence):
            raise ValueError("All input lists must have the same length")
        
        total_relevant = 0
        total_retrieved = 0
        exact_matches = 0
        semantic_matches = 0
        
        relevance_scores = []
        
        for claim, retrieved, ground_truth in zip(claims, retrieved_evidence, ground_truth_evidence):
            if not ground_truth:  # Skip if no ground truth
                continue
            
            claim_relevant = 0
            claim_total = len(retrieved)
            total_retrieved += claim_total
            
            # Exact matching
            for evidence in retrieved:
                if evidence in ground_truth:
                    claim_relevant += 1
                    exact_matches += 1
                
                # Semantic similarity matching
                if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
                    max_similarity = self._compute_max_similarity(evidence, ground_truth)
                    if max_similarity >= relevance_threshold:
                        semantic_matches += 1
                        if evidence not in ground_truth:  # Don't double count exact matches
                            claim_relevant += 1
            
            total_relevant += claim_relevant
            
            # Claim-level relevance score
            claim_relevance = claim_relevant / claim_total if claim_total > 0 else 0.0
            relevance_scores.append(claim_relevance)
        
        # Aggregate metrics
        overall_relevance = total_relevant / total_retrieved if total_retrieved > 0 else 0.0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        return {
            'overall_relevance': overall_relevance,
            'average_relevance': avg_relevance,
            'exact_match_rate': exact_matches / total_retrieved if total_retrieved > 0 else 0.0,
            'semantic_match_rate': semantic_matches / total_retrieved if total_retrieved > 0 else 0.0,
            'relevance_std': np.std(relevance_scores) if relevance_scores else 0.0
        }
    
    def evaluate_redundancy(
        self,
        retrieved_evidence: List[List[str]],
        similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """
        Evaluate redundancy in retrieved evidence.
        
        Args:
            retrieved_evidence: List of retrieved evidence for each claim
            similarity_threshold: Threshold for considering evidence redundant
            
        Returns:
            Dictionary with redundancy metrics
        """
        total_pairs = 0
        redundant_pairs = 0
        redundancy_scores = []
        
        for evidence_list in retrieved_evidence:
            if len(evidence_list) < 2:
                continue
            
            claim_redundant = 0
            claim_pairs = 0
            
            # Check all pairs of evidence within this claim
            for i in range(len(evidence_list)):
                for j in range(i + 1, len(evidence_list)):
                    claim_pairs += 1
                    total_pairs += 1
                    
                    # Compute similarity
                    similarity = self._compute_text_similarity(evidence_list[i], evidence_list[j])
                    
                    if similarity >= similarity_threshold:
                        claim_redundant += 1
                        redundant_pairs += 1
            
            # Claim-level redundancy score
            claim_redundancy = claim_redundant / claim_pairs if claim_pairs > 0 else 0.0
            redundancy_scores.append(claim_redundancy)
        
        # Aggregate metrics
        overall_redundancy = redundant_pairs / total_pairs if total_pairs > 0 else 0.0
        avg_redundancy = np.mean(redundancy_scores) if redundancy_scores else 0.0
        
        return {
            'overall_redundancy': overall_redundancy,
            'average_redundancy': avg_redundancy,
            'redundant_pairs': redundant_pairs,
            'total_pairs': total_pairs,
            'redundancy_std': np.std(redundancy_scores) if redundancy_scores else 0.0
        }
    
    def evaluate_diversity(
        self,
        retrieved_evidence: List[List[str]],
        source_extraction_func: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate diversity of retrieved evidence.
        
        Args:
            retrieved_evidence: List of retrieved evidence for each claim
            source_extraction_func: Function to extract source from evidence text
            
        Returns:
            Dictionary with diversity metrics
        """
        if source_extraction_func is None:
            source_extraction_func = self._extract_source_from_text
        
        diversity_scores = []
        source_diversity_scores = []
        content_diversity_scores = []
        
        for evidence_list in retrieved_evidence:
            if len(evidence_list) < 2:
                diversity_scores.append(1.0)  # Single evidence is maximally diverse
                source_diversity_scores.append(1.0)
                content_diversity_scores.append(1.0)
                continue
            
            # Source diversity (unique sources)
            sources = [source_extraction_func(evidence) for evidence in evidence_list]
            unique_sources = len(set(sources))
            source_diversity = unique_sources / len(evidence_list)
            source_diversity_scores.append(source_diversity)
            
            # Content diversity (average pairwise dissimilarity)
            if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
                similarities = []
                for i in range(len(evidence_list)):
                    for j in range(i + 1, len(evidence_list)):
                        similarity = self._compute_text_similarity(evidence_list[i], evidence_list[j])
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                content_diversity = 1.0 - avg_similarity  # Diversity is inverse of similarity
                content_diversity_scores.append(content_diversity)
            else:
                content_diversity_scores.append(0.5)  # Default if no similarity computation available
            
            # Overall diversity (combination of source and content diversity)
            overall_diversity = (source_diversity + content_diversity_scores[-1]) / 2
            diversity_scores.append(overall_diversity)
        
        return {
            'overall_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'source_diversity': np.mean(source_diversity_scores) if source_diversity_scores else 0.0,
            'content_diversity': np.mean(content_diversity_scores) if content_diversity_scores else 0.0,
            'diversity_std': np.std(diversity_scores) if diversity_scores else 0.0
        }
    
    def evaluate_coverage(
        self,
        claims: List[str],
        retrieved_evidence: List[List[str]],
        ground_truth_evidence: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate coverage of ground truth evidence by retrieved evidence.
        
        Args:
            claims: List of claim texts
            retrieved_evidence: List of retrieved evidence for each claim
            ground_truth_evidence: List of ground truth evidence for each claim
            
        Returns:
            Dictionary with coverage metrics
        """
        coverage_scores = []
        total_gt_covered = 0
        total_gt_evidence = 0
        
        for claim, retrieved, ground_truth in zip(claims, retrieved_evidence, ground_truth_evidence):
            if not ground_truth:
                continue
            
            total_gt_evidence += len(ground_truth)
            covered_count = 0
            
            # Check how many ground truth evidence pieces are covered
            for gt_evidence in ground_truth:
                # Check for exact match
                if gt_evidence in retrieved:
                    covered_count += 1
                # Check for semantic match
                elif SKLEARN_AVAILABLE and self.tfidf_vectorizer:
                    max_similarity = self._compute_max_similarity(gt_evidence, retrieved)
                    if max_similarity >= 0.7:  # Threshold for semantic coverage
                        covered_count += 1
            
            total_gt_covered += covered_count
            
            # Claim-level coverage
            claim_coverage = covered_count / len(ground_truth)
            coverage_scores.append(claim_coverage)
        
        # Aggregate metrics
        overall_coverage = total_gt_covered / total_gt_evidence if total_gt_evidence > 0 else 0.0
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        
        return {
            'overall_coverage': overall_coverage,
            'average_coverage': avg_coverage,
            'covered_evidence': total_gt_covered,
            'total_ground_truth': total_gt_evidence,
            'coverage_std': np.std(coverage_scores) if coverage_scores else 0.0
        }
    
    def evaluate_ranking_quality(
        self,
        claims: List[str],
        retrieved_evidence: List[List[str]],
        ground_truth_evidence: List[List[str]],
        relevance_scores: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of evidence ranking using NDCG@k and MAP.
        
        Args:
            claims: List of claim texts
            retrieved_evidence: List of retrieved evidence for each claim
            ground_truth_evidence: List of ground truth evidence for each claim
            relevance_scores: Optional relevance scores for retrieved evidence
            
        Returns:
            Dictionary with ranking quality metrics
        """
        # Use FactCheckMetrics for ranking evaluation
        ranking_metrics = self.metrics_calc.compute_retrieval_metrics(
            retrieved_evidence, ground_truth_evidence, relevance_scores
        )
        
        # Extract ranking-specific metrics
        ranking_quality = {
            'ndcg@3': ranking_metrics.get('ndcg@3', 0.0),
            'ndcg@5': ranking_metrics.get('ndcg@5', 0.0),
            'ndcg@10': ranking_metrics.get('ndcg@10', 0.0),
            'map': ranking_metrics.get('map', 0.0),
            'mrr': ranking_metrics.get('mrr', 0.0)
        }
        
        return ranking_quality
    
    def analyze_source_distribution(
        self,
        retrieved_evidence: List[List[str]],
        source_extraction_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Analyze distribution of evidence sources.
        
        Args:
            retrieved_evidence: List of retrieved evidence for each claim
            source_extraction_func: Function to extract source from evidence text
            
        Returns:
            Dictionary with source distribution analysis
        """
        if source_extraction_func is None:
            source_extraction_func = self._extract_source_from_text
        
        all_sources = []
        source_counts = Counter()
        
        for evidence_list in retrieved_evidence:
            for evidence in evidence_list:
                source = source_extraction_func(evidence)
                all_sources.append(source)
                source_counts[source] += 1
        
        # Calculate diversity metrics
        total_evidence = len(all_sources)
        unique_sources = len(source_counts)
        
        # Source entropy (measure of distribution uniformity)
        if total_evidence > 0:
            probabilities = [count / total_evidence for count in source_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(unique_sources) if unique_sources > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy = normalized_entropy = 0.0
        
        return {
            'total_evidence_pieces': total_evidence,
            'unique_sources': unique_sources,
            'source_counts': dict(source_counts.most_common()),
            'source_entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'top_sources': dict(source_counts.most_common(10))
        }
    
    def evaluate_evidence_quality(
        self,
        claims: List[str],
        retrieved_evidence: List[List[str]],
        ground_truth_evidence: List[List[str]],
        source_extraction_func: Optional[callable] = None,
        relevance_scores: Optional[List[List[float]]] = None
    ) -> EvidenceQualityMetrics:
        """
        Comprehensive evidence quality evaluation.
        
        Args:
            claims: List of claim texts
            retrieved_evidence: List of retrieved evidence for each claim
            ground_truth_evidence: List of ground truth evidence for each claim
            source_extraction_func: Function to extract source from evidence text
            relevance_scores: Optional relevance scores for retrieved evidence
            
        Returns:
            EvidenceQualityMetrics object with all evaluation results
        """
        self.logger.info("Starting comprehensive evidence quality evaluation")
        
        # Relevance evaluation
        relevance_metrics = self.evaluate_relevance(claims, retrieved_evidence, ground_truth_evidence)
        
        # Redundancy evaluation
        redundancy_metrics = self.evaluate_redundancy(retrieved_evidence)
        
        # Diversity evaluation
        diversity_metrics = self.evaluate_diversity(retrieved_evidence, source_extraction_func)
        
        # Coverage evaluation
        coverage_metrics = self.evaluate_coverage(claims, retrieved_evidence, ground_truth_evidence)
        
        # Ranking quality evaluation
        ranking_quality = self.evaluate_ranking_quality(
            claims, retrieved_evidence, ground_truth_evidence, relevance_scores
        )
        
        # Source distribution analysis
        source_analysis = self.analyze_source_distribution(retrieved_evidence, source_extraction_func)
        
        # Create comprehensive metrics object
        quality_metrics = EvidenceQualityMetrics(
            relevance_score=relevance_metrics['average_relevance'],
            redundancy_score=redundancy_metrics['average_redundancy'],
            diversity_score=diversity_metrics['overall_diversity'],
            coverage_score=coverage_metrics['average_coverage'],
            ranking_quality=ranking_quality,
            source_distribution=source_analysis['top_sources'],
            detailed_analysis={
                'relevance_metrics': relevance_metrics,
                'redundancy_metrics': redundancy_metrics,
                'diversity_metrics': diversity_metrics,
                'coverage_metrics': coverage_metrics,
                'source_analysis': source_analysis
            }
        )
        
        self.logger.info("Evidence quality evaluation completed")
        return quality_metrics
    
    def generate_evidence_report(
        self,
        quality_metrics: EvidenceQualityMetrics,
        title: str = "Evidence Quality Evaluation Report"
    ) -> str:
        """
        Generate human-readable evidence quality report.
        
        Args:
            quality_metrics: EvidenceQualityMetrics object
            title: Report title
            
        Returns:
            Formatted report string
        """
        report_lines = [
            f"\n{'='*70}",
            f"{title:^70}",
            f"{'='*70}\n"
        ]
        
        # Summary metrics
        report_lines.extend([
            "SUMMARY METRICS",
            "-" * 40,
            f"{'Relevance Score':<20}: {quality_metrics.relevance_score:.4f}",
            f"{'Redundancy Score':<20}: {quality_metrics.redundancy_score:.4f}",
            f"{'Diversity Score':<20}: {quality_metrics.diversity_score:.4f}",
            f"{'Coverage Score':<20}: {quality_metrics.coverage_score:.4f}",
            ""
        ])
        
        # Ranking quality
        report_lines.extend([
            "RANKING QUALITY",
            "-" * 40
        ])
        
        for metric, value in quality_metrics.ranking_quality.items():
            report_lines.append(f"{metric.upper():<20}: {value:.4f}")
        
        report_lines.append("")
        
        # Detailed analysis
        detailed = quality_metrics.detailed_analysis
        
        # Relevance details
        if 'relevance_metrics' in detailed:
            rel_metrics = detailed['relevance_metrics']
            report_lines.extend([
                "RELEVANCE ANALYSIS",
                "-" * 40,
                f"{'Exact Match Rate':<25}: {rel_metrics.get('exact_match_rate', 0):.4f}",
                f"{'Semantic Match Rate':<25}: {rel_metrics.get('semantic_match_rate', 0):.4f}",
                f"{'Relevance Std Dev':<25}: {rel_metrics.get('relevance_std', 0):.4f}",
                ""
            ])
        
        # Diversity details
        if 'diversity_metrics' in detailed:
            div_metrics = detailed['diversity_metrics']
            report_lines.extend([
                "DIVERSITY ANALYSIS",
                "-" * 40,
                f"{'Source Diversity':<25}: {div_metrics.get('source_diversity', 0):.4f}",
                f"{'Content Diversity':<25}: {div_metrics.get('content_diversity', 0):.4f}",
                ""
            ])
        
        # Source distribution
        if quality_metrics.source_distribution:
            report_lines.extend([
                "TOP EVIDENCE SOURCES",
                "-" * 40
            ])
            
            for source, count in list(quality_metrics.source_distribution.items())[:10]:
                report_lines.append(f"{'  ' + source:<30}: {count:>5} occurrences")
            
            report_lines.append("")
        
        # Coverage details
        if 'coverage_metrics' in detailed:
            cov_metrics = detailed['coverage_metrics']
            report_lines.extend([
                "COVERAGE ANALYSIS",
                "-" * 40,
                f"{'Covered Evidence':<25}: {cov_metrics.get('covered_evidence', 0)}/{cov_metrics.get('total_ground_truth', 0)}",
                f"{'Overall Coverage':<25}: {cov_metrics.get('overall_coverage', 0):.4f}",
                ""
            ])
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def save_evidence_report(
        self,
        quality_metrics: EvidenceQualityMetrics,
        output_dir: str,
        filename_prefix: str = "evidence_evaluation"
    ):
        """
        Save evidence evaluation report in multiple formats.
        
        Args:
            quality_metrics: EvidenceQualityMetrics object
            output_dir: Output directory path
            filename_prefix: Prefix for output filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = output_path / f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(quality_metrics.to_dict(), f, indent=2)
        
        # Save human-readable report
        txt_file = output_path / f"{filename_prefix}_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write(self.generate_evidence_report(quality_metrics))
        
        # Save CSV summary
        csv_file = output_path / f"{filename_prefix}_{timestamp}.csv"
        summary_data = {
            'relevance_score': quality_metrics.relevance_score,
            'redundancy_score': quality_metrics.redundancy_score,
            'diversity_score': quality_metrics.diversity_score,
            'coverage_score': quality_metrics.coverage_score,
            **quality_metrics.ranking_quality
        }
        
        df = pd.DataFrame([summary_data])
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Evidence evaluation reports saved to {output_path}")
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings."""
        
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            elif not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union
        
        try:
            # Fit TF-IDF on both texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _compute_max_similarity(self, text: str, text_list: List[str]) -> float:
        """Compute maximum similarity between text and list of texts."""
        
        if not text_list:
            return 0.0
        
        similarities = [self._compute_text_similarity(text, other_text) for other_text in text_list]
        return max(similarities)
    
    def _extract_source_from_text(self, evidence_text: str) -> str:
        """Extract source identifier from evidence text."""
        
        # Simple heuristic: extract domain from URLs or use first few words
        url_pattern = r'https?://([a-zA-Z0-9.-]+)'
        url_match = re.search(url_pattern, evidence_text)
        
        if url_match:
            return url_match.group(1)
        
        # Fallback: use first 3 words as source identifier
        words = evidence_text.split()[:3]
        return "_".join(words) if words else "unknown_source"


def main():
    """Example usage of EvidenceEvaluator."""
    
    # Initialize evaluator
    evaluator = EvidenceEvaluator()
    
    print("=== EvidenceEvaluator Example ===")
    
    # Example data
    claims = [
        "COVID-19 vaccines are effective",
        "Climate change is caused by human activities",
        "The Earth is flat"
    ]
    
    retrieved_evidence = [
        [
            "Clinical trials show 90-95% efficacy for COVID-19 vaccines",
            "Real-world data confirms vaccine effectiveness",
            "Vaccines reduce hospitalization rates significantly"
        ],
        [
            "Scientists agree that greenhouse gases cause climate change",
            "Human activities increase CO2 levels",
            "Global warming is linked to fossil fuel use"
        ],
        [
            "Satellite images show Earth's curvature",
            "Ships disappear over horizon due to curvature",
            "Gravity explains planetary formation"
        ]
    ]
    
    ground_truth_evidence = [
        [
            "Clinical trials show 90-95% efficacy for COVID-19 vaccines",
            "Vaccination programs reduce COVID-19 deaths"
        ],
        [
            "Scientists agree that greenhouse gases cause climate change",
            "Human carbon emissions drive global warming"
        ],
        [
            "Earth is a sphere confirmed by space exploration",
            "Gravity theory explains planetary shapes"
        ]
    ]
    
    print("Evaluating evidence quality...")
    
    # Comprehensive evaluation
    quality_metrics = evaluator.evaluate_evidence_quality(
        claims, retrieved_evidence, ground_truth_evidence
    )
    
    # Generate and display report
    report = evaluator.generate_evidence_report(quality_metrics)
    print(report)
    
    # Save reports
    output_dir = Path("fact_verification/evaluation/results")
    evaluator.save_evidence_report(quality_metrics, output_dir, "example_evidence")
    
    print(f"\nEvidence evaluation reports saved to {output_dir}")


if __name__ == "__main__":
    main()
