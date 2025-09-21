#!/usr/bin/env python3
"""
Diversity Metrics for Paraphrase Evaluation

Comprehensive lexical and semantic diversity analysis including distinct-n metrics,
self-BLEU redundancy measurement, entropy-based diversity, and embedding-based
semantic diversity using sentence transformers.

Example Usage:
    >>> from paraphrasing.utils import DiversityMetrics
    >>> 
    >>> # Initialize diversity analyzer
    >>> diversity = DiversityMetrics()
    >>> 
    >>> # Analyze generated paraphrases
    >>> predictions = [
    ...     "The weather is nice today",
    ...     "Today's weather is pleasant", 
    ...     "It's a beautiful day",
    ...     "The day has lovely weather"
    ... ]
    >>> diversity.update(predictions)
    >>> 
    >>> # Compute diversity metrics
    >>> metrics = diversity.compute()
    >>> print(f"Distinct-2: {metrics['distinct_2']:.3f}")
    >>> print(f"Self-BLEU: {metrics['self_bleu']:.3f}")
    >>> print(f"Entropy: {metrics['entropy']:.3f}")
    >>> print(f"Embedding diversity: {metrics['embedding_diversity']:.3f}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
from collections import defaultdict, Counter
import math
import logging
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Import evaluation dependencies
try:
    from shared.utils.metrics import calculate_bleu_score
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Import sentence transformers for semantic diversity
try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional visualization support
try:
    from shared.utils.visualization import create_diversity_plot
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class DiversityConfig:
    """Configuration for diversity metrics computation."""
    
    # N-gram diversity
    ngram_range: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Self-BLEU configuration
    compute_self_bleu: bool = True
    self_bleu_n: int = 4
    
    # Entropy configuration
    compute_entropy: bool = True
    entropy_level: str = "word"  # "word", "char", "ngram"
    
    # Embedding diversity
    compute_embedding_diversity: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_metric: str = "cosine"  # "cosine", "euclidean"
    
    # Text preprocessing
    lowercase: bool = True
    remove_punctuation: bool = False
    min_text_length: int = 1
    
    # Advanced metrics
    compute_ttr: bool = True  # Type-token ratio
    compute_msttr: bool = True  # Mean segmental TTR
    msttr_segment_length: int = 50
    
    # Performance settings
    batch_size: int = 32
    max_samples_for_pairwise: int = 1000


class DiversityMetrics:
    """
    Comprehensive diversity metrics analyzer for generated text.
    
    Computes lexical diversity (distinct-n, TTR, entropy) and semantic diversity 
    (embedding-based variance, self-BLEU) with support for batch processing
    and accumulation across multiple updates.
    """
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        """
        Initialize diversity metrics analyzer.
        
        Args:
            config: Diversity computation configuration
        """
        self.config = config or DiversityConfig()
        self.logger = get_logger("DiversityMetrics")
        
        # Initialize sentence transformer for semantic diversity
        self.sentence_model = None
        if self.config.compute_embedding_diversity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info(f"Loaded sentence transformer: {self.config.embedding_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
        
        # Storage for accumulated predictions
        self.predictions = []
        
        self.logger.info("Initialized DiversityMetrics analyzer")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text based on configuration."""
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def _extract_ngrams(self, text: str, n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from text."""
        
        tokens = text.split()
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def compute_distinct_n(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute distinct-n metrics for lexical diversity.
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Dictionary with distinct-n scores
        """
        if not predictions:
            return {f'distinct_{n}': 0.0 for n in self.config.ngram_range}
        
        distinct_scores = {}
        
        for n in self.config.ngram_range:
            all_ngrams = []
            total_ngrams = 0
            
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                ngrams = self._extract_ngrams(processed_pred, n)
                all_ngrams.extend(ngrams)
                total_ngrams += len(ngrams)
            
            if total_ngrams > 0:
                unique_ngrams = len(set(all_ngrams))
                distinct_scores[f'distinct_{n}'] = unique_ngrams / total_ngrams
            else:
                distinct_scores[f'distinct_{n}'] = 0.0
        
        return distinct_scores
    
    def compute_self_bleu(self, predictions: List[str]) -> float:
        """
        Compute self-BLEU score to measure redundancy.
        
        Higher self-BLEU indicates more redundancy (lower diversity).
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Average self-BLEU score
        """
        if not self.config.compute_self_bleu or not METRICS_AVAILABLE:
            return 0.0
        
        if len(predictions) < 2:
            return 0.0
        
        # Limit samples for computational efficiency
        if len(predictions) > self.config.max_samples_for_pairwise:
            import random
            predictions = random.sample(predictions, self.config.max_samples_for_pairwise)
        
        self_bleu_scores = []
        
        for i, pred in enumerate(predictions):
            # Use all other predictions as references
            references = [predictions[j] for j in range(len(predictions)) if j != i]
            
            if references:
                try:
                    # Compute BLEU score against other predictions
                    bleu_score = calculate_bleu_score(references, pred)
                    self_bleu_scores.append(bleu_score)
                except:
                    continue
        
        return np.mean(self_bleu_scores) if self_bleu_scores else 0.0
    
    def compute_entropy(self, predictions: List[str]) -> float:
        """
        Compute entropy-based diversity measure.
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Entropy score (higher = more diverse)
        """
        if not self.config.compute_entropy or not predictions:
            return 0.0
        
        if self.config.entropy_level == "word":
            # Word-level entropy
            all_tokens = []
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                tokens = processed_pred.split()
                all_tokens.extend(tokens)
            
        elif self.config.entropy_level == "char":
            # Character-level entropy
            all_tokens = []
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                chars = list(processed_pred.replace(' ', ''))
                all_tokens.extend(chars)
            
        elif self.config.entropy_level == "ngram":
            # N-gram entropy (using bigrams)
            all_tokens = []
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                ngrams = self._extract_ngrams(processed_pred, 2)
                all_tokens.extend([' '.join(ngram) for ngram in ngrams])
        
        else:
            return 0.0
        
        if not all_tokens:
            return 0.0
        
        # Compute token frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def compute_type_token_ratio(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute Type-Token Ratio (TTR) and Mean Segmental TTR (MSTTR).
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Dictionary with TTR metrics
        """
        if not predictions:
            return {'ttr': 0.0, 'msttr': 0.0}
        
        results = {}
        
        # Simple TTR
        if self.config.compute_ttr:
            all_tokens = []
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                tokens = processed_pred.split()
                all_tokens.extend(tokens)
            
            if all_tokens:
                unique_tokens = len(set(all_tokens))
                total_tokens = len(all_tokens)
                results['ttr'] = unique_tokens / total_tokens
            else:
                results['ttr'] = 0.0
        
        # Mean Segmental TTR (MSTTR)
        if self.config.compute_msttr:
            all_tokens = []
            for pred in predictions:
                processed_pred = self._preprocess_text(pred)
                tokens = processed_pred.split()
                all_tokens.extend(tokens)
            
            if len(all_tokens) >= self.config.msttr_segment_length:
                segment_ttrs = []
                segment_length = self.config.msttr_segment_length
                
                for i in range(0, len(all_tokens), segment_length):
                    segment = all_tokens[i:i + segment_length]
                    if len(segment) == segment_length:  # Only use complete segments
                        unique_in_segment = len(set(segment))
                        segment_ttr = unique_in_segment / len(segment)
                        segment_ttrs.append(segment_ttr)
                
                results['msttr'] = np.mean(segment_ttrs) if segment_ttrs else 0.0
            else:
                results['msttr'] = results.get('ttr', 0.0)
        
        return results
    
    def compute_embedding_diversity(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute semantic diversity using sentence embeddings.
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Dictionary with embedding-based diversity metrics
        """
        if (not self.config.compute_embedding_diversity or 
            not self.sentence_model or 
            len(predictions) < 2):
            return {'embedding_diversity': 0.0, 'embedding_variance': 0.0}
        
        try:
            # Encode all predictions
            embeddings = self.sentence_model.encode(
                predictions, 
                batch_size=self.config.batch_size,
                convert_to_tensor=True
            )
            
            # Compute pairwise similarities
            if self.config.similarity_metric == "cosine":
                # Normalize embeddings for cosine similarity
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            
            elif self.config.similarity_metric == "euclidean":
                # Compute euclidean distances and convert to similarities
                distances = torch.cdist(embeddings, embeddings, p=2)
                # Convert distances to similarities (higher distance = lower similarity)
                max_distance = torch.max(distances)
                similarity_matrix = 1.0 - (distances / (max_distance + 1e-8))
            
            else:
                return {'embedding_diversity': 0.0, 'embedding_variance': 0.0}
            
            # Extract upper triangular part (excluding diagonal)
            triu_indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), offset=1)
            pairwise_similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
            
            # Diversity metrics
            mean_similarity = torch.mean(pairwise_similarities).item()
            variance_similarity = torch.var(pairwise_similarities).item()
            
            # Diversity = 1 - mean_similarity (higher diversity = lower similarity)
            diversity = 1.0 - mean_similarity
            
            return {
                'embedding_diversity': max(0.0, diversity),
                'embedding_variance': variance_similarity,
                'mean_pairwise_similarity': mean_similarity
            }
            
        except Exception as e:
            self.logger.warning(f"Embedding diversity computation failed: {e}")
            return {'embedding_diversity': 0.0, 'embedding_variance': 0.0}
    
    def update(self, predictions: List[str]):
        """
        Update diversity analyzer with new predictions.
        
        Args:
            predictions: List of generated texts to add
        """
        # Filter out very short texts if configured
        filtered_predictions = [
            pred for pred in predictions 
            if len(pred.split()) >= self.config.min_text_length
        ]
        
        self.predictions.extend(filtered_predictions)
        
        self.logger.debug(f"Updated with {len(filtered_predictions)} predictions (total: {len(self.predictions)})")
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all diversity metrics on accumulated predictions.
        
        Returns:
            Dictionary of diversity metrics
        """
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # Distinct-n metrics
        distinct_metrics = self.compute_distinct_n(self.predictions)
        metrics.update(distinct_metrics)
        
        # Self-BLEU
        if self.config.compute_self_bleu:
            metrics['self_bleu'] = self.compute_self_bleu(self.predictions)
        
        # Entropy
        if self.config.compute_entropy:
            metrics['entropy'] = self.compute_entropy(self.predictions)
        
        # Type-token ratios
        ttr_metrics = self.compute_type_token_ratio(self.predictions)
        metrics.update(ttr_metrics)
        
        # Embedding diversity
        embedding_metrics = self.compute_embedding_diversity(self.predictions)
        metrics.update(embedding_metrics)
        
        # Additional derived metrics
        if 'distinct_1' in metrics and 'distinct_2' in metrics:
            # Average distinct score
            distinct_scores = [metrics[f'distinct_{n}'] for n in self.config.ngram_range if f'distinct_{n}' in metrics]
            metrics['avg_distinct'] = np.mean(distinct_scores)
        
        # Overall diversity score (weighted combination)
        diversity_components = []
        weights = []
        
        if 'avg_distinct' in metrics:
            diversity_components.append(metrics['avg_distinct'])
            weights.append(0.4)
        
        if 'entropy' in metrics and metrics['entropy'] > 0:
            # Normalize entropy (rough normalization)
            normalized_entropy = min(1.0, metrics['entropy'] / 10.0)
            diversity_components.append(normalized_entropy)
            weights.append(0.3)
        
        if 'embedding_diversity' in metrics:
            diversity_components.append(metrics['embedding_diversity'])
            weights.append(0.3)
        
        if diversity_components:
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            metrics['overall_diversity'] = np.sum(np.array(diversity_components) * weights)
        
        self.logger.info(f"Computed diversity metrics for {len(self.predictions)} predictions")
        return metrics
    
    def reset(self):
        """Reset accumulated predictions."""
        self.predictions.clear()
        self.logger.debug("Reset diversity metrics analyzer")
    
    def get_sample_count(self) -> int:
        """Get number of accumulated predictions."""
        return len(self.predictions)
    
    def create_diversity_visualization(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[str]:
        """
        Create visualization of diversity metrics.
        
        Args:
            save_path: Path to save plot
            show: Whether to display plot
            
        Returns:
            Path to saved plot or None if visualization unavailable
        """
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization not available")
            return None
        
        if not self.predictions:
            self.logger.warning("No predictions available for visualization")
            return None
        
        try:
            # Compute current metrics
            metrics = self.compute()
            
            # Create diversity plot
            plot_path = create_diversity_plot(
                metrics, 
                predictions=self.predictions[:100],  # Sample for visualization
                save_path=save_path,
                show=show
            )
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Diversity visualization failed: {e}")
            return None


def main():
    """Example usage of DiversityMetrics."""
    
    # Sample generated paraphrases
    predictions = [
        "The weather is beautiful today.",
        "Today's weather is lovely and nice.",
        "It's a gorgeous day outside.",
        "The day has wonderful weather conditions.",
        "Beautiful weather graces us today.",
        "Today we have pleasant weather.",
        "The weather conditions are excellent today.",
        "It's a beautiful day with nice weather.",
        "Today brings us lovely weather.",
        "The weather is absolutely wonderful today."
    ]
    
    # Initialize diversity analyzer
    config = DiversityConfig(
        ngram_range=[1, 2, 3],
        compute_self_bleu=True,
        compute_entropy=True,
        compute_embedding_diversity=True
    )
    
    diversity = DiversityMetrics(config)
    
    print("=== Diversity Metrics Example ===")
    print(f"Analyzing {len(predictions)} generated paraphrases")
    
    # Update with predictions
    diversity.update(predictions)
    
    # Compute metrics
    metrics = diversity.compute()
    
    print("\n=== Diversity Results ===")
    for metric_name, score in metrics.items():
        print(f"{metric_name:>25}: {score:.4f}")
    
    # Test incremental updates
    print("\n=== Incremental Updates ===")
    diversity.reset()
    
    # Add predictions in batches
    batch_size = 3
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i + batch_size]
        diversity.update(batch)
        
        current_metrics = diversity.compute()
        distinct_2 = current_metrics.get('distinct_2', 0)
        entropy = current_metrics.get('entropy', 0)
        
        print(f"After batch {i//batch_size + 1}: {diversity.get_sample_count()} samples, "
              f"Distinct-2: {distinct_2:.3f}, Entropy: {entropy:.3f}")
    
    # Test individual metric computations
    print("\n=== Individual Metrics ===")
    
    # Test distinct-n
    distinct_scores = diversity.compute_distinct_n(predictions)
    print("Distinct-n scores:", {k: f"{v:.3f}" for k, v in distinct_scores.items()})
    
    # Test self-BLEU
    self_bleu = diversity.compute_self_bleu(predictions)
    print(f"Self-BLEU (redundancy): {self_bleu:.3f}")
    
    # Test entropy
    entropy = diversity.compute_entropy(predictions)
    print(f"Word-level entropy: {entropy:.3f}")
    
    # Test TTR
    ttr_scores = diversity.compute_type_token_ratio(predictions)
    print("TTR scores:", {k: f"{v:.3f}" for k, v in ttr_scores.items()})
    
    # Test embedding diversity
    embedding_scores = diversity.compute_embedding_diversity(predictions)
    print("Embedding diversity:", {k: f"{v:.3f}" for k, v in embedding_scores.items()})


if __name__ == "__main__":
    main()
