#!/usr/bin/env python3
"""
Semantic Similarity Analysis for Paraphrase Evaluation

Provides sentence-transformer-based semantic similarity computation with support
for threshold-based classification, distribution analysis, and visualization hooks
for comprehensive paraphrase semantic analysis.

Example Usage:
    >>> from paraphrasing.evaluation import SemanticSimilarity
    >>> 
    >>> # Initialize similarity computer
    >>> similarity = SemanticSimilarity(model_name='all-MiniLM-L6-v2')
    >>> 
    >>> # Compute similarities
    >>> sources = ["The weather is nice", "I enjoy coding"]
    >>> candidates = ["It's a beautiful day", "Programming is fun"]
    >>> similarities = similarity.compute_similarity(sources, candidates)
    >>> print(f"Similarities: {similarities}")
    >>> 
    >>> # Analyze distribution
    >>> stats = similarity.analyze_distribution(similarities)
    >>> print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass, field
import logging
import warnings

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    import torch
    import torch.nn.functional as F
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install with: pip install sentence-transformers")

# Optional imports for visualization and dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class SimilarityConfig:
    """Configuration for semantic similarity computation."""
    
    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # "auto", "cpu", "cuda"
    normalize_embeddings: bool = True
    
    # Similarity computation
    similarity_metric: str = "cosine"  # "cosine", "dot", "euclidean"
    batch_size: int = 32
    
    # Classification thresholds
    paraphrase_threshold: float = 0.7
    high_similarity_threshold: float = 0.8
    low_similarity_threshold: float = 0.4
    
    # Analysis configuration
    compute_statistics: bool = True
    enable_caching: bool = True
    
    # Visualization configuration
    enable_visualization: bool = False
    dimensionality_reduction: str = "pca"  # "pca", "tsne", "umap"
    n_components: int = 2


def load_sentence_transformer(model_name: str, device: str = "auto") -> Optional[SentenceTransformer]:
    """
    Load sentence transformer model with proper device handling.
    
    Args:
        model_name: Name of the sentence transformer model
        device: Device to load model on ("auto", "cpu", "cuda")
        
    Returns:
        Loaded sentence transformer model or None if unavailable
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        return model
        
    except Exception as e:
        logging.getLogger("SemanticSimilarity").error(f"Failed to load model {model_name}: {e}")
        return None


def compute_similarity(
    a: List[str],
    b: List[str],
    model: SentenceTransformer,
    metric: str = "cosine",
    batch_size: int = 32,
    normalize: bool = True
) -> List[float]:
    """
    Compute semantic similarity between two lists of texts.
    
    Args:
        a: First list of texts
        b: Second list of texts (must be same length as a)
        model: Sentence transformer model
        metric: Similarity metric ("cosine", "dot", "euclidean")
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        
    Returns:
        List of similarity scores
    """
    if len(a) != len(b):
        raise ValueError(f"Lists must have same length: {len(a)} vs {len(b)}")
    
    if not model:
        return [0.0] * len(a)
    
    try:
        # Encode texts in batches
        embeddings_a = model.encode(
            a,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        
        embeddings_b = model.encode(
            b,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        
        # Compute similarities
        if metric == "cosine":
            similarities = F.cosine_similarity(embeddings_a, embeddings_b, dim=1)
        elif metric == "dot":
            similarities = torch.sum(embeddings_a * embeddings_b, dim=1)
        elif metric == "euclidean":
            # Convert to similarity (inverse of distance)
            distances = F.pairwise_distance(embeddings_a, embeddings_b)
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarities.cpu().tolist()
        
    except Exception as e:
        logging.getLogger("SemanticSimilarity").error(f"Similarity computation failed: {e}")
        return [0.0] * len(a)


class SemanticSimilarity:
    """
    Semantic similarity analyzer for paraphrase evaluation.
    
    Provides comprehensive semantic similarity analysis including computation,
    threshold-based classification, statistical analysis, and optional visualization.
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        """
        Initialize semantic similarity analyzer.
        
        Args:
            config: Similarity configuration. If None, uses default config.
        """
        self.config = config or SimilarityConfig()
        self.logger = get_logger("SemanticSimilarity")
        
        # Load sentence transformer model
        self.model = load_sentence_transformer(
            self.config.model_name,
            self.config.device
        )
        
        # Caching for embeddings
        self.embedding_cache = {} if self.config.enable_caching else None
        
        # Statistics tracking
        self.similarity_history = []
        
        if self.model:
            self.logger.info(f"Initialized SemanticSimilarity with model: {self.config.model_name}")
        else:
            self.logger.warning("Sentence transformers not available - using fallback methods")
    
    def _get_cached_embeddings(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Get embeddings from cache if available."""
        
        if not self.embedding_cache:
            return None
        
        cache_key = "|".join(texts)
        return self.embedding_cache.get(cache_key)
    
    def _cache_embeddings(self, texts: List[str], embeddings: torch.Tensor):
        """Cache embeddings for future use."""
        
        if not self.embedding_cache:
            return
        
        cache_key = "|".join(texts)
        self.embedding_cache[cache_key] = embeddings
    
    def encode_texts(self, texts: List[str]) -> Optional[torch.Tensor]:
        """
        Encode texts to embeddings with caching support.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Tensor of embeddings or None if model unavailable
        """
        if not self.model:
            return None
        
        # Check cache first
        cached_embeddings = self._get_cached_embeddings(texts)
        if cached_embeddings is not None:
            return cached_embeddings
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            # Cache embeddings
            self._cache_embeddings(texts, embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Text encoding failed: {e}")
            return None
    
    def compute_similarity(
        self,
        a: List[str],
        b: Optional[List[str]] = None
    ) -> Union[List[float], np.ndarray]:
        """
        Compute semantic similarity scores.
        
        Args:
            a: First list of texts
            b: Second list of texts. If None, computes pairwise similarities within a
            
        Returns:
            Similarity scores as list (if b provided) or matrix (if b is None)
        """
        if not self.model:
            # Fallback: use simple word overlap
            return self._compute_fallback_similarity(a, b)
        
        if b is not None:
            # Pairwise similarities between two lists
            similarities = compute_similarity(
                a, b, self.model,
                metric=self.config.similarity_metric,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize_embeddings
            )
            
            # Track statistics
            if self.config.compute_statistics:
                self.similarity_history.extend(similarities)
            
            return similarities
        
        else:
            # All-pairs similarity matrix within list a
            embeddings_a = self.encode_texts(a)
            if embeddings_a is None:
                return np.zeros((len(a), len(a)))
            
            # Compute similarity matrix
            if self.config.similarity_metric == "cosine":
                similarity_matrix = torch.mm(embeddings_a, embeddings_a.t())
            elif self.config.similarity_metric == "dot":
                similarity_matrix = torch.mm(embeddings_a, embeddings_a.t())
            else:
                # Compute pairwise distances and convert to similarities
                from torch.nn.functional import pdist, squareform
                distances = pdist(embeddings_a, p=2)
                distance_matrix = squareform(distances)
                similarity_matrix = 1.0 / (1.0 + distance_matrix)
            
            return similarity_matrix.cpu().numpy()
    
    def _compute_fallback_similarity(
        self,
        a: List[str],
        b: Optional[List[str]] = None
    ) -> Union[List[float], np.ndarray]:
        """Compute similarity using word overlap as fallback."""
        
        def word_overlap_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        if b is not None:
            # Pairwise similarities
            similarities = []
            for text1, text2 in zip(a, b):
                sim = word_overlap_similarity(text1, text2)
                similarities.append(sim)
            return similarities
        
        else:
            # All-pairs matrix
            n = len(a)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sim = word_overlap_similarity(a[i], a[j])
                        similarity_matrix[i, j] = sim
            
            return similarity_matrix
    
    def classify_paraphrases(
        self,
        sources: List[str],
        candidates: List[str]
    ) -> Dict[str, List[bool]]:
        """
        Classify candidate texts as paraphrases based on similarity thresholds.
        
        Args:
            sources: Source texts
            candidates: Candidate paraphrase texts
            
        Returns:
            Dictionary with classification results at different thresholds
        """
        similarities = self.compute_similarity(sources, candidates)
        
        classifications = {
            'is_paraphrase': [s >= self.config.paraphrase_threshold for s in similarities],
            'high_similarity': [s >= self.config.high_similarity_threshold for s in similarities],
            'low_similarity': [s <= self.config.low_similarity_threshold for s in similarities]
        }
        
        return classifications
    
    def analyze_distribution(self, similarities: List[float]) -> Dict[str, float]:
        """
        Analyze statistical distribution of similarity scores.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Dictionary of statistical measures
        """
        if not similarities:
            return {}
        
        similarities_array = np.array(similarities)
        
        stats = {
            'count': len(similarities),
            'mean': float(np.mean(similarities_array)),
            'median': float(np.median(similarities_array)),
            'std': float(np.std(similarities_array)),
            'min': float(np.min(similarities_array)),
            'max': float(np.max(similarities_array)),
            'q25': float(np.percentile(similarities_array, 25)),
            'q75': float(np.percentile(similarities_array, 75)),
        }
        
        # Add threshold-based counts
        stats['above_paraphrase_threshold'] = int(np.sum(similarities_array >= self.config.paraphrase_threshold))
        stats['above_high_threshold'] = int(np.sum(similarities_array >= self.config.high_similarity_threshold))
        stats['below_low_threshold'] = int(np.sum(similarities_array <= self.config.low_similarity_threshold))
        
        # Add percentages
        total = len(similarities)
        stats['pct_above_paraphrase_threshold'] = stats['above_paraphrase_threshold'] / total * 100
        stats['pct_above_high_threshold'] = stats['above_high_threshold'] / total * 100
        stats['pct_below_low_threshold'] = stats['below_low_threshold'] / total * 100
        
        return stats
    
    def get_similarity_history_stats(self) -> Dict[str, float]:
        """Get statistics for all computed similarities."""
        
        if not self.similarity_history:
            return {}
        
        return self.analyze_distribution(self.similarity_history)
    
    def create_embeddings_visualization(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Create 2D visualization of text embeddings using dimensionality reduction.
        
        Args:
            texts: List of texts to visualize
            labels: Optional labels for coloring points
            save_path: Optional path to save the visualization
            
        Returns:
            2D coordinates array or None if visualization unavailable
        """
        if not self.model or not PLOTTING_AVAILABLE or not SKLEARN_AVAILABLE:
            self.logger.warning("Visualization dependencies not available")
            return None
        
        # Get embeddings
        embeddings = self.encode_texts(texts)
        if embeddings is None:
            return None
        
        embeddings_np = embeddings.cpu().numpy()
        
        # Apply dimensionality reduction
        if self.config.dimensionality_reduction == "pca":
            reducer = PCA(n_components=self.config.n_components)
        elif self.config.dimensionality_reduction == "tsne":
            reducer = TSNE(n_components=self.config.n_components, random_state=42)
        elif self.config.dimensionality_reduction == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=self.config.n_components, random_state=42)
        else:
            # Fallback to PCA
            reducer = PCA(n_components=self.config.n_components)
        
        # Reduce dimensions
        coords_2d = reducer.fit_transform(embeddings_np)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels:
            # Color by labels
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(
                    coords_2d[mask, 0], coords_2d[mask, 1],
                    c=[colors[i]], label=label, alpha=0.7, s=50
                )
            plt.legend()
        else:
            plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, s=50)
        
        plt.title(f'Text Embeddings Visualization ({self.config.dimensionality_reduction.upper()})')
        plt.xlabel(f'{self.config.dimensionality_reduction.upper()} Component 1')
        plt.ylabel(f'{self.config.dimensionality_reduction.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return coords_2d
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.embedding_cache:
            self.embedding_cache.clear()
    
    def clear_history(self):
        """Clear similarity computation history."""
        self.similarity_history.clear()


def main():
    """Example usage of SemanticSimilarity."""
    
    # Sample data
    sources = [
        "The weather is beautiful today.",
        "I love programming in Python.",
        "This movie is really interesting.",
        "Machine learning is fascinating.",
        "The cat is sleeping on the couch."
    ]
    
    candidates = [
        "Today's weather is lovely.",
        "Python programming is enjoyable.", 
        "The film is quite engaging.",
        "AI algorithms are captivating.",
        "A feline rests on the sofa."
    ]
    
    # Initialize semantic similarity
    config = SimilarityConfig(
        model_name="all-MiniLM-L6-v2",
        paraphrase_threshold=0.7,
        compute_statistics=True
    )
    
    similarity = SemanticSimilarity(config)
    
    print("=== Semantic Similarity Example ===")
    print(f"Sources: {len(sources)}")
    print(f"Candidates: {len(candidates)}")
    
    # Compute pairwise similarities
    print("\n=== Pairwise Similarities ===")
    similarities = similarity.compute_similarity(sources, candidates)
    
    for i, (src, cand, sim) in enumerate(zip(sources, candidates, similarities)):
        print(f"{i+1}. Similarity: {sim:.3f}")
        print(f"   Source: {src}")
        print(f"   Candidate: {cand}")
        print()
    
    # Analyze distribution
    print("=== Distribution Analysis ===")
    stats = similarity.analyze_distribution(similarities)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Classify paraphrases
    print("\n=== Paraphrase Classification ===")
    classifications = similarity.classify_paraphrases(sources, candidates)
    
    for key, values in classifications.items():
        count = sum(values)
        print(f"{key}: {count}/{len(values)} ({count/len(values)*100:.1f}%)")
    
    # All-pairs similarity matrix
    print("\n=== All-pairs Similarity Matrix ===")
    matrix = similarity.compute_similarity(sources[:3])  # Use subset for clarity
    print("Matrix shape:", matrix.shape)
    print("Sample similarities:")
    for i in range(min(3, matrix.shape[0])):
        for j in range(min(3, matrix.shape[1])):
            print(f"  [{i},{j}]: {matrix[i,j]:.3f}")
    
    # Get overall statistics
    print("\n=== Overall Statistics ===")
    history_stats = similarity.get_similarity_history_stats()
    if history_stats:
        print(f"Total computations: {history_stats.get('count', 0)}")
        print(f"Mean similarity: {history_stats.get('mean', 0):.3f}")
        print(f"Std deviation: {history_stats.get('std', 0):.3f}")


if __name__ == "__main__":
    main()
