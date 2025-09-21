#!/usr/bin/env python3
"""
Hybrid Retrieval combining Dense and Sparse Methods

Combines dense semantic retrieval with sparse lexical retrieval using
configurable weighting and ranking fusion strategies for optimal coverage.

Example Usage:
    >>> from fact_verification.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
    >>> 
    >>> # Initialize component retrievers
    >>> dense = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
    >>> sparse = SparseRetriever(method="bm25")
    >>> 
    >>> # Create hybrid retriever
    >>> hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    >>> 
    >>> # Build indexes
    >>> corpus = ["Document 1 text", "Document 2 text", ...]
    >>> hybrid.build_index(corpus)
    >>> 
    >>> # Retrieve with fusion
    >>> results = hybrid.retrieve("COVID vaccine effectiveness", top_k=10, alpha=0.6)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
import math

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger
from .dense_retriever import DenseRetriever, RetrievalResult
from .sparse_retriever import SparseRetriever, SparseRetrievalResult

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class HybridRetrievalResult:
    """Container for hybrid retrieval results."""
    
    text: str
    score: float
    dense_score: float
    sparse_score: float
    doc_id: int
    rank_position: int
    fusion_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    term_matches: List[str] = field(default_factory=list)


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retriever."""
    
    # Fusion parameters
    default_alpha: float = 0.5  # Weight for dense vs sparse (0=sparse only, 1=dense only)
    fusion_method: str = "weighted_sum"  # weighted_sum, rrf, convex_combination
    rrf_k: int = 60  # Parameter for Reciprocal Rank Fusion
    
    # Score normalization
    normalize_scores: bool = True
    normalization_method: str = "min_max"  # min_max, z_score, rank
    
    # Retrieval parameters
    dense_top_k_multiplier: float = 2.0  # Retrieve more from each component
    sparse_top_k_multiplier: float = 2.0
    
    # Result filtering
    min_dense_score: float = 0.0
    min_sparse_score: float = 0.0
    min_combined_score: float = 0.0
    
    # Diversity
    enable_diversity_penalty: bool = False
    diversity_lambda: float = 0.1


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Provides sophisticated fusion strategies including weighted combination,
    Reciprocal Rank Fusion (RRF), and advanced score normalization for
    optimal retrieval performance across different query types.
    """
    
    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        config: Optional[HybridRetrieverConfig] = None,
        results_dir: str = "fact_verification/retrieval/results",
        logger: Optional[Any] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            config: Hybrid retriever configuration
            results_dir: Directory for saving results
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("HybridRetriever")
        
        # Configuration
        self.config = config or HybridRetrieverConfig()
        
        # Component retrievers
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        
        # Results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation
        if not self.dense_retriever and not self.sparse_retriever:
            raise ValueError("At least one retriever (dense or sparse) must be provided")
        
        # State tracking
        self.is_indexed = False
        self.corpus = []
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'dense_queries': 0,
            'sparse_queries': 0,
            'fusion_method_usage': defaultdict(int),
            'average_retrieval_time': 0.0
        }
        
        self.logger.info("Initialized HybridRetriever")
        self.logger.info(f"Dense retriever: {'Available' if dense_retriever else 'Not available'}")
        self.logger.info(f"Sparse retriever: {'Available' if sparse_retriever else 'Not available'}")
    
    def build_index(
        self,
        corpus: List[Union[str, Dict[str, Any]]],
        save_index: bool = True,
        index_name: str = "hybrid_index",
        rebuild: bool = False
    ) -> bool:
        """
        Build indexes for both dense and sparse retrievers.
        
        Args:
            corpus: List of documents
            save_index: Whether to save indexes
            index_name: Name prefix for saved indexes
            rebuild: Whether to rebuild existing indexes
            
        Returns:
            True if successful
        """
        self.logger.info(f"Building hybrid index with {len(corpus)} documents")
        
        success = True
        self.corpus = corpus
        
        # Build dense index
        if self.dense_retriever:
            self.logger.info("Building dense index...")
            dense_success = self.dense_retriever.build_index(
                corpus, save_index=save_index, 
                index_name=f"{index_name}_dense", rebuild=rebuild
            )
            if not dense_success:
                self.logger.error("Failed to build dense index")
                success = False
        
        # Build sparse index
        if self.sparse_retriever:
            self.logger.info("Building sparse index...")
            sparse_success = self.sparse_retriever.build_index(
                corpus, save_index=save_index,
                index_name=f"{index_name}_sparse", rebuild=rebuild
            )
            if not sparse_success:
                self.logger.error("Failed to build sparse index")
                success = False
        
        self.is_indexed = success
        
        if success:
            self.logger.info("Hybrid index built successfully")
        else:
            self.logger.error("Failed to build hybrid index")
        
        return success
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None,
        fusion_method: Optional[str] = None,
        include_component_scores: bool = True,
        save_results: bool = False
    ) -> List[HybridRetrievalResult]:
        """
        Retrieve documents using hybrid fusion.
        
        Args:
            query: Query string
            top_k: Number of results to return
            alpha: Fusion weight (0=sparse only, 1=dense only)
            fusion_method: Fusion method to use
            include_component_scores: Whether to include individual scores
            save_results: Whether to save results to disk
            
        Returns:
            List of HybridRetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Set defaults
        alpha = alpha if alpha is not None else self.config.default_alpha
        fusion_method = fusion_method or self.config.fusion_method
        
        # Compute retrieval amounts for each component
        dense_k = int(top_k * self.config.dense_top_k_multiplier)
        sparse_k = int(top_k * self.config.sparse_top_k_multiplier)
        
        # Retrieve from dense retriever
        dense_results = []
        if self.dense_retriever and alpha > 0:
            try:
                dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)
                self.retrieval_stats['dense_queries'] += 1
            except Exception as e:
                self.logger.warning(f"Dense retrieval failed: {e}")
        
        # Retrieve from sparse retriever
        sparse_results = []
        if self.sparse_retriever and alpha < 1:
            try:
                sparse_results = self.sparse_retriever.retrieve(query, top_k=sparse_k)
                self.retrieval_stats['sparse_queries'] += 1
            except Exception as e:
                self.logger.warning(f"Sparse retrieval failed: {e}")
        
        # Fusion
        if fusion_method == "weighted_sum":
            hybrid_results = self._weighted_sum_fusion(dense_results, sparse_results, alpha)
        elif fusion_method == "rrf":
            hybrid_results = self._reciprocal_rank_fusion(dense_results, sparse_results, alpha)
        elif fusion_method == "convex_combination":
            hybrid_results = self._convex_combination_fusion(dense_results, sparse_results, alpha)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Apply diversity penalty if enabled
        if self.config.enable_diversity_penalty:
            hybrid_results = self._apply_diversity_penalty(hybrid_results, query)
        
        # Filter and take top-k
        filtered_results = self._filter_results(hybrid_results)
        final_results = filtered_results[:top_k]
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.retrieval_stats['total_queries'] += 1
        self.retrieval_stats['fusion_method_usage'][fusion_method] += 1
        
        # Update average time (exponential moving average)
        if self.retrieval_stats['average_retrieval_time'] == 0:
            self.retrieval_stats['average_retrieval_time'] = retrieval_time
        else:
            alpha_ema = 0.1
            self.retrieval_stats['average_retrieval_time'] = (
                alpha_ema * retrieval_time + 
                (1 - alpha_ema) * self.retrieval_stats['average_retrieval_time']
            )
        
        # Save results if requested
        if save_results:
            self._save_retrieval_results(query, final_results, {
                'top_k': top_k,
                'alpha': alpha,
                'fusion_method': fusion_method,
                'retrieval_time': retrieval_time,
                'num_dense_results': len(dense_results),
                'num_sparse_results': len(sparse_results)
            })
        
        self.logger.debug(f"Retrieved {len(final_results)} results in {retrieval_time:.3f}s")
        return final_results
    
    def _weighted_sum_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[SparseRetrievalResult],
        alpha: float
    ) -> List[HybridRetrievalResult]:
        """Combine results using weighted sum fusion."""
        
        # Collect all unique documents
        all_docs = {}
        
        # Normalize dense scores
        if dense_results and self.config.normalize_scores:
            dense_scores = [r.score for r in dense_results]
            dense_scores_norm = self._normalize_scores(dense_scores)
        else:
            dense_scores_norm = [r.score for r in dense_results]
        
        # Add dense results
        for i, (result, norm_score) in enumerate(zip(dense_results, dense_scores_norm)):
            doc_id = result.doc_id
            all_docs[doc_id] = {
                'text': result.text,
                'dense_score': norm_score,
                'sparse_score': 0.0,
                'doc_id': doc_id,
                'metadata': result.metadata,
                'term_matches': [],
                'dense_rank': i + 1
            }
        
        # Normalize sparse scores
        if sparse_results and self.config.normalize_scores:
            sparse_scores = [r.score for r in sparse_results]
            sparse_scores_norm = self._normalize_scores(sparse_scores)
        else:
            sparse_scores_norm = [r.score for r in sparse_results]
        
        # Add sparse results
        for i, (result, norm_score) in enumerate(zip(sparse_results, sparse_scores_norm)):
            doc_id = result.doc_id
            
            if doc_id in all_docs:
                all_docs[doc_id]['sparse_score'] = norm_score
                all_docs[doc_id]['term_matches'] = result.term_matches
                all_docs[doc_id]['sparse_rank'] = i + 1
            else:
                all_docs[doc_id] = {
                    'text': result.text,
                    'dense_score': 0.0,
                    'sparse_score': norm_score,
                    'doc_id': doc_id,
                    'metadata': result.metadata,
                    'term_matches': result.term_matches,
                    'sparse_rank': i + 1
                }
        
        # Compute hybrid scores
        hybrid_results = []
        
        for doc_id, doc_info in all_docs.items():
            # Weighted sum
            combined_score = (
                alpha * doc_info['dense_score'] + 
                (1 - alpha) * doc_info['sparse_score']
            )
            
            hybrid_result = HybridRetrievalResult(
                text=doc_info['text'],
                score=combined_score,
                dense_score=doc_info['dense_score'],
                sparse_score=doc_info['sparse_score'],
                doc_id=doc_id,
                rank_position=0,  # Will be set after sorting
                fusion_method="weighted_sum",
                metadata=doc_info['metadata'],
                term_matches=doc_info['term_matches']
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by combined score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Set rank positions
        for i, result in enumerate(hybrid_results):
            result.rank_position = i + 1
        
        return hybrid_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[SparseRetrievalResult],
        alpha: float
    ) -> List[HybridRetrievalResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        
        # Collect document rankings
        all_docs = {}
        k = self.config.rrf_k
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.doc_id
            rrf_score = 1.0 / (k + rank)
            
            all_docs[doc_id] = {
                'text': result.text,
                'dense_rrf': rrf_score,
                'sparse_rrf': 0.0,
                'dense_score': result.score,
                'sparse_score': 0.0,
                'doc_id': doc_id,
                'metadata': result.metadata,
                'term_matches': [],
                'dense_rank': rank
            }
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.doc_id
            rrf_score = 1.0 / (k + rank)
            
            if doc_id in all_docs:
                all_docs[doc_id]['sparse_rrf'] = rrf_score
                all_docs[doc_id]['sparse_score'] = result.score
                all_docs[doc_id]['term_matches'] = result.term_matches
                all_docs[doc_id]['sparse_rank'] = rank
            else:
                all_docs[doc_id] = {
                    'text': result.text,
                    'dense_rrf': 0.0,
                    'sparse_rrf': rrf_score,
                    'dense_score': 0.0,
                    'sparse_score': result.score,
                    'doc_id': doc_id,
                    'metadata': result.metadata,
                    'term_matches': result.term_matches,
                    'sparse_rank': rank
                }
        
        # Compute combined RRF scores
        hybrid_results = []
        
        for doc_id, doc_info in all_docs.items():
            # Weighted RRF combination
            combined_rrf = (
                alpha * doc_info['dense_rrf'] + 
                (1 - alpha) * doc_info['sparse_rrf']
            )
            
            hybrid_result = HybridRetrievalResult(
                text=doc_info['text'],
                score=combined_rrf,
                dense_score=doc_info['dense_score'],
                sparse_score=doc_info['sparse_score'],
                doc_id=doc_id,
                rank_position=0,
                fusion_method="rrf",
                metadata=doc_info['metadata'],
                term_matches=doc_info['term_matches']
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by RRF score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Set rank positions
        for i, result in enumerate(hybrid_results):
            result.rank_position = i + 1
        
        return hybrid_results
    
    def _convex_combination_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[SparseRetrievalResult],
        alpha: float
    ) -> List[HybridRetrievalResult]:
        """Combine results using convex combination with rank normalization."""
        
        # Similar to weighted sum but with rank-based normalization
        all_docs = {}
        
        # Process dense results with rank normalization
        max_dense_rank = len(dense_results)
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.doc_id
            rank_norm = 1.0 - (rank - 1) / max_dense_rank if max_dense_rank > 0 else 0.0
            
            all_docs[doc_id] = {
                'text': result.text,
                'dense_rank_norm': rank_norm,
                'sparse_rank_norm': 0.0,
                'dense_score': result.score,
                'sparse_score': 0.0,
                'doc_id': doc_id,
                'metadata': result.metadata,
                'term_matches': []
            }
        
        # Process sparse results with rank normalization
        max_sparse_rank = len(sparse_results)
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.doc_id
            rank_norm = 1.0 - (rank - 1) / max_sparse_rank if max_sparse_rank > 0 else 0.0
            
            if doc_id in all_docs:
                all_docs[doc_id]['sparse_rank_norm'] = rank_norm
                all_docs[doc_id]['sparse_score'] = result.score
                all_docs[doc_id]['term_matches'] = result.term_matches
            else:
                all_docs[doc_id] = {
                    'text': result.text,
                    'dense_rank_norm': 0.0,
                    'sparse_rank_norm': rank_norm,
                    'dense_score': 0.0,
                    'sparse_score': result.score,
                    'doc_id': doc_id,
                    'metadata': result.metadata,
                    'term_matches': result.term_matches
                }
        
        # Combine using convex combination
        hybrid_results = []
        
        for doc_id, doc_info in all_docs.items():
            combined_score = (
                alpha * doc_info['dense_rank_norm'] + 
                (1 - alpha) * doc_info['sparse_rank_norm']
            )
            
            hybrid_result = HybridRetrievalResult(
                text=doc_info['text'],
                score=combined_score,
                dense_score=doc_info['dense_score'],
                sparse_score=doc_info['sparse_score'],
                doc_id=doc_id,
                rank_position=0,
                fusion_method="convex_combination",
                metadata=doc_info['metadata'],
                term_matches=doc_info['term_matches']
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort and rank
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(hybrid_results):
            result.rank_position = i + 1
        
        return hybrid_results
    
    def _normalize_scores(self, scores: List[float], method: str = None) -> List[float]:
        """Normalize scores using specified method."""
        
        if not scores:
            return []
        
        method = method or self.config.normalization_method
        scores_array = np.array(scores)
        
        if method == "min_max":
            if len(scores) == 1:
                return [1.0]
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score - min_score == 0:
                return [1.0] * len(scores)
            return ((scores_array - min_score) / (max_score - min_score)).tolist()
        
        elif method == "z_score":
            mean_score = scores_array.mean()
            std_score = scores_array.std()
            if std_score == 0:
                return [0.0] * len(scores)
            return ((scores_array - mean_score) / std_score).tolist()
        
        elif method == "rank":
            # Convert to rank-based scores
            sorted_indices = np.argsort(scores_array)[::-1]
            rank_scores = np.zeros_like(scores_array)
            for i, idx in enumerate(sorted_indices):
                rank_scores[idx] = 1.0 - i / len(scores)
            return rank_scores.tolist()
        
        else:
            return scores
    
    def _apply_diversity_penalty(
        self,
        results: List[HybridRetrievalResult],
        query: str
    ) -> List[HybridRetrievalResult]:
        """Apply diversity penalty to reduce redundant results."""
        
        if len(results) <= 1:
            return results
        
        # Simple diversity penalty based on text similarity
        diversity_results = []
        selected_texts = []
        
        for result in results:
            # Compute similarity to already selected results
            max_similarity = 0.0
            
            for selected_text in selected_texts:
                # Simple word overlap similarity
                result_words = set(result.text.lower().split())
                selected_words = set(selected_text.lower().split())
                
                if result_words and selected_words:
                    overlap = len(result_words & selected_words)
                    union = len(result_words | selected_words)
                    similarity = overlap / union
                    max_similarity = max(max_similarity, similarity)
            
            # Apply diversity penalty
            diversity_penalty = self.config.diversity_lambda * max_similarity
            adjusted_score = result.score * (1 - diversity_penalty)
            
            # Update result
            result.score = adjusted_score
            diversity_results.append(result)
            selected_texts.append(result.text)
        
        # Re-sort after diversity adjustment
        diversity_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(diversity_results):
            result.rank_position = i + 1
        
        return diversity_results
    
    def _filter_results(self, results: List[HybridRetrievalResult]) -> List[HybridRetrievalResult]:
        """Filter results based on minimum score thresholds."""
        
        filtered = []
        
        for result in results:
            # Apply filters
            if result.dense_score < self.config.min_dense_score:
                continue
            if result.sparse_score < self.config.min_sparse_score:
                continue
            if result.score < self.config.min_combined_score:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        alpha: Optional[float] = None,
        fusion_method: Optional[str] = None,
        save_results: bool = False
    ) -> List[List[HybridRetrievalResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            alpha: Fusion weight
            fusion_method: Fusion method to use
            save_results: Whether to save batch results
            
        Returns:
            List of result lists, one per query
        """
        all_results = []
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            results = self.retrieve(
                query, top_k=top_k, alpha=alpha, 
                fusion_method=fusion_method, save_results=False
            )
            all_results.append(results)
        
        batch_time = time.time() - start_time
        
        # Save batch results if requested
        if save_results:
            self._save_batch_results(queries, all_results, {
                'top_k': top_k,
                'alpha': alpha or self.config.default_alpha,
                'fusion_method': fusion_method or self.config.fusion_method,
                'batch_time': batch_time,
                'num_queries': len(queries)
            })
        
        self.logger.info(f"Batch retrieval completed: {len(queries)} queries in {batch_time:.2f}s")
        return all_results
    
    def _save_retrieval_results(
        self,
        query: str,
        results: List[HybridRetrievalResult],
        metadata: Dict[str, Any]
    ):
        """Save retrieval results to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrieval_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Prepare data for saving
        results_data = {
            'query': query,
            'metadata': metadata,
            'results': []
        }
        
        for result in results:
            result_data = {
                'text': result.text,
                'score': result.score,
                'dense_score': result.dense_score,
                'sparse_score': result.sparse_score,
                'doc_id': result.doc_id,
                'rank_position': result.rank_position,
                'fusion_method': result.fusion_method,
                'metadata': result.metadata,
                'term_matches': result.term_matches
            }
            results_data['results'].append(result_data)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.debug(f"Retrieval results saved to {filepath}")
    
    def _save_batch_results(
        self,
        queries: List[str],
        batch_results: List[List[HybridRetrievalResult]],
        metadata: Dict[str, Any]
    ):
        """Save batch retrieval results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_retrieval_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Prepare batch data
        batch_data = {
            'metadata': metadata,
            'queries': []
        }
        
        for query, results in zip(queries, batch_results):
            query_data = {
                'query': query,
                'results': []
            }
            
            for result in results:
                result_data = {
                    'text': result.text,
                    'score': result.score,
                    'dense_score': result.dense_score,
                    'sparse_score': result.sparse_score,
                    'doc_id': result.doc_id,
                    'rank_position': result.rank_position,
                    'fusion_method': result.fusion_method
                }
                query_data['results'].append(result_data)
            
            batch_data['queries'].append(query_data)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        # Also save as CSV if pandas available
        if PANDAS_AVAILABLE:
            csv_filename = f"batch_retrieval_{timestamp}.csv"
            csv_filepath = self.results_dir / csv_filename
            
            # Flatten data for CSV
            csv_data = []
            for query, results in zip(queries, batch_results):
                for result in results:
                    csv_row = {
                        'query': query,
                        'text': result.text[:200],  # Truncate for CSV
                        'score': result.score,
                        'dense_score': result.dense_score,
                        'sparse_score': result.sparse_score,
                        'doc_id': result.doc_id,
                        'rank_position': result.rank_position,
                        'fusion_method': result.fusion_method
                    }
                    csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filepath, index=False)
        
        self.logger.info(f"Batch results saved to {filepath}")
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about hybrid retriever configuration and status."""
        
        return {
            'is_indexed': self.is_indexed,
            'corpus_size': len(self.corpus),
            'config': {
                'default_alpha': self.config.default_alpha,
                'fusion_method': self.config.fusion_method,
                'normalize_scores': self.config.normalize_scores,
                'normalization_method': self.config.normalization_method,
                'rrf_k': self.config.rrf_k
            },
            'retrievers': {
                'dense_available': self.dense_retriever is not None,
                'sparse_available': self.sparse_retriever is not None,
                'dense_info': self.dense_retriever.get_retriever_info() if self.dense_retriever else None,
                'sparse_info': self.sparse_retriever.get_retriever_info() if self.sparse_retriever else None
            },
            'statistics': self.retrieval_stats,
            'results_dir': str(self.results_dir)
        }


def main():
    """Example usage of HybridRetriever."""
    
    # Import here to avoid circular imports
    from .dense_retriever import DenseRetriever
    from .sparse_retriever import SparseRetriever
    
    print("=== HybridRetriever Example ===")
    
    # Sample corpus
    corpus = [
        "COVID-19 vaccines have been proven to be 90-95% effective in preventing severe illness and hospitalization.",
        "Climate change is primarily caused by human activities, particularly greenhouse gas emissions from fossil fuels.",
        "The Earth is approximately 4.5 billion years old according to geological and radiometric evidence.",
        "Artificial intelligence has made significant advances in natural language processing and computer vision.",
        "Regular exercise and a balanced diet are essential for maintaining good health and preventing chronic diseases.",
        "The COVID-19 pandemic began in late 2019 and has affected millions of people worldwide.",
        "Renewable energy sources like solar and wind power are becoming increasingly cost-effective.",
        "Machine learning algorithms can be trained to recognize patterns in large datasets.",
        "Social media platforms have transformed how people communicate and share information globally.",
        "Electric vehicles are becoming more popular as battery technology continues to improve rapidly."
    ]
    
    # Initialize component retrievers
    print("Initializing component retrievers...")
    
    try:
        # Use lightweight models for demo
        dense = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            enable_gpu=False
        )
        dense_available = True
    except:
        print("Dense retriever not available")
        dense = None
        dense_available = False
    
    sparse = SparseRetriever(method="bm25")
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        config=HybridRetrieverConfig(
            default_alpha=0.6,
            fusion_method="weighted_sum",
            normalize_scores=True
        )
    )
    
    print(f"Hybrid retriever info: {hybrid.get_retriever_info()}")
    
    # Build index
    print(f"\nBuilding hybrid index with {len(corpus)} documents...")
    success = hybrid.build_index(corpus, save_index=False, index_name="demo_hybrid")
    
    if success:
        print("Hybrid index built successfully!")
        
        # Test queries
        test_queries = [
            "COVID vaccine effectiveness",
            "climate change fossil fuels",
            "artificial intelligence machine learning"
        ]
        
        # Test different fusion methods
        fusion_methods = ["weighted_sum", "rrf"]
        if dense_available:
            fusion_methods.append("convex_combination")
        
        for fusion_method in fusion_methods:
            print(f"\n=== Testing {fusion_method.upper()} Fusion ===")
            
            for query in test_queries[:2]:  # Limit for demo
                print(f"\nQuery: {query}")
                
                results = hybrid.retrieve(
                    query, 
                    top_k=3, 
                    alpha=0.6,
                    fusion_method=fusion_method,
                    save_results=False
                )
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Combined: {result.score:.3f} "
                          f"(Dense: {result.dense_score:.3f}, Sparse: {result.sparse_score:.3f})")
                    print(f"     Text: {result.text[:80]}...")
                    if result.term_matches:
                        print(f"     Matches: {', '.join(result.term_matches[:3])}")
        
        # Test different alpha values
        print(f"\n=== Testing Different Alpha Values ===")
        query = "COVID vaccine effectiveness"
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            if alpha == 0.0 and not sparse.is_indexed:
                continue
            if alpha == 1.0 and not dense_available:
                continue
            
            print(f"\nAlpha = {alpha} ({'sparse only' if alpha == 0 else 'dense only' if alpha == 1 else 'hybrid'})")
            
            results = hybrid.retrieve(query, top_k=2, alpha=alpha)
            
            for result in results:
                print(f"  Score: {result.score:.3f} "
                      f"(D: {result.dense_score:.3f}, S: {result.sparse_score:.3f})")
        
        # Test batch retrieval
        print(f"\n=== Batch Retrieval Test ===")
        batch_results = hybrid.batch_retrieve(
            test_queries[:2], 
            top_k=2, 
            alpha=0.5,
            save_results=True
        )
        
        print(f"Batch retrieval completed: {len(batch_results)} query results")
        for query, results in zip(test_queries[:2], batch_results):
            print(f"  {query}: {len(results)} results")
        
        # Display statistics
        print(f"\n=== Retrieval Statistics ===")
        stats = hybrid.retrieval_stats
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    else:
        print("Failed to build hybrid index")


if __name__ == "__main__":
    main()
