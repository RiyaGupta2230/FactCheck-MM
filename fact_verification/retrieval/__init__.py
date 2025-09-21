"""
Fact Verification Retrieval Module

This module provides comprehensive evidence retrieval capabilities including
dense transformer-based retrieval, sparse lexical retrieval, and hybrid
approaches for robust fact verification evidence gathering.

Key components:
- DenseRetriever: Transformer-based dense passage retrieval with FAISS indexing
- SparseRetriever: BM25/TF-IDF lexical retrieval for fast word-overlap matching  
- HybridRetriever: Combined dense and sparse retrieval with score fusion

Example Usage:
    >>> from fact_verification.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
    >>> 
    >>> # Dense retrieval with transformer embeddings
    >>> dense = DenseRetriever(model_name="facebook/dpr-ctx_encoder-single-nq-base")
    >>> dense.build_index(corpus)
    >>> results = dense.retrieve("COVID vaccines are effective", top_k=5)
    >>> 
    >>> # Fast sparse retrieval
    >>> sparse = SparseRetriever(method="bm25")  
    >>> sparse.build_index(corpus)
    >>> results = sparse.retrieve("Climate change causes", top_k=10)
    >>> 
    >>> # Hybrid approach combining both
    >>> hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    >>> results = hybrid.retrieve("Political claim about economy", alpha=0.6)
"""

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .hybrid_retriever import HybridRetriever

__all__ = [
    "DenseRetriever",
    "SparseRetriever", 
    "HybridRetriever"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
