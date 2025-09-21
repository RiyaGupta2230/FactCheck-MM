#!/usr/bin/env python3
"""
Sparse Retrieval using BM25 and TF-IDF

Fast lexical retrieval based on word overlap and term frequency statistics
for efficient CPU-based evidence retrieval with inverted indexing.

Example Usage:
    >>> from fact_verification.retrieval import SparseRetriever
    >>> 
    >>> # Initialize with BM25
    >>> retriever = SparseRetriever(method="bm25", k1=1.5, b=0.75)
    >>> 
    >>> # Build index from corpus
    >>> corpus = ["Document 1 text", "Document 2 text", ...]
    >>> retriever.build_index(corpus, save_index=True)
    >>> 
    >>> # Retrieve relevant passages
    >>> results = retriever.retrieve("COVID vaccines effectiveness", top_k=10)
    >>> for result in results:
    ...     print(f"Score: {result['score']:.3f}, Text: {result['text'][:100]}...")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import pickle
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import math
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional dependencies
try:
    from rank_bm25 import BM25Okapi, BM25L, BM25Plus
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class SparseRetrievalResult:
    """Container for sparse retrieval results."""
    
    text: str
    score: float
    doc_id: int
    term_matches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SparseRetrieverConfig:
    """Configuration for sparse retriever."""
    
    method: str = "bm25"  # bm25, tfidf, count
    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    # TF-IDF parameters
    max_features: int = 100000
    ngram_range: Tuple[int, int] = (1, 2)
    max_df: float = 0.95
    min_df: int = 1
    # Text preprocessing
    lowercase: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = False
    apply_lemmatization: bool = False
    min_token_length: int = 2


class SparseRetriever:
    """
    Fast lexical retrieval using BM25 or TF-IDF.
    
    Provides efficient CPU-based retrieval with inverted indexing for
    fast word overlap matching and lexical similarity computation.
    """
    
    def __init__(
        self,
        method: str = "bm25",
        config: Optional[SparseRetrieverConfig] = None,
        index_dir: str = "data/indexes/sparse",
        logger: Optional[Any] = None
    ):
        """
        Initialize sparse retriever.
        
        Args:
            method: Retrieval method ('bm25', 'tfidf', 'count')
            config: Retriever configuration
            index_dir: Directory for storing indexes
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("SparseRetriever")
        self.method = method
        
        # Configuration
        self.config = config or SparseRetrieverConfig(method=method)
        
        # Index directory
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing components
        self.stopwords = set()
        self.stemmer = None
        self.lemmatizer = None
        self._init_preprocessing()
        
        # Index components
        self.corpus = []
        self.processed_corpus = []
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index = defaultdict(list)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.is_indexed = False
        
        # Vocabulary and statistics
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_frequencies = Counter()
        
        self.logger.info(f"Initialized SparseRetriever with method: {method}")
    
    def _init_preprocessing(self):
        """Initialize text preprocessing components."""
        
        # Load stopwords
        if self.config.remove_stopwords and NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                try:
                    nltk.download('stopwords', quiet=True)
                    self.stopwords = set(stopwords.words('english'))
                except:
                    self.logger.warning("Could not load stopwords")
                    self.stopwords = set()
        
        # Initialize stemmer
        if self.config.apply_stemming and NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
        
        # Initialize lemmatizer
        if self.config.apply_lemmatization and NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Test if wordnet is available
                self.lemmatizer.lemmatize("test")
            except:
                try:
                    nltk.download('wordnet', quiet=True)
                    self.lemmatizer = WordNetLemmatizer()
                except:
                    self.logger.warning("Could not load lemmatizer")
                    self.lemmatizer = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for indexing and retrieval.
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Tokenization
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower() if self.config.lowercase else text)
            except:
                tokens = text.lower().split() if self.config.lowercase else text.split()
        else:
            tokens = text.lower().split() if self.config.lowercase else text.split()
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= self.config.min_token_length]
        
        # Remove stopwords
        if self.config.remove_stopwords and self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming
        if self.config.apply_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if self.config.apply_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def build_index(
        self,
        corpus: List[Union[str, Dict[str, Any]]],
        save_index: bool = True,
        index_name: str = "sparse_index",
        rebuild: bool = False
    ) -> bool:
        """
        Build sparse index from corpus.
        
        Args:
            corpus: List of documents (strings or dictionaries)
            save_index: Whether to save index to disk
            index_name: Name for saved index
            rebuild: Whether to rebuild existing index
            
        Returns:
            True if successful
        """
        self.logger.info(f"Building sparse index with {len(corpus)} documents")
        
        # Check for existing index
        index_path = self.index_dir / f"{index_name}.pkl"
        if index_path.exists() and not rebuild:
            self.logger.info("Loading existing index...")
            return self.load_index(index_name)
        
        # Normalize corpus
        self.corpus = []
        texts_to_index = []
        
        for i, doc in enumerate(corpus):
            if isinstance(doc, str):
                doc_text = doc
                doc_metadata = {"doc_id": i}
            elif isinstance(doc, dict):
                doc_text = doc.get('text', str(doc))
                doc_metadata = {k: v for k, v in doc.items() if k != 'text'}
                doc_metadata['doc_id'] = i
            else:
                doc_text = str(doc)
                doc_metadata = {"doc_id": i}
            
            self.corpus.append({
                'text': doc_text,
                'metadata': doc_metadata
            })
            texts_to_index.append(doc_text)
        
        # Preprocess documents
        self.processed_corpus = []
        for text in texts_to_index:
            tokens = self.preprocess_text(text)
            self.processed_corpus.append(tokens)
        
        # Build method-specific index
        start_time = time.time()
        
        if self.method == "bm25":
            self._build_bm25_index()
        elif self.method == "tfidf":
            self._build_tfidf_index(texts_to_index)
        elif self.method == "count":
            self._build_count_index()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Build inverted index for term matching
        self._build_inverted_index()
        
        # Compute statistics
        self._compute_statistics()
        
        build_time = time.time() - start_time
        self.is_indexed = True
        
        # Save index
        if save_index:
            self.save_index(index_name)
        
        self.logger.info(f"Sparse index built successfully in {build_time:.2f}s")
        return True
    
    def _build_bm25_index(self):
        """Build BM25 index."""
        
        if RANK_BM25_AVAILABLE:
            # Use rank_bm25 library
            self.bm25_index = BM25Okapi(self.processed_corpus, 
                                      k1=self.config.k1, 
                                      b=self.config.b, 
                                      epsilon=self.config.epsilon)
        else:
            # Manual BM25 implementation
            self.logger.info("rank_bm25 not available, using manual implementation")
            self._build_manual_bm25()
    
    def _build_manual_bm25(self):
        """Manual BM25 implementation."""
        
        # Compute document frequencies
        self.doc_frequencies = Counter()
        for doc_tokens in self.processed_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
        
        # Compute document lengths
        self.doc_lengths = [len(doc) for doc in self.processed_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Create vocabulary
        all_tokens = set()
        for doc_tokens in self.processed_corpus:
            all_tokens.update(doc_tokens)
        
        self.vocabulary = {token: i for i, token in enumerate(sorted(all_tokens))}
        
        # Compute IDF scores
        n_docs = len(self.processed_corpus)
        for token in self.vocabulary:
            df = self.doc_frequencies[token]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5))
            self.idf_scores[token] = max(idf, 0.25)  # Minimum IDF
    
    def _build_tfidf_index(self, texts: List[str]):
        """Build TF-IDF index using sklearn."""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for TF-IDF indexing")
        
        # Custom preprocessor to use our preprocessing
        def custom_preprocessor(text):
            tokens = self.preprocess_text(text)
            return ' '.join(tokens)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            max_df=self.config.max_df,
            min_df=self.config.min_df,
            preprocessor=custom_preprocessor,
            tokenizer=lambda x: x.split(),  # Already preprocessed
            lowercase=False  # Already lowercased in preprocessing
        )
        
        # Fit and transform corpus
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Store feature names for analysis
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
    
    def _build_count_index(self):
        """Build simple count-based index."""
        
        if not SKLEARN_AVAILABLE:
            # Manual count implementation
            self._build_manual_count()
        else:
            # Use sklearn CountVectorizer
            def custom_preprocessor(text):
                tokens = self.preprocess_text(text)
                return ' '.join(tokens)
            
            texts = [doc['text'] for doc in self.corpus]
            
            count_vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                max_df=self.config.max_df,
                min_df=self.config.min_df,
                preprocessor=custom_preprocessor,
                tokenizer=lambda x: x.split(),
                lowercase=False
            )
            
            self.tfidf_matrix = count_vectorizer.fit_transform(texts)
            self.tfidf_vectorizer = count_vectorizer
            self.feature_names = count_vectorizer.get_feature_names_out()
    
    def _build_manual_count(self):
        """Manual count-based implementation."""
        
        # Similar to manual BM25 but simpler
        self.doc_frequencies = Counter()
        for doc_tokens in self.processed_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
        
        # Create vocabulary
        all_tokens = set()
        for doc_tokens in self.processed_corpus:
            all_tokens.update(doc_tokens)
        
        self.vocabulary = {token: i for i, token in enumerate(sorted(all_tokens))}
    
    def _build_inverted_index(self):
        """Build inverted index for fast term lookup."""
        
        self.inverted_index = defaultdict(list)
        
        for doc_id, doc_tokens in enumerate(self.processed_corpus):
            for token in set(doc_tokens):  # Use set to avoid duplicates
                self.inverted_index[token].append(doc_id)
    
    def _compute_statistics(self):
        """Compute index statistics."""
        
        self.index_stats = {
            'num_documents': len(self.corpus),
            'num_tokens': sum(len(doc) for doc in self.processed_corpus),
            'avg_doc_length': np.mean([len(doc) for doc in self.processed_corpus]) if self.processed_corpus else 0,
            'vocabulary_size': len(self.vocabulary) if hasattr(self, 'vocabulary') else 0,
            'method': self.method
        }
        
        if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            self.index_stats['sparsity'] = 1.0 - (self.tfidf_matrix.nnz / self.tfidf_matrix.size)
    
    def save_index(self, index_name: str) -> bool:
        """Save index to disk."""
        
        try:
            index_data = {
                'corpus': self.corpus,
                'processed_corpus': self.processed_corpus,
                'method': self.method,
                'config': self.config,
                'inverted_index': dict(self.inverted_index),
                'index_stats': getattr(self, 'index_stats', {}),
                'created_at': datetime.now().isoformat()
            }
            
            # Method-specific data
            if self.method == "bm25":
                if RANK_BM25_AVAILABLE and self.bm25_index:
                    # Save BM25 parameters
                    index_data['bm25_params'] = {
                        'k1': self.config.k1,
                        'b': self.config.b,
                        'epsilon': self.config.epsilon
                    }
                    # Note: rank_bm25 objects are not directly serializable
                    # We'll rebuild on load
                else:
                    # Manual BM25 data
                    index_data.update({
                        'doc_frequencies': dict(self.doc_frequencies),
                        'doc_lengths': self.doc_lengths,
                        'avg_doc_length': self.avg_doc_length,
                        'vocabulary': self.vocabulary,
                        'idf_scores': self.idf_scores
                    })
            
            elif self.method in ["tfidf", "count"]:
                if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                    # Save vectorizer and matrix separately
                    vectorizer_path = self.index_dir / f"{index_name}_vectorizer.pkl"
                    with open(vectorizer_path, 'wb') as f:
                        pickle.dump(self.tfidf_vectorizer, f)
                    
                    matrix_path = self.index_dir / f"{index_name}_matrix.pkl"
                    with open(matrix_path, 'wb') as f:
                        pickle.dump(self.tfidf_matrix, f)
                    
                    index_data['has_sklearn_components'] = True
            
            # Save main index data
            index_path = self.index_dir / f"{index_name}.pkl"
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            self.logger.info(f"Index saved to {index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_name: str) -> bool:
        """Load index from disk."""
        
        try:
            # Load main index data
            index_path = self.index_dir / f"{index_name}.pkl"
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Restore basic data
            self.corpus = index_data['corpus']
            self.processed_corpus = index_data['processed_corpus']
            self.method = index_data['method']
            self.config = index_data.get('config', SparseRetrieverConfig(method=self.method))
            self.inverted_index = defaultdict(list, index_data.get('inverted_index', {}))
            self.index_stats = index_data.get('index_stats', {})
            
            # Method-specific restoration
            if self.method == "bm25":
                if 'bm25_params' in index_data and RANK_BM25_AVAILABLE:
                    # Rebuild BM25 index
                    self.bm25_index = BM25Okapi(
                        self.processed_corpus,
                        k1=index_data['bm25_params']['k1'],
                        b=index_data['bm25_params']['b'],
                        epsilon=index_data['bm25_params']['epsilon']
                    )
                else:
                    # Manual BM25 data
                    self.doc_frequencies = Counter(index_data.get('doc_frequencies', {}))
                    self.doc_lengths = index_data.get('doc_lengths', [])
                    self.avg_doc_length = index_data.get('avg_doc_length', 0)
                    self.vocabulary = index_data.get('vocabulary', {})
                    self.idf_scores = index_data.get('idf_scores', {})
            
            elif self.method in ["tfidf", "count"]:
                if index_data.get('has_sklearn_components') and SKLEARN_AVAILABLE:
                    # Load vectorizer and matrix
                    vectorizer_path = self.index_dir / f"{index_name}_vectorizer.pkl"
                    matrix_path = self.index_dir / f"{index_name}_matrix.pkl"
                    
                    if vectorizer_path.exists() and matrix_path.exists():
                        with open(vectorizer_path, 'rb') as f:
                            self.tfidf_vectorizer = pickle.load(f)
                        
                        with open(matrix_path, 'rb') as f:
                            self.tfidf_matrix = pickle.load(f)
                        
                        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            self.is_indexed = True
            self.logger.info(f"Index loaded successfully: {len(self.corpus)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_term_matches: bool = True
    ) -> List[SparseRetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            include_term_matches: Whether to include matching terms
            
        Returns:
            List of SparseRetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            return []
        
        # Method-specific retrieval
        if self.method == "bm25":
            scores = self._retrieve_bm25(query_tokens)
        elif self.method == "tfidf":
            scores = self._retrieve_tfidf(query, query_tokens)
        elif self.method == "count":
            scores = self._retrieve_count(query, query_tokens)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Get top-k results
        if isinstance(scores, list):
            # Scores for each document
            scored_docs = [(score, i) for i, score in enumerate(scores) if score > 0]
        else:
            # Sparse matrix or other format
            scored_docs = [(scores[i], i) for i in range(len(self.corpus)) if scores[i] > 0]
        
        # Sort by score and take top-k
        scored_docs.sort(reverse=True)
        top_docs = scored_docs[:top_k]
        
        # Build results
        results = []
        for score, doc_id in top_docs:
            doc = self.corpus[doc_id]
            
            # Find term matches if requested
            term_matches = []
            if include_term_matches:
                doc_tokens = set(self.processed_corpus[doc_id])
                term_matches = list(set(query_tokens) & doc_tokens)
            
            result = SparseRetrievalResult(
                text=doc['text'],
                score=float(score),
                doc_id=doc_id,
                term_matches=term_matches,
                metadata=doc['metadata']
            )
            results.append(result)
        
        return results
    
    def _retrieve_bm25(self, query_tokens: List[str]) -> List[float]:
        """Retrieve using BM25."""
        
        if RANK_BM25_AVAILABLE and self.bm25_index:
            return self.bm25_index.get_scores(query_tokens)
        else:
            # Manual BM25 scoring
            return self._manual_bm25_scores(query_tokens)
    
    def _manual_bm25_scores(self, query_tokens: List[str]) -> List[float]:
        """Manual BM25 scoring implementation."""
        
        scores = [0.0] * len(self.processed_corpus)
        
        for doc_id, doc_tokens in enumerate(self.processed_corpus):
            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            
            # Count term frequencies in document
            doc_term_counts = Counter(doc_tokens)
            
            for term in query_tokens:
                if term in self.vocabulary:
                    tf = doc_term_counts[term]
                    if tf > 0:
                        idf = self.idf_scores[term]
                        
                        # BM25 formula
                        numerator = tf * (self.config.k1 + 1)
                        denominator = tf + self.config.k1 * (
                            1 - self.config.b + self.config.b * doc_length / self.avg_doc_length
                        )
                        
                        score += idf * (numerator / denominator)
            
            scores[doc_id] = score
        
        return scores
    
    def _retrieve_tfidf(self, query: str, query_tokens: List[str]) -> List[float]:
        """Retrieve using TF-IDF."""
        
        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            # Fallback to simple term matching
            return self._simple_term_matching(query_tokens)
        
        # Transform query using fitted vectorizer
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Compute cosine similarity
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            return similarities.tolist()
        
        except Exception as e:
            self.logger.warning(f"TF-IDF retrieval failed: {e}")
            return self._simple_term_matching(query_tokens)
    
    def _retrieve_count(self, query: str, query_tokens: List[str]) -> List[float]:
        """Retrieve using count-based similarity."""
        
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None:
            # Use sklearn count vectorizer
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                return similarities.tolist()
            except Exception as e:
                self.logger.warning(f"Count-based retrieval failed: {e}")
        
        # Fallback to simple term matching
        return self._simple_term_matching(query_tokens)
    
    def _simple_term_matching(self, query_tokens: List[str]) -> List[float]:
        """Simple term overlap scoring."""
        
        scores = []
        
        for doc_tokens in self.processed_corpus:
            doc_token_set = set(doc_tokens)
            query_token_set = set(query_tokens)
            
            # Jaccard similarity
            intersection = len(doc_token_set & query_token_set)
            union = len(doc_token_set | query_token_set)
            
            score = intersection / union if union > 0 else 0.0
            scores.append(score)
        
        return scores
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        include_term_matches: bool = False
    ) -> List[List[SparseRetrievalResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            include_term_matches: Whether to include matching terms
            
        Returns:
            List of result lists, one per query
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        all_results = []
        
        for query in queries:
            results = self.retrieve(query, top_k, include_term_matches)
            all_results.append(results)
        
        return all_results
    
    def add_documents(
        self,
        new_docs: List[Union[str, Dict[str, Any]]],
        rebuild_index: bool = False
    ) -> bool:
        """
        Add new documents to existing index.
        
        Args:
            new_docs: New documents to add
            rebuild_index: Whether to rebuild entire index (vs incremental)
            
        Returns:
            True if successful
        """
        if not self.is_indexed:
            return self.build_index(new_docs)
        
        self.logger.info(f"Adding {len(new_docs)} documents to index")
        
        # Normalize new documents
        new_corpus_entries = []
        new_processed_docs = []
        
        start_doc_id = len(self.corpus)
        
        for i, doc in enumerate(new_docs):
            if isinstance(doc, str):
                doc_text = doc
                doc_metadata = {"doc_id": start_doc_id + i}
            elif isinstance(doc, dict):
                doc_text = doc.get('text', str(doc))
                doc_metadata = {k: v for k, v in doc.items() if k != 'text'}
                doc_metadata['doc_id'] = start_doc_id + i
            else:
                doc_text = str(doc)
                doc_metadata = {"doc_id": start_doc_id + i}
            
            new_corpus_entries.append({
                'text': doc_text,
                'metadata': doc_metadata
            })
            
            # Preprocess
            tokens = self.preprocess_text(doc_text)
            new_processed_docs.append(tokens)
        
        if rebuild_index:
            # Rebuild entire index
            all_docs = [doc['text'] for doc in self.corpus] + [doc['text'] for doc in new_corpus_entries]
            return self.build_index(all_docs, save_index=False)
        else:
            # Incremental update
            self.corpus.extend(new_corpus_entries)
            self.processed_corpus.extend(new_processed_docs)
            
            # Update inverted index
            for doc_id_offset, doc_tokens in enumerate(new_processed_docs):
                doc_id = start_doc_id + doc_id_offset
                for token in set(doc_tokens):
                    self.inverted_index[token].append(doc_id)
            
            # Method-specific updates
            if self.method == "bm25" and RANK_BM25_AVAILABLE:
                # Need to rebuild BM25 index
                self.bm25_index = BM25Okapi(self.processed_corpus, 
                                          k1=self.config.k1, 
                                          b=self.config.b, 
                                          epsilon=self.config.epsilon)
            elif self.method in ["tfidf", "count"]:
                # Need to rebuild sklearn components
                if SKLEARN_AVAILABLE:
                    all_texts = [doc['text'] for doc in self.corpus]
                    self._build_tfidf_index(all_texts)
            
            # Update statistics
            self._compute_statistics()
        
        self.logger.info(f"Successfully added {len(new_docs)} documents")
        return True
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about retriever configuration and status."""
        
        info = {
            'method': self.method,
            'is_indexed': self.is_indexed,
            'corpus_size': len(self.corpus),
            'config': {
                'k1': self.config.k1,
                'b': self.config.b,
                'max_features': self.config.max_features,
                'ngram_range': self.config.ngram_range,
                'remove_stopwords': self.config.remove_stopwords,
                'apply_stemming': self.config.apply_stemming
            },
            'components_available': {
                'rank_bm25': RANK_BM25_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE,
                'nltk': NLTK_AVAILABLE
            },
            'index_dir': str(self.index_dir)
        }
        
        if hasattr(self, 'index_stats'):
            info['index_stats'] = self.index_stats
        
        return info


def main():
    """Example usage of SparseRetriever."""
    
    # Initialize retriever with BM25
    retriever = SparseRetriever(method="bm25", config=SparseRetrieverConfig(
        k1=1.2, b=0.75, remove_stopwords=True
    ))
    
    print("=== SparseRetriever Example ===")
    print(f"Retriever info: {retriever.get_retriever_info()}")
    
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
    
    print(f"\nBuilding BM25 index with {len(corpus)} documents...")
    
    # Build index
    success = retriever.build_index(corpus, save_index=True, index_name="demo_sparse_index")
    
    if success:
        print("Index built successfully!")
        
        # Test queries
        test_queries = [
            "COVID vaccine effectiveness",
            "climate change causes fossil fuels",
            "artificial intelligence machine learning",
            "electric vehicles battery"
        ]
        
        print("\n=== Single Query Retrieval ===")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     Text: {result.text[:80]}...")
                if result.term_matches:
                    print(f"     Matches: {', '.join(result.term_matches[:5])}")
        
        # Test batch retrieval
        print("\n=== Batch Retrieval ===")
        batch_results = retriever.batch_retrieve(test_queries[:2], top_k=2)
        
        for query, results in zip(test_queries[:2], batch_results):
            print(f"\nQuery: {query}")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}, Doc ID: {result.doc_id}")
        
        # Test different methods
        print("\n=== Testing Different Methods ===")
        
        methods_to_test = ["bm25"]
        if SKLEARN_AVAILABLE:
            methods_to_test.extend(["tfidf", "count"])
        
        for method in methods_to_test:
            print(f"\nTesting {method.upper()}:")
            
            method_retriever = SparseRetriever(method=method)
            method_retriever.build_index(corpus[:5], save_index=False)  # Smaller corpus for speed
            
            results = method_retriever.retrieve("COVID vaccine", top_k=2)
            for result in results:
                print(f"  Score: {result.score:.3f}, Text: {result.text[:60]}...")
        
        # Test adding documents
        print("\n=== Adding New Documents ===")
        new_docs = [
            "Quantum computing could revolutionize cryptography and optimization problems in the future.",
            "Remote work has become more common since the COVID-19 pandemic changed workplace dynamics."
        ]
        
        add_success = retriever.add_documents(new_docs, rebuild_index=True)
        if add_success:
            print(f"Added {len(new_docs)} new documents")
            print(f"Updated corpus size: {len(retriever.corpus)}")
            
            # Test retrieval with new documents
            results = retriever.retrieve("quantum computing remote work", top_k=3)
            print("\nRetrieval after adding documents:")
            for result in results:
                print(f"  Score: {result.score:.3f}, Text: {result.text[:70]}...")
        
        # Test index persistence
        print("\n=== Index Persistence Test ===")
        retriever.save_index("demo_sparse_index")
        
        # Create new retriever and load index
        new_retriever = SparseRetriever(method="bm25")
        load_success = new_retriever.load_index("demo_sparse_index")
        
        if load_success:
            print("Index loaded successfully in new retriever instance")
            
            # Test retrieval with loaded index
            results = new_retriever.retrieve("vaccine", top_k=2)
            print("Retrieval with loaded index:")
            for result in results:
                print(f"  Score: {result.score:.3f}, Text: {result.text[:70]}...")
    
    else:
        print("Failed to build index")


if __name__ == "__main__":
    main()
