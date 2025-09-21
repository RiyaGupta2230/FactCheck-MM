#!/usr/bin/env python3
"""
Dense Passage Retrieval

Transformer-based dense retrieval using pre-trained models like DPR and ColBERT
for semantic similarity-based evidence retrieval with FAISS indexing support.

Example Usage:
    >>> from fact_verification.retrieval import DenseRetriever
    >>> 
    >>> # Initialize with DPR model
    >>> retriever = DenseRetriever(
    ...     model_name="facebook/dpr-ctx_encoder-single-nq-base",
    ...     enable_gpu=True
    ... )
    >>> 
    >>> # Build index from corpus
    >>> corpus = ["Document 1 text", "Document 2 text", ...]
    >>> retriever.build_index(corpus, save_index=True)
    >>> 
    >>> # Retrieve relevant passages
    >>> results = retriever.retrieve("COVID vaccines effectiveness", top_k=5)
    >>> for result in results:
    ...     print(f"Score: {result['score']:.3f}, Text: {result['text'][:100]}...")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional dependencies
try:
    from transformers import (
        AutoModel, AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder,
        DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    
    text: str
    score: float
    doc_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseRetrieverConfig:
    """Configuration for dense retriever."""
    
    model_name: str = "facebook/dpr-ctx_encoder-single-nq-base"
    query_model_name: Optional[str] = None  # If different from context encoder
    max_seq_length: int = 512
    embedding_dim: int = 768
    batch_size: int = 16
    enable_gpu: bool = True
    normalize_embeddings: bool = True
    pooling_strategy: str = "cls"  # cls, mean, max
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    faiss_nprobe: int = 32


class DenseRetriever:
    """
    Dense passage retrieval using transformer-based encoders.
    
    Supports various architectures including DPR, BERT-based models, and
    sentence transformers with FAISS indexing for efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        query_model_name: Optional[str] = None,
        config: Optional[DenseRetrieverConfig] = None,
        index_dir: str = "data/indexes/dense",
        enable_gpu: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Context encoder model name
            query_model_name: Query encoder model name (if different)
            config: Retriever configuration
            index_dir: Directory for storing indexes
            enable_gpu: Enable GPU acceleration
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("DenseRetriever")
        
        # Configuration
        self.config = config or DenseRetrieverConfig(
            model_name=model_name,
            query_model_name=query_model_name,
            enable_gpu=enable_gpu
        )
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and enable_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Index directory
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.context_encoder = None
        self.query_encoder = None
        self.context_tokenizer = None
        self.query_tokenizer = None
        
        self._load_models()
        
        # Index components
        self.faiss_index = None
        self.corpus = []
        self.doc_embeddings = None
        self.is_indexed = False
        
        # Knowledge base integration
        self.kb_connector = None
        self._try_load_kb_connector()
        
        self.logger.info(f"Initialized DenseRetriever with model: {model_name}")
    
    def _load_models(self):
        """Load transformer models and tokenizers."""
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for DenseRetriever")
        
        model_name = self.config.model_name
        query_model_name = self.config.query_model_name or model_name
        
        try:
            # Check if it's a DPR model
            if 'dpr' in model_name.lower():
                self.context_encoder = DPRContextEncoder.from_pretrained(model_name)
                self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
                
                if 'ctx_encoder' in model_name and query_model_name == model_name:
                    # Load corresponding question encoder
                    question_model = model_name.replace('ctx_encoder', 'question_encoder')
                    try:
                        self.query_encoder = DPRQuestionEncoder.from_pretrained(question_model)
                        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model)
                    except:
                        # Fallback to same model for both
                        self.query_encoder = self.context_encoder
                        self.query_tokenizer = self.context_tokenizer
                else:
                    self.query_encoder = DPRQuestionEncoder.from_pretrained(query_model_name)
                    self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_model_name)
            
            # Try sentence-transformers
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer(model_name, device=str(self.device))
                self.context_encoder = self.sentence_model
                self.query_encoder = self.sentence_model
                self.logger.info("Using SentenceTransformer model")
            
            # Generic transformer model
            else:
                self.context_encoder = AutoModel.from_pretrained(model_name)
                self.context_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.query_encoder = self.context_encoder
                self.query_tokenizer = self.context_tokenizer
            
            # Move to device
            if hasattr(self.context_encoder, 'to'):
                self.context_encoder.to(self.device)
            if hasattr(self.query_encoder, 'to') and self.query_encoder != self.context_encoder:
                self.query_encoder.to(self.device)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def _try_load_kb_connector(self):
        """Try to load knowledge base connector for enhanced retrieval."""
        
        try:
            from fact_verification.utils.knowledge_bases import KnowledgeBaseConnector
            self.kb_connector = KnowledgeBaseConnector(enable_cache=True)
            self.logger.info("Knowledge base connector loaded")
        except Exception as e:
            self.logger.warning(f"Knowledge base connector not available: {e}")
    
    def encode_texts(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode texts into dense vectors.
        
        Args:
            texts: List of texts to encode
            is_query: Whether texts are queries (vs documents)
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.config.batch_size
        
        # Use sentence-transformers if available
        if hasattr(self, 'sentence_model'):
            embeddings = self.sentence_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            return embeddings
        
        # Use DPR or generic transformer
        encoder = self.query_encoder if is_query else self.context_encoder
        tokenizer = self.query_tokenizer if is_query else self.context_tokenizer
        
        if encoder is None or tokenizer is None:
            raise ValueError("Models not properly initialized")
        
        embeddings = []
        encoder.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = encoder(**inputs)
                
                if hasattr(outputs, 'pooler_output'):
                    # BERT-style models
                    batch_embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Extract embeddings based on pooling strategy
                    hidden_states = outputs.last_hidden_state
                    
                    if self.config.pooling_strategy == 'cls':
                        batch_embeddings = hidden_states[:, 0, :]  # CLS token
                    elif self.config.pooling_strategy == 'mean':
                        # Mean pooling with attention mask
                        attention_mask = inputs['attention_mask'].unsqueeze(-1)
                        batch_embeddings = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                    elif self.config.pooling_strategy == 'max':
                        batch_embeddings = torch.max(hidden_states, dim=1)[0]
                    else:
                        batch_embeddings = hidden_states[:, 0, :]  # Default to CLS
                else:
                    # DPR models
                    batch_embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def build_index(
        self,
        corpus: List[Union[str, Dict[str, Any]]],
        save_index: bool = True,
        index_name: str = "dense_index",
        rebuild: bool = False
    ) -> bool:
        """
        Build FAISS index from corpus.
        
        Args:
            corpus: List of documents (strings or dictionaries)
            save_index: Whether to save index to disk
            index_name: Name for saved index
            rebuild: Whether to rebuild existing index
            
        Returns:
            True if successful
        """
        self.logger.info(f"Building dense index with {len(corpus)} documents")
        
        # Check for existing index
        index_path = self.index_dir / f"{index_name}.faiss"
        metadata_path = self.index_dir / f"{index_name}_metadata.json"
        
        if index_path.exists() and not rebuild:
            self.logger.info("Loading existing index...")
            return self.load_index(index_name)
        
        # Normalize corpus
        self.corpus = []
        texts_to_encode = []
        
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
            texts_to_encode.append(doc_text)
        
        # Encode documents
        start_time = time.time()
        self.doc_embeddings = self.encode_texts(texts_to_encode, is_query=False)
        encoding_time = time.time() - start_time
        
        self.logger.info(f"Encoded {len(texts_to_encode)} documents in {encoding_time:.2f}s")
        
        # Build FAISS index
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, using simple cosine similarity")
            self.faiss_index = None
        else:
            self._build_faiss_index()
        
        self.is_indexed = True
        
        # Save index
        if save_index:
            self.save_index(index_name)
        
        self.logger.info("Dense index built successfully")
        return True
    
    def _build_faiss_index(self):
        """Build FAISS index from embeddings."""
        
        embedding_dim = self.doc_embeddings.shape[1]
        num_docs = self.doc_embeddings.shape[0]
        
        if self.config.faiss_index_type == "flat":
            # Flat index for exact search
            if self.config.normalize_embeddings:
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
            else:
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        
        elif self.config.faiss_index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(int(np.sqrt(num_docs)), 1000)  # Number of clusters
            
            if self.config.normalize_embeddings:
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            else:
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            
            # Train the index
            self.faiss_index.train(self.doc_embeddings.astype(np.float32))
            self.faiss_index.nprobe = min(self.config.faiss_nprobe, nlist)
        
        elif self.config.faiss_index_type == "hnsw":
            # HNSW index for very fast approximate search
            M = 16  # Number of connections
            self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, M)
            if not self.config.normalize_embeddings:
                self.faiss_index.metric_type = faiss.METRIC_L2
        
        else:
            # Default to flat index
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to index
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))
        
        self.logger.info(f"Built {self.config.faiss_index_type} FAISS index with {num_docs} documents")
    
    def save_index(self, index_name: str) -> bool:
        """Save index to disk."""
        
        try:
            # Save FAISS index
            if self.faiss_index:
                index_path = self.index_dir / f"{index_name}.faiss"
                faiss.write_index(self.faiss_index, str(index_path))
            
            # Save embeddings
            embeddings_path = self.index_dir / f"{index_name}_embeddings.npy"
            if self.doc_embeddings is not None:
                np.save(embeddings_path, self.doc_embeddings)
            
            # Save corpus and metadata
            metadata = {
                'corpus_size': len(self.corpus),
                'embedding_dim': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
                'config': {
                    'model_name': self.config.model_name,
                    'max_seq_length': self.config.max_seq_length,
                    'normalize_embeddings': self.config.normalize_embeddings,
                    'faiss_index_type': self.config.faiss_index_type,
                    'embedding_dim': self.config.embedding_dim
                },
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.index_dir / f"{index_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save corpus
            corpus_path = self.index_dir / f"{index_name}_corpus.pkl"
            with open(corpus_path, 'wb') as f:
                pickle.dump(self.corpus, f)
            
            self.logger.info(f"Index saved to {self.index_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_name: str) -> bool:
        """Load index from disk."""
        
        try:
            # Load metadata
            metadata_path = self.index_dir / f"{index_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Loading index created at {metadata['created_at']}")
            
            # Load FAISS index
            index_path = self.index_dir / f"{index_name}.faiss"
            if index_path.exists() and FAISS_AVAILABLE:
                self.faiss_index = faiss.read_index(str(index_path))
            
            # Load embeddings
            embeddings_path = self.index_dir / f"{index_name}_embeddings.npy"
            if embeddings_path.exists():
                self.doc_embeddings = np.load(embeddings_path)
            
            # Load corpus
            corpus_path = self.index_dir / f"{index_name}_corpus.pkl"
            if corpus_path.exists():
                with open(corpus_path, 'rb') as f:
                    self.corpus = pickle.load(f)
            
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
        include_scores: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            include_scores: Whether to include similarity scores
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encode_texts([query], is_query=True)[0]
        
        if self.faiss_index:
            # Use FAISS for search
            scores, doc_ids = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                top_k
            )
            scores = scores[0]
            doc_ids = doc_ids[0]
        else:
            # Fallback to numpy cosine similarity
            if self.config.normalize_embeddings:
                similarities = np.dot(self.doc_embeddings, query_embedding)
            else:
                # Compute cosine similarity manually
                query_norm = np.linalg.norm(query_embedding)
                doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
                similarities = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[top_indices]
            doc_ids = top_indices
        
        # Build results
        results = []
        for score, doc_id in zip(scores, doc_ids):
            if doc_id < len(self.corpus):
                doc = self.corpus[doc_id]
                
                result = RetrievalResult(
                    text=doc['text'],
                    score=float(score) if include_scores else 0.0,
                    doc_id=int(doc_id),
                    metadata=doc['metadata']
                )
                results.append(result)
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        batch_size: Optional[int] = None
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            batch_size: Batch size for query encoding
            
        Returns:
            List of result lists, one per query
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode all queries
        query_embeddings = self.encode_texts(queries, is_query=True, batch_size=batch_size)
        
        all_results = []
        
        if self.faiss_index:
            # Batch search with FAISS
            scores_batch, doc_ids_batch = self.faiss_index.search(
                query_embeddings.astype(np.float32),
                top_k
            )
            
            for i, (scores, doc_ids) in enumerate(zip(scores_batch, doc_ids_batch)):
                results = []
                for score, doc_id in zip(scores, doc_ids):
                    if doc_id < len(self.corpus):
                        doc = self.corpus[doc_id]
                        result = RetrievalResult(
                            text=doc['text'],
                            score=float(score),
                            doc_id=int(doc_id),
                            metadata=doc['metadata']
                        )
                        results.append(result)
                
                all_results.append(results)
        
        else:
            # Fallback batch computation
            for query_embedding in query_embeddings:
                if self.config.normalize_embeddings:
                    similarities = np.dot(self.doc_embeddings, query_embedding)
                else:
                    query_norm = np.linalg.norm(query_embedding)
                    doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
                    similarities = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
                
                top_indices = np.argsort(similarities)[::-1][:top_k]
                scores = similarities[top_indices]
                
                results = []
                for score, doc_id in zip(scores, top_indices):
                    doc = self.corpus[doc_id]
                    result = RetrievalResult(
                        text=doc['text'],
                        score=float(score),
                        doc_id=int(doc_id),
                        metadata=doc['metadata']
                    )
                    results.append(result)
                
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
        texts_to_encode = []
        
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
            texts_to_encode.append(doc_text)
        
        # Encode new documents
        new_embeddings = self.encode_texts(texts_to_encode, is_query=False)
        
        if rebuild_index:
            # Rebuild entire index
            self.corpus.extend(new_corpus_entries)
            all_texts = [doc['text'] for doc in self.corpus]
            self.doc_embeddings = self.encode_texts(all_texts, is_query=False)
            
            if FAISS_AVAILABLE:
                self._build_faiss_index()
        else:
            # Incremental update
            self.corpus.extend(new_corpus_entries)
            
            if self.doc_embeddings is not None:
                self.doc_embeddings = np.vstack([self.doc_embeddings, new_embeddings])
            else:
                self.doc_embeddings = new_embeddings
            
            # Add to FAISS index if available
            if self.faiss_index:
                self.faiss_index.add(new_embeddings.astype(np.float32))
        
        self.logger.info(f"Successfully added {len(new_docs)} documents")
        return True
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about retriever configuration and status."""
        
        return {
            'model_name': self.config.model_name,
            'device': str(self.device),
            'is_indexed': self.is_indexed,
            'corpus_size': len(self.corpus),
            'embedding_dim': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
            'faiss_available': FAISS_AVAILABLE,
            'faiss_index_type': self.config.faiss_index_type,
            'index_dir': str(self.index_dir),
            'kb_connector_available': self.kb_connector is not None,
            'config': {
                'max_seq_length': self.config.max_seq_length,
                'batch_size': self.config.batch_size,
                'normalize_embeddings': self.config.normalize_embeddings,
                'pooling_strategy': self.config.pooling_strategy
            }
        }


def main():
    """Example usage of DenseRetriever."""
    
    # Initialize retriever
    retriever = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight model for demo
        enable_gpu=False  # Use CPU for demo
    )
    
    print("=== DenseRetriever Example ===")
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
        "Machine learning algorithms can be trained to recognize patterns in large datasets."
    ]
    
    print(f"\nBuilding index with {len(corpus)} documents...")
    
    # Build index
    success = retriever.build_index(corpus, save_index=True, index_name="demo_index")
    
    if success:
        print("Index built successfully!")
        
        # Test queries
        test_queries = [
            "COVID vaccine effectiveness",
            "causes of climate change",
            "artificial intelligence advances"
        ]
        
        print("\n=== Single Query Retrieval ===")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     Text: {result.text[:100]}...")
        
        # Test batch retrieval
        print("\n=== Batch Retrieval ===")
        batch_results = retriever.batch_retrieve(test_queries, top_k=2)
        
        for query, results in zip(test_queries, batch_results):
            print(f"\nQuery: {query}")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}, Doc ID: {result.doc_id}")
        
        # Test adding new documents
        print("\n=== Adding New Documents ===")
        new_docs = [
            "Electric vehicles are becoming more popular as battery technology improves.",
            "Space exploration has led to many technological innovations that benefit life on Earth."
        ]
        
        add_success = retriever.add_documents(new_docs)
        if add_success:
            print(f"Added {len(new_docs)} new documents")
            print(f"Updated corpus size: {len(retriever.corpus)}")
            
            # Test retrieval with new documents
            results = retriever.retrieve("electric cars and space technology", top_k=3)
            print("\nRetrieval after adding documents:")
            for result in results:
                print(f"  Score: {result.score:.3f}, Text: {result.text[:80]}...")
        
        # Test index saving/loading
        print("\n=== Index Persistence Test ===")
        retriever.save_index("demo_index")
        
        # Create new retriever and load index
        new_retriever = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            enable_gpu=False
        )
        
        load_success = new_retriever.load_index("demo_index")
        if load_success:
            print("Index loaded successfully in new retriever instance")
            
            # Test retrieval with loaded index
            results = new_retriever.retrieve("vaccine", top_k=2)
            print("Retrieval with loaded index:")
            for result in results:
                print(f"  Score: {result.score:.3f}, Text: {result.text[:80]}...")
    
    else:
        print("Failed to build index")


if __name__ == "__main__":
    main()
