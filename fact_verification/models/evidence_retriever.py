#!/usr/bin/env python3
"""
Evidence Retrieval Model for Fact Verification

Implements dual-mode evidence retrieval with dense retrieval (DPR-style dual encoder)
and sparse retrieval (BM25) capabilities, plus hybrid fusion for comprehensive
evidence discovery in fact-checking pipelines.

Example Usage:
    >>> from fact_verification.models import EvidenceRetriever
    >>> 
    >>> # Initialize with hybrid retrieval
    >>> retriever = EvidenceRetriever(mode='hybrid')
    >>> 
    >>> # Retrieve evidence for claim
    >>> claim = "COVID-19 vaccines are effective against severe illness"
    >>> evidence = retriever.retrieve(claim, top_k=5)
    >>> 
    >>> for i, ev in enumerate(evidence):
    ...     print(f"{i+1}. {ev['text'][:100]}... (score: {ev['score']:.3f})")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import numpy as np
import json
from dataclasses import dataclass, field
import pickle

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.preprocessing.text_processor import TextProcessor, TextProcessorConfig
from shared.utils.logging_utils import get_logger

# Optional imports for sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class EvidenceRetrieverConfig:
    """Configuration for evidence retrieval model."""
    
    # Model architecture
    model_name: str = "roberta-base"
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # Retrieval configuration
    mode: str = "hybrid"  # "dense", "sparse", "hybrid"
    max_query_length: int = 128
    max_evidence_length: int = 256
    
    # Dense retrieval (DPR-style)
    projection_dim: int = 256  # Dimension for similarity computation
    temperature: float = 0.05  # Temperature for contrastive learning
    normalize_embeddings: bool = True
    
    # Sparse retrieval (BM25)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # Hybrid fusion
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    # Evidence corpus
    evidence_corpus_path: Optional[str] = None
    max_corpus_size: int = 1000000
    
    # Performance
    batch_size: int = 32
    index_batch_size: int = 256
    use_gpu_for_search: bool = True
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: str = "cache/evidence_retrieval"


class DensePassageRetriever(nn.Module):
    """Dense Passage Retrieval (DPR) style dual encoder."""
    
    def __init__(self, config: EvidenceRetrieverConfig):
        super().__init__()
        
        self.config = config
        
        # Load pre-trained models
        self.roberta_config = RobertaConfig.from_pretrained(config.model_name)
        
        # Query encoder
        self.query_encoder = RobertaModel.from_pretrained(
            config.model_name, config=self.roberta_config
        )
        
        # Context encoder (evidence)
        self.context_encoder = RobertaModel.from_pretrained(
            config.model_name, config=self.roberta_config
        )
        
        # Projection layers
        if config.projection_dim != config.hidden_size:
            self.query_projection = nn.Linear(config.hidden_size, config.projection_dim)
            self.context_projection = nn.Linear(config.hidden_size, config.projection_dim)
        else:
            self.query_projection = nn.Identity()
            self.context_projection = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def encode_query(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode query (claim) into dense representation."""
        
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Project to target dimension
        query_embedding = self.query_projection(pooled_output)
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        
        return query_embedding
    
    def encode_context(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode context (evidence) into dense representation."""
        
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Project to target dimension
        context_embedding = self.context_projection(pooled_output)
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            context_embedding = F.normalize(context_embedding, p=2, dim=-1)
        
        return context_embedding
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training with contrastive loss."""
        
        # Encode queries and contexts
        query_embeddings = self.encode_query(query_input_ids, query_attention_mask)
        context_embeddings = self.encode_context(context_input_ids, context_attention_mask)
        
        # Compute similarity scores
        similarity_scores = torch.matmul(query_embeddings, context_embeddings.t()) / self.config.temperature
        
        outputs = {
            'query_embeddings': query_embeddings,
            'context_embeddings': context_embeddings,
            'similarity_scores': similarity_scores
        }
        
        # Compute contrastive loss if labels provided
        if labels is not None:
            # In-batch negatives contrastive loss
            batch_size = query_embeddings.size(0)
            labels = torch.arange(batch_size, device=query_embeddings.device)
            
            loss = F.cross_entropy(similarity_scores, labels)
            outputs['loss'] = loss
        
        return outputs


class EvidenceRetriever(BaseMultimodalModel):
    """
    Comprehensive evidence retrieval system with dense, sparse, and hybrid modes.
    
    Supports DPR-style dual encoder for dense retrieval, BM25 for sparse retrieval,
    and fusion strategies for combining multiple retrieval approaches.
    """
    
    def __init__(self, config: Optional[EvidenceRetrieverConfig] = None):
        """
        Initialize evidence retriever.
        
        Args:
            config: Retrieval configuration
        """
        super().__init__()
        
        self.config = config or EvidenceRetrieverConfig()
        self.logger = get_logger("EvidenceRetriever")
        
        # Initialize text processor
        processor_config = TextProcessorConfig(
            model_name=self.config.model_name,
            max_length=max(self.config.max_query_length, self.config.max_evidence_length)
        )
        self.text_processor = TextProcessor(processor_config)
        self.tokenizer = self.text_processor.tokenizer
        
        # Initialize retrieval components based on mode
        self.dense_retriever = None
        self.sparse_retriever = None
        self.evidence_corpus = []
        self.dense_index = None
        
        if self.config.mode in ["dense", "hybrid"]:
            self.dense_retriever = DensePassageRetriever(self.config)
            self._setup_dense_index()
        
        if self.config.mode in ["sparse", "hybrid"]:
            self._setup_sparse_retriever()
        
        # Load evidence corpus
        self._load_evidence_corpus()
        
        self.logger.info(f"Initialized EvidenceRetriever in '{self.config.mode}' mode")
        self.logger.info(f"Evidence corpus size: {len(self.evidence_corpus):,}")
    
    def _setup_dense_index(self):
        """Setup FAISS index for dense retrieval."""
        
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available for dense retrieval")
            return
        
        # Initialize FAISS index
        if self.config.use_gpu_for_search and torch.cuda.is_available():
            # GPU index
            self.dense_index = faiss.IndexFlatIP(self.config.projection_dim)
            self.dense_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.dense_index)
        else:
            # CPU index
            self.dense_index = faiss.IndexFlatIP(self.config.projection_dim)
        
        self.logger.info(f"Setup FAISS index with dimension {self.config.projection_dim}")
    
    def _setup_sparse_retriever(self):
        """Setup BM25 sparse retriever."""
        
        if not BM25_AVAILABLE:
            self.logger.warning("BM25 not available for sparse retrieval")
            self.sparse_retriever = None
            return
        
        # BM25 will be initialized when evidence corpus is loaded
        pass
    
    def _load_evidence_corpus(self):
        """Load evidence corpus for retrieval."""
        
        if self.config.evidence_corpus_path:
            corpus_path = Path(self.config.evidence_corpus_path)
            
            if corpus_path.exists():
                try:
                    if corpus_path.suffix == '.json':
                        with open(corpus_path, 'r') as f:
                            self.evidence_corpus = json.load(f)
                    elif corpus_path.suffix == '.jsonl':
                        with open(corpus_path, 'r') as f:
                            self.evidence_corpus = [json.loads(line) for line in f]
                    else:
                        # Assume text file with one evidence per line
                        with open(corpus_path, 'r') as f:
                            lines = f.readlines()
                            self.evidence_corpus = [
                                {'id': i, 'text': line.strip()} 
                                for i, line in enumerate(lines) if line.strip()
                            ]
                    
                    # Limit corpus size if configured
                    if len(self.evidence_corpus) > self.config.max_corpus_size:
                        self.evidence_corpus = self.evidence_corpus[:self.config.max_corpus_size]
                    
                    self.logger.info(f"Loaded {len(self.evidence_corpus):,} evidence passages")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load evidence corpus: {e}")
                    self._create_dummy_corpus()
            else:
                self.logger.warning(f"Evidence corpus not found at {corpus_path}")
                self._create_dummy_corpus()
        else:
            self._create_dummy_corpus()
        
        # Initialize retrievers with corpus
        if self.dense_retriever and self.dense_index and self.evidence_corpus:
            self._build_dense_index()
        
        if self.config.mode in ["sparse", "hybrid"] and BM25_AVAILABLE and self.evidence_corpus:
            self._build_sparse_index()
    
    def _create_dummy_corpus(self):
        """Create dummy evidence corpus for testing."""
        
        dummy_evidence = [
            {"id": 0, "text": "COVID-19 vaccines have been shown to be highly effective in preventing severe illness, hospitalization, and death."},
            {"id": 1, "text": "Clinical trials demonstrate that approved COVID-19 vaccines have efficacy rates of 70-95% against symptomatic infection."},
            {"id": 2, "text": "The Earth is the third planet from the Sun and the only known planet to harbor life."},
            {"id": 3, "text": "Climate change refers to long-term shifts in global temperatures and weather patterns."},
            {"id": 4, "text": "Artificial intelligence has applications in healthcare, transportation, and scientific research."}
        ]
        
        self.evidence_corpus = dummy_evidence
        self.logger.info("Using dummy evidence corpus for testing")
    
    def _build_dense_index(self):
        """Build dense retrieval index from evidence corpus."""
        
        if not self.dense_retriever or not self.dense_index:
            return
        
        self.logger.info("Building dense retrieval index...")
        
        # Check for cached embeddings
        cache_path = Path(self.config.cache_dir) / "dense_embeddings.pkl"
        
        if self.config.cache_embeddings and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    evidence_embeddings = pickle.load(f)
                self.logger.info("Loaded cached evidence embeddings")
            except:
                evidence_embeddings = self._compute_evidence_embeddings()
                self._cache_embeddings(evidence_embeddings, cache_path)
        else:
            evidence_embeddings = self._compute_evidence_embeddings()
            if self.config.cache_embeddings:
                self._cache_embeddings(evidence_embeddings, cache_path)
        
        # Add to FAISS index
        if evidence_embeddings is not None:
            self.dense_index.add(evidence_embeddings.astype('float32'))
            self.logger.info(f"Added {evidence_embeddings.shape[0]} embeddings to dense index")
    
    def _compute_evidence_embeddings(self) -> Optional[np.ndarray]:
        """Compute embeddings for all evidence passages."""
        
        if not self.dense_retriever:
            return None
        
        self.dense_retriever.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(self.evidence_corpus), self.config.index_batch_size):
                batch = self.evidence_corpus[i:i + self.config.index_batch_size]
                batch_texts = [item['text'] for item in batch]
                
                # Process batch
                inputs = self.text_processor.process_batch(
                    batch_texts,
                    max_length=self.config.max_evidence_length,
                    padding=True,
                    truncation=True
                )
                
                # Move to device
                device = next(self.dense_retriever.parameters()).device
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                
                # Encode
                embeddings = self.dense_retriever.encode_context(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        return None
    
    def _cache_embeddings(self, embeddings: np.ndarray, cache_path: Path):
        """Cache computed embeddings."""
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            self.logger.info(f"Cached embeddings to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache embeddings: {e}")
    
    def _build_sparse_index(self):
        """Build BM25 sparse retrieval index."""
        
        if not BM25_AVAILABLE:
            return
        
        # Prepare corpus for BM25
        corpus_texts = [item['text'] for item in self.evidence_corpus]
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        
        # Initialize BM25
        self.sparse_retriever = BM25Okapi(
            tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        
        self.logger.info(f"Built BM25 index with {len(tokenized_corpus)} documents")
    
    def retrieve_dense(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence using dense retrieval."""
        
        if not self.dense_retriever or not self.dense_index:
            return []
        
        self.dense_retriever.eval()
        
        with torch.no_grad():
            # Process query
            inputs = self.text_processor.process_text(
                query,
                max_length=self.config.max_query_length,
                truncation=True
            )
            
            # Move to device
            device = next(self.dense_retriever.parameters()).device
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0).to(device)
            
            # Encode query
            query_embedding = self.dense_retriever.encode_query(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            # Search
            query_vector = query_embedding.cpu().numpy().astype('float32')
            scores, indices = self.dense_index.search(query_vector, top_k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.evidence_corpus):
                    evidence = self.evidence_corpus[idx]
                    results.append({
                        'id': evidence.get('id', idx),
                        'text': evidence['text'],
                        'score': float(score),
                        'rank': i + 1,
                        'retrieval_type': 'dense'
                    })
        
        return results
    
    def retrieve_sparse(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence using sparse (BM25) retrieval."""
        
        if not self.sparse_retriever:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.sparse_retriever.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.evidence_corpus):
                evidence = self.evidence_corpus[idx]
                results.append({
                    'id': evidence.get('id', idx),
                    'text': evidence['text'],
                    'score': float(scores[idx]),
                    'rank': i + 1,
                    'retrieval_type': 'sparse'
                })
        
        return results
    
    def retrieve_hybrid(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence using hybrid dense + sparse retrieval."""
        
        # Get results from both retrievers
        dense_results = self.retrieve_dense(query, top_k * 2)  # Get more for fusion
        sparse_results = self.retrieve_sparse(query, top_k * 2)
        
        # Normalize scores to [0, 1] range
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            min_dense = min(r['score'] for r in dense_results)
            if max_dense > min_dense:
                for r in dense_results:
                    r['normalized_score'] = (r['score'] - min_dense) / (max_dense - min_dense)
            else:
                for r in dense_results:
                    r['normalized_score'] = 1.0
        
        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            min_sparse = min(r['score'] for r in sparse_results)
            if max_sparse > min_sparse:
                for r in sparse_results:
                    r['normalized_score'] = (r['score'] - min_sparse) / (max_sparse - min_sparse)
            else:
                for r in sparse_results:
                    r['normalized_score'] = 1.0
        
        # Combine results with weighted fusion
        combined_scores = {}
        
        # Add dense scores
        for result in dense_results:
            doc_id = result['id']
            combined_scores[doc_id] = {
                'dense_score': result['normalized_score'],
                'sparse_score': 0.0,
                'evidence': result
            }
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['sparse_score'] = result['normalized_score']
            else:
                combined_scores[doc_id] = {
                    'dense_score': 0.0,
                    'sparse_score': result['normalized_score'],
                    'evidence': result
                }
        
        # Compute hybrid scores
        hybrid_results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = (
                self.config.dense_weight * scores['dense_score'] +
                self.config.sparse_weight * scores['sparse_score']
            )
            
            evidence = scores['evidence']
            evidence['score'] = hybrid_score
            evidence['dense_score'] = scores['dense_score']
            evidence['sparse_score'] = scores['sparse_score']
            evidence['retrieval_type'] = 'hybrid'
            
            hybrid_results.append(evidence)
        
        # Sort by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results[:top_k]):
            result['rank'] = i + 1
        
        return hybrid_results[:top_k]
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval interface.
        
        Args:
            query: Query text (claim)
            top_k: Number of results to return
            mode: Retrieval mode override
            
        Returns:
            List of retrieved evidence with scores and metadata
        """
        retrieval_mode = mode or self.config.mode
        
        if retrieval_mode == "dense":
            return self.retrieve_dense(query, top_k)
        elif retrieval_mode == "sparse":
            return self.retrieve_sparse(query, top_k)
        elif retrieval_mode == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training dense retriever."""
        
        if self.dense_retriever:
            return self.dense_retriever(
                query_input_ids, query_attention_mask,
                context_input_ids, context_attention_mask,
                labels
            )
        else:
            raise ValueError("Dense retriever not initialized")
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration."""
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save dense retriever if available
        if self.dense_retriever:
            torch.save(self.dense_retriever.state_dict(), save_path / "dense_retriever.bin")
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save evidence corpus metadata
        corpus_info = {
            'corpus_size': len(self.evidence_corpus),
            'sample_evidence': self.evidence_corpus[:3] if self.evidence_corpus else []
        }
        
        with open(save_path / "corpus_info.json", 'w') as f:
            json.dump(corpus_info, f, indent=2)
        
        self.logger.info(f"EvidenceRetriever saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[EvidenceRetrieverConfig] = None
    ) -> 'EvidenceRetriever':
        """Load model from pretrained checkpoint."""
        
        model_path = Path(model_path)
        
        # Load configuration
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = EvidenceRetrieverConfig(**config_dict)
            else:
                config = EvidenceRetrieverConfig()
        
        # Create model instance
        model = cls(config)
        
        # Load dense retriever weights if available
        dense_weights_path = model_path / "dense_retriever.bin"
        if dense_weights_path.exists() and model.dense_retriever:
            state_dict = torch.load(dense_weights_path, map_location='cpu')
            model.dense_retriever.load_state_dict(state_dict)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        return {
            'model_type': 'EvidenceRetriever',
            'retrieval_mode': self.config.mode,
            'base_model': self.config.model_name,
            'evidence_corpus_size': len(self.evidence_corpus),
            'projection_dim': self.config.projection_dim,
            'max_query_length': self.config.max_query_length,
            'max_evidence_length': self.config.max_evidence_length,
            'dense_available': self.dense_retriever is not None,
            'sparse_available': self.sparse_retriever is not None,
            'faiss_available': FAISS_AVAILABLE,
            'bm25_available': BM25_AVAILABLE
        }


def main():
    """Example usage of EvidenceRetriever."""
    
    # Initialize retriever
    config = EvidenceRetrieverConfig(
        mode='hybrid',
        max_query_length=64,
        max_evidence_length=128,
        projection_dim=256
    )
    
    retriever = EvidenceRetriever(config)
    
    print("=== Evidence Retrieval Example ===")
    print(f"Model info: {retriever.get_model_info()}")
    
    # Test queries
    test_queries = [
        "COVID-19 vaccines are effective",
        "The Earth is flat",
        "Climate change is caused by human activities",
        "Artificial intelligence will replace human workers"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Test different retrieval modes
        modes = []
        if config.mode == 'hybrid':
            modes = ['dense', 'sparse', 'hybrid']
        else:
            modes = [config.mode]
        
        for mode in modes:
            try:
                results = retriever.retrieve(query, top_k=3, mode=mode)
                
                print(f"\n{mode.upper()} Retrieval:")
                for j, result in enumerate(results):
                    print(f"  {j+1}. Score: {result['score']:.3f}")
                    print(f"     Text: {result['text'][:80]}...")
                    
                    if mode == 'hybrid':
                        print(f"     Dense: {result.get('dense_score', 0):.3f}, "
                              f"Sparse: {result.get('sparse_score', 0):.3f}")
                
            except Exception as e:
                print(f"{mode.upper()} retrieval failed: {e}")
    
    # Test model serialization
    print("\n=== Model Serialization Test ===")
    save_path = "test_evidence_retriever"
    retriever.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    try:
        loaded_retriever = EvidenceRetriever.from_pretrained(save_path)
        print("Model loaded successfully")
        
        # Test loaded model
        test_result = loaded_retriever.retrieve("Test query", top_k=2)
        print(f"Loaded model test: {len(test_result)} results")
        
    except Exception as e:
        print(f"Model loading failed: {e}")


if __name__ == "__main__":
    main()
