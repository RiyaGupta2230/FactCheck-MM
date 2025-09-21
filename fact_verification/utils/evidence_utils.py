#!/usr/bin/env python3
"""
Evidence Handling Utilities

Comprehensive utilities for evidence processing including formatting, deduplication,
chunking, highlighting, and multimodal evidence integration for fact verification
systems.

Example Usage:
    >>> from fact_verification.utils import EvidenceUtils
    >>> 
    >>> # Initialize evidence processor
    >>> evidence_utils = EvidenceUtils()
    >>> 
    >>> # Prepare evidence batch for processing
    >>> formatted_evidence = evidence_utils.prepare_evidence_batch(
    ...     evidence_list, claims_list, max_length=512
    ... )
    >>> 
    >>> # Deduplicate retrieved documents
    >>> unique_evidence = evidence_utils.deduplicate_evidence(evidence_list)
    >>> 
    >>> # Highlight relevant spans using attention
    >>> highlighted = evidence_utils.highlight_relevant_spans(
    ...     claim, evidence_text, attention_weights
    ... )
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import re
import json
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class ProcessedEvidence:
    """Container for processed evidence information."""
    
    original_text: str
    chunks: List[str] = field(default_factory=list)
    summary: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relevance_score: float = 0.0
    source_info: Dict[str, Any] = field(default_factory=dict)
    highlighted_spans: List[Tuple[int, int]] = field(default_factory=list)
    multimodal_content: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBatch:
    """Container for batched evidence processing results."""
    
    processed_evidence: List[ProcessedEvidence]
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    deduplication_stats: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)


class EvidenceUtils:
    """
    Comprehensive evidence processing utilities for fact verification.
    
    Provides advanced evidence handling including formatting, deduplication,
    chunking, relevance scoring, span highlighting, and multimodal content
    integration with support for various retrieval systems.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "roberta-base",
        max_chunk_length: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85,
        lightweight_mode: bool = False,
        logger: Optional[Any] = None
    ):
        """
        Initialize evidence utilities.
        
        Args:
            tokenizer_name: Tokenizer name for text processing
            max_chunk_length: Maximum length for evidence chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Threshold for deduplication
            lightweight_mode: Use lightweight processing
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("EvidenceUtils")
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.lightweight_mode = lightweight_mode
        
        # Initialize tokenizer
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE and not lightweight_mode:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.logger.info(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer: {e}")
        
        # Initialize similarity calculator
        self.similarity_calculator = None
        if SKLEARN_AVAILABLE:
            self.similarity_calculator = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Text processing patterns
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`[\]]+')
        self.citation_pattern = re.compile(r'\[[0-9]+\]|\([^)]*\d{4}[^)]*\)')
        self.whitespace_pattern = re.compile(r'\s+')
        
        self.logger.info(f"Initialized EvidenceUtils (lightweight: {lightweight_mode})")
    
    def prepare_evidence_batch(
        self,
        evidence_list: List[Union[str, Dict[str, Any]]],
        claims: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        include_metadata: bool = True,
        deduplicate: bool = True
    ) -> EvidenceBatch:
        """
        Prepare evidence batch for model processing.
        
        Args:
            evidence_list: List of evidence texts or dictionaries
            claims: Optional list of corresponding claims
            max_length: Maximum length for evidence formatting
            include_metadata: Whether to include processing metadata
            deduplicate: Whether to deduplicate evidence
            
        Returns:
            EvidenceBatch with processed evidence
        """
        self.logger.info(f"Preparing evidence batch with {len(evidence_list)} items")
        
        max_length = max_length or self.max_chunk_length
        
        # Normalize evidence input
        normalized_evidence = self._normalize_evidence_input(evidence_list)
        
        # Deduplicate if requested
        if deduplicate:
            unique_evidence, dedup_stats = self.deduplicate_evidence(
                normalized_evidence, return_stats=True
            )
        else:
            unique_evidence = normalized_evidence
            dedup_stats = {}
        
        # Process each evidence item
        processed_evidence = []
        
        for i, evidence_item in enumerate(unique_evidence):
            try:
                # Get corresponding claim if available
                claim = claims[i] if claims and i < len(claims) else None
                
                processed = self._process_single_evidence(
                    evidence_item, claim, max_length, include_metadata
                )
                processed_evidence.append(processed)
                
            except Exception as e:
                self.logger.warning(f"Failed to process evidence item {i}: {e}")
                # Add minimal processed evidence
                fallback_evidence = ProcessedEvidence(
                    original_text=str(evidence_item),
                    chunks=[str(evidence_item)[:max_length]],
                    metadata={'processing_error': str(e)}
                )
                processed_evidence.append(fallback_evidence)
        
        # Compute batch statistics
        batch_stats = self._compute_batch_stats(processed_evidence)
        
        # Create batch result
        evidence_batch = EvidenceBatch(
            processed_evidence=processed_evidence,
            batch_metadata={
                'original_count': len(evidence_list),
                'processed_count': len(processed_evidence),
                'max_length': max_length,
                'deduplicated': deduplicate
            },
            deduplication_stats=dedup_stats,
            processing_stats=batch_stats
        )
        
        self.logger.info(f"Evidence batch preparation completed: {len(processed_evidence)} items")
        return evidence_batch
    
    def _normalize_evidence_input(
        self,
        evidence_list: List[Union[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Normalize evidence input to consistent format."""
        
        normalized = []
        
        for item in evidence_list:
            if isinstance(item, str):
                normalized.append({
                    'text': item,
                    'source': 'unknown',
                    'metadata': {}
                })
            elif isinstance(item, dict):
                # Ensure required fields
                normalized_item = {
                    'text': item.get('text', str(item)),
                    'source': item.get('source', 'unknown'),
                    'metadata': item.get('metadata', {})
                }
                
                # Preserve additional fields
                for key, value in item.items():
                    if key not in normalized_item:
                        normalized_item[key] = value
                
                normalized.append(normalized_item)
            else:
                # Convert to string representation
                normalized.append({
                    'text': str(item),
                    'source': 'unknown',
                    'metadata': {'original_type': type(item).__name__}
                })
        
        return normalized
    
    def _process_single_evidence(
        self,
        evidence_item: Dict[str, Any],
        claim: Optional[str] = None,
        max_length: int = 512,
        include_metadata: bool = True
    ) -> ProcessedEvidence:
        """Process a single evidence item."""
        
        text = evidence_item.get('text', '')
        
        # Clean and preprocess text
        cleaned_text = self._clean_evidence_text(text)
        
        # Create chunks
        chunks = self._create_text_chunks(cleaned_text, max_length)
        
        # Compute relevance score if claim provided
        relevance_score = 0.0
        if claim and SKLEARN_AVAILABLE:
            relevance_score = self._compute_relevance_score(cleaned_text, claim)
        
        # Extract entities (simplified)
        entities = self._extract_evidence_entities(cleaned_text)
        
        # Create summary (first chunk or truncated text)
        summary = chunks[0] if chunks else cleaned_text[:200]
        
        # Extract source information
        source_info = {
            'source': evidence_item.get('source', 'unknown'),
            'url': self._extract_urls(text),
            'domain': self._extract_domain(evidence_item.get('source', '')),
            'timestamp': evidence_item.get('timestamp'),
            'confidence': evidence_item.get('confidence', 1.0)
        }
        
        # Create processed evidence
        processed = ProcessedEvidence(
            original_text=text,
            chunks=chunks,
            summary=summary,
            entities=entities,
            relevance_score=relevance_score,
            source_info=source_info,
            metadata={
                'cleaned_text': cleaned_text,
                'original_length': len(text),
                'chunk_count': len(chunks),
                'processing_mode': 'lightweight' if self.lightweight_mode else 'full'
            } if include_metadata else {}
        )
        
        return processed
    
    def _clean_evidence_text(self, text: str) -> str:
        """Clean evidence text by removing noise and formatting."""
        
        if not text:
            return ""
        
        # Remove URLs
        cleaned = self.url_pattern.sub(' [URL] ', text)
        
        # Remove citations
        cleaned = self.citation_pattern.sub('', cleaned)
        
        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned)
        
        # Remove HTML tags (basic)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        return cleaned.strip()
    
    def _create_text_chunks(self, text: str, max_length: int) -> List[str]:
        """Create overlapping chunks from text."""
        
        if not text:
            return []
        
        # Use tokenizer if available
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            
            if len(tokens) <= max_length:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = start + max_length
                chunk_tokens = tokens[start:end]
                
                # Convert back to text
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                
                if start >= len(tokens):
                    break
            
            return chunks
        
        # Fallback to character-based chunking
        else:
            if len(text) <= max_length:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + max_length
                
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for sentence end within overlap range
                    overlap_start = max(start, end - self.chunk_overlap)
                    sentence_break = text.rfind('.', overlap_start, end)
                    
                    if sentence_break > overlap_start:
                        end = sentence_break + 1
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                
                if start >= len(text):
                    break
            
            return chunks
    
    def _compute_relevance_score(self, evidence_text: str, claim: str) -> float:
        """Compute relevance score between evidence and claim."""
        
        if not evidence_text or not claim or not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            # Use TF-IDF similarity
            texts = [evidence_text, claim]
            tfidf_matrix = self.similarity_calculator.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute relevance score: {e}")
            return 0.0
    
    def _extract_evidence_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from evidence text (simplified)."""
        
        entities = []
        
        # Simple regex-based entity extraction
        patterns = {
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'ORGANIZATION': r'\b[A-Z][A-Z\s&]+\b',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*(?:million|billion|trillion)\b',
            'PERCENT': r'\d+(?:\.\d+)?%',
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': 0.8  # Fixed confidence for regex matches
                })
        
        return entities
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        
        urls = self.url_pattern.findall(text)
        return list(set(urls))  # Remove duplicates
    
    def _extract_domain(self, source: str) -> str:
        """Extract domain from source string."""
        
        # Try to extract domain from URL
        url_match = self.url_pattern.search(source)
        if url_match:
            url = url_match.group()
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                return domain_match.group(1)
        
        # Return source as-is if no URL found
        return source
    
    def deduplicate_evidence(
        self,
        evidence_list: List[Union[str, Dict[str, Any]]],
        method: str = 'similarity',
        return_stats: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Deduplicate evidence using various methods.
        
        Args:
            evidence_list: List of evidence items
            method: Deduplication method ('exact', 'similarity', 'hash')
            return_stats: Whether to return deduplication statistics
            
        Returns:
            Deduplicated evidence list, optionally with statistics
        """
        if not evidence_list:
            return ([], {}) if return_stats else []
        
        # Normalize input
        normalized_evidence = self._normalize_evidence_input(evidence_list)
        
        if method == 'exact':
            unique_evidence = self._deduplicate_exact(normalized_evidence)
        elif method == 'hash':
            unique_evidence = self._deduplicate_hash(normalized_evidence)
        elif method == 'similarity':
            unique_evidence = self._deduplicate_similarity(normalized_evidence)
        else:
            self.logger.warning(f"Unknown deduplication method: {method}")
            unique_evidence = normalized_evidence
        
        stats = {
            'original_count': len(normalized_evidence),
            'unique_count': len(unique_evidence),
            'duplicates_removed': len(normalized_evidence) - len(unique_evidence),
            'deduplication_rate': (len(normalized_evidence) - len(unique_evidence)) / len(normalized_evidence) if normalized_evidence else 0,
            'method': method
        }
        
        self.logger.info(f"Deduplication completed: {stats['original_count']} → {stats['unique_count']} ({stats['deduplication_rate']:.1%} reduction)")
        
        return (unique_evidence, stats) if return_stats else unique_evidence
    
    def _deduplicate_exact(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate using exact text matching."""
        
        seen_texts = set()
        unique_evidence = []
        
        for evidence in evidence_list:
            text = evidence.get('text', '').strip().lower()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_evidence.append(evidence)
        
        return unique_evidence
    
    def _deduplicate_hash(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate using text hashing."""
        
        seen_hashes = set()
        unique_evidence = []
        
        for evidence in evidence_list:
            text = evidence.get('text', '').strip()
            if text:
                # Create hash of normalized text
                normalized = re.sub(r'\s+', ' ', text.lower())
                text_hash = hashlib.md5(normalized.encode()).hexdigest()
                
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    unique_evidence.append(evidence)
        
        return unique_evidence
    
    def _deduplicate_similarity(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate using similarity thresholding."""
        
        if not SKLEARN_AVAILABLE or len(evidence_list) <= 1:
            return evidence_list
        
        try:
            # Extract texts
            texts = [evidence.get('text', '') for evidence in evidence_list]
            
            # Compute TF-IDF similarity matrix
            tfidf_matrix = self.similarity_calculator.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates based on similarity threshold
            unique_indices = []
            seen = set()
            
            for i in range(len(evidence_list)):
                if i in seen:
                    continue
                
                unique_indices.append(i)
                
                # Mark similar items as seen
                for j in range(i + 1, len(evidence_list)):
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        seen.add(j)
            
            return [evidence_list[i] for i in unique_indices]
            
        except Exception as e:
            self.logger.warning(f"Similarity-based deduplication failed: {e}")
            return self._deduplicate_hash(evidence_list)  # Fallback
    
    def highlight_relevant_spans(
        self,
        claim: str,
        evidence_text: str,
        attention_weights: Optional[np.ndarray] = None,
        method: str = 'attention'
    ) -> Dict[str, Any]:
        """
        Highlight relevant spans in evidence text.
        
        Args:
            claim: Claim text
            evidence_text: Evidence text to highlight
            attention_weights: Optional attention weights from model
            method: Highlighting method ('attention', 'similarity', 'keywords')
            
        Returns:
            Dictionary with highlighted spans and metadata
        """
        if not evidence_text or not claim:
            return {
                'highlighted_text': evidence_text,
                'spans': [],
                'method': method,
                'relevance_score': 0.0
            }
        
        if method == 'attention' and attention_weights is not None:
            return self._highlight_with_attention(claim, evidence_text, attention_weights)
        elif method == 'similarity':
            return self._highlight_with_similarity(claim, evidence_text)
        elif method == 'keywords':
            return self._highlight_with_keywords(claim, evidence_text)
        else:
            return self._highlight_with_similarity(claim, evidence_text)
    
    def _highlight_with_attention(
        self,
        claim: str,
        evidence_text: str,
        attention_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Highlight spans using attention weights."""
        
        # Tokenize evidence
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(evidence_text)
        else:
            tokens = evidence_text.split()
        
        # Ensure attention weights match tokens
        if len(attention_weights) != len(tokens):
            # Truncate or pad attention weights
            if len(attention_weights) > len(tokens):
                attention_weights = attention_weights[:len(tokens)]
            else:
                padding = np.zeros(len(tokens) - len(attention_weights))
                attention_weights = np.concatenate([attention_weights, padding])
        
        # Find high-attention spans
        threshold = np.percentile(attention_weights, 75)  # Top 25% of attention
        highlighted_indices = np.where(attention_weights >= threshold)[0]
        
        # Group consecutive indices into spans
        spans = []
        if len(highlighted_indices) > 0:
            start_idx = highlighted_indices[0]
            end_idx = highlighted_indices[0]
            
            for idx in highlighted_indices[1:]:
                if idx == end_idx + 1:
                    end_idx = idx
                else:
                    spans.append((start_idx, end_idx + 1))
                    start_idx = idx
                    end_idx = idx
            
            spans.append((start_idx, end_idx + 1))
        
        # Convert token indices to character positions (approximation)
        char_spans = []
        for start_token, end_token in spans:
            # Simple approximation: assume average token length
            avg_token_length = len(evidence_text) / len(tokens) if tokens else 1
            start_char = int(start_token * avg_token_length)
            end_char = int(end_token * avg_token_length)
            
            # Clamp to text bounds
            start_char = max(0, min(start_char, len(evidence_text)))
            end_char = max(start_char, min(end_char, len(evidence_text)))
            
            char_spans.append((start_char, end_char))
        
        # Create highlighted text (simplified)
        highlighted_text = evidence_text
        
        return {
            'highlighted_text': highlighted_text,
            'spans': char_spans,
            'method': 'attention',
            'relevance_score': float(np.mean(attention_weights)) if len(attention_weights) > 0 else 0.0,
            'attention_weights': attention_weights.tolist()
        }
    
    def _highlight_with_similarity(
        self,
        claim: str,
        evidence_text: str
    ) -> Dict[str, Any]:
        """Highlight spans using similarity matching."""
        
        # Split evidence into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(evidence_text)
        else:
            sentences = re.split(r'[.!?]+', evidence_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Compute similarity between claim and each sentence
        similarities = []
        
        if SKLEARN_AVAILABLE and sentences:
            try:
                all_texts = [claim] + sentences
                tfidf_matrix = self.similarity_calculator.fit_transform(all_texts)
                
                # Compute similarities between claim and each sentence
                claim_vector = tfidf_matrix[0:1]
                sentence_vectors = tfidf_matrix[1:]
                
                similarity_scores = cosine_similarity(claim_vector, sentence_vectors)[0]
                similarities = similarity_scores.tolist()
                
            except Exception as e:
                self.logger.warning(f"Similarity computation failed: {e}")
                similarities = [0.0] * len(sentences)
        else:
            similarities = [0.0] * len(sentences)
        
        # Find high-similarity sentences
        if similarities:
            threshold = np.percentile(similarities, 70)  # Top 30% of sentences
            
            # Find character spans for relevant sentences
            spans = []
            current_pos = 0
            
            for i, sentence in enumerate(sentences):
                start_pos = evidence_text.find(sentence, current_pos)
                
                if start_pos != -1 and similarities[i] >= threshold:
                    end_pos = start_pos + len(sentence)
                    spans.append((start_pos, end_pos))
                    current_pos = end_pos
                else:
                    current_pos += len(sentence)
        else:
            spans = []
        
        return {
            'highlighted_text': evidence_text,
            'spans': spans,
            'method': 'similarity',
            'relevance_score': float(np.mean(similarities)) if similarities else 0.0,
            'sentence_similarities': similarities
        }
    
    def _highlight_with_keywords(
        self,
        claim: str,
        evidence_text: str
    ) -> Dict[str, Any]:
        """Highlight spans containing claim keywords."""
        
        # Extract keywords from claim (simple approach)
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        claim_keywords = claim_words - stopwords
        
        # Find keyword occurrences in evidence
        spans = []
        evidence_lower = evidence_text.lower()
        
        for keyword in claim_keywords:
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', evidence_lower):
                spans.append((match.start(), match.end()))
        
        # Sort spans by position
        spans.sort()
        
        # Merge overlapping spans
        merged_spans = []
        for start, end in spans:
            if merged_spans and start <= merged_spans[-1][1]:
                # Extend last span
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
            else:
                merged_spans.append((start, end))
        
        # Calculate relevance score
        total_keyword_chars = sum(end - start for start, end in merged_spans)
        relevance_score = total_keyword_chars / len(evidence_text) if evidence_text else 0.0
        
        return {
            'highlighted_text': evidence_text,
            'spans': merged_spans,
            'method': 'keywords',
            'relevance_score': relevance_score,
            'keywords_found': list(claim_keywords)
        }
    
    def merge_multimodal_evidence(
        self,
        text_evidence: List[str],
        image_captions: List[str],
        other_modalities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Merge evidence from different modalities.
        
        Args:
            text_evidence: List of text evidence
            image_captions: List of image captions
            other_modalities: Optional other modality data
            
        Returns:
            List of merged multimodal evidence items
        """
        merged_evidence = []
        
        # Add text evidence
        for i, text in enumerate(text_evidence):
            evidence_item = {
                'type': 'text',
                'content': text,
                'modality': 'text',
                'index': i,
                'metadata': {'source': 'text_evidence'}
            }
            merged_evidence.append(evidence_item)
        
        # Add image captions
        for i, caption in enumerate(image_captions):
            evidence_item = {
                'type': 'image_caption',
                'content': caption,
                'modality': 'visual',
                'index': i,
                'metadata': {'source': 'image_caption'}
            }
            merged_evidence.append(evidence_item)
        
        # Add other modalities
        if other_modalities:
            for i, modality_data in enumerate(other_modalities):
                evidence_item = {
                    'type': modality_data.get('type', 'unknown'),
                    'content': modality_data.get('content', ''),
                    'modality': modality_data.get('modality', 'unknown'),
                    'index': i,
                    'metadata': modality_data.get('metadata', {})
                }
                merged_evidence.append(evidence_item)
        
        # Sort by relevance if possible
        # This is simplified - in practice would use cross-modal similarity
        return merged_evidence
    
    def _compute_batch_stats(
        self,
        processed_evidence: List[ProcessedEvidence]
    ) -> Dict[str, Any]:
        """Compute statistics for processed evidence batch."""
        
        if not processed_evidence:
            return {}
        
        total_items = len(processed_evidence)
        total_chunks = sum(len(ev.chunks) for ev in processed_evidence)
        total_entities = sum(len(ev.entities) for ev in processed_evidence)
        
        relevance_scores = [ev.relevance_score for ev in processed_evidence if ev.relevance_score > 0]
        
        stats = {
            'total_items': total_items,
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'avg_chunks_per_item': total_chunks / total_items,
            'avg_entities_per_item': total_entities / total_items,
            'items_with_entities': sum(1 for ev in processed_evidence if ev.entities),
            'avg_relevance_score': np.mean(relevance_scores) if relevance_scores else 0.0,
            'items_with_multimodal': sum(1 for ev in processed_evidence if ev.multimodal_content),
        }
        
        return stats
    
    def get_utils_info(self) -> Dict[str, Any]:
        """Get information about utils configuration and capabilities."""
        
        return {
            'max_chunk_length': self.max_chunk_length,
            'chunk_overlap': self.chunk_overlap,
            'similarity_threshold': self.similarity_threshold,
            'lightweight_mode': self.lightweight_mode,
            'components': {
                'tokenizer_available': self.tokenizer is not None,
                'similarity_calculator_available': self.similarity_calculator is not None,
                'sklearn_available': SKLEARN_AVAILABLE,
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'nltk_available': NLTK_AVAILABLE
            }
        }


def main():
    """Example usage of EvidenceUtils."""
    
    # Initialize evidence utils
    evidence_utils = EvidenceUtils(
        max_chunk_length=200,  # Short chunks for demo
        lightweight_mode=True
    )
    
    print("=== EvidenceUtils Example ===")
    print(f"Utils info: {evidence_utils.get_utils_info()}")
    
    # Test evidence
    test_evidence = [
        "The COVID-19 vaccine has been shown to be 95% effective in preventing severe illness according to clinical trials conducted in 2020-2021.",
        "Recent studies from multiple universities confirm that climate change is primarily caused by human activities, particularly greenhouse gas emissions from fossil fuels.",
        "The COVID-19 vaccine is highly effective and safe according to multiple clinical trials.",  # Duplicate-ish
        "NASA's satellite data shows that global temperatures have risen by 1.1°C since pre-industrial times, with the most rapid warming occurring in recent decades."
    ]
    
    test_claims = [
        "COVID-19 vaccines are effective",
        "Climate change is caused by humans",
        "Vaccines are safe and effective",
        "Global warming is real"
    ]
    
    print(f"\nProcessing {len(test_evidence)} evidence items...")
    
    # Prepare evidence batch
    evidence_batch = evidence_utils.prepare_evidence_batch(
        test_evidence,
        claims=test_claims,
        max_length=150,
        deduplicate=True
    )
    
    print(f"\nBatch Results:")
    print(f"Original count: {evidence_batch.batch_metadata['original_count']}")
    print(f"Processed count: {evidence_batch.batch_metadata['processed_count']}")
    print(f"Deduplication stats: {evidence_batch.deduplication_stats}")
    
    # Show processed evidence details
    for i, processed in enumerate(evidence_batch.processed_evidence):
        print(f"\n--- Evidence {i+1} ---")
        print(f"Original length: {len(processed.original_text)}")
        print(f"Chunks: {len(processed.chunks)}")
        print(f"Entities: {len(processed.entities)}")
        print(f"Relevance score: {processed.relevance_score:.3f}")
        print(f"Summary: {processed.summary[:80]}...")
        
        if processed.entities:
            print("  Entities found:")
            for entity in processed.entities[:3]:
                print(f"    {entity['text']} ({entity['label']})")
    
    # Test deduplication separately
    print(f"\n=== Deduplication Test ===")
    
    duplicate_evidence = [
        "The Earth is round.",
        "Our planet is spherical in shape.",
        "The Earth is round.",  # Exact duplicate
        "Climate change is real."
    ]
    
    for method in ['exact', 'hash', 'similarity']:
        deduplicated, stats = evidence_utils.deduplicate_evidence(
            duplicate_evidence, method=method, return_stats=True
        )
        print(f"{method.title()} method: {stats['original_count']} → {stats['unique_count']} ({stats['deduplication_rate']:.1%} reduction)")
    
    # Test span highlighting
    print(f"\n=== Span Highlighting Test ===")
    
    claim = "COVID-19 vaccines are effective"
    evidence = "Clinical trials have shown that COVID-19 vaccines are highly effective in preventing severe illness and death."
    
    for method in ['similarity', 'keywords']:
        highlighted = evidence_utils.highlight_relevant_spans(claim, evidence, method=method)
        print(f"\n{method.title()} highlighting:")
        print(f"  Relevance score: {highlighted['relevance_score']:.3f}")
        print(f"  Spans found: {len(highlighted['spans'])}")
        
        for start, end in highlighted['spans'][:3]:
            span_text = evidence[start:end]
            print(f"    \"{span_text}\"")
    
    # Test multimodal merging
    print(f"\n=== Multimodal Evidence Test ===")
    
    text_evidence = ["Text evidence about vaccines"]
    image_captions = ["Image showing vaccination process", "Chart of vaccine effectiveness"]
    
    merged = evidence_utils.merge_multimodal_evidence(text_evidence, image_captions)
    
    print(f"Merged {len(merged)} multimodal evidence items:")
    for item in merged:
        print(f"  {item['type']}: {item['content'][:50]}...")


if __name__ == "__main__":
    main()
