#!/usr/bin/env python3
"""
Claim Processing Utilities

Advanced claim preprocessing including normalization, NER, coreference resolution,
and claim boundary detection for robust fact verification pipelines.

Example Usage:
    >>> from fact_verification.utils import ClaimProcessor
    >>> 
    >>> # Initialize processor with full NLP pipeline
    >>> processor = ClaimProcessor(enable_ner=True, enable_coref=True)
    >>> 
    >>> # Process individual claim
    >>> claim = "President Biden visited Paris last year"
    >>> processed = processor.process_single_claim(claim)
    >>> print(f"Entities: {processed['entities']}")
    >>> print(f"Normalized: {processed['normalized_text']}")
    >>> 
    >>> # Process dataset batch
    >>> from fact_verification.data import FeverDataset
    >>> dataset = FeverDataset('train')
    >>> processed_dataset = processor.process_claims(dataset)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import json
from dataclasses import dataclass, field
from collections import defaultdict
import unicodedata

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional NLP dependencies
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import neuralcoref
    NEURALCOREF_AVAILABLE = True
except ImportError:
    NEURALCOREF_AVAILABLE = False


@dataclass
class ProcessedClaim:
    """Container for processed claim information."""
    
    original_text: str
    normalized_text: str
    sentences: List[str]
    entities: List[Dict[str, Any]] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)
    claim_boundaries: List[Tuple[int, int]] = field(default_factory=list)
    coreferences: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaimProcessor:
    """
    Advanced claim processor with NLP capabilities for fact verification.
    
    Provides comprehensive text processing including normalization, entity
    extraction, coreference resolution, and claim boundary detection with
    support for both lightweight and full-featured processing modes.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_ner: bool = True,
        enable_coref: bool = False,
        enable_claim_detection: bool = True,
        lightweight_mode: bool = False,
        logger: Optional[Any] = None
    ):
        """
        Initialize claim processor.
        
        Args:
            spacy_model: SpaCy model name for NLP processing
            enable_ner: Enable named entity recognition
            enable_coref: Enable coreference resolution
            enable_claim_detection: Enable claim boundary detection
            lightweight_mode: Use lightweight processing (for resource constraints)
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("ClaimProcessor")
        self.enable_ner = enable_ner
        self.enable_coref = enable_coref
        self.enable_claim_detection = enable_claim_detection
        self.lightweight_mode = lightweight_mode
        
        # Initialize NLP components
        self.nlp = None
        self.ner_pipeline = None
        self.claim_detector = None
        
        self._initialize_nlp_components(spacy_model)
        
        # Text normalization patterns
        self.normalization_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'[^\w\s.,!?;:()-]', ''),  # Remove special characters except basic punctuation
            (r'\.{2,}', '.'),  # Multiple dots to single dot
            (r'[,;]{2,}', ','),  # Multiple commas/semicolons to single comma
        ]
        
        # Stopwords
        self.stopwords = set()
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                # Download stopwords if not available
                try:
                    nltk.download('stopwords', quiet=True)
                    self.stopwords = set(stopwords.words('english'))
                except:
                    pass
        
        # Stemmer and lemmatizer
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        
        # Claim boundary detection patterns (fallback regex)
        self.claim_patterns = [
            r'\b(?:claim|assert|state|argue|maintain|contend)\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'\b(?:according\s+to|reports?\s+(?:indicate|suggest|show))\s+(.+?)(?:\.|$)',
            r'\b(?:it\s+is\s+(?:true|false|certain|uncertain))\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'^(.+?)(?:\s+is\s+(?:true|false|correct|incorrect|accurate|inaccurate))\.?$',
        ]
        
        self.logger.info(f"Initialized ClaimProcessor (lightweight: {lightweight_mode})")
    
    def _initialize_nlp_components(self, spacy_model: str):
        """Initialize NLP processing components."""
        
        # SpaCy initialization
        if SPACY_AVAILABLE and not self.lightweight_mode:
            try:
                self.nlp = spacy.load(spacy_model)
                
                # Add neuralcoref if available
                if self.enable_coref and NEURALCOREF_AVAILABLE:
                    try:
                        neuralcoref.add_to_pipe(self.nlp)
                        self.logger.info("Added neuralcoref to SpaCy pipeline")
                    except Exception as e:
                        self.logger.warning(f"Failed to add neuralcoref: {e}")
                        self.enable_coref = False
                
                self.logger.info(f"Loaded SpaCy model: {spacy_model}")
                
            except OSError:
                self.logger.warning(f"SpaCy model {spacy_model} not found, falling back to basic processing")
                self.nlp = None
        
        # Transformer-based NER (lightweight alternative)
        if self.enable_ner and TRANSFORMERS_AVAILABLE and (self.lightweight_mode or not SPACY_AVAILABLE):
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                self.logger.info("Loaded transformer-based NER pipeline")
            except Exception as e:
                self.logger.warning(f"Failed to load NER pipeline: {e}")
                self.enable_ner = False
        
        # Claim detection model (transformer-based)
        if self.enable_claim_detection and TRANSFORMERS_AVAILABLE:
            try:
                # Use a general text classification model as placeholder
                # In practice, this would be a fine-tuned claim detection model
                self.claim_detector = pipeline(
                    "text-classification",
                    model="textattack/roberta-base-CoLA"  # Grammar acceptability as proxy
                )
                self.logger.info("Loaded claim detection pipeline")
            except Exception as e:
                self.logger.warning(f"Failed to load claim detection model: {e}")
                self.enable_claim_detection = False
    
    def normalize_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
        apply_lemmatization: bool = False
    ) -> str:
        """
        Normalize text with various preprocessing options.
        
        Args:
            text: Input text to normalize
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_stopwords: Remove stop words
            apply_stemming: Apply stemming
            apply_lemmatization: Apply lemmatization
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', text)
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Lowercase
        if lowercase:
            normalized = normalized.lower()
        
        # Remove punctuation
        if remove_punctuation:
            normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Tokenize for further processing
        if remove_stopwords or apply_stemming or apply_lemmatization:
            if NLTK_AVAILABLE:
                tokens = word_tokenize(normalized)
            else:
                tokens = normalized.split()
            
            # Remove stopwords
            if remove_stopwords and self.stopwords:
                tokens = [token for token in tokens if token.lower() not in self.stopwords]
            
            # Apply stemming
            if apply_stemming and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            # Apply lemmatization
            if apply_lemmatization and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            normalized = ' '.join(tokens)
        
        # Clean up extra spaces
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return normalized
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences using available NLP tools.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Use SpaCy if available
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        
        # Fallback to NLTK
        elif NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                return [sent.strip() for sent in sentences if sent.strip()]
            except:
                pass
        
        # Basic regex fallback
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List of entity dictionaries with type, text, and position
        """
        entities = []
        
        if not self.enable_ner or not text:
            return entities
        
        # Use SpaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'confidence': getattr(ent, 'score', 1.0)
                })
        
        # Fallback to transformer NER
        elif self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                for result in ner_results:
                    entities.append({
                        'text': result['word'],
                        'label': result['entity_group'],
                        'start_char': result.get('start', 0),
                        'end_char': result.get('end', 0),
                        'confidence': result.get('score', 0.0)
                    })
            except Exception as e:
                self.logger.warning(f"NER extraction failed: {e}")
        
        return entities
    
    def resolve_coreferences(self, text: str) -> Dict[str, Any]:
        """
        Resolve coreferences in text.
        
        Args:
            text: Input text for coreference resolution
            
        Returns:
            Dictionary with resolved text and coreference clusters
        """
        if not self.enable_coref or not text or not self.nlp:
            return {
                'resolved_text': text,
                'clusters': [],
                'has_coreferences': False
            }
        
        try:
            doc = self.nlp(text)
            
            if hasattr(doc._, 'coref_clusters') and doc._.coref_clusters:
                resolved_text = doc._.coref_resolved
                clusters = []
                
                for cluster in doc._.coref_clusters:
                    cluster_info = {
                        'main_mention': cluster.main.text,
                        'mentions': [mention.text for mention in cluster.mentions],
                        'resolved': cluster.main.text
                    }
                    clusters.append(cluster_info)
                
                return {
                    'resolved_text': resolved_text,
                    'clusters': clusters,
                    'has_coreferences': True
                }
            else:
                return {
                    'resolved_text': text,
                    'clusters': [],
                    'has_coreferences': False
                }
        
        except Exception as e:
            self.logger.warning(f"Coreference resolution failed: {e}")
            return {
                'resolved_text': text,
                'clusters': [],
                'has_coreferences': False
            }
    
    def detect_claim_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect claim boundaries in text.
        
        Args:
            text: Input text for claim boundary detection
            
        Returns:
            List of (start, end) character positions for detected claims
        """
        boundaries = []
        
        if not text:
            return boundaries
        
        # Use transformer model if available
        if self.enable_claim_detection and self.claim_detector:
            try:
                # Split text into sentences for claim detection
                sentences = self.segment_sentences(text)
                current_pos = 0
                
                for sentence in sentences:
                    # Find sentence position in original text
                    start_pos = text.find(sentence, current_pos)
                    if start_pos != -1:
                        end_pos = start_pos + len(sentence)
                        
                        # Check if sentence is a claim using the model
                        result = self.claim_detector(sentence)
                        
                        # Interpret result (depends on model)
                        # This is simplified - in practice would use a proper claim detection model
                        if isinstance(result, list) and result:
                            score = result[0].get('score', 0.0)
                            if score > 0.5:  # Threshold for claim detection
                                boundaries.append((start_pos, end_pos))
                        
                        current_pos = end_pos
            
            except Exception as e:
                self.logger.warning(f"Transformer-based claim detection failed: {e}")
        
        # Fallback to regex patterns
        if not boundaries:
            for pattern in self.claim_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match.group(1):  # Has captured group
                        boundaries.append(match.span(1))
                    else:
                        boundaries.append(match.span())
        
        # If no claims detected, treat entire text as one claim
        if not boundaries and text.strip():
            boundaries.append((0, len(text)))
        
        return boundaries
    
    def process_single_claim(
        self,
        claim_text: str,
        normalize_options: Optional[Dict[str, bool]] = None
    ) -> ProcessedClaim:
        """
        Process a single claim with full NLP pipeline.
        
        Args:
            claim_text: Input claim text
            normalize_options: Normalization options override
            
        Returns:
            ProcessedClaim object with all processing results
        """
        if not claim_text:
            return ProcessedClaim(
                original_text="",
                normalized_text="",
                sentences=[]
            )
        
        # Default normalization options
        norm_opts = {
            'lowercase': True,
            'remove_punctuation': False,
            'remove_stopwords': False,
            'apply_stemming': False,
            'apply_lemmatization': False
        }
        
        if normalize_options:
            norm_opts.update(normalize_options)
        
        # Basic processing
        normalized_text = self.normalize_text(claim_text, **norm_opts)
        sentences = self.segment_sentences(claim_text)
        
        # Entity extraction
        entities = self.extract_entities(claim_text)
        
        # Coreference resolution
        coref_result = self.resolve_coreferences(claim_text)
        
        # Claim boundary detection
        claim_boundaries = self.detect_claim_boundaries(claim_text)
        
        # Tokenization and POS tagging
        tokens = []
        pos_tags = []
        
        if self.nlp:
            doc = self.nlp(claim_text)
            tokens = [token.text for token in doc]
            pos_tags = [(token.text, token.pos_) for token in doc]
        elif NLTK_AVAILABLE:
            tokens = word_tokenize(claim_text)
            try:
                pos_tags = nltk.pos_tag(tokens)
            except:
                pos_tags = [(token, 'UNK') for token in tokens]
        else:
            tokens = claim_text.split()
            pos_tags = [(token, 'UNK') for token in tokens]
        
        # Create processed claim object
        processed = ProcessedClaim(
            original_text=claim_text,
            normalized_text=normalized_text,
            sentences=sentences,
            entities=entities,
            tokens=tokens,
            pos_tags=pos_tags,
            claim_boundaries=claim_boundaries,
            coreferences=coref_result['clusters'],
            metadata={
                'has_coreferences': coref_result['has_coreferences'],
                'resolved_text': coref_result['resolved_text'],
                'entity_count': len(entities),
                'sentence_count': len(sentences),
                'token_count': len(tokens)
            }
        )
        
        return processed
    
    def process_claims(
        self,
        claims: Union[List[str], Any],
        batch_size: int = 32,
        save_processed: bool = False,
        output_dir: str = "fact_verification/data/processed"
    ) -> List[ProcessedClaim]:
        """
        Process multiple claims or a dataset.
        
        Args:
            claims: List of claim strings or dataset object
            batch_size: Processing batch size
            save_processed: Whether to save processed results
            output_dir: Output directory for saved results
            
        Returns:
            List of ProcessedClaim objects
        """
        self.logger.info("Starting batch claim processing")
        
        # Handle different input types
        if hasattr(claims, '__len__') and hasattr(claims, '__getitem__'):
            # Dataset or list-like object
            if hasattr(claims, 'get_claims'):
                # Custom dataset with get_claims method
                claim_texts = claims.get_claims()
            elif isinstance(claims, list):
                claim_texts = claims
            else:
                # Assume iterable dataset
                claim_texts = []
                for i, item in enumerate(claims):
                    if isinstance(item, dict):
                        claim_text = item.get('claim', item.get('claim_text', str(item)))
                    elif isinstance(item, str):
                        claim_text = item
                    else:
                        claim_text = str(item)
                    
                    claim_texts.append(claim_text)
                    
                    # Limit processing for very large datasets
                    if len(claim_texts) >= 10000:
                        self.logger.warning("Limiting processing to first 10,000 claims")
                        break
        else:
            claim_texts = list(claims)
        
        # Process claims in batches
        processed_claims = []
        
        for i in range(0, len(claim_texts), batch_size):
            batch = claim_texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(claim_texts) + batch_size - 1)//batch_size}")
            
            for claim_text in batch:
                try:
                    processed = self.process_single_claim(claim_text)
                    processed_claims.append(processed)
                except Exception as e:
                    self.logger.warning(f"Failed to process claim: {claim_text[:50]}... Error: {e}")
                    # Add empty processed claim to maintain alignment
                    processed_claims.append(ProcessedClaim(
                        original_text=claim_text,
                        normalized_text=claim_text,
                        sentences=[claim_text],
                        metadata={'processing_error': str(e)}
                    ))
        
        # Save processed results if requested
        if save_processed:
            self._save_processed_claims(processed_claims, output_dir)
        
        self.logger.info(f"Completed processing {len(processed_claims)} claims")
        return processed_claims
    
    def _save_processed_claims(
        self,
        processed_claims: List[ProcessedClaim],
        output_dir: str
    ):
        """Save processed claims to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_claims = []
        
        for claim in processed_claims:
            claim_data = {
                'original_text': claim.original_text,
                'normalized_text': claim.normalized_text,
                'sentences': claim.sentences,
                'entities': claim.entities,
                'tokens': claim.tokens,
                'pos_tags': claim.pos_tags,
                'claim_boundaries': claim.claim_boundaries,
                'coreferences': claim.coreferences,
                'metadata': claim.metadata
            }
            serializable_claims.append(claim_data)
        
        # Save as JSON
        json_file = output_path / "processed_claims.json"
        with open(json_file, 'w') as f:
            json.dump(serializable_claims, f, indent=2)
        
        # Save summary statistics
        stats = self._compute_processing_stats(processed_claims)
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Processed claims saved to {output_path}")
    
    def _compute_processing_stats(
        self,
        processed_claims: List[ProcessedClaim]
    ) -> Dict[str, Any]:
        """Compute processing statistics."""
        
        if not processed_claims:
            return {}
        
        total_claims = len(processed_claims)
        total_entities = sum(len(claim.entities) for claim in processed_claims)
        total_sentences = sum(len(claim.sentences) for claim in processed_claims)
        total_tokens = sum(len(claim.tokens) for claim in processed_claims)
        
        claims_with_entities = sum(1 for claim in processed_claims if claim.entities)
        claims_with_coref = sum(1 for claim in processed_claims if claim.coreferences)
        claims_with_multiple_sentences = sum(1 for claim in processed_claims if len(claim.sentences) > 1)
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for claim in processed_claims:
            for entity in claim.entities:
                entity_types[entity.get('label', 'UNKNOWN')] += 1
        
        stats = {
            'total_claims': total_claims,
            'total_entities': total_entities,
            'total_sentences': total_sentences,
            'total_tokens': total_tokens,
            'avg_entities_per_claim': total_entities / total_claims if total_claims > 0 else 0,
            'avg_sentences_per_claim': total_sentences / total_claims if total_claims > 0 else 0,
            'avg_tokens_per_claim': total_tokens / total_claims if total_claims > 0 else 0,
            'claims_with_entities_pct': claims_with_entities / total_claims * 100 if total_claims > 0 else 0,
            'claims_with_coref_pct': claims_with_coref / total_claims * 100 if total_claims > 0 else 0,
            'claims_with_multiple_sentences_pct': claims_with_multiple_sentences / total_claims * 100 if total_claims > 0 else 0,
            'entity_type_distribution': dict(entity_types),
            'processing_mode': 'lightweight' if self.lightweight_mode else 'full'
        }
        
        return stats
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about processor configuration and capabilities."""
        
        return {
            'lightweight_mode': self.lightweight_mode,
            'spacy_available': SPACY_AVAILABLE and self.nlp is not None,
            'nltk_available': NLTK_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'ner_enabled': self.enable_ner,
            'coref_enabled': self.enable_coref and NEURALCOREF_AVAILABLE,
            'claim_detection_enabled': self.enable_claim_detection,
            'components': {
                'spacy_model': self.nlp is not None,
                'ner_pipeline': self.ner_pipeline is not None,
                'claim_detector': self.claim_detector is not None,
                'stemmer': self.stemmer is not None,
                'lemmatizer': self.lemmatizer is not None,
                'stopwords': len(self.stopwords) > 0
            }
        }


def main():
    """Example usage of ClaimProcessor."""
    
    # Initialize processor
    processor = ClaimProcessor(
        enable_ner=True,
        enable_coref=False,  # Disable for example (requires model)
        lightweight_mode=True  # Use lightweight mode for demo
    )
    
    print("=== ClaimProcessor Example ===")
    print(f"Processor info: {processor.get_processor_info()}")
    
    # Test claims
    test_claims = [
        "President Biden visited Paris last year to discuss climate change.",
        "The COVID-19 vaccine is 95% effective according to clinical trials.",
        "Climate change is caused by human activities, scientists say.",
        "The Earth is flat and NASA has been lying to us all along."
    ]
    
    print(f"\nProcessing {len(test_claims)} test claims...")
    
    # Process individual claims
    for i, claim in enumerate(test_claims, 1):
        print(f"\n--- Claim {i} ---")
        print(f"Original: {claim}")
        
        processed = processor.process_single_claim(claim)
        
        print(f"Normalized: {processed.normalized_text}")
        print(f"Sentences: {len(processed.sentences)}")
        print(f"Entities: {len(processed.entities)}")
        
        if processed.entities:
            print("  Detected entities:")
            for entity in processed.entities[:3]:  # Show first 3
                print(f"    {entity['text']} ({entity['label']})")
        
        print(f"Tokens: {len(processed.tokens)}")
        print(f"Claim boundaries: {processed.claim_boundaries}")
        
        if processed.metadata.get('has_coreferences'):
            print("  Has coreferences: Yes")
        
    # Test batch processing
    print(f"\n=== Batch Processing Test ===")
    processed_batch = processor.process_claims(test_claims)
    
    print(f"Processed {len(processed_batch)} claims")
    
    # Compute and display statistics
    stats = processor._compute_processing_stats(processed_batch)
    
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test normalization options
    print(f"\n=== Normalization Options Test ===")
    test_text = "The U.S. President Biden visited Paris, France in 2023!!!"
    
    norm_options = [
        {'lowercase': True},
        {'lowercase': True, 'remove_punctuation': True},
        {'lowercase': True, 'remove_stopwords': True},
    ]
    
    for opts in norm_options:
        normalized = processor.normalize_text(test_text, **opts)
        print(f"Options {opts}: {normalized}")


if __name__ == "__main__":
    main()
