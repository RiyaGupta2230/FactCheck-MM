#!/usr/bin/env python3
"""
Post-Processing Utilities for Generated Paraphrases

Comprehensive text refinement tools including redundancy removal, spacing/punctuation
fixing, truecasing, detokenization, and length filtering. Designed to be compatible
with sarcasm-aware generation by preserving special tokens.

Example Usage:
    >>> from paraphrasing.utils import PostProcessor
    >>> 
    >>> # Initialize post-processor
    >>> processor = PostProcessor()
    >>> 
    >>> # Clean generated texts
    >>> raw_texts = [
    ...     "the   weather is  nice nice today .",
    ...     "TODAY'S weather IS lovely",
    ...     "<sarcastic> oh great , another monday </sarcastic>",
    ...     "this is way way way too repetitive text"
    ... ]
    >>> 
    >>> cleaned_texts = processor.clean_batch(raw_texts)
    >>> for original, cleaned in zip(raw_texts, cleaned_texts):
    ...     print(f"Original: {original}")
    ...     print(f"Cleaned:  {cleaned}")
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional NLTK for advanced processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Import base config if available
try:
    from config.base_config import BaseConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@dataclass
class PostProcessingConfig:
    """Configuration for text post-processing."""
    
    # Basic cleaning
    fix_spacing: bool = True
    fix_punctuation: bool = True
    remove_extra_whitespace: bool = True
    
    # Redundancy removal
    remove_word_repetitions: bool = True
    remove_phrase_repetitions: bool = True
    max_word_repetitions: int = 2
    max_phrase_repetitions: int = 1
    min_phrase_length: int = 3  # Minimum words in a phrase
    
    # Capitalization
    apply_truecasing: bool = True
    capitalize_sentences: bool = True
    preserve_all_caps: bool = False
    
    # Length filtering
    filter_by_length: bool = True
    min_length_words: int = 3
    max_length_words: int = 100
    min_length_chars: int = 10
    max_length_chars: int = 500
    
    # Detokenization
    detokenize: bool = True
    join_subwords: bool = True  # For BPE/WordPiece tokens
    
    # Sarcasm-aware processing
    preserve_sarcasm_tokens: bool = True
    sarcasm_tokens: Set[str] = field(default_factory=lambda: {
        '<sarcastic>', '</sarcastic>', '<non_sarcastic>', '</non_sarcastic>',
        '<preserve_sarcasm>', '<formal>', '<informal>'
    })
    
    # Advanced processing
    remove_incomplete_sentences: bool = False
    fix_grammar: bool = False  # Requires additional dependencies
    
    # Performance settings
    batch_size: int = 32
    enable_parallel: bool = False


class PostProcessor:
    """
    Comprehensive post-processing for generated paraphrases.
    
    Provides text cleaning, redundancy removal, truecasing, and length filtering
    with special handling for sarcasm-aware generation tokens.
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        """
        Initialize post-processor.
        
        Args:
            config: Post-processing configuration
        """
        # Load config from base config if available
        if config is None and CONFIG_AVAILABLE:
            try:
                base_config = BaseConfig()
                config = getattr(base_config, 'post_processing', PostProcessingConfig())
            except:
                config = PostProcessingConfig()
        
        self.config = config or PostProcessingConfig()
        self.logger = get_logger("PostProcessor")
        
        # Download NLTK data if needed
        if NLTK_AVAILABLE and (self.config.apply_truecasing or self.config.capitalize_sentences):
            self._setup_nltk()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        self.logger.info("Initialized PostProcessor")
        self.logger.info(f"NLTK available: {NLTK_AVAILABLE}")
    
    def _setup_nltk(self):
        """Setup NLTK dependencies."""
        try:
            # Try to access required data, download if missing
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def _compile_patterns(self):
        """Compile regex patterns for text processing."""
        
        # Spacing patterns
        self.extra_whitespace_pattern = re.compile(r'\s+')
        self.punct_space_pattern = re.compile(r'\s*([.!?,:;])\s*')
        
        # Repetition patterns
        self.word_repetition_pattern = re.compile(
            r'\b(\w+)(?:\s+\1\b){' + str(self.config.max_word_repetitions) + r',}',
            re.IGNORECASE
        )
        
        # Sarcasm token patterns
        if self.config.preserve_sarcasm_tokens:
            sarcasm_pattern = '|'.join(re.escape(token) for token in self.config.sarcasm_tokens)
            self.sarcasm_token_pattern = re.compile(f'({sarcasm_pattern})', re.IGNORECASE)
        
        # Sentence boundary patterns
        self.sentence_end_pattern = re.compile(r'[.!?]+')
        
        # Detokenization patterns
        self.subword_pattern = re.compile(r'(\w+)(##\w+)+')  # WordPiece
        self.bpe_pattern = re.compile(r'(\w+@@\s*)+\w+')  # BPE
    
    def remove_redundancy(self, text: str) -> str:
        """
        Remove redundant words and phrases from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with redundancy removed
        """
        if not (self.config.remove_word_repetitions or self.config.remove_phrase_repetitions):
            return text
        
        # Preserve sarcasm tokens
        sarcasm_tokens = []
        processed_text = text
        
        if self.config.preserve_sarcasm_tokens:
            # Extract and temporarily remove sarcasm tokens
            matches = list(self.sarcasm_token_pattern.finditer(text))
            for i, match in enumerate(reversed(matches)):  # Reverse to maintain indices
                placeholder = f"__SARCASM_TOKEN_{len(matches)-1-i}__"
                sarcasm_tokens.insert(0, (placeholder, match.group()))
                processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
        
        # Remove word repetitions
        if self.config.remove_word_repetitions:
            def replace_word_reps(match):
                word = match.group(1)
                return ' '.join([word] * (self.config.max_word_repetitions + 1))
            
            processed_text = self.word_repetition_pattern.sub(replace_word_reps, processed_text)
        
        # Remove phrase repetitions
        if self.config.remove_phrase_repetitions:
            processed_text = self._remove_phrase_repetitions(processed_text)
        
        # Restore sarcasm tokens
        for placeholder, original_token in sarcasm_tokens:
            processed_text = processed_text.replace(placeholder, original_token)
        
        return processed_text
    
    def _remove_phrase_repetitions(self, text: str) -> str:
        """Remove repetitive phrases from text."""
        
        words = text.split()
        if len(words) < self.config.min_phrase_length * 2:
            return text
        
        # Find repetitive phrases
        cleaned_words = []
        i = 0
        
        while i < len(words):
            # Look for phrase repetitions
            found_repetition = False
            
            for phrase_len in range(self.config.min_phrase_length, min(len(words) - i, 10)):
                if i + phrase_len * 2 > len(words):
                    break
                
                phrase1 = words[i:i + phrase_len]
                phrase2 = words[i + phrase_len:i + phrase_len * 2]
                
                # Check if phrases are identical (case-insensitive)
                if [w.lower() for w in phrase1] == [w.lower() for w in phrase2]:
                    # Check how many times this phrase repeats
                    repetitions = 1
                    pos = i + phrase_len
                    
                    while pos + phrase_len <= len(words):
                        next_phrase = words[pos:pos + phrase_len]
                        if [w.lower() for w in phrase1] == [w.lower() for w in next_phrase]:
                            repetitions += 1
                            pos += phrase_len
                        else:
                            break
                    
                    # If repetitions exceed threshold, keep only allowed number
                    if repetitions > self.config.max_phrase_repetitions + 1:
                        # Keep max_phrase_repetitions + 1 instances
                        keep_repetitions = self.config.max_phrase_repetitions + 1
                        cleaned_words.extend(words[i:i + phrase_len * keep_repetitions])
                        i = pos
                        found_repetition = True
                        break
            
            if not found_repetition:
                cleaned_words.append(words[i])
                i += 1
        
        return ' '.join(cleaned_words)
    
    def fix_spacing_punctuation(self, text: str) -> str:
        """
        Fix spacing and punctuation issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed spacing and punctuation
        """
        if not (self.config.fix_spacing or self.config.fix_punctuation):
            return text
        
        # Fix extra whitespace
        if self.config.remove_extra_whitespace:
            text = self.extra_whitespace_pattern.sub(' ', text)
        
        # Fix punctuation spacing
        if self.config.fix_punctuation:
            # Remove spaces before punctuation, add space after
            text = self.punct_space_pattern.sub(r'\1 ', text)
            
            # Handle special cases
            text = re.sub(r'\s*\.\s*\.\s*\.', '...', text)  # Ellipsis
            text = re.sub(r'\s*!\s*!\s*!', '!!!', text)      # Multiple exclamation
            text = re.sub(r'\s*\?\s*\?\s*\?', '???', text)   # Multiple question
            
            # Remove space before sentence end, ensure space after
            text = re.sub(r'\s+([.!?])', r'\1', text)
            text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def apply_truecasing(self, text: str) -> str:
        """
        Apply truecasing to restore proper capitalization.
        
        Args:
            text: Input text
            
        Returns:
            Text with proper capitalization
        """
        if not self.config.apply_truecasing:
            return text
        
        # Simple truecasing heuristics
        words = text.split()
        if not words:
            return text
        
        truecased_words = []
        
        for i, word in enumerate(words):
            # Skip sarcasm tokens
            if self.config.preserve_sarcasm_tokens and word in self.config.sarcasm_tokens:
                truecased_words.append(word)
                continue
            
            # Preserve all caps if configured
            if self.config.preserve_all_caps and word.isupper() and len(word) > 1:
                truecased_words.append(word)
                continue
            
            # First word of sentence should be capitalized
            if i == 0 or (i > 0 and words[i-1].endswith(('.', '!', '?'))):
                if word.isalpha():
                    truecased_words.append(word.capitalize())
                else:
                    truecased_words.append(word)
            
            # Common proper nouns (simple heuristic)
            elif word.lower() in {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                                  'january', 'february', 'march', 'april', 'may', 'june',
                                  'july', 'august', 'september', 'october', 'november', 'december',
                                  'python', 'java', 'javascript', 'english', 'french'}:
                truecased_words.append(word.capitalize())
            
            # Acronyms (all caps, length <= 5)
            elif word.isupper() and len(word) <= 5 and word.isalpha():
                truecased_words.append(word)
            
            # Default to lowercase
            else:
                truecased_words.append(word.lower())
        
        result = ' '.join(truecased_words)
        
        # Capitalize sentences if configured
        if self.config.capitalize_sentences and NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(result)
                capitalized_sentences = []
                
                for sentence in sentences:
                    if sentence and sentence[0].isalpha():
                        capitalized_sentences.append(sentence[0].upper() + sentence[1:])
                    else:
                        capitalized_sentences.append(sentence)
                
                result = ' '.join(capitalized_sentences)
            except:
                # Fallback to simple capitalization
                result = '. '.join(s.strip().capitalize() for s in result.split('.') if s.strip())
        
        return result
    
    def detokenize(self, text: str) -> str:
        """
        Detokenize text by joining subwords and fixing tokenization artifacts.
        
        Args:
            text: Input text
            
        Returns:
            Detokenized text
        """
        if not self.config.detokenize:
            return text
        
        # Join subwords
        if self.config.join_subwords:
            # Handle WordPiece tokens (##)
            text = re.sub(r'\s*##', '', text)
            
            # Handle BPE tokens (@@ )
            text = re.sub(r'@@\s*', '', text)
            
            # Handle SentencePiece tokens (▁)
            text = text.replace('▁', ' ')
            
            # Clean up extra spaces
            text = self.extra_whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def filter_by_length(self, text: str) -> bool:
        """
        Check if text meets length requirements.
        
        Args:
            text: Input text
            
        Returns:
            True if text passes length filters
        """
        if not self.config.filter_by_length:
            return True
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Check word count limits
        if word_count < self.config.min_length_words or word_count > self.config.max_length_words:
            return False
        
        # Check character count limits
        if char_count < self.config.min_length_chars or char_count > self.config.max_length_chars:
            return False
        
        return True
    
    def remove_incomplete_sentences(self, text: str) -> str:
        """
        Remove incomplete sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with incomplete sentences removed
        """
        if not self.config.remove_incomplete_sentences:
            return text
        
        # Simple heuristic: sentences should end with proper punctuation
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Last sentence might be incomplete if no ending punctuation
            if i == len(sentences) - 1 and not text.rstrip().endswith(('.', '!', '?')):
                # Check if it looks complete (has subject and predicate)
                words = sentence.split()
                if len(words) >= 3:  # Minimum for complete sentence
                    complete_sentences.append(sentence)
            else:
                complete_sentences.append(sentence)
        
        result = '. '.join(complete_sentences)
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result
    
    def clean_single(self, text: str) -> str:
        """
        Apply all post-processing steps to a single text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Apply processing steps in order
        processed_text = text
        
        # 1. Detokenization (first, to normalize token structure)
        processed_text = self.detokenize(processed_text)
        
        # 2. Remove redundancy
        processed_text = self.remove_redundancy(processed_text)
        
        # 3. Fix spacing and punctuation
        processed_text = self.fix_spacing_punctuation(processed_text)
        
        # 4. Apply truecasing
        processed_text = self.apply_truecasing(processed_text)
        
        # 5. Remove incomplete sentences
        processed_text = self.remove_incomplete_sentences(processed_text)
        
        # 6. Final cleanup
        processed_text = processed_text.strip()
        
        return processed_text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Apply post-processing to a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        if not texts:
            return []
        
        cleaned_texts = []
        processed_count = 0
        filtered_count = 0
        
        # Process texts
        if self.config.enable_parallel and len(texts) > 10:
            # Placeholder for parallel processing
            # In practice, would use multiprocessing or joblib
            self.logger.info("Parallel processing not implemented, using sequential")
            cleaned_texts = [self.clean_single(text) for text in texts]
        else:
            for text in texts:
                cleaned_text = self.clean_single(text)
                
                # Apply length filtering
                if self.filter_by_length(cleaned_text):
                    cleaned_texts.append(cleaned_text)
                    processed_count += 1
                else:
                    cleaned_texts.append("")  # Keep index alignment
                    filtered_count += 1
        
        self.logger.info(f"Post-processed {len(texts)} texts: "
                        f"{processed_count} kept, {filtered_count} filtered")
        
        return cleaned_texts
    
    def get_processing_stats(self, original_texts: List[str], cleaned_texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics on post-processing effects.
        
        Args:
            original_texts: Original texts
            cleaned_texts: Cleaned texts
            
        Returns:
            Dictionary of processing statistics
        """
        if len(original_texts) != len(cleaned_texts):
            raise ValueError("Original and cleaned text lists must have same length")
        
        stats = {
            'total_texts': len(original_texts),
            'kept_texts': sum(1 for text in cleaned_texts if text.strip()),
            'filtered_texts': sum(1 for text in cleaned_texts if not text.strip()),
            'avg_length_change': 0.0,
            'avg_word_change': 0.0,
            'length_reduction_ratio': 0.0
        }
        
        if stats['kept_texts'] > 0:
            length_changes = []
            word_changes = []
            
            for orig, cleaned in zip(original_texts, cleaned_texts):
                if cleaned.strip():  # Only compare kept texts
                    length_change = len(cleaned) - len(orig)
                    word_change = len(cleaned.split()) - len(orig.split())
                    
                    length_changes.append(length_change)
                    word_changes.append(word_change)
            
            if length_changes:
                stats['avg_length_change'] = sum(length_changes) / len(length_changes)
                stats['avg_word_change'] = sum(word_changes) / len(word_changes)
                
                original_total_length = sum(len(orig) for orig in original_texts)
                cleaned_total_length = sum(len(cleaned) for cleaned in cleaned_texts if cleaned.strip())
                
                if original_total_length > 0:
                    stats['length_reduction_ratio'] = 1.0 - (cleaned_total_length / original_total_length)
        
        return stats


def main():
    """Example usage of PostProcessor."""
    
    # Sample texts with various issues
    raw_texts = [
        "the   weather is  nice nice nice today .",
        "TODAY'S weather IS LOVELY AND WONDERFUL",
        "<sarcastic> oh great , another monday meeting </sarcastic>",
        "this is way way way too repetitive repetitive text text text",
        "hello world hello world hello world this repeats",
        "i love programming programming i love programming programming",
        "   spaced    badly   punctuation,like this!and   this?  ",
        "ALLCAPSTEXT needs fixing NOW PLEASE",
        "incomplete sentence without",
        "t5 token ##ization needs fix ##ing",
        "This is a@@@ proper@@@ BPE@@@ token@@@ example"
    ]
    
    # Initialize post-processor with custom config
    config = PostProcessingConfig(
        fix_spacing=True,
        fix_punctuation=True,
        remove_word_repetitions=True,
        remove_phrase_repetitions=True,
        max_word_repetitions=1,
        max_phrase_repetitions=1,
        apply_truecasing=True,
        capitalize_sentences=True,
        detokenize=True,
        filter_by_length=True,
        min_length_words=2,
        preserve_sarcasm_tokens=True
    )
    
    processor = PostProcessor(config)
    
    print("=== Post-Processing Example ===")
    print(f"Processing {len(raw_texts)} raw texts")
    
    # Clean batch
    cleaned_texts = processor.clean_batch(raw_texts)
    
    print("\n=== Results ===")
    for i, (original, cleaned) in enumerate(zip(raw_texts, cleaned_texts)):
        print(f"\n{i+1}. Original: {repr(original)}")
        print(f"   Cleaned:  {repr(cleaned)}")
        if not cleaned.strip():
            print("   [FILTERED OUT]")
    
    # Get processing statistics
    stats = processor.get_processing_stats(raw_texts, cleaned_texts)
    
    print(f"\n=== Processing Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test individual processing steps
    print(f"\n=== Individual Processing Steps ===")
    test_text = "the   weather is  nice nice nice today ."
    
    print(f"Original: {repr(test_text)}")
    
    # Step by step
    step1 = processor.detokenize(test_text)
    print(f"Detokenized: {repr(step1)}")
    
    step2 = processor.remove_redundancy(step1)
    print(f"Redundancy removed: {repr(step2)}")
    
    step3 = processor.fix_spacing_punctuation(step2)
    print(f"Spacing fixed: {repr(step3)}")
    
    step4 = processor.apply_truecasing(step3)
    print(f"Truecased: {repr(step4)}")
    
    final = processor.clean_single(test_text)
    print(f"Final result: {repr(final)}")


if __name__ == "__main__":
    main()
