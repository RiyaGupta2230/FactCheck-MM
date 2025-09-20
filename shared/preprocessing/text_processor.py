"""
Text Preprocessing for FactCheck-MM
Handles tokenization, cleaning, and sarcasm-specific preprocessing.
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

from ..utils import get_logger


class TextProcessor:
    """
    Comprehensive text processor for multimodal fact-checking.
    Handles tokenization, cleaning, and task-specific preprocessing.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-large",
        max_length: int = 512,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        add_sarcasm_markers: bool = True,
        clean_html: bool = True,
        normalize_whitespace: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize text processor.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to apply lemmatization
            add_sarcasm_markers: Whether to add sarcasm detection markers
            clean_html: Whether to clean HTML tags
            normalize_whitespace: Whether to normalize whitespace
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.add_sarcasm_markers = add_sarcasm_markers
        self.clean_html = clean_html
        self.normalize_whitespace = normalize_whitespace
        
        self.logger = get_logger("TextProcessor")
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                use_fast=True
            )
            self.logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer {model_name}: {e}")
            raise
        
        # Initialize NLP tools if needed
        self.nlp_tools = {}
        
        if remove_stopwords:
            try:
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except:
                self.logger.warning("Failed to load stopwords, disabling stop word removal")
                self.remove_stopwords = False
        
        if lemmatize:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except:
                self.logger.warning("Failed to load lemmatizer, disabling lemmatization")
                self.lemmatize = False
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Sarcasm markers
        self.sarcasm_patterns = [
            r'/s\b',  # Reddit-style sarcasm marker
            r'\bsarcasm\b',
            r'\birony\b',
            r'\boh really\b',
            r'\byeah right\b',
            r'\bsure thing\b',
            r'\bof course\b(?=.*[.!?])',  # "of course" with punctuation
        ]
        
        self.logger.info("Text processor initialized successfully")
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        self.patterns = {
            'html': re.compile(r'<[^<]+?>'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'mentions': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtags': re.compile(r'#[A-Za-z0-9_]+'),
            'whitespace': re.compile(r'\s+'),
            'non_ascii': re.compile(r'[^\x00-\x7F]+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),  # More than 2 repeated characters
            'numbers': re.compile(r'\b\d+\b'),
        }
    
    def clean_text(
        self, 
        text: str,
        preserve_case: bool = False,
        preserve_urls: bool = False,
        preserve_mentions: bool = False
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            preserve_urls: Whether to preserve URLs
            preserve_mentions: Whether to preserve @mentions
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove HTML tags
        if self.clean_html:
            text = self.patterns['html'].sub(' ', text)
        
        # Remove URLs unless preserved
        if not preserve_urls:
            text = self.patterns['urls'].sub(' [URL] ', text)
        
        # Handle mentions
        if not preserve_mentions:
            text = self.patterns['mentions'].sub(' [USER] ', text)
        else:
            text = self.patterns['mentions'].sub(' ', text)
        
        # Remove emails
        text = self.patterns['emails'].sub(' [EMAIL] ', text)
        
        # Handle hashtags (preserve the text part)
        text = self.patterns['hashtags'].sub(lambda m: m.group(0)[1:], text)
        
        # Normalize repeated characters (but keep some for emotion: "sooo" -> "soo")
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Handle case
        if self.lowercase and not preserve_case:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.patterns['whitespace'].sub(' ', text).strip()
        
        return text
    
    def detect_sarcasm_markers(self, text: str) -> Dict[str, Any]:
        """
        Detect sarcasm markers in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sarcasm detection results
        """
        results = {
            'has_markers': False,
            'markers_found': [],
            'confidence': 0.0,
            'processed_text': text
        }
        
        if not self.add_sarcasm_markers:
            return results
        
        text_lower = text.lower()
        markers_found = []
        
        for pattern in self.sarcasm_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                markers_found.extend(matches)
        
        # Additional heuristics
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        quotation_marks = text.count('"') + text.count("'")
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Simple scoring
        score = len(markers_found) * 0.4
        score += min(question_marks * 0.1, 0.3)
        score += min(exclamation_marks * 0.1, 0.2)
        score += min(caps_ratio * 0.3, 0.2)
        
        results.update({
            'has_markers': len(markers_found) > 0,
            'markers_found': list(set(markers_found)),
            'confidence': min(score, 1.0),
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'caps_ratio': caps_ratio
        })
        
        return results
    
    def preprocess_text(
        self,
        text: str,
        task: str = "classification",
        add_special_tokens: bool = True
    ) -> str:
        """
        Apply task-specific text preprocessing.
        
        Args:
            text: Input text
            task: Task type ('classification', 'generation', 'fact_checking')
            add_special_tokens: Whether to add task-specific tokens
            
        Returns:
            Preprocessed text
        """
        # Clean text
        processed = self.clean_text(text)
        
        # Task-specific processing
        if task == "sarcasm_detection" and self.add_sarcasm_markers:
            sarcasm_info = self.detect_sarcasm_markers(processed)
            if sarcasm_info['has_markers']:
                processed = f"[SARCASM_DETECTED] {processed}"
        
        elif task == "fact_verification":
            # Add claim markers for fact verification
            if add_special_tokens:
                processed = f"[CLAIM] {processed}"
        
        elif task == "paraphrasing":
            # Add paraphrase markers
            if add_special_tokens:
                processed = f"[PARAPHRASE] {processed}"
        
        # Apply NLP preprocessing if enabled
        if self.remove_stopwords or self.lemmatize:
            processed = self._apply_nlp_preprocessing(processed)
        
        return processed
    
    def _apply_nlp_preprocessing(self, text: str) -> str:
        """Apply NLP preprocessing like stopword removal and lemmatization."""
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
        
        return ' '.join(tokens)
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using the configured tokenizer.
        
        Args:
            text: Input text(s)
            text_pair: Optional second text(s) for pair tasks
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized inputs
        """
        try:
            return self.tokenizer(
                text,
                text_pair=text_pair,
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_tensors=return_tensors,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise
    
    def process_batch(
        self,
        texts: List[str],
        task: str = "classification",
        **tokenize_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
            task: Task type
            **tokenize_kwargs: Additional tokenization arguments
            
        Returns:
            Batch of tokenized inputs
        """
        # Preprocess all texts
        processed_texts = [
            self.preprocess_text(text, task=task) 
            for text in texts
        ]
        
        # Tokenize batch
        return self.tokenize(processed_texts, **tokenize_kwargs)
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text(s)
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'cls_token_id': getattr(self.tokenizer, 'cls_token_id', None),
            'sep_token_id': getattr(self.tokenizer, 'sep_token_id', None),
            'mask_token_id': getattr(self.tokenizer, 'mask_token_id', None),
        }
    
    def __repr__(self) -> str:
        return (
            f"TextProcessor(\n"
            f"  model_name='{self.model_name}',\n"
            f"  max_length={self.max_length},\n"
            f"  vocab_size={self.get_vocab_size()},\n"
            f"  lowercase={self.lowercase},\n"
            f"  sarcasm_markers={self.add_sarcasm_markers}\n"
            f")"
        )
