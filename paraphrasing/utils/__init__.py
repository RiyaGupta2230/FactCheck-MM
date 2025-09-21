"""
Paraphrasing Utilities Module

This module provides essential utilities for paraphrase generation including
text generation/decoding strategies, diversity metrics computation, and 
post-processing refinement tools.

Key components:
- generate_text: Unified interface for text generation with various decoding strategies
- DiversityMetrics: Comprehensive lexical and semantic diversity analysis
- PostProcessor: Text cleaning and refinement utilities

Example Usage:
    >>> from paraphrasing.utils import generate_text, DiversityMetrics, PostProcessor
    >>> 
    >>> # Generate text with beam search
    >>> generated = generate_text(model, tokenizer, inputs, strategy="beam", num_beams=4)
    >>> 
    >>> # Analyze diversity
    >>> diversity = DiversityMetrics()
    >>> diversity.update(generated_texts)
    >>> metrics = diversity.compute()
    >>> print(f"Distinct-2: {metrics['distinct_2']:.3f}")
    >>> 
    >>> # Clean generated text
    >>> processor = PostProcessor()
    >>> cleaned = processor.clean_batch(generated_texts)
"""

from .text_generation import generate_text
from .diversity_metrics import DiversityMetrics  
from .post_processing import PostProcessor

__all__ = [
    "generate_text",
    "DiversityMetrics",
    "PostProcessor"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
