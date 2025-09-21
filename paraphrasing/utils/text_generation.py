#!/usr/bin/env python3
"""
Text Generation Utilities for Paraphrase Models

Comprehensive decoding strategies for seq2seq models including greedy search,
beam search, top-k sampling, and nucleus sampling with temperature control
and repetition penalty. Supports both single and batch generation.

Example Usage:
    >>> from paraphrasing.utils import generate_text
    >>> from transformers import T5ForConditionalGeneration, T5Tokenizer
    >>> 
    >>> # Load model and tokenizer
    >>> model = T5ForConditionalGeneration.from_pretrained("t5-base")
    >>> tokenizer = T5Tokenizer.from_pretrained("t5-base")
    >>> 
    >>> # Prepare inputs
    >>> inputs = tokenizer("paraphrase: The weather is nice", return_tensors="pt")
    >>> 
    >>> # Generate with different strategies
    >>> beam_output = generate_text(model, tokenizer, inputs, strategy="beam", num_beams=4)
    >>> nucleus_output = generate_text(model, tokenizer, inputs, strategy="nucleus", top_p=0.9)
    >>> print(f"Beam search: {beam_output}")
    >>> print(f"Nucleus sampling: {nucleus_output}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Import transformers components
try:
    from transformers import (
        PreTrainedModel, PreTrainedTokenizer, 
        GenerationConfig, GenerationMixin
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    

@dataclass
class GenerationParams:
    """Parameters for text generation."""
    
    # Decoding strategy
    strategy: str = "beam"  # "greedy", "beam", "top_k", "nucleus"
    
    # Common parameters
    max_length: int = 128
    min_length: int = 1
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 2
    
    # Beam search parameters
    num_beams: int = 4
    num_beam_groups: int = 1
    early_stopping: bool = True
    
    # Sampling parameters
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Batch parameters
    num_return_sequences: int = 1
    batch_size: int = 1
    
    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # Advanced parameters
    diversity_penalty: float = 0.0
    output_scores: bool = False
    return_dict_in_generate: bool = True


def greedy_search(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 128,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Greedy search decoding.
    
    Args:
        model: Pre-trained seq2seq model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        max_length: Maximum generation length
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Generated token IDs [batch_size, generated_length]
    """
    logger = get_logger("greedy_search")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )
        
        return generated_ids
        
    except Exception as e:
        logger.error(f"Greedy search failed: {e}")
        raise


def beam_search(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_beams: int = 4,
    max_length: int = 128,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    **kwargs
) -> torch.Tensor:
    """
    Beam search decoding.
    
    Args:
        model: Pre-trained seq2seq model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        num_beams: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor
        early_stopping: Whether to stop early when EOS is generated
        num_return_sequences: Number of sequences to return per input
        
    Returns:
        Generated token IDs [batch_size * num_return_sequences, generated_length]
    """
    logger = get_logger("beam_search")
    
    if num_return_sequences > num_beams:
        logger.warning(f"num_return_sequences ({num_return_sequences}) > num_beams ({num_beams})")
        num_return_sequences = num_beams
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                **kwargs
            )
        
        return generated_ids
        
    except Exception as e:
        logger.error(f"Beam search failed: {e}")
        raise


def top_k_sampling(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 50,
    temperature: float = 1.0,
    max_length: int = 128,
    repetition_penalty: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Top-k sampling decoding.
    
    Args:
        model: Pre-trained seq2seq model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        top_k: Number of top tokens to sample from
        temperature: Sampling temperature
        max_length: Maximum generation length
        repetition_penalty: Repetition penalty factor
        
    Returns:
        Generated token IDs [batch_size, generated_length]
    """
    logger = get_logger("top_k_sampling")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=top_k,
                top_p=1.0,  # Disable nucleus sampling
                temperature=temperature,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        return generated_ids
        
    except Exception as e:
        logger.error(f"Top-k sampling failed: {e}")
        raise


def nucleus_sampling(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    top_p: float = 0.95,
    temperature: float = 1.0,
    max_length: int = 128,
    repetition_penalty: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling decoding.
    
    Args:
        model: Pre-trained seq2seq model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        top_p: Nucleus sampling threshold
        temperature: Sampling temperature
        max_length: Maximum generation length
        repetition_penalty: Repetition penalty factor
        
    Returns:
        Generated token IDs [batch_size, generated_length]
    """
    logger = get_logger("nucleus_sampling")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=0,  # Disable top-k
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        return generated_ids
        
    except Exception as e:
        logger.error(f"Nucleus sampling failed: {e}")
        raise


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
    strategy: str = "beam",
    decode: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
    **kwargs
) -> Union[List[str], torch.Tensor]:
    """
    Unified text generation interface with multiple decoding strategies.
    
    Args:
        model: Pre-trained seq2seq model
        tokenizer: Model tokenizer
        inputs: Input tensors (dict with input_ids, attention_mask) or input_ids tensor
        strategy: Decoding strategy ("greedy", "beam", "top_k", "nucleus")
        decode: Whether to decode token IDs to text
        skip_special_tokens: Whether to skip special tokens in decoding
        clean_up_tokenization_spaces: Whether to clean up tokenization spaces
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text(s) if decode=True, otherwise token IDs
        
    Example:
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-base")
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-base")
        >>> inputs = tokenizer("paraphrase: Hello world", return_tensors="pt")
        >>> 
        >>> # Beam search generation
        >>> output = generate_text(model, tokenizer, inputs, strategy="beam", num_beams=4)
        >>> print(f"Generated: {output}")
        >>> 
        >>> # Nucleus sampling
        >>> output = generate_text(model, tokenizer, inputs, strategy="nucleus", top_p=0.9)
        >>> print(f"Generated: {output}")
    """
    logger = get_logger("generate_text")
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available")
    
    # Extract input components
    if isinstance(inputs, dict):
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
    else:
        input_ids = inputs
        attention_mask = None
    
    if input_ids is None:
        raise ValueError("input_ids must be provided")
    
    # Set default token IDs if not provided
    if 'pad_token_id' not in kwargs:
        kwargs['pad_token_id'] = tokenizer.pad_token_id
    if 'eos_token_id' not in kwargs:
        kwargs['eos_token_id'] = tokenizer.eos_token_id
    
    # Generation parameters
    generation_params = GenerationParams(**kwargs)
    
    logger.debug(f"Generating text with strategy: {strategy}")
    
    try:
        # Route to appropriate generation function
        if strategy == "greedy":
            generated_ids = greedy_search(
                model, input_ids, attention_mask,
                max_length=generation_params.max_length,
                pad_token_id=generation_params.pad_token_id,
                eos_token_id=generation_params.eos_token_id,
                repetition_penalty=generation_params.repetition_penalty,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size
            )
            
        elif strategy == "beam":
            generated_ids = beam_search(
                model, input_ids, attention_mask,
                num_beams=generation_params.num_beams,
                max_length=generation_params.max_length,
                length_penalty=generation_params.length_penalty,
                early_stopping=generation_params.early_stopping,
                num_return_sequences=generation_params.num_return_sequences,
                pad_token_id=generation_params.pad_token_id,
                eos_token_id=generation_params.eos_token_id,
                repetition_penalty=generation_params.repetition_penalty,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size
            )
            
        elif strategy == "top_k":
            generated_ids = top_k_sampling(
                model, input_ids, attention_mask,
                top_k=generation_params.top_k,
                temperature=generation_params.temperature,
                max_length=generation_params.max_length,
                repetition_penalty=generation_params.repetition_penalty,
                pad_token_id=generation_params.pad_token_id,
                eos_token_id=generation_params.eos_token_id,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size
            )
            
        elif strategy == "nucleus":
            generated_ids = nucleus_sampling(
                model, input_ids, attention_mask,
                top_p=generation_params.top_p,
                temperature=generation_params.temperature,
                max_length=generation_params.max_length,
                repetition_penalty=generation_params.repetition_penalty,
                pad_token_id=generation_params.pad_token_id,
                eos_token_id=generation_params.eos_token_id,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size
            )
            
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")
        
        # Return token IDs if decode=False
        if not decode:
            return generated_ids
        
        # Decode to text
        if generated_ids.dim() == 2:
            # Multiple sequences
            generated_texts = []
            for sequence in generated_ids:
                text = tokenizer.decode(
                    sequence,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces
                )
                generated_texts.append(text)
            
            # Return single text if only one sequence
            if len(generated_texts) == 1:
                return generated_texts[0]
            return generated_texts
        else:
            # Single sequence
            return tokenizer.decode(
                generated_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            
    except Exception as e:
        logger.error(f"Text generation failed with strategy '{strategy}': {e}")
        raise


def batch_generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_batch: List[Dict[str, torch.Tensor]],
    strategy: str = "beam",
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Generate text for a batch of inputs with automatic batching.
    
    Args:
        model: Pre-trained seq2seq model
        tokenizer: Model tokenizer
        input_batch: List of input tensors
        strategy: Generation strategy
        batch_size: Processing batch size
        **kwargs: Generation parameters
        
    Returns:
        List of generated texts
    """
    logger = get_logger("batch_generate_text")
    
    all_outputs = []
    
    for i in range(0, len(input_batch), batch_size):
        batch_inputs = input_batch[i:i + batch_size]
        
        # Combine batch inputs
        combined_inputs = {}
        for key in batch_inputs[0].keys():
            combined_inputs[key] = torch.cat([inp[key] for inp in batch_inputs], dim=0)
        
        try:
            # Generate for batch
            batch_outputs = generate_text(
                model, tokenizer, combined_inputs,
                strategy=strategy, **kwargs
            )
            
            # Ensure outputs is a list
            if isinstance(batch_outputs, str):
                batch_outputs = [batch_outputs]
            
            all_outputs.extend(batch_outputs)
            
        except Exception as e:
            logger.error(f"Batch generation failed for batch {i//batch_size + 1}: {e}")
            # Add empty strings for failed batch
            all_outputs.extend([""] * len(batch_inputs))
    
    logger.info(f"Generated {len(all_outputs)} texts using {strategy} strategy")
    return all_outputs


def configure_generation_for_sarcasm(
    generation_params: GenerationParams,
    preserve_sarcasm_tokens: bool = True
) -> GenerationParams:
    """
    Configure generation parameters for sarcasm-aware models.
    
    Args:
        generation_params: Base generation parameters
        preserve_sarcasm_tokens: Whether to preserve sarcasm control tokens
        
    Returns:
        Modified generation parameters
    """
    # Adjust parameters for sarcasm-aware generation
    modified_params = generation_params
    
    if preserve_sarcasm_tokens:
        # Lower repetition penalty to allow sarcasm tokens
        modified_params.repetition_penalty = max(1.1, generation_params.repetition_penalty - 0.1)
        
        # Reduce no_repeat_ngram_size for flexibility
        modified_params.no_repeat_ngram_size = max(1, generation_params.no_repeat_ngram_size - 1)
    
    return modified_params


def main():
    """Example usage of text generation utilities."""
    
    print("=== Text Generation Utilities Example ===")
    
    # Note: This example requires actual model loading
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        print("Loading T5 model...")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        # Test inputs
        test_texts = [
            "paraphrase: The weather is beautiful today.",
            "paraphrase: I love programming in Python."
        ]
        
        print(f"\nTesting generation strategies:")
        
        for i, text in enumerate(test_texts):
            print(f"\n--- Input {i+1}: {text} ---")
            
            # Prepare inputs
            inputs = tokenizer(text, return_tensors="pt")
            
            # Test different strategies
            strategies = {
                "greedy": {"strategy": "greedy"},
                "beam": {"strategy": "beam", "num_beams": 3},
                "top_k": {"strategy": "top_k", "top_k": 50, "temperature": 0.8},
                "nucleus": {"strategy": "nucleus", "top_p": 0.9, "temperature": 0.8}
            }
            
            for strategy_name, params in strategies.items():
                try:
                    output = generate_text(model, tokenizer, inputs, **params)
                    print(f"{strategy_name:>8}: {output}")
                except Exception as e:
                    print(f"{strategy_name:>8}: Error - {e}")
        
        # Test batch generation
        print(f"\n--- Batch Generation ---")
        batch_inputs = [tokenizer(text, return_tensors="pt") for text in test_texts]
        
        try:
            batch_outputs = batch_generate_text(
                model, tokenizer, batch_inputs,
                strategy="beam", num_beams=2, batch_size=2
            )
            
            for i, output in enumerate(batch_outputs):
                print(f"Batch {i+1}: {output}")
                
        except Exception as e:
            print(f"Batch generation error: {e}")
        
    except ImportError:
        print("Transformers library not available - showing parameter examples only")
        
        # Show parameter configuration examples
        params = GenerationParams(
            strategy="beam",
            num_beams=4,
            max_length=64,
            temperature=0.8,
            repetition_penalty=1.2
        )
        
        print(f"Example parameters: {params}")
        
        # Show sarcasm configuration
        sarcasm_params = configure_generation_for_sarcasm(params, preserve_sarcasm_tokens=True)
        print(f"Sarcasm-aware parameters: {sarcasm_params}")


if __name__ == "__main__":
    main()
