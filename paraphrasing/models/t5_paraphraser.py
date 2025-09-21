#!/usr/bin/env python3
"""
T5-Based Paraphrase Generation Model

Implements T5ForConditionalGeneration for paraphrase generation with advanced
decoding strategies including beam search, nucleus sampling, and top-k sampling.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, T5Config,
    GenerationConfig, get_linear_schedule_with_warmup
)
from dataclasses import dataclass, field
import json
import logging
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.utils.checkpoint_manager import CheckpointManager
from shared.utils.logging_utils import get_logger
from shared.utils.metrics import calculate_bleu_score, calculate_rouge_score


@dataclass
class T5ParaphraserConfig:
    """Configuration for T5-based paraphraser."""
    
    # Model architecture
    model_name: str = "t5-base"
    vocab_size: int = 32128
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    
    # Training parameters
    max_input_length: int = 128
    max_target_length: int = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Generation parameters
    num_beams: int = 4
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    
    # Special tokens
    task_prefix: str = "paraphrase: "
    bos_token: str = "<pad>"
    eos_token: str = "</s>"
    
    # Hardware optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4


class T5Paraphraser(BaseMultimodalModel):
    """
    T5-based paraphrase generation model with advanced decoding capabilities.
    
    Supports teacher forcing training, multiple generation strategies,
    and comprehensive checkpoint management.
    """
    
    def __init__(self, config: T5ParaphraserConfig):
        """
        Initialize T5 paraphraser.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger("T5Paraphraser")
        
        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        self._setup_special_tokens()
        
        # Initialize model
        if hasattr(config, 'from_scratch') and config.from_scratch:
            t5_config = T5Config.from_pretrained(config.model_name)
            t5_config.vocab_size = len(self.tokenizer)
            self.model = T5ForConditionalGeneration(t5_config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        
        # Resize token embeddings if vocabulary changed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_length=config.max_target_length,
            num_beams=config.num_beams,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            do_sample=True,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.pad_token_id
        )
        
        self.logger.info(f"Initialized T5 paraphraser with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _setup_special_tokens(self):
        """Setup special tokens for paraphrasing task."""
        
        special_tokens = {
            'additional_special_tokens': ['<paraphrase>', '<sarcastic>', '<formal>', '<informal>']
        }
        
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            self.logger.info(f"Added {num_added_tokens} special tokens")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for T5 paraphraser.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            labels: Target labels for training
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            **kwargs
        )
        
        return {
            'loss': outputs.loss if outputs.loss is not None else torch.tensor(0.0),
            'logits': outputs.logits,
            'hidden_states': outputs.encoder_last_hidden_state,
            'decoder_hidden_states': outputs.last_hidden_state
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_strategy: str = "beam_search",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate paraphrases using specified strategy.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            generation_strategy: Generation strategy ("beam_search", "nucleus", "top_k", "greedy")
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        self.model.eval()
        
        with torch.no_grad():
            # Update generation config based on strategy
            gen_config = self._get_generation_config(generation_strategy, **generation_kwargs)
            
            # Generate sequences
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode generated sequences
            generated_sequences = []
            for seq in generated_ids.sequences:
                # Remove special tokens and decode
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated_sequences.append(decoded)
            
            # Calculate generation metrics
            scores = generated_ids.sequences_scores if hasattr(generated_ids, 'sequences_scores') else None
            
            return {
                'generated_sequences': generated_sequences,
                'generated_ids': generated_ids.sequences,
                'scores': scores,
                'generation_config': gen_config
            }
    
    def _get_generation_config(self, strategy: str, **kwargs) -> GenerationConfig:
        """Get generation configuration for specified strategy."""
        
        config = GenerationConfig(**self.generation_config.to_dict())
        
        if strategy == "beam_search":
            config.update({
                'num_beams': kwargs.get('num_beams', self.config.num_beams),
                'do_sample': False,
                'early_stopping': True
            })
        
        elif strategy == "nucleus":
            config.update({
                'do_sample': True,
                'top_p': kwargs.get('top_p', self.config.top_p),
                'top_k': 0,
                'num_beams': 1,
                'temperature': kwargs.get('temperature', self.config.temperature)
            })
        
        elif strategy == "top_k":
            config.update({
                'do_sample': True,
                'top_k': kwargs.get('top_k', self.config.top_k),
                'top_p': 1.0,
                'num_beams': 1,
                'temperature': kwargs.get('temperature', self.config.temperature)
            })
        
        elif strategy == "greedy":
            config.update({
                'do_sample': False,
                'num_beams': 1,
                'top_k': 0,
                'top_p': 1.0
            })
        
        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            labels: Target labels
            
        Returns:
            Training loss
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs['loss']
    
    def prepare_inputs(
        self,
        source_texts: List[str],
        target_texts: Optional[List[str]] = None,
        add_task_prefix: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training or generation.
        
        Args:
            source_texts: Source texts to paraphrase
            target_texts: Target paraphrases (for training)
            add_task_prefix: Whether to add task prefix
            
        Returns:
            Dictionary of tokenized inputs
        """
        # Add task prefix if requested
        if add_task_prefix:
            source_texts = [self.config.task_prefix + text for text in source_texts]
        
        # Tokenize source texts
        source_encoding = self.tokenizer(
            source_texts,
            max_length=self.config.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {
            'input_ids': source_encoding.input_ids,
            'attention_mask': source_encoding.attention_mask
        }
        
        # Tokenize target texts if provided
        if target_texts:
            target_encoding = self.tokenizer(
                target_texts,
                max_length=self.config.max_target_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            inputs.update({
                'labels': target_encoding.input_ids,
                'decoder_attention_mask': target_encoding.attention_mask
            })
        
        return inputs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            scaler: Mixed precision scaler
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Forward pass
        if scaler and self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(**batch)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self.compute_loss(**batch)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        return {'loss': loss.item()}
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        target_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model on a batch.
        
        Args:
            batch: Evaluation batch
            target_texts: Target texts for metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate paraphrases
            generated = self.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                generation_strategy="beam_search"
            )
            
            generated_texts = generated['generated_sequences']
            
            # Calculate metrics
            metrics = {}
            
            # BLEU score
            bleu_scores = []
            for gen_text, target_text in zip(generated_texts, target_texts):
                bleu = calculate_bleu_score([target_text], gen_text)
                bleu_scores.append(bleu)
            
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
            
            # ROUGE score
            rouge_scores = []
            for gen_text, target_text in zip(generated_texts, target_texts):
                rouge = calculate_rouge_score(target_text, gen_text)
                rouge_scores.append(rouge['rouge-l']['f'])
            
            metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores)
            
            # Loss
            loss = self.compute_loss(**batch)
            metrics['loss'] = loss.item()
        
        return metrics
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and tokenizer.
        
        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = save_path / "paraphraser_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[T5ParaphraserConfig] = None
    ) -> 'T5Paraphraser':
        """
        Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to saved model
            config: Optional configuration override
            
        Returns:
            Loaded T5Paraphraser instance
        """
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "paraphraser_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = T5ParaphraserConfig(**config_dict)
            else:
                config = T5ParaphraserConfig()
        
        # Create instance
        instance = cls(config)
        
        # Load model state
        instance.model = T5ForConditionalGeneration.from_pretrained(model_path)
        instance.tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'T5Paraphraser',
            'base_model': self.config.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'vocab_size': len(self.tokenizer),
            'max_input_length': self.config.max_input_length,
            'max_target_length': self.config.max_target_length
        }


def main():
    """Example usage of T5 paraphraser."""
    
    # Configuration
    config = T5ParaphraserConfig(
        model_name="t5-small",  # Use small model for testing
        max_input_length=64,
        max_target_length=64
    )
    
    # Create model
    model = T5Paraphraser(config)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test text preparation
    source_texts = [
        "The weather is beautiful today.",
        "I love programming in Python."
    ]
    target_texts = [
        "Today's weather is wonderful.",
        "Python programming is something I enjoy."
    ]
    
    # Prepare inputs
    inputs = model.prepare_inputs(source_texts, target_texts)
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items() if torch.is_tensor(v)]}")
    
    # Test forward pass
    outputs = model(**inputs)
    print(f"Training loss: {outputs['loss']:.4f}")
    
    # Test generation
    generated = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        generation_strategy="beam_search"
    )
    
    print("\nGenerated paraphrases:")
    for i, (source, generated_text) in enumerate(zip(source_texts, generated['generated_sequences'])):
        print(f"{i+1}. Source: {source}")
        print(f"   Generated: {generated_text}")
    
    # Test saving and loading
    save_dir = "test_t5_model"
    model.save_pretrained(save_dir)
    
    # Load model
    loaded_model = T5Paraphraser.from_pretrained(save_dir)
    print(f"Loaded model info: {loaded_model.get_model_info()}")


if __name__ == "__main__":
    main()
