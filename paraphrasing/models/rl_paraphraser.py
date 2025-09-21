#!/usr/bin/env python3
"""
Reinforcement Learning-Based Paraphrase Generation

Implements REINFORCE algorithm for paraphrase generation with composite reward function
including semantic similarity, fluency, diversity, and copying penalty.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig
from dataclasses import dataclass, field
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import math

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.utils.logging_utils import get_logger
from .t5_paraphraser import T5Paraphraser, T5ParaphraserConfig
from .bart_paraphraser import BARTParaphraser, BARTParaphraserConfig
from .quality_scorer import QualityScorer


@dataclass
class RLParaphraserConfig:
    """Configuration for reinforcement learning paraphraser."""
    
    # Base model configuration
    base_model_type: str = "t5"  # "t5" or "bart"
    base_model_name: str = "t5-base"
    pretrained_model_path: Optional[str] = None
    
    # RL training parameters
    rl_learning_rate: float = 1e-5
    baseline_momentum: float = 0.9
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    
    # Reward function weights
    semantic_similarity_weight: float = 0.4
    fluency_weight: float = 0.3
    diversity_weight: float = 0.2
    copy_penalty_weight: float = 0.1
    
    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_length: int = 128
    
    # Reward computation
    similarity_model_name: str = "all-MiniLM-L6-v2"
    fluency_model_path: Optional[str] = None
    quality_scorer_path: Optional[str] = None
    
    # Training configuration
    num_samples_per_input: int = 4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Diversity metrics
    distinct_n: int = 3  # For distinct-n diversity metric
    
    # Hardware optimization
    mixed_precision: bool = True
    device: str = "auto"


class RewardFunction:
    """Composite reward function for paraphrase quality assessment."""
    
    def __init__(self, config: RLParaphraserConfig):
        """
        Initialize reward function.
        
        Args:
            config: RL paraphraser configuration
        """
        self.config = config
        self.logger = get_logger("RewardFunction")
        
        # Initialize semantic similarity model
        self.similarity_model = SentenceTransformer(config.similarity_model_name)
        
        # Initialize fluency model (GPT-2 for perplexity)
        self.fluency_model = None
        self.fluency_tokenizer = None
        self._setup_fluency_model()
        
        # Initialize quality scorer if available
        self.quality_scorer = None
        if config.quality_scorer_path:
            self.quality_scorer = self._load_quality_scorer()
        
        self.logger.info("Initialized composite reward function")
    
    def _setup_fluency_model(self):
        """Setup fluency model for perplexity computation."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            model_name = self.config.fluency_model_path or "gpt2"
            self.fluency_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.fluency_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            # Set pad token
            if self.fluency_tokenizer.pad_token is None:
                self.fluency_tokenizer.pad_token = self.fluency_tokenizer.eos_token
            
            self.fluency_model.eval()
            
        except Exception as e:
            self.logger.warning(f"Failed to load fluency model: {e}")
    
    def _load_quality_scorer(self) -> Optional[QualityScorer]:
        """Load pretrained quality scorer."""
        try:
            return QualityScorer.from_pretrained(self.config.quality_scorer_path)
        except Exception as e:
            self.logger.warning(f"Failed to load quality scorer: {e}")
            return None
    
    def compute_semantic_similarity(
        self,
        source_texts: List[str],
        generated_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute semantic similarity between source and generated texts.
        
        Args:
            source_texts: Source texts
            generated_texts: Generated paraphrases
            
        Returns:
            Similarity scores [batch_size]
        """
        # Encode texts
        source_embeddings = self.similarity_model.encode(source_texts, convert_to_tensor=True)
        generated_embeddings = self.similarity_model.encode(generated_texts, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(source_embeddings, generated_embeddings, dim=1)
        
        return similarities.clamp(0, 1)  # Ensure positive values
    
    def compute_fluency(self, texts: List[str]) -> torch.Tensor:
        """
        Compute fluency scores based on language model perplexity.
        
        Args:
            texts: Texts to evaluate
            
        Returns:
            Fluency scores [batch_size]
        """
        if self.fluency_model is None:
            return torch.ones(len(texts))
        
        scores = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                inputs = self.fluency_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Compute perplexity
                outputs = self.fluency_model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                # Convert to fluency score (inverse of perplexity, normalized)
                fluency = 1.0 / (1.0 + perplexity.item() / 100.0)
                scores.append(fluency)
        
        return torch.tensor(scores)
    
    def compute_diversity(self, texts: List[str]) -> torch.Tensor:
        """
        Compute diversity scores using distinct-n metric.
        
        Args:
            texts: Texts to evaluate
            
        Returns:
            Diversity scores [batch_size]
        """
        scores = []
        
        for text in texts:
            tokens = text.lower().split()
            
            if len(tokens) < self.config.distinct_n:
                scores.append(0.0)
                continue
            
            # Generate n-grams
            ngrams = set()
            for i in range(len(tokens) - self.config.distinct_n + 1):
                ngram = tuple(tokens[i:i + self.config.distinct_n])
                ngrams.add(ngram)
            
            # Calculate distinct-n ratio
            total_ngrams = len(tokens) - self.config.distinct_n + 1
            distinct_ratio = len(ngrams) / max(total_ngrams, 1)
            
            scores.append(distinct_ratio)
        
        return torch.tensor(scores)
    
    def compute_copy_penalty(
        self,
        source_texts: List[str],
        generated_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute penalty for copying input verbatim.
        
        Args:
            source_texts: Source texts
            generated_texts: Generated paraphrases
            
        Returns:
            Copy penalty scores [batch_size]
        """
        penalties = []
        
        for source, generated in zip(source_texts, generated_texts):
            source_tokens = set(source.lower().split())
            generated_tokens = set(generated.lower().split())
            
            if len(generated_tokens) == 0:
                penalties.append(1.0)  # Maximum penalty for empty output
                continue
            
            # Calculate token overlap
            overlap = len(source_tokens & generated_tokens)
            total_generated = len(generated_tokens)
            
            overlap_ratio = overlap / total_generated
            penalty = overlap_ratio  # Higher overlap = higher penalty
            
            penalties.append(penalty)
        
        return torch.tensor(penalties)
    
    def __call__(
        self,
        source_texts: List[str],
        generated_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute composite reward scores.
        
        Args:
            source_texts: Source texts
            generated_texts: Generated paraphrases
            
        Returns:
            Reward scores [batch_size]
        """
        # Compute individual reward components
        semantic_scores = self.compute_semantic_similarity(source_texts, generated_texts)
        fluency_scores = self.compute_fluency(generated_texts)
        diversity_scores = self.compute_diversity(generated_texts)
        copy_penalties = self.compute_copy_penalty(source_texts, generated_texts)
        
        # Use quality scorer if available
        if self.quality_scorer is not None:
            quality_scores = self.quality_scorer.predict_quality(source_texts, generated_texts)
        else:
            quality_scores = torch.ones_like(semantic_scores)
        
        # Combine rewards with weights
        composite_rewards = (
            self.config.semantic_similarity_weight * semantic_scores +
            self.config.fluency_weight * fluency_scores +
            self.config.diversity_weight * diversity_scores +
            0.2 * quality_scores -  # Additional quality bonus
            self.config.copy_penalty_weight * copy_penalties
        )
        
        return composite_rewards.clamp(0, 1)  # Normalize to [0, 1]


class RLParaphraser(BaseMultimodalModel):
    """
    Reinforcement learning-based paraphrase generation model.
    
    Uses REINFORCE algorithm with composite reward function for
    optimizing paraphrase quality across multiple dimensions.
    """
    
    def __init__(self, config: RLParaphraserConfig):
        """
        Initialize RL paraphraser.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger("RLParaphraser")
        
        # Initialize base model
        self.base_model = self._create_base_model()
        
        # Initialize reward function
        self.reward_function = RewardFunction(config)
        
        # Value function for baseline (reduces variance)
        self.value_function = nn.Sequential(
            nn.Linear(self.base_model.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # RL training state
        self.baseline_value = 0.0
        self.rl_step_count = 0
        
        # Setup device
        self.device = self._setup_device()
        
        self.logger.info(f"Initialized RL paraphraser with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _create_base_model(self) -> Union[T5Paraphraser, BARTParaphraser]:
        """Create base paraphraser model."""
        
        if self.config.base_model_type.lower() == "t5":
            base_config = T5ParaphraserConfig(
                model_name=self.config.base_model_name,
                mixed_precision=self.config.mixed_precision
            )
            model = T5Paraphraser(base_config)
        else:
            base_config = BARTParaphraserConfig(
                model_name=self.config.base_model_name,
                mixed_precision=self.config.mixed_precision
            )
            model = BARTParaphraser(base_config)
        
        # Load pretrained weights if available
        if self.config.pretrained_model_path:
            try:
                if self.config.base_model_type.lower() == "t5":
                    model = T5Paraphraser.from_pretrained(self.config.pretrained_model_path)
                else:
                    model = BARTParaphraser.from_pretrained(self.config.pretrained_model_path)
                self.logger.info("Loaded pretrained base model")
            except Exception as e:
                self.logger.warning(f"Failed to load pretrained model: {e}")
        
        return model
    
    def _setup_device(self) -> str:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            
        Returns:
            Model outputs
        """
        return self.base_model.forward(input_ids, attention_mask, **kwargs)
    
    def sample_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int = None
    ) -> Dict[str, Any]:
        """
        Sample sequences for RL training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            num_samples: Number of samples per input
            
        Returns:
            Sampled sequences and log probabilities
        """
        if num_samples is None:
            num_samples = self.config.num_samples_per_input
        
        batch_size = input_ids.shape[0]
        
        # Expand inputs for multiple samples
        expanded_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
        
        # Sample with temperature
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            max_length=self.config.max_length,
            pad_token_id=self.base_model.tokenizer.pad_token_id,
            eos_token_id=self.base_model.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Generate sequences
        with torch.no_grad():
            outputs = self.base_model.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                generation_config=gen_config
            )
        
        # Calculate log probabilities
        sequences = outputs.sequences
        scores = outputs.scores
        
        # Compute log probabilities for generated tokens
        log_probs = []
        for i, score in enumerate(scores):
            token_probs = F.softmax(score, dim=-1)
            token_indices = sequences[:, i + expanded_input_ids.shape[1]]
            token_log_probs = torch.log(token_probs.gather(1, token_indices.unsqueeze(1)).squeeze(1))
            log_probs.append(token_log_probs)
        
        # Sum log probabilities for each sequence
        total_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        
        # Reshape to [batch_size, num_samples, ...]
        sequences = sequences.view(batch_size, num_samples, -1)
        total_log_probs = total_log_probs.view(batch_size, num_samples)
        
        return {
            'sequences': sequences,
            'log_probs': total_log_probs,
            'input_ids': expanded_input_ids.view(batch_size, num_samples, -1)
        }
    
    def compute_value_estimates(
        self,
        encoder_outputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value estimates for baseline.
        
        Args:
            encoder_outputs: Encoder hidden states
            attention_mask: Attention mask
            
        Returns:
            Value estimates [batch_size]
        """
        # Pool encoder outputs
        pooled_outputs = torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1)
        pooled_outputs = pooled_outputs / attention_mask.sum(dim=1, keepdim=True)
        
        # Compute value estimates
        values = self.value_function(pooled_outputs).squeeze(-1)
        
        return values
    
    def reinforce_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute REINFORCE loss with baseline.
        
        Args:
            log_probs: Log probabilities [batch_size, num_samples]
            rewards: Reward scores [batch_size, num_samples]
            values: Value estimates [batch_size]
            
        Returns:
            Loss components
        """
        # Expand values to match samples
        expanded_values = values.unsqueeze(1).expand_as(rewards)
        
        # Compute advantages
        advantages = rewards - expanded_values
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value function loss
        value_loss = F.mse_loss(expanded_values, rewards)
        
        # Entropy bonus for exploration
        entropy_bonus = -(log_probs * log_probs.exp()).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_loss_coefficient * value_loss -
            self.config.entropy_coefficient * entropy_bonus
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_bonus': entropy_bonus,
            'mean_reward': rewards.mean(),
            'mean_advantage': advantages.mean()
        }
    
    def rl_training_step(
        self,
        input_texts: List[str],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform single RL training step.
        
        Args:
            input_texts: Input texts for paraphrasing
            optimizer: Optimizer
            
        Returns:
            Training metrics
        """
        # Prepare inputs
        inputs = self.base_model.prepare_inputs(input_texts, add_task_prefix=False)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get encoder outputs for value function
        encoder_outputs = self.base_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Compute value estimates
        values = self.compute_value_estimates(encoder_outputs, attention_mask)
        
        # Sample sequences
        sampling_outputs = self.sample_sequences(input_ids, attention_mask)
        sequences = sampling_outputs['sequences']
        log_probs = sampling_outputs['log_probs']
        
        # Decode generated sequences
        generated_texts = []
        for batch_idx in range(sequences.shape[0]):
            batch_generated = []
            for sample_idx in range(sequences.shape[1]):
                seq = sequences[batch_idx, sample_idx]
                decoded = self.base_model.tokenizer.decode(seq, skip_special_tokens=True)
                batch_generated.append(decoded)
            generated_texts.append(batch_generated)
        
        # Compute rewards
        all_rewards = []
        for batch_idx in range(len(input_texts)):
            source_text = input_texts[batch_idx]
            batch_generated = generated_texts[batch_idx]
            
            # Compute rewards for all samples
            batch_rewards = self.reward_function(
                [source_text] * len(batch_generated),
                batch_generated
            )
            all_rewards.append(batch_rewards)
        
        rewards = torch.stack(all_rewards).to(self.device)
        
        # Compute REINFORCE loss
        loss_dict = self.reinforce_loss(log_probs, rewards, values)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        
        # Update baseline
        self.baseline_value = (
            self.config.baseline_momentum * self.baseline_value +
            (1 - self.config.baseline_momentum) * loss_dict['mean_reward'].item()
        )
        
        self.rl_step_count += 1
        
        # Convert to metrics
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics['baseline_value'] = self.baseline_value
        
        return metrics
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_strategy: str = "beam_search",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate paraphrases using trained model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            generation_strategy: Generation strategy
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated sequences and metadata
        """
        return self.base_model.generate(
            input_ids, attention_mask, generation_strategy, **generation_kwargs
        )
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute standard supervised learning loss.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            labels: Target labels
            
        Returns:
            Supervised loss
        """
        return self.base_model.compute_loss(input_ids, attention_mask, labels, **kwargs)
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration.
        
        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save base model
        base_model_path = save_path / "base_model"
        self.base_model.save_pretrained(str(base_model_path))
        
        # Save RL-specific components
        rl_components = {
            'value_function': self.value_function.state_dict(),
            'baseline_value': self.baseline_value,
            'rl_step_count': self.rl_step_count
        }
        
        torch.save(rl_components, save_path / "rl_components.pt")
        
        # Save config
        config_path = save_path / "rl_paraphraser_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"RL paraphraser saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[RLParaphraserConfig] = None
    ) -> 'RLParaphraser':
        """
        Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to saved model
            config: Optional configuration override
            
        Returns:
            Loaded RLParaphraser instance
        """
        model_path = Path(model_path)
        
        # Load config
        if config is None:
            config_path = model_path / "rl_paraphraser_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = RLParaphraserConfig(**config_dict)
            else:
                config = RLParaphraserConfig()
        
        # Create instance
        instance = cls(config)
        
        # Load base model
        base_model_path = model_path / "base_model"
        if instance.config.base_model_type.lower() == "t5":
            instance.base_model = T5Paraphraser.from_pretrained(str(base_model_path))
        else:
            instance.base_model = BARTParaphraser.from_pretrained(str(base_model_path))
        
        # Load RL components
        rl_components_path = model_path / "rl_components.pt"
        if rl_components_path.exists():
            components = torch.load(rl_components_path, map_location='cpu')
            
            instance.value_function.load_state_dict(components['value_function'])
            instance.baseline_value = components.get('baseline_value', 0.0)
            instance.rl_step_count = components.get('rl_step_count', 0)
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        base_info = self.base_model.get_model_info()
        
        return {
            **base_info,
            'model_type': 'RLParaphraser',
            'base_model_type': self.config.base_model_type,
            'rl_learning_rate': self.config.rl_learning_rate,
            'reward_weights': {
                'semantic_similarity': self.config.semantic_similarity_weight,
                'fluency': self.config.fluency_weight,
                'diversity': self.config.diversity_weight,
                'copy_penalty': self.config.copy_penalty_weight
            },
            'baseline_value': self.baseline_value,
            'rl_step_count': self.rl_step_count
        }


def main():
    """Example usage of RL paraphraser."""
    
    # Configuration
    config = RLParaphraserConfig(
        base_model_type="t5",
        base_model_name="t5-small",
        num_samples_per_input=2,
        batch_size=2
    )
    
    # Create model
    model = RLParaphraser(config)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test RL training step
    input_texts = [
        "The weather is beautiful today.",
        "I love programming in Python."
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.rl_learning_rate)
    
    # Perform RL training step
    metrics = model.rl_training_step(input_texts, optimizer)
    
    print("\nRL Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test generation
    inputs = model.base_model.prepare_inputs(input_texts)
    
    generated = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        generation_strategy="beam_search"
    )
    
    print("\nGenerated paraphrases:")
    for i, (source, generated_text) in enumerate(zip(input_texts, generated['generated_sequences'])):
        print(f"{i+1}. Source: {source}")
        print(f"   Generated: {generated_text}")


if __name__ == "__main__":
    main()
