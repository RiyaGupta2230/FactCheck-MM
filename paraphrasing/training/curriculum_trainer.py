#!/usr/bin/env python3
"""
Curriculum Learning Training Script for Paraphrase Generation

Implements curriculum learning by gradually increasing training complexity
from simple/short sentences to more complex/ambiguous cases.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime
import math
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paraphrasing.models import T5Paraphraser, BARTParaphraser, SarcasmAwareParaphraser
from paraphrasing.models import T5ParaphraserConfig, BARTParaphraserConfig, SarcasmAwareConfig
from paraphrasing.data import UnifiedParaphraseDataset, UnifiedParaphraseConfig
from paraphrasing.training.train_generation import GenerationTrainer
from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.checkpoint_manager import CheckpointManager
from shared.utils.device_utils import setup_device

# Logging imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Curriculum progression
    num_stages: int = 4
    stage_epochs: List[int] = field(default_factory=lambda: [2, 3, 3, 4])
    
    # Complexity criteria
    complexity_criteria: List[str] = field(default_factory=lambda: [
        'length', 'ambiguity', 'syntactic_complexity', 'semantic_similarity'
    ])
    
    # Stage progression thresholds
    length_thresholds: List[int] = field(default_factory=lambda: [20, 40, 80, 150])
    ambiguity_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])
    
    # Training parameters per stage
    learning_rates: List[float] = field(default_factory=lambda: [5e-5, 3e-5, 2e-5, 1e-5])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 12, 8, 6])
    
    # Data filtering
    min_samples_per_stage: int = 1000
    max_samples_per_stage: int = 10000
    
    # Smoothing between stages
    stage_overlap: float = 0.2  # Overlap between consecutive stages
    
    # Model configuration
    model_type: str = "t5"  # "t5", "bart", "sarcasm_aware"
    base_model_name: str = "t5-base"


class ComplexityAnalyzer:
    """Analyzes text complexity for curriculum learning."""
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize complexity analyzer.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config
        self.logger = get_logger("ComplexityAnalyzer")
        
        # Initialize complexity models if needed
        self._setup_complexity_models()
    
    def _setup_complexity_models(self):
        """Setup models for complexity analysis."""
        
        # For semantic similarity analysis
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded sentence transformer for similarity analysis")
        except Exception as e:
            self.similarity_model = None
            self.logger.warning(f"Failed to load sentence transformer: {e}")
        
        # For syntactic complexity (could add spacy parser here)
        self.syntactic_parser = None
    
    def analyze_length_complexity(self, text: str) -> float:
        """Analyze text length complexity."""
        words = text.split()
        chars = len(text)
        
        # Normalize by typical text lengths
        word_complexity = min(len(words) / 100.0, 1.0)  # Cap at 100 words = complexity 1.0
        char_complexity = min(chars / 500.0, 1.0)  # Cap at 500 chars = complexity 1.0
        
        return (word_complexity + char_complexity) / 2.0
    
    def analyze_ambiguity_complexity(self, source: str, target: str) -> float:
        """Analyze semantic ambiguity/similarity between source and target."""
        
        if not self.similarity_model:
            # Fallback: use word overlap as proxy for ambiguity
            source_words = set(source.lower().split())
            target_words = set(target.lower().split())
            
            if len(source_words) == 0 or len(target_words) == 0:
                return 1.0
            
            overlap = len(source_words & target_words)
            union = len(source_words | target_words)
            
            # Higher ambiguity = lower word overlap (harder to paraphrase)
            return 1.0 - (overlap / union)
        
        try:
            # Use semantic similarity - lower similarity = higher ambiguity
            embeddings = self.similarity_model.encode([source, target])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Convert similarity to ambiguity (inverse relationship)
            ambiguity = 1.0 - max(0.0, similarity)
            return ambiguity
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.5  # Default ambiguity
    
    def analyze_syntactic_complexity(self, text: str) -> float:
        """Analyze syntactic complexity of text."""
        
        # Simple heuristics for syntactic complexity
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Count complex structures
        complexity_markers = [
            ',', ';', ':', '(', ')', '[', ']', '{', '}',  # Punctuation
            ' which ', ' that ', ' where ', ' when ',    # Relative clauses
            ' because ', ' since ', ' although ', ' however ',  # Conjunctions
            ' not ', ' never ', ' hardly ', ' barely '   # Negations
        ]
        
        marker_count = sum(text.lower().count(marker) for marker in complexity_markers)
        marker_density = marker_count / max(len(text.split()), 1)
        
        # Normalize complexity score
        length_complexity = min(avg_sentence_length / 20.0, 1.0)
        marker_complexity = min(marker_density, 1.0)
        
        return (length_complexity + marker_complexity) / 2.0
    
    def compute_overall_complexity(
        self,
        source: str,
        target: str
    ) -> Dict[str, float]:
        """
        Compute overall complexity score for a text pair.
        
        Args:
            source: Source text
            target: Target text
            
        Returns:
            Dictionary of complexity scores
        """
        complexity_scores = {}
        
        # Length complexity (use average of source and target)
        length_complexity = (
            self.analyze_length_complexity(source) +
            self.analyze_length_complexity(target)
        ) / 2.0
        complexity_scores['length'] = length_complexity
        
        # Ambiguity complexity
        complexity_scores['ambiguity'] = self.analyze_ambiguity_complexity(source, target)
        
        # Syntactic complexity (use average)
        syntactic_complexity = (
            self.analyze_syntactic_complexity(source) +
            self.analyze_syntactic_complexity(target)
        ) / 2.0
        complexity_scores['syntactic_complexity'] = syntactic_complexity
        
        # Overall complexity (weighted average)
        weights = {'length': 0.3, 'ambiguity': 0.4, 'syntactic_complexity': 0.3}
        overall = sum(weights[k] * v for k, v in complexity_scores.items())
        complexity_scores['overall'] = overall
        
        return complexity_scores


class CurriculumTrainer:
    """
    Curriculum learning trainer for paraphrase generation.
    
    Gradually increases training complexity across multiple stages,
    allowing models to learn from simple to complex examples.
    """
    
    def __init__(
        self,
        model: Union[T5Paraphraser, BARTParaphraser, SarcasmAwareParaphraser],
        dataset: UnifiedParaphraseDataset,
        val_dataset: Optional[UnifiedParaphraseDataset] = None,
        config: Optional[CurriculumConfig] = None
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Paraphrasing model to train
            dataset: Full training dataset
            val_dataset: Validation dataset
            config: Curriculum configuration
        """
        self.model = model
        self.full_dataset = dataset
        self.val_dataset = val_dataset
        self.config = config or CurriculumConfig()
        
        # Setup logging
        self.logger = get_logger("CurriculumTrainer")
        
        # Setup device
        self.device = setup_device()
        self.model.to(self.device)
        
        # Initialize complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer(self.config)
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=f"paraphrasing/checkpoints/curriculum",
            max_checkpoints=10
        )
        
        # Setup logging
        self._setup_logging()
        
        # Analyze dataset complexity
        self.dataset_complexities = self._analyze_dataset_complexity()
        
        # Create curriculum stages
        self.stage_datasets = self._create_curriculum_stages()
        
        # Training state
        self.current_stage = 0
        self.stage_history = []
        
        self.logger.info(f"Initialized curriculum trainer with {self.config.num_stages} stages")
        self.logger.info(f"Dataset complexity analysis completed for {len(self.full_dataset)} samples")
    
    def _setup_logging(self):
        """Setup logging for curriculum training."""
        
        if TENSORBOARD_AVAILABLE:
            log_dir = Path("logs/paraphrasing/curriculum") / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.tensorboard_writer = None
        
        if WANDB_AVAILABLE:
            wandb.init(
                project="factcheck-mm-curriculum",
                name=f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.__dict__
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def _analyze_dataset_complexity(self) -> List[Dict[str, float]]:
        """Analyze complexity of all samples in the dataset."""
        
        self.logger.info("Analyzing dataset complexity...")
        complexities = []
        
        # Sample a subset for efficiency if dataset is very large
        analysis_size = min(len(self.full_dataset), 10000)
        indices = np.random.choice(len(self.full_dataset), analysis_size, replace=False)
        
        for idx in tqdm(indices, desc="Analyzing complexity"):
            try:
                sample = self.full_dataset[idx]
                source_text = sample.get('text1', sample.get('reference_text', ''))
                target_text = sample.get('text2', sample.get('paraphrase_text', ''))
                
                if source_text and target_text:
                    complexity = self.complexity_analyzer.compute_overall_complexity(
                        source_text, target_text
                    )
                    complexity['dataset_index'] = idx
                    complexities.append(complexity)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze sample {idx}: {e}")
                continue
        
        self.logger.info(f"Analyzed {len(complexities)} samples")
        return complexities
    
    def _create_curriculum_stages(self) -> List[Subset]:
        """Create curriculum stages based on complexity analysis."""
        
        self.logger.info("Creating curriculum stages...")
        
        stage_datasets = []
        
        # Sort samples by overall complexity
        sorted_complexities = sorted(self.dataset_complexities, key=lambda x: x['overall'])
        
        # Create stages with overlapping complexity ranges
        for stage_idx in range(self.config.num_stages):
            # Calculate complexity range for this stage
            if stage_idx == 0:
                # First stage: simplest samples
                complexity_min = 0.0
                complexity_max = 0.25 + stage_idx * 0.25
            elif stage_idx == self.config.num_stages - 1:
                # Last stage: all remaining samples
                complexity_min = (stage_idx - 1) * 0.25 - self.config.stage_overlap
                complexity_max = 1.0
            else:
                # Middle stages: overlapping ranges
                complexity_min = (stage_idx - 1) * 0.25 - self.config.stage_overlap
                complexity_max = stage_idx * 0.25 + 0.25
            
            # Filter samples for this stage
            stage_samples = []
            for complexity_data in sorted_complexities:
                if complexity_min <= complexity_data['overall'] <= complexity_max:
                    stage_samples.append(complexity_data['dataset_index'])
            
            # Ensure minimum and maximum samples per stage
            if len(stage_samples) < self.config.min_samples_per_stage:
                # Add more samples if needed
                remaining_samples = [c['dataset_index'] for c in sorted_complexities 
                                   if c['dataset_index'] not in stage_samples]
                additional_needed = self.config.min_samples_per_stage - len(stage_samples)
                stage_samples.extend(remaining_samples[:additional_needed])
            
            elif len(stage_samples) > self.config.max_samples_per_stage:
                # Random subsample if too many
                stage_samples = np.random.choice(
                    stage_samples, self.config.max_samples_per_stage, replace=False
                ).tolist()
            
            # Create subset
            stage_dataset = Subset(self.full_dataset, stage_samples)
            stage_datasets.append(stage_dataset)
            
            self.logger.info(
                f"Stage {stage_idx + 1}: {len(stage_samples)} samples "
                f"(complexity {complexity_min:.2f}-{complexity_max:.2f})"
            )
        
        return stage_datasets
    
    def train_stage(self, stage_idx: int) -> Dict[str, float]:
        """Train model on a specific curriculum stage."""
        
        self.logger.info(f"Training curriculum stage {stage_idx + 1}/{self.config.num_stages}")
        
        # Get stage configuration
        stage_epochs = self.config.stage_epochs[stage_idx]
        stage_lr = self.config.learning_rates[stage_idx]
        stage_batch_size = self.config.batch_sizes[stage_idx]
        
        # Create data loaders for this stage
        stage_dataloader = DataLoader(
            self.stage_datasets[stage_idx],
            batch_size=stage_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        val_dataloader = None
        if self.val_dataset:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=stage_batch_size * 2,
                shuffle=False,
                num_workers=2,
                collate_fn=self._collate_fn
            )
        
        # Create stage-specific training configuration
        stage_config = {
            'num_epochs': stage_epochs,
            'learning_rate': stage_lr,
            'batch_size': stage_batch_size,
            'checkpoint_dir': f"paraphrasing/checkpoints/curriculum/stage_{stage_idx + 1}",
            'log_dir': f"logs/paraphrasing/curriculum/stage_{stage_idx + 1}",
            'experiment_name': f"curriculum_stage_{stage_idx + 1}",
            'mixed_precision': True,
            'eval_steps': max(100, len(stage_dataloader) // 4),
            'save_steps': max(200, len(stage_dataloader) // 2),
            'logging_steps': 50
        }
        
        # Create trainer for this stage
        stage_trainer = GenerationTrainer(
            model=self.model,
            train_dataloader=stage_dataloader,
            val_dataloader=val_dataloader,
            config=stage_config
        )
        
        # Train the stage
        stage_trainer.train()
        
        # Get stage metrics
        stage_metrics = stage_trainer.training_history[-1] if stage_trainer.training_history else {}
        
        # Log stage completion
        self._log_stage_metrics(stage_idx, stage_metrics)
        
        return stage_metrics
    
    def train_curriculum(self):
        """Train complete curriculum from simple to complex."""
        
        self.logger.info("Starting curriculum training...")
        
        try:
            for stage_idx in range(self.config.num_stages):
                self.current_stage = stage_idx
                
                # Train stage
                stage_metrics = self.train_stage(stage_idx)
                
                # Record stage history
                self.stage_history.append({
                    'stage': stage_idx,
                    'metrics': stage_metrics,
                    'num_samples': len(self.stage_datasets[stage_idx]),
                    'epochs': self.config.stage_epochs[stage_idx],
                    'learning_rate': self.config.learning_rates[stage_idx]
                })
                
                # Save curriculum checkpoint
                self._save_curriculum_checkpoint()
                
                self.logger.info(f"Completed curriculum stage {stage_idx + 1}/{self.config.num_stages}")
        
        except KeyboardInterrupt:
            self.logger.info("Curriculum training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Curriculum training failed: {e}")
            raise
        
        finally:
            # Close logging
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            if self.use_wandb:
                wandb.finish()
            
            self.logger.info("Curriculum training completed!")
    
    def _collate_fn(self, batch):
        """Collate function for curriculum training batches."""
        # Use the same collate function as the unified dataset
        from paraphrasing.data.unified_loader import _collate_unified_batch
        return _collate_unified_batch(batch)
    
    def _log_stage_metrics(self, stage_idx: int, metrics: Dict[str, Any]):
        """Log metrics for a completed stage."""
        
        # TensorBoard logging
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"curriculum/stage_{key}", value, stage_idx)
        
        # WandB logging
        if self.use_wandb:
            wandb_metrics = {f"curriculum/{k}": v for k, v in metrics.items() 
                           if isinstance(v, (int, float))}
            wandb.log(wandb_metrics, step=stage_idx)
    
    def _save_curriculum_checkpoint(self):
        """Save curriculum training checkpoint."""
        
        checkpoint_data = {
            'current_stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'stage_history': self.stage_history,
            'dataset_complexities': self.dataset_complexities
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            metric_value=self.current_stage,
            is_best=False
        )
        
        self.logger.info(f"Curriculum checkpoint saved: {checkpoint_path}")


def train_curriculum(
    config_path: Optional[str] = None,
    **kwargs
) -> Union[T5Paraphraser, BARTParaphraser, SarcasmAwareParaphraser]:
    """
    Train model with curriculum learning.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Override configuration parameters
        
    Returns:
        Trained model
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = {}
    
    # Override with kwargs
    config_dict.update(kwargs)
    
    # Create curriculum configuration
    curriculum_config = CurriculumConfig(**config_dict)
    
    # Create model based on configuration
    if curriculum_config.model_type.lower() == "t5":
        model_config = T5ParaphraserConfig(
            model_name=curriculum_config.base_model_name,
            mixed_precision=config_dict.get('mixed_precision', True)
        )
        model = T5Paraphraser(model_config)
    
    elif curriculum_config.model_type.lower() == "bart":
        model_config = BARTParaphraserConfig(
            model_name=curriculum_config.base_model_name,
            mixed_precision=config_dict.get('mixed_precision', True)
        )
        model = BARTParaphraser(model_config)
    
    elif curriculum_config.model_type.lower() == "sarcasm_aware":
        model_config = SarcasmAwareConfig(
            base_model_name=curriculum_config.base_model_name,
            mixed_precision=config_dict.get('mixed_precision', True)
        )
        model = SarcasmAwareParaphraser(model_config)
    
    else:
        raise ValueError(f"Unknown model type: {curriculum_config.model_type}")
    
    # Create dataset
    data_config = UnifiedParaphraseConfig(
        use_paranmt=config_dict.get('use_paranmt', True),
        use_mrpc=config_dict.get('use_mrpc', True),
        use_quora=config_dict.get('use_quora', True),
        balance_datasets=config_dict.get('balance_datasets', True)
    )
    
    train_dataset = UnifiedParaphraseDataset(data_config, 'train')
    val_dataset = UnifiedParaphraseDataset(data_config, 'val')
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        config=curriculum_config
    )
    
    # Train with curriculum
    trainer.train_curriculum()
    
    return model


def main():
    """Main curriculum training script entry point."""
    
    parser = argparse.ArgumentParser(description="Curriculum learning for paraphrase generation")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--model-type', type=str, choices=['t5', 'bart', 'sarcasm_aware'],
                       default='t5', help='Model type to train')
    parser.add_argument('--base-model', type=str, default='t5-base', help='Base model name')
    parser.add_argument('--num-stages', type=int, default=4, help='Number of curriculum stages')
    parser.add_argument('--stage-epochs', type=int, nargs='+', default=[2, 3, 3, 4],
                       help='Epochs per stage')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Configuration
    config = {
        'model_type': args.model_type,
        'base_model_name': args.base_model,
        'num_stages': args.num_stages,
        'stage_epochs': args.stage_epochs
    }
    
    # Train with curriculum learning
    trained_model = train_curriculum(args.config, **config)
    
    print(f"Curriculum training completed successfully!")
    print(f"Model info: {trained_model.get_model_info()}")


if __name__ == "__main__":
    main()
