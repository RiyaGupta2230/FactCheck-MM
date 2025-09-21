#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Paraphrase Generation

Implements REINFORCE algorithm for fine-tuning paraphrase generation models
using composite reward functions with semantic similarity, fluency, diversity,
and copy penalty components.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime
import math
from collections import deque

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paraphrasing.models import RLParaphraser, RLParaphraserConfig, QualityScorer
from paraphrasing.data import UnifiedParaphraseDataset, UnifiedParaphraseConfig, create_unified_dataloader
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


class RLTrainer:
    """
    Reinforcement Learning trainer for paraphrase generation.
    
    Implements REINFORCE algorithm with baseline subtraction for variance reduction
    and comprehensive reward function combining multiple quality metrics.
    """
    
    def __init__(
        self,
        model: RLParaphraser,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        quality_scorer: Optional[QualityScorer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RL trainer.
        
        Args:
            model: RL paraphraser model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            quality_scorer: Optional quality scorer for additional rewards
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.quality_scorer = quality_scorer
        self.config = config or self._get_default_config()
        
        # Setup logging
        self.logger = get_logger("RLTrainer")
        
        # Setup device
        self.device = setup_device(self.config.get('device', 'auto'))
        self.model.to(self.device)
        if self.quality_scorer:
            self.quality_scorer.to(self.device)
        
        # Setup mixed precision
        self.use_mixed_precision = self.config.get('mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Setup optimizer for RL parameters
        self.rl_optimizer = self._create_rl_optimizer()
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config['checkpoint_dir'],
            max_checkpoints=self.config.get('max_checkpoints', 5)
        )
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_reward = float('-inf')
        self.reward_history = deque(maxlen=1000)
        self.training_history = []
        
        # RL-specific tracking
        self.policy_loss_history = deque(maxlen=100)
        self.value_loss_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        
        self.logger.info(f"Initialized RL trainer with device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Quality scorer available: {self.quality_scorer is not None}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default RL training configuration."""
        return {
            'num_epochs': 10,
            'rl_learning_rate': 1e-5,
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'max_grad_norm': 1.0,
            'mixed_precision': True,
            
            # RL specific parameters
            'num_samples_per_input': 4,
            'baseline_momentum': 0.9,
            'entropy_coefficient': 0.01,
            'value_loss_coefficient': 0.5,
            'reward_scaling': 1.0,
            
            # Reward function weights
            'reward_weights': {
                'semantic_similarity': 0.3,
                'fluency': 0.25,
                'diversity': 0.2,
                'copy_penalty': 0.15,
                'quality_scorer': 0.1
            },
            
            # Training schedule
            'eval_steps': 200,
            'save_steps': 500,
            'logging_steps': 50,
            'warmup_steps': 100,
            
            # Generation parameters
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            
            # Directories
            'checkpoint_dir': 'paraphrasing/checkpoints/rl',
            'log_dir': 'logs/paraphrasing/rl',
        }
    
    def _create_rl_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for RL training."""
        
        # Only optimize policy parameters (base model + value function)
        rl_parameters = []
        
        # Base model parameters
        rl_parameters.extend(self.model.base_model.parameters())
        
        # Value function parameters
        rl_parameters.extend(self.model.value_function.parameters())
        
        return torch.optim.AdamW(
            rl_parameters,
            lr=self.config['rl_learning_rate'],
            weight_decay=0.01,
            eps=1e-8
        )
    
    def _setup_logging(self):
        """Setup TensorBoard and WandB logging."""
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(self.config['log_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.tensorboard_writer = None
        
        # WandB
        if WANDB_AVAILABLE and self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'factcheck-mm-rl'),
                name=self.config.get('experiment_name', f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config
            )
            self.use_wandb = True
            self.logger.info("WandB logging initialized")
        else:
            self.use_wandb = False
    
    def compute_composite_rewards(
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
            Reward scores [batch_size * num_samples]
        """
        # Use model's reward function
        base_rewards = self.model.reward_function(source_texts, generated_texts)
        
        # Add quality scorer rewards if available
        if self.quality_scorer:
            try:
                quality_rewards = self.quality_scorer.predict_quality(source_texts, generated_texts)
                quality_weight = self.config['reward_weights'].get('quality_scorer', 0.1)
                
                # Combine rewards
                total_rewards = (
                    (1 - quality_weight) * base_rewards +
                    quality_weight * quality_rewards
                )
            except Exception as e:
                self.logger.warning(f"Quality scorer failed: {e}")
                total_rewards = base_rewards
        else:
            total_rewards = base_rewards
        
        # Scale rewards
        total_rewards = total_rewards * self.config['reward_scaling']
        
        return total_rewards.to(self.device)
    
    def rl_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single RL training step."""
        
        # Extract source texts
        source_texts = batch['text1'] if 'text1' in batch else batch.get('reference_text', [])
        
        # Perform RL training step using model's method
        step_metrics = self.model.rl_training_step(source_texts, self.rl_optimizer)
        
        # Track metrics history
        self.reward_history.extend([step_metrics['mean_reward']] * len(source_texts))
        self.policy_loss_history.append(step_metrics['policy_loss'])
        self.value_loss_history.append(step_metrics['value_loss'])
        self.entropy_history.append(step_metrics['entropy_bonus'])
        
        return step_metrics
    
    def evaluate_rl(self) -> Dict[str, float]:
        """Evaluate RL model performance."""
        
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        
        eval_rewards = []
        eval_metrics = {
            'reward': 0.0,
            'diversity': 0.0,
            'samples_processed': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="RL Evaluation", leave=False):
                source_texts = batch['text1'] if 'text1' in batch else batch.get('reference_text', [])
                
                if not source_texts:
                    continue
                
                # Prepare inputs
                inputs = self.model.base_model.prepare_inputs(source_texts)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                # Generate samples
                sampling_outputs = self.model.sample_sequences(
                    input_ids, attention_mask, num_samples=2
                )
                
                # Decode generated texts
                generated_texts = []
                for batch_idx in range(sampling_outputs['sequences'].shape[0]):
                    for sample_idx in range(sampling_outputs['sequences'].shape[1]):
                        seq = sampling_outputs['sequences'][batch_idx, sample_idx]
                        decoded = self.model.base_model.tokenizer.decode(seq, skip_special_tokens=True)
                        generated_texts.append(decoded)
                
                # Expand source texts to match generated texts
                expanded_sources = []
                for source in source_texts:
                    expanded_sources.extend([source] * 2)
                
                # Compute rewards
                if generated_texts and expanded_sources:
                    batch_rewards = self.compute_composite_rewards(expanded_sources, generated_texts)
                    eval_rewards.extend(batch_rewards.cpu().tolist())
                    
                    eval_metrics['samples_processed'] += len(source_texts)
        
        # Calculate final metrics
        if eval_rewards:
            eval_metrics['reward'] = np.mean(eval_rewards)
            eval_metrics['reward_std'] = np.std(eval_rewards)
            
            # Calculate diversity (1 - self-BLEU)
            if len(generated_texts) > 1:
                from shared.utils.metrics import calculate_bleu_score
                self_bleu_scores = []
                
                for i in range(0, len(generated_texts), 2):
                    if i + 1 < len(generated_texts):
                        try:
                            bleu = calculate_bleu_score([generated_texts[i]], generated_texts[i + 1])
                            self_bleu_scores.append(bleu)
                        except:
                            continue
                
                if self_bleu_scores:
                    eval_metrics['diversity'] = 1.0 - np.mean(self_bleu_scores)
        
        return eval_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train RL model for one epoch."""
        
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_bonus': 0.0,
            'mean_reward': 0.0,
            'samples_processed': 0
        }
        
        # Progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"RL Epoch {self.current_epoch + 1}/{self.config['num_epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Skip empty batches
            source_texts = batch['text1'] if 'text1' in batch else batch.get('reference_text', [])
            if not source_texts:
                continue
            
            # RL training step
            try:
                step_metrics = self.rl_training_step(batch)
                
                # Update epoch metrics
                for key in epoch_metrics.keys():
                    if key in step_metrics and key != 'samples_processed':
                        epoch_metrics[key] += step_metrics[key]
                
                epoch_metrics['samples_processed'] += len(source_texts)
                
                # Logging
                if (self.current_step + 1) % self.config['logging_steps'] == 0:
                    self._log_metrics(step_metrics, 'train', self.current_step)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'reward': f"{step_metrics.get('mean_reward', 0):.4f}",
                        'loss': f"{step_metrics.get('total_loss', 0):.4f}",
                        'baseline': f"{step_metrics.get('baseline_value', 0):.4f}"
                    })
                
                # Evaluation
                if self.val_dataloader and (self.current_step + 1) % self.config['eval_steps'] == 0:
                    val_metrics = self.evaluate_rl()
                    self._log_metrics(val_metrics, 'val', self.current_step)
                    
                    # Check for best model
                    val_reward = val_metrics.get('reward', 0)
                    if val_reward > self.best_reward:
                        self.best_reward = val_reward
                        self._save_checkpoint(is_best=True)
                
                # Save checkpoint
                if (self.current_step + 1) % self.config['save_steps'] == 0:
                    self._save_checkpoint()
                
                self.current_step += 1
                
            except Exception as e:
                self.logger.warning(f"RL training step failed: {e}")
                continue
        
        # Average metrics over epoch
        num_batches = len(self.train_dataloader)
        for key in ['total_loss', 'policy_loss', 'value_loss', 'entropy_bonus', 'mean_reward']:
            if key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self):
        """Complete RL training loop."""
        
        self.logger.info("Starting RL training...")
        self.logger.info(f"Total epochs: {self.config['num_epochs']}")
        self.logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        
        try:
            for epoch in range(self.config['num_epochs']):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Log epoch metrics
                self._log_metrics(train_metrics, 'train_epoch', epoch)
                
                # Evaluate
                if self.val_dataloader:
                    val_metrics = self.evaluate_rl()
                    self._log_metrics(val_metrics, 'val_epoch', epoch)
                    
                    # Save training history
                    self.training_history.append({
                        'epoch': epoch,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    })
                    
                    self.logger.info(
                        f"RL Epoch {epoch + 1}: "
                        f"Reward: {train_metrics.get('mean_reward', 0):.4f}, "
                        f"Val Reward: {val_metrics.get('reward', 0):.4f}, "
                        f"Diversity: {val_metrics.get('diversity', 0):.4f}"
                    )
                else:
                    self.training_history.append({
                        'epoch': epoch,
                        'train_metrics': train_metrics
                    })
                    
                    self.logger.info(
                        f"RL Epoch {epoch + 1}: Reward: {train_metrics.get('mean_reward', 0):.4f}"
                    )
                
                # Save checkpoint at end of epoch
                self._save_checkpoint()
        
        except KeyboardInterrupt:
            self.logger.info("RL training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"RL training failed with error: {e}")
            raise
        
        finally:
            # Close logging
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            if self.use_wandb:
                wandb.finish()
            
            self.logger.info("RL training completed!")
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str, step: int):
        """Log metrics to TensorBoard and WandB."""
        
        # TensorBoard logging
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"{phase}/{key}", value, step)
        
        # WandB logging
        if self.use_wandb:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            wandb.log(wandb_metrics, step=step)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save RL training checkpoint."""
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.rl_optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': self.config,
            'training_history': self.training_history,
            'reward_history': list(self.reward_history)
        }
        
        if self.scaler:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            metric_value=self.best_reward,
            is_best=is_best
        )
        
        # Save best model
        if is_best:
            best_model_dir = Path(self.config['checkpoint_dir']) / 'best_rl_model'
            self.model.save_pretrained(str(best_model_dir))
            self.logger.info(f"Best RL model saved to: {best_model_dir}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load RL training checkpoint."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model and optimizer state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.rl_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_reward = checkpoint['best_reward']
            self.training_history = checkpoint.get('training_history', [])
            self.reward_history = deque(checkpoint.get('reward_history', []), maxlen=1000)
            
            self.logger.info(f"RL checkpoint loaded from: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load RL checkpoint: {e}")
            return False


def train_rl(
    pretrained_model_path: str,
    config_path: Optional[str] = None,
    **kwargs
) -> RLParaphraser:
    """
    Train RL paraphraser model.
    
    Args:
        pretrained_model_path: Path to pretrained base model
        config_path: Path to configuration file
        **kwargs: Override configuration parameters
        
    Returns:
        Trained RL model
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with kwargs
    config.update(kwargs)
    
    # RL model configuration
    rl_config = RLParaphraserConfig(
        base_model_type=config.get('base_model_type', 't5'),
        base_model_name=config.get('base_model_name', 't5-base'),
        pretrained_model_path=pretrained_model_path,
        rl_learning_rate=config.get('rl_learning_rate', 1e-5),
        mixed_precision=config.get('mixed_precision', True)
    )
    
    # Create RL model
    model = RLParaphraser(rl_config)
    
    # Load quality scorer if specified
    quality_scorer = None
    if config.get('quality_scorer_path'):
        try:
            from paraphrasing.models import QualityScorer
            quality_scorer = QualityScorer.from_pretrained(config['quality_scorer_path'])
            print(f"Loaded quality scorer from: {config['quality_scorer_path']}")
        except Exception as e:
            print(f"Failed to load quality scorer: {e}")
    
    # Data configuration
    data_config = UnifiedParaphraseConfig(
        use_paranmt=config.get('use_paranmt', True),
        use_mrpc=config.get('use_mrpc', True),
        use_quora=config.get('use_quora', True),
        balance_datasets=config.get('balance_datasets', True),
        max_samples_per_dataset=config.get('max_samples_per_dataset', None)
    )
    
    # Create data loaders with smaller batches for RL
    train_dataloader = create_unified_dataloader(
        data_config, 'train',
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    
    val_dataloader = create_unified_dataloader(
        data_config, 'val',
        batch_size=config.get('eval_batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    # Create RL trainer
    trainer = RLTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        quality_scorer=quality_scorer,
        config=config
    )
    
    # Resume from checkpoint if specified
    if config.get('resume_from_checkpoint'):
        trainer.load_checkpoint(config['resume_from_checkpoint'])
    
    # Train model
    trainer.train()
    
    return model


def main():
    """Main RL training script entry point."""
    
    parser = argparse.ArgumentParser(description="RL fine-tuning for paraphrase generation")
    parser.add_argument('--pretrained-model', type=str, required=True,
                       help='Path to pretrained base model')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of RL training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='RL training batch size')
    parser.add_argument('--rl-learning-rate', type=float, default=1e-5, help='RL learning rate')
    parser.add_argument('--quality-scorer', type=str, help='Path to quality scorer model')
    parser.add_argument('--checkpoint-dir', type=str, default='paraphrasing/checkpoints/rl',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Configuration
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'rl_learning_rate': args.rl_learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'quality_scorer_path': args.quality_scorer,
        'use_wandb': args.use_wandb,
        'experiment_name': args.experiment_name or f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    # Train RL model
    trained_model = train_rl(args.pretrained_model, args.config, **config)
    
    print(f"RL training completed successfully!")
    print(f"Model info: {trained_model.get_model_info()}")


if __name__ == "__main__":
    main()
