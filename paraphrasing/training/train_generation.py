#!/usr/bin/env python3
"""
Paraphrase Generation Training Script

Comprehensive training script for T5/BART paraphrase generation models with
support for teacher forcing, mixed precision, gradient accumulation, and
distributed training.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
from dataclasses import asdict
import logging
from datetime import datetime
import math

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paraphrasing.models import T5Paraphraser, BARTParaphraser, T5ParaphraserConfig, BARTParaphraserConfig
from paraphrasing.data import UnifiedParaphraseDataset, UnifiedParaphraseConfig, create_unified_dataloader
from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.checkpoint_manager import CheckpointManager
from shared.utils.metrics import calculate_bleu_score, calculate_rouge_score
from shared.utils.device_utils import get_device_info, setup_device


# Tensorboard and WandB imports
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


class GenerationTrainer:
    """
    Comprehensive trainer for paraphrase generation models.
    
    Supports T5 and BART models with advanced training features including
    mixed precision, gradient accumulation, and comprehensive logging.
    """
    
    def __init__(
        self,
        model: Union[T5Paraphraser, BARTParaphraser],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize generation trainer.
        
        Args:
            model: T5 or BART paraphraser model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            optimizer: Optimizer (will be created if None)
            scheduler: Learning rate scheduler (will be created if None)
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config or self._get_default_config()
        
        # Setup logging
        self.logger = get_logger("GenerationTrainer")
        
        # Setup device and distributed training
        self.device = setup_device(self.config.get('device', 'auto'))
        self.model.to(self.device)
        
        # Setup mixed precision
        self.use_mixed_precision = self.config.get('mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Setup optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
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
        self.best_metric = float('-inf')
        self.training_history = []
        
        self.logger.info(f"Initialized trainer with device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'num_epochs': 5,
            'learning_rate': 3e-5,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 4,
            'eval_steps': 500,
            'save_steps': 1000,
            'logging_steps': 100,
            'mixed_precision': True,
            'label_smoothing': 0.1,
            'checkpoint_dir': 'paraphrasing/checkpoints',
            'log_dir': 'logs/paraphrasing',
            'generation_eval_samples': 100,
            'scheduler_type': 'linear',  # 'linear', 'cosine', 'constant'
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        
        # Create parameter groups for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            eps=1e-8
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        
        num_training_steps = len(self.train_dataloader) * self.config['num_epochs']
        num_training_steps = num_training_steps // self.config['gradient_accumulation_steps']
        
        if self.config['scheduler_type'] == 'linear':
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps
            )
        elif self.config['scheduler_type'] == 'cosine':
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps
            )
        else:
            # Constant learning rate
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
    
    def _setup_logging(self):
        """Setup TensorBoard and WandB logging."""
        
        # Setup TensorBoard
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(self.config['log_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.tensorboard_writer = None
            self.logger.warning("TensorBoard not available")
        
        # Setup WandB
        if WANDB_AVAILABLE and self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'factcheck-mm-paraphrasing'),
                name=self.config.get('experiment_name', f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config
            )
            self.use_wandb = True
            self.logger.info("WandB logging initialized")
        else:
            self.use_wandb = False
    
    def _apply_label_smoothing(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to cross-entropy loss."""
        
        if self.config.get('label_smoothing', 0.0) <= 0:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        # Label smoothing
        eps = self.config['label_smoothing']
        n_class = logits.size(-1)
        
        one_hot = torch.zeros_like(logits).scatter(2, labels.unsqueeze(2), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logits, dim=-1)
        
        # Mask out ignored tokens
        mask = (labels != -100).unsqueeze(-1).expand_as(log_prb)
        loss = -(one_hot * log_prb * mask).sum() / mask.sum()
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step."""
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch.get('decoder_attention_mask')
                )
                loss = outputs['loss'] / self.config['gradient_accumulation_steps']
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                decoder_attention_mask=batch.get('decoder_attention_mask')
            )
            loss = outputs['loss'] / self.config['gradient_accumulation_steps']
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if gradient accumulation step is reached
        if (self.current_step + 1) % self.config['gradient_accumulation_steps'] == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            'loss': loss.item() * self.config['gradient_accumulation_steps'],
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single evaluation step."""
        
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                decoder_attention_mask=batch.get('decoder_attention_mask')
            )
            
            loss = outputs['loss']
            
            return {'loss': loss.item()}
    
    def generate_and_evaluate(self, dataloader: DataLoader, num_samples: int = 100) -> Dict[str, float]:
        """Generate paraphrases and compute evaluation metrics."""
        
        self.model.eval()
        
        source_texts = []
        target_texts = []
        generated_texts = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate paraphrases
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_strategy="beam_search",
                    num_beams=4,
                    max_length=128
                )
                
                # Collect texts
                batch_source = batch['text1'] if 'text1' in batch else batch.get('reference_text', [])
                batch_target = batch['text2'] if 'text2' in batch else batch.get('paraphrase_text', [])
                batch_generated = generated['generated_sequences']
                
                source_texts.extend(batch_source[:num_samples - sample_count])
                target_texts.extend(batch_target[:num_samples - sample_count])
                generated_texts.extend(batch_generated[:num_samples - sample_count])
                
                sample_count += len(batch_generated)
        
        # Calculate metrics
        metrics = {}
        
        if generated_texts:
            # BLEU scores
            bleu_scores = []
            for src, gen in zip(source_texts, generated_texts):
                if gen.strip():  # Only calculate if generation is not empty
                    bleu = calculate_bleu_score([src], gen)
                    bleu_scores.append(bleu)
            
            if bleu_scores:
                metrics['bleu'] = np.mean(bleu_scores)
            
            # ROUGE scores
            rouge_scores = []
            for src, gen in zip(source_texts, generated_texts):
                if gen.strip():
                    rouge = calculate_rouge_score(src, gen)
                    rouge_scores.append(rouge['rouge-l']['f'])
            
            if rouge_scores:
                metrics['rouge_l'] = np.mean(rouge_scores)
            
            # Self-BLEU (diversity measure)
            if len(generated_texts) > 1:
                self_bleu_scores = []
                for i, gen1 in enumerate(generated_texts):
                    for j, gen2 in enumerate(generated_texts):
                        if i != j and gen1.strip() and gen2.strip():
                            self_bleu = calculate_bleu_score([gen1], gen2)
                            self_bleu_scores.append(self_bleu)
                
                if self_bleu_scores:
                    metrics['self_bleu'] = np.mean(self_bleu_scores)
                    metrics['diversity'] = 1.0 - metrics['self_bleu']  # Higher diversity = lower self-BLEU
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'learning_rate': 0.0,
            'samples_processed': 0
        }
        
        # Progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            step_metrics = self.train_step(batch)
            
            # Update metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            epoch_metrics['samples_processed'] += batch['input_ids'].size(0)
            
            # Logging
            if (self.current_step + 1) % self.config['logging_steps'] == 0:
                self._log_metrics(step_metrics, 'train', self.current_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{step_metrics['loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.2e}"
                })
            
            # Evaluation
            if self.val_dataloader and (self.current_step + 1) % self.config['eval_steps'] == 0:
                val_metrics = self.evaluate()
                self._log_metrics(val_metrics, 'val', self.current_step)
                
                # Check for best model
                val_score = val_metrics.get('bleu', val_metrics.get('loss', 0))
                if val_score > self.best_metric:
                    self.best_metric = val_score
                    self._save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (self.current_step + 1) % self.config['save_steps'] == 0:
                self._save_checkpoint()
            
            self.current_step += 1
        
        # Average metrics over epoch
        for key in ['loss', 'learning_rate']:
            if key in epoch_metrics:
                epoch_metrics[key] /= len(self.train_dataloader)
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        
        eval_metrics = {'loss': 0.0, 'samples_processed': 0}
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                step_metrics = self.evaluate_step(batch)
                
                for key, value in step_metrics.items():
                    if key in eval_metrics:
                        eval_metrics[key] += value
                
                eval_metrics['samples_processed'] += batch['input_ids'].size(0)
        
        # Average metrics
        eval_metrics['loss'] /= len(self.val_dataloader)
        
        # Generate samples for additional metrics
        generation_metrics = self.generate_and_evaluate(
            self.val_dataloader,
            self.config['generation_eval_samples']
        )
        eval_metrics.update(generation_metrics)
        
        return eval_metrics
    
    def train(self):
        """Complete training loop."""
        
        self.logger.info("Starting training...")
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
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics, 'val_epoch', epoch)
                    
                    # Save training history
                    self.training_history.append({
                        'epoch': epoch,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    })
                    
                    self.logger.info(
                        f"Epoch {epoch + 1}: "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val BLEU: {val_metrics.get('bleu', 0):.4f}"
                    )
                else:
                    self.training_history.append({
                        'epoch': epoch,
                        'train_metrics': train_metrics
                    })
                    
                    self.logger.info(
                        f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}"
                    )
                
                # Save checkpoint at end of epoch
                self._save_checkpoint()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final evaluation
            if self.test_dataloader:
                self.logger.info("Running final evaluation on test set...")
                test_metrics = self.evaluate_test()
                self.logger.info(f"Test metrics: {test_metrics}")
                self._log_metrics(test_metrics, 'test', self.current_step)
            
            # Close logging
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            if self.use_wandb:
                wandb.finish()
            
            self.logger.info("Training completed!")
    
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        
        if not self.test_dataloader:
            return {}
        
        # Temporarily switch validation dataloader
        temp_val_loader = self.val_dataloader
        self.val_dataloader = self.test_dataloader
        
        test_metrics = self.evaluate()
        
        # Restore validation dataloader
        self.val_dataloader = temp_val_loader
        
        return test_metrics
    
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
        """Save model checkpoint."""
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            metric_value=self.best_metric,
            is_best=is_best
        )
        
        # Also save model using model's save_pretrained method
        if is_best:
            best_model_dir = Path(self.config['checkpoint_dir']) / 'best_model'
            self.model.save_pretrained(str(best_model_dir))
            
            self.logger.info(f"Best model saved to: {best_model_dir}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load checkpoint and resume training."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_metric = checkpoint['best_metric']
            self.training_history = checkpoint.get('training_history', [])
            
            self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


def train_t5(
    config_path: Optional[str] = None,
    **kwargs
) -> T5Paraphraser:
    """
    Train T5 paraphraser model.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Override configuration parameters
        
    Returns:
        Trained T5 model
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with kwargs
    config.update(kwargs)
    
    # Model configuration
    model_config = T5ParaphraserConfig(
        model_name=config.get('model_name', 't5-base'),
        max_input_length=config.get('max_input_length', 128),
        max_target_length=config.get('max_target_length', 128),
        mixed_precision=config.get('mixed_precision', True)
    )
    
    # Create model
    model = T5Paraphraser(model_config)
    
    # Data configuration
    data_config = UnifiedParaphraseConfig(
        use_paranmt=config.get('use_paranmt', True),
        use_mrpc=config.get('use_mrpc', True),
        use_quora=config.get('use_quora', True),
        balance_datasets=config.get('balance_datasets', True),
        max_samples_per_dataset=config.get('max_samples_per_dataset', None)
    )
    
    # Create data loaders
    train_dataloader = create_unified_dataloader(
        data_config, 'train',
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    
    val_dataloader = create_unified_dataloader(
        data_config, 'val',
        batch_size=config.get('eval_batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    # Create trainer
    trainer = GenerationTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if config.get('resume_from_checkpoint'):
        trainer.load_checkpoint(config['resume_from_checkpoint'])
    
    # Train model
    trainer.train()
    
    return model


def train_bart(
    config_path: Optional[str] = None,
    **kwargs
) -> BARTParaphraser:
    """
    Train BART paraphraser model.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Override configuration parameters
        
    Returns:
        Trained BART model
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with kwargs
    config.update(kwargs)
    
    # Model configuration
    model_config = BARTParaphraserConfig(
        model_name=config.get('model_name', 'facebook/bart-base'),
        max_input_length=config.get('max_input_length', 128),
        max_target_length=config.get('max_target_length', 128),
        mixed_precision=config.get('mixed_precision', True)
    )
    
    # Create model
    model = BARTParaphraser(model_config)
    
    # Data configuration
    data_config = UnifiedParaphraseConfig(
        use_paranmt=config.get('use_paranmt', True),
        use_mrpc=config.get('use_mrpc', True),
        use_quora=config.get('use_quora', True),
        balance_datasets=config.get('balance_datasets', True),
        max_samples_per_dataset=config.get('max_samples_per_dataset', None)
    )
    
    # Create data loaders
    train_dataloader = create_unified_dataloader(
        data_config, 'train',
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    
    val_dataloader = create_unified_dataloader(
        data_config, 'val',
        batch_size=config.get('eval_batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    # Create trainer
    trainer = GenerationTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if config.get('resume_from_checkpoint'):
        trainer.load_checkpoint(config['resume_from_checkpoint'])
    
    # Train model
    trainer.train()
    
    return model


def main():
    """Main training script entry point."""
    
    parser = argparse.ArgumentParser(description="Train paraphrase generation models")
    parser.add_argument('--model', type=str, choices=['t5', 'bart'], required=True,
                       help='Model type to train')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='paraphrasing/checkpoints',
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
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'experiment_name': args.experiment_name or f"{args.model}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    # Train model
    if args.model == 't5':
        trained_model = train_t5(args.config, **config)
    elif args.model == 'bart':
        trained_model = train_bart(args.config, **config)
    
    print(f"Training completed successfully!")
    print(f"Model info: {trained_model.get_model_info()}")



if __name__ == "__main__":
    main()
