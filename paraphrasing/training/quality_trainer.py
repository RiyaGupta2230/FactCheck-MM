#!/usr/bin/env python3
"""
Quality Scorer Training Script

Trains the paraphrase quality scorer model using synthetic labels (BLEU/ROUGE)
and human ratings when available. Optimizes with regression loss (MSE/Huber)
for use as reward function in RL training.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pickle

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paraphrasing.models import QualityScorer, QualityScorerConfig
from paraphrasing.data import UnifiedParaphraseDataset, UnifiedParaphraseConfig, create_unified_dataloader
from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.checkpoint_manager import CheckpointManager
from shared.utils.device_utils import setup_device
from shared.utils.metrics import calculate_bleu_score, calculate_rouge_score

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


class QualityDataset(Dataset):
    """
    Dataset for training quality scorer with triplets of (source, target, paraphrase).
    
    Generates synthetic quality labels based on automatic metrics and optionally
    incorporates human quality ratings.
    """
    
    def __init__(
        self,
        base_dataset: UnifiedParaphraseDataset,
        paraphrase_generator: Optional[callable] = None,
        human_ratings_path: Optional[str] = None,
        num_synthetic_per_sample: int = 3,
        quality_noise_std: float = 0.1
    ):
        """
        Initialize quality dataset.
        
        Args:
            base_dataset: Base paraphrasing dataset
            paraphrase_generator: Function to generate paraphrases
            human_ratings_path: Path to human quality ratings file
            num_synthetic_per_sample: Number of synthetic paraphrases per sample
            quality_noise_std: Standard deviation for quality label noise
        """
        self.base_dataset = base_dataset
        self.paraphrase_generator = paraphrase_generator
        self.num_synthetic_per_sample = num_synthetic_per_sample
        self.quality_noise_std = quality_noise_std
        
        self.logger = get_logger("QualityDataset")
        
        # Load human ratings if available
        self.human_ratings = self._load_human_ratings(human_ratings_path)
        
        # Generate quality training samples
        self.quality_samples = self._generate_quality_samples()
        
        self.logger.info(f"Generated {len(self.quality_samples)} quality training samples")
    
    def _load_human_ratings(self, ratings_path: Optional[str]) -> Dict[str, float]:
        """Load human quality ratings if available."""
        
        if not ratings_path or not Path(ratings_path).exists():
            return {}
        
        try:
            if ratings_path.endswith('.json'):
                with open(ratings_path, 'r') as f:
                    ratings = json.load(f)
            elif ratings_path.endswith('.pkl'):
                with open(ratings_path, 'rb') as f:
                    ratings = pickle.load(f)
            else:
                # Assume CSV format with columns: source, target, paraphrase, rating
                import pandas as pd
                df = pd.read_csv(ratings_path)
                ratings = {}
                for _, row in df.iterrows():
                    key = f"{row['source']}|{row['target']}|{row['paraphrase']}"
                    ratings[key] = float(row['rating'])
            
            self.logger.info(f"Loaded {len(ratings)} human quality ratings")
            return ratings
            
        except Exception as e:
            self.logger.warning(f"Failed to load human ratings: {e}")
            return {}
    
    def _generate_paraphrases(self, text: str, num_paraphrases: int = 3) -> List[str]:
        """Generate paraphrases for quality evaluation."""
        
        if self.paraphrase_generator:
            try:
                return self.paraphrase_generator(text, num_paraphrases)
            except Exception as e:
                self.logger.warning(f"Paraphrase generation failed: {e}")
        
        # Fallback: simple rule-based paraphrases for demonstration
        paraphrases = []
        
        # Rule 1: Word order variation
        words = text.split()
        if len(words) > 2:
            # Move first word to end
            para1 = ' '.join(words[1:] + [words[0]])
            paraphrases.append(para1)
        
        # Rule 2: Synonym replacement (simple)
        synonyms = {
            'good': 'great', 'bad': 'terrible', 'big': 'large', 'small': 'tiny',
            'fast': 'quick', 'slow': 'sluggish', 'hot': 'warm', 'cold': 'chilly',
            'happy': 'joyful', 'sad': 'unhappy', 'beautiful': 'gorgeous'
        }
        
        para_words = text.lower().split()
        for i, word in enumerate(para_words):
            if word in synonyms:
                para_words[i] = synonyms[word]
                break
        
        paraphrases.append(' '.join(para_words))
        
        # Rule 3: Add filler words (lower quality)
        fillers = ['really', 'very', 'quite', 'somewhat', 'rather']
        para3 = f"{np.random.choice(fillers)} {text}"
        paraphrases.append(para3)
        
        # Pad with original text if needed
        while len(paraphrases) < num_paraphrases:
            paraphrases.append(text)
        
        return paraphrases[:num_paraphrases]
    
    def _compute_synthetic_quality(
        self,
        source: str,
        target: str,
        paraphrase: str
    ) -> float:
        """Compute synthetic quality score based on automatic metrics."""
        
        # Check for human rating first
        human_key = f"{source}|{target}|{paraphrase}"
        if human_key in self.human_ratings:
            return self.human_ratings[human_key]
        
        # Compute automatic metrics
        scores = []
        
        # BLEU score against source
        try:
            bleu_source = calculate_bleu_score([source], paraphrase)
            scores.append(bleu_source)
        except:
            scores.append(0.0)
        
        # BLEU score against target
        try:
            bleu_target = calculate_bleu_score([target], paraphrase)
            scores.append(bleu_target)
        except:
            scores.append(0.0)
        
        # ROUGE-L score against source
        try:
            rouge_source = calculate_rouge_score(source, paraphrase)
            scores.append(rouge_source['rouge-l']['f'])
        except:
            scores.append(0.0)
        
        # ROUGE-L score against target
        try:
            rouge_target = calculate_rouge_score(target, paraphrase)
            scores.append(rouge_target['rouge-l']['f'])
        except:
            scores.append(0.0)
        
        # Length ratio penalty
        source_len = len(source.split())
        para_len = len(paraphrase.split())
        length_ratio = min(para_len, source_len) / max(para_len, source_len, 1)
        scores.append(length_ratio)
        
        # Compute weighted average
        weights = [0.2, 0.3, 0.2, 0.2, 0.1]  # Emphasize target similarity
        quality = sum(w * s for w, s in zip(weights, scores))
        
        # Add noise for robustness
        noise = np.random.normal(0, self.quality_noise_std)
        quality = np.clip(quality + noise, 0.0, 1.0)
        
        return quality
    
    def _generate_quality_samples(self) -> List[Dict[str, Any]]:
        """Generate quality training samples from base dataset."""
        
        quality_samples = []
        
        # Sample from base dataset
        num_base_samples = min(len(self.base_dataset), 5000)  # Limit for efficiency
        sample_indices = np.random.choice(len(self.base_dataset), num_base_samples, replace=False)
        
        for idx in tqdm(sample_indices, desc="Generating quality samples"):
            try:
                sample = self.base_dataset[idx]
                
                source_text = sample.get('text1', sample.get('reference_text', ''))
                target_text = sample.get('text2', sample.get('paraphrase_text', ''))
                
                if not source_text or not target_text:
                    continue
                
                # Generate synthetic paraphrases
                generated_paraphrases = self._generate_paraphrases(
                    source_text, self.num_synthetic_per_sample
                )
                
                # Create quality samples
                for paraphrase in generated_paraphrases:
                    if paraphrase.strip():
                        quality_score = self._compute_synthetic_quality(
                            source_text, target_text, paraphrase
                        )
                        
                        quality_samples.append({
                            'source_text': source_text,
                            'target_text': target_text,
                            'paraphrase_text': paraphrase,
                            'quality_score': quality_score
                        })
                
                # Also include the original target as a high-quality sample
                target_quality = self._compute_synthetic_quality(
                    source_text, target_text, target_text
                )
                
                quality_samples.append({
                    'source_text': source_text,
                    'target_text': target_text,
                    'paraphrase_text': target_text,
                    'quality_score': min(target_quality + 0.2, 1.0)  # Boost original target
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        return quality_samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.quality_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get quality training sample."""
        
        sample = self.quality_samples[idx]
        
        return {
            'source_text': sample['source_text'],
            'target_text': sample.get('target_text', ''),
            'paraphrase_text': sample['paraphrase_text'],
            'quality_score': sample['quality_score']
        }


class QualityTrainer:
    """
    Trainer for paraphrase quality scorer model.
    
    Supports both regression and classification training with comprehensive
    evaluation metrics and hardware optimization.
    """
    
    def __init__(
        self,
        model: QualityScorer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize quality trainer.
        
        Args:
            model: Quality scorer model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config or self._get_default_config()
        
        # Setup logging
        self.logger = get_logger("QualityTrainer")
        
        # Setup device
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
        self.best_metric = float('inf') if self.config.get('minimize_metric', True) else float('-inf')
        self.training_history = []
        
        self.logger.info(f"Initialized quality trainer with device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Task type: {self.model.config.task_type}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'num_epochs': 10,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 2,
            'eval_steps': 200,
            'save_steps': 500,
            'logging_steps': 50,
            'mixed_precision': True,
            'loss_type': 'mse',  # 'mse', 'huber', 'mae'
            'huber_delta': 0.1,
            'minimize_metric': True,  # True for loss, False for correlation
            'checkpoint_dir': 'paraphrasing/checkpoints/quality',
            'log_dir': 'logs/paraphrasing/quality',
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        
        # Separate encoder and quality head parameters
        encoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'quality_head' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)
        
        # Use different learning rates for encoder and head
        optimizer_grouped_parameters = [
            {
                "params": encoder_params,
                "lr": self.config['learning_rate'],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": head_params,
                "lr": self.config['learning_rate'] * 2,  # Higher LR for head
                "weight_decay": self.config['weight_decay'] * 0.1,
            },
        ]
        
        return torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        
        num_training_steps = len(self.train_dataloader) * self.config['num_epochs']
        num_training_steps = num_training_steps // self.config['gradient_accumulation_steps']
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=num_training_steps
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
                project=self.config.get('wandb_project', 'factcheck-mm-quality'),
                name=self.config.get('experiment_name', f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config
            )
            self.use_wandb = True
            self.logger.info("WandB logging initialized")
        else:
            self.use_wandb = False
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss based on configuration."""
        
        if self.model.config.task_type == "classification":
            return F.cross_entropy(predictions, targets.long())
        
        # Regression losses
        if self.config['loss_type'] == 'mse':
            return F.mse_loss(predictions.squeeze(), targets)
        
        elif self.config['loss_type'] == 'huber':
            return F.smooth_l1_loss(
                predictions.squeeze(), targets, 
                beta=self.config.get('huber_delta', 0.1)
            )
        
        elif self.config['loss_type'] == 'mae':
            return F.l1_loss(predictions.squeeze(), targets)
        
        else:
            # Default to MSE
            return F.mse_loss(predictions.squeeze(), targets)
    
    def train_step(self, batch: List[Dict[str, str]]) -> Dict[str, float]:
        """Perform single training step."""
        
        self.model.train()
        
        # Extract batch data
        source_texts = [item['source_text'] for item in batch]
        target_texts = [item.get('target_text', '') for item in batch]
        paraphrase_texts = [item['paraphrase_text'] for item in batch]
        quality_scores = torch.tensor(
            [item['quality_score'] for item in batch],
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(
                    source_texts=source_texts,
                    paraphrases=paraphrase_texts,
                    target_texts=target_texts if target_texts[0] else None,
                    quality_labels=quality_scores
                )
                loss = outputs['loss'] / self.config['gradient_accumulation_steps']
        else:
            outputs = self.model(
                source_texts=source_texts,
                paraphrases=paraphrase_texts,
                target_texts=target_texts if target_texts[0] else None,
                quality_labels=quality_scores
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
    
    def evaluate_step(self, batch: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single evaluation step."""
        
        self.model.eval()
        
        with torch.no_grad():
            # Extract batch data
            source_texts = [item['source_text'] for item in batch]
            target_texts = [item.get('target_text', '') for item in batch]
            paraphrase_texts = [item['paraphrase_text'] for item in batch]
            quality_scores = torch.tensor(
                [item['quality_score'] for item in batch],
                dtype=torch.float32
            )
            
            # Forward pass
            predictions = self.model.predict_quality(
                source_texts=source_texts,
                paraphrases=paraphrase_texts,
                target_texts=target_texts if target_texts[0] else None
            )
            
            return predictions.cpu(), quality_scores
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                try:
                    predictions, targets = self.evaluate_step(batch)
                    
                    all_predictions.extend(predictions.tolist())
                    all_targets.extend(targets.tolist())
                    
                    # Compute loss
                    batch_loss = self.compute_loss(
                        predictions.to(self.device),
                        targets.to(self.device)
                    )
                    total_loss += batch_loss.item()
                    
                except Exception as e:
                    self.logger.warning(f"Evaluation step failed: {e}")
                    continue
        
        # Calculate metrics
        metrics = {'loss': total_loss / len(self.val_dataloader)}
        
        if all_predictions and all_targets:
            pred_array = np.array(all_predictions)
            target_array = np.array(all_targets)
            
            # Regression metrics
            metrics.update(self.model.evaluate_predictions(
                torch.tensor(pred_array),
                torch.tensor(target_array)
            ))
        
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
            try:
                # Training step
                step_metrics = self.train_step(batch)
                
                # Update metrics
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                
                epoch_metrics['samples_processed'] += len(batch)
                
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
                    metric_value = val_metrics.get('loss', float('inf'))
                    if self.config['minimize_metric']:
                        is_best = metric_value < self.best_metric
                    else:
                        is_best = metric_value > self.best_metric
                    
                    if is_best:
                        self.best_metric = metric_value
                        self._save_checkpoint(is_best=True)
                
                # Save checkpoint
                if (self.current_step + 1) % self.config['save_steps'] == 0:
                    self._save_checkpoint()
                
                self.current_step += 1
                
            except Exception as e:
                self.logger.warning(f"Training step failed: {e}")
                continue
        
        # Average metrics over epoch
        for key in ['loss', 'learning_rate']:
            if key in epoch_metrics:
                epoch_metrics[key] /= len(self.train_dataloader)
        
        return epoch_metrics
    
    def train(self):
        """Complete training loop."""
        
        self.logger.info("Starting quality scorer training...")
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
                        f"Val R²: {val_metrics.get('r2', 0):.4f}"
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
            
            self.logger.info("Quality scorer training completed!")
    
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
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            metric_value=self.best_metric,
            is_best=is_best
        )
        
        # Save best model
        if is_best:
            best_model_dir = Path(self.config['checkpoint_dir']) / 'best_model'
            self.model.save_pretrained(str(best_model_dir))
            self.logger.info(f"Best quality scorer saved to: {best_model_dir}")
    
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
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


def train_quality(
    config_path: Optional[str] = None,
    **kwargs
) -> QualityScorer:
    """
    Train quality scorer model.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Override configuration parameters
        
    Returns:
        Trained quality scorer
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
    model_config = QualityScorerConfig(
        base_model_name=config.get('base_model_name', 'roberta-base'),
        max_length=config.get('max_length', 256),
        task_type=config.get('task_type', 'regression'),
        use_explicit_features=config.get('use_explicit_features', True),
        mixed_precision=config.get('mixed_precision', True)
    )
    
    # Create model
    model = QualityScorer(model_config)
    
    # Create base dataset
    data_config = UnifiedParaphraseConfig(
        use_paranmt=config.get('use_paranmt', True),
        use_mrpc=config.get('use_mrpc', True),
        use_quora=config.get('use_quora', True),
        balance_datasets=config.get('balance_datasets', True),
        max_samples_per_dataset=config.get('max_samples_per_dataset', {'paranmt': 5000, 'mrpc': None, 'quora': 2000})
    )
    
    base_dataset = UnifiedParaphraseDataset(data_config, 'train')
    
    # Create quality dataset
    quality_dataset = QualityDataset(
        base_dataset=base_dataset,
        human_ratings_path=config.get('human_ratings_path'),
        num_synthetic_per_sample=config.get('num_synthetic_per_sample', 3),
        quality_noise_std=config.get('quality_noise_std', 0.1)
    )
    
    # Split dataset
    train_size = int(0.8 * len(quality_dataset))
    val_size = int(0.1 * len(quality_dataset))
    test_size = len(quality_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        quality_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    def collate_fn(batch):
        return batch  # Return list of dictionaries
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.get('eval_batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.get('eval_batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn
    )
    
    # Create trainer
    trainer = QualityTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if config.get('resume_from_checkpoint'):
        trainer.load_checkpoint(config['resume_from_checkpoint'])
    
    # Train model
    trainer.train()
    
    return model


def main():
    """Main quality trainer script entry point."""
    
    parser = argparse.ArgumentParser(description="Train paraphrase quality scorer")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--base-model', type=str, default='roberta-base', help='Base model name')
    parser.add_argument('--task-type', type=str, choices=['regression', 'classification'],
                       default='regression', help='Task type')
    parser.add_argument('--loss-type', type=str, choices=['mse', 'huber', 'mae'],
                       default='mse', help='Loss function type')
    parser.add_argument('--human-ratings', type=str, help='Path to human quality ratings')
    parser.add_argument('--checkpoint-dir', type=str, default='paraphrasing/checkpoints/quality',
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
        'base_model_name': args.base_model,
        'task_type': args.task_type,
        'loss_type': args.loss_type,
        'human_ratings_path': args.human_ratings,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'experiment_name': args.experiment_name or f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    # Train quality scorer
    trained_model = train_quality(args.config, **config)
    
    print(f"Quality scorer training completed successfully!")
    print(f"Model info: {trained_model.get_model_info()}")


if __name__ == "__main__":
    main()
