#!/usr/bin/env python3
"""
Domain Adaptation for Fact Verification Models

Implements domain adaptation strategies including feature-based adaptation,
adversarial adaptation, and few-shot learning for domain-specific fact checking.

Example Usage:
    python domain_adaptation.py --source_domain general --target_domain political --adaptation_method feature
    python domain_adaptation.py --few_shot --target_samples 100 --adaptation_method adversarial
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
from fact_verification.models import FactVerifier, FactVerifierConfig
from shared.utils.logging_utils import get_logger
from shared.utils.checkpoint_manager import CheckpointManager
from shared.datasets.data_loaders import ChunkedDataLoader

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import get_cosine_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer implementation."""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation."""
    
    def __init__(self, input_size: int, num_domains: int, hidden_size: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_domains)
        )
    
    def forward(self, x):
        return self.classifier(x)


class DomainAdaptationTrainer:
    """Trainer for domain adaptation of fact verification models."""
    
    def __init__(
        self,
        model: FactVerifier,
        source_loader: DataLoader,
        target_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger or get_logger("DomainAdaptationTrainer")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Domain adaptation components
        self.domain_classifier = None
        self.gradient_reversal = None
        
        if args.adaptation_method == 'adversarial':
            self._setup_adversarial_components()
        
        # Initialize optimizers
        self.model_optimizer = self._setup_model_optimizer()
        self.domain_optimizer = self._setup_domain_optimizer()
        
        # Initialize schedulers
        self.model_scheduler = self._setup_scheduler(self.model_optimizer)
        self.domain_scheduler = self._setup_scheduler(self.domain_optimizer) if self.domain_optimizer else None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir = Path(args.checkpoint_dir) if hasattr(args, "checkpoint_dir") and args.checkpoint_dir else Path("fact_verification/checkpoints"),
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=args.max_checkpoints
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_target_accuracy = 0.0
        
        # Metrics tracking
        self.adaptation_metrics = []
        
        self.logger.info(f"Initialized DomainAdaptationTrainer on device: {self.device}")
        self.logger.info(f"Adaptation method: {args.adaptation_method}")
        self.logger.info(f"Source domain: {args.source_domain}, Target domain: {args.target_domain}")
    
    def _setup_adversarial_components(self):
        """Setup components for adversarial domain adaptation."""
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(alpha=self.args.domain_lambda)
        
        # Domain classifier
        self.domain_classifier = DomainClassifier(
            input_size=self.model.config.hidden_size,
            num_domains=2,  # Source and target
            hidden_size=self.args.domain_classifier_hidden
        )
        
        self.domain_classifier.to(self.device)
        
        self.logger.info("Setup adversarial domain adaptation components")
    
    def _setup_model_optimizer(self) -> optim.Optimizer:
        """Setup optimizer for the main model."""
        
        # Different learning rates for different parts
        encoder_params = list(self.model.roberta.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        param_groups = [
            {'params': encoder_params, 'lr': self.args.model_lr},
            {'params': classifier_params, 'lr': self.args.model_lr * self.args.classifier_lr_multiplier}
        ]
        
        return optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
    
    def _setup_domain_optimizer(self) -> Optional[optim.Optimizer]:
        """Setup optimizer for domain classifier."""
        
        if self.domain_classifier:
            return optim.AdamW(
                self.domain_classifier.parameters(),
                lr=self.args.domain_lr,
                weight_decay=self.args.weight_decay
            )
        return None
    
    def _setup_scheduler(self, optimizer):
        """Setup learning rate scheduler."""
        
        if not TRANSFORMERS_AVAILABLE:
            return optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        
        # Estimate total steps
        max_loader_len = max(len(self.source_loader), len(self.target_loader))
        total_steps = max_loader_len * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def _create_combined_batches(self):
        """Create combined batches from source and target data."""
        
        # Create iterators
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        
        combined_batches = []
        
        # Iterate until one of the iterators is exhausted
        while True:
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
                
                # Add domain labels
                source_batch['domain_label'] = torch.zeros(source_batch['input_ids'].size(0), dtype=torch.long)
                target_batch['domain_label'] = torch.ones(target_batch['input_ids'].size(0), dtype=torch.long)
                
                combined_batches.append((source_batch, target_batch))
                
            except StopIteration:
                break
        
        # If one loader is shorter, continue with the remaining batches
        try:
            while True:
                source_batch = next(source_iter)
                source_batch['domain_label'] = torch.zeros(source_batch['input_ids'].size(0), dtype=torch.long)
                combined_batches.append((source_batch, None))
        except StopIteration:
            pass
        
        try:
            while True:
                target_batch = next(target_iter)
                target_batch['domain_label'] = torch.ones(target_batch['input_ids'].size(0), dtype=torch.long)
                combined_batches.append((None, target_batch))
        except StopIteration:
            pass
        
        return combined_batches
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with domain adaptation."""
        
        self.model.train()
        if self.domain_classifier:
            self.domain_classifier.train()
        
        # Update gradient reversal alpha
        if self.gradient_reversal and self.args.dynamic_lambda:
            # Gradually increase domain adversarial strength
            progress = self.epoch / self.args.epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1
            self.gradient_reversal.alpha = alpha * self.args.domain_lambda
        
        # Training metrics
        total_task_loss = 0.0
        total_domain_loss = 0.0
        total_combined_loss = 0.0
        
        task_predictions = []
        task_labels = []
        domain_predictions = []
        domain_labels = []
        
        num_batches = 0
        
        # Create combined batches
        combined_batches = self._create_combined_batches()
        
        progress_bar = tqdm(combined_batches, desc=f"Epoch {self.epoch} (DA)")
        
        for batch_idx, (source_batch, target_batch) in enumerate(progress_bar):
            self.model_optimizer.zero_grad()
            if self.domain_optimizer:
                self.domain_optimizer.zero_grad()
            
            batch_task_loss = 0.0
            batch_domain_loss = 0.0
            batch_size = 0
            
            # Process source batch
            if source_batch is not None:
                source_batch = self._move_batch_to_device(source_batch)
                batch_size += source_batch['input_ids'].size(0)
                
                # Task loss on source data
                if self.args.adaptation_method in ['feature', 'adversarial']:
                    task_loss, task_preds = self._compute_task_loss(source_batch)
                    batch_task_loss += task_loss
                    
                    task_predictions.extend(task_preds.cpu().numpy())
                    task_labels.extend(source_batch['label'].cpu().numpy())
                
                # Domain loss on source data
                if self.args.adaptation_method == 'adversarial':
                    domain_loss, domain_preds = self._compute_domain_loss(source_batch, is_source=True)
                    batch_domain_loss += domain_loss
                    
                    domain_predictions.extend(domain_preds.cpu().numpy())
                    domain_labels.extend(source_batch['domain_label'].cpu().numpy())
            
            # Process target batch
            if target_batch is not None:
                target_batch = self._move_batch_to_device(target_batch)
                batch_size += target_batch['input_ids'].size(0)
                
                # Task loss on target data (if supervised)
                if self.args.adaptation_method == 'feature' and not self.args.unsupervised:
                    task_loss, task_preds = self._compute_task_loss(target_batch)
                    batch_task_loss += task_loss
                    
                    task_predictions.extend(task_preds.cpu().numpy())
                    task_labels.extend(target_batch['label'].cpu().numpy())
                
                # Domain loss on target data
                if self.args.adaptation_method == 'adversarial':
                    domain_loss, domain_preds = self._compute_domain_loss(target_batch, is_source=False)
                    batch_domain_loss += domain_loss
                    
                    domain_predictions.extend(domain_preds.cpu().numpy())
                    domain_labels.extend(target_batch['domain_label'].cpu().numpy())
            
            # Combine losses
            if self.args.adaptation_method == 'adversarial':
                combined_loss = batch_task_loss + self.args.domain_weight * batch_domain_loss
            else:
                combined_loss = batch_task_loss
            
            # Backward pass
            if self.args.gradient_accumulation_steps > 1:
                combined_loss = combined_loss / self.args.gradient_accumulation_steps
            
            combined_loss.backward()
            
            # Gradient accumulation and optimization
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if self.domain_classifier:
                        torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), self.args.max_grad_norm)
                
                self.model_optimizer.step()
                self.model_scheduler.step()
                
                if self.domain_optimizer:
                    self.domain_optimizer.step()
                    if self.domain_scheduler:
                        self.domain_scheduler.step()
                
                self.global_step += 1
            
            # Track metrics
            total_task_loss += batch_task_loss.item() if isinstance(batch_task_loss, torch.Tensor) else batch_task_loss
            total_domain_loss += batch_domain_loss.item() if isinstance(batch_domain_loss, torch.Tensor) else batch_domain_loss
            total_combined_loss += combined_loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            task_acc = accuracy_score(task_labels, task_predictions) if task_labels else 0.0
            domain_acc = accuracy_score(domain_labels, domain_predictions) if domain_labels else 0.0
            
            progress_bar.set_postfix({
                'task_loss': f'{total_task_loss / num_batches:.4f}',
                'domain_loss': f'{total_domain_loss / num_batches:.4f}',
                'task_acc': f'{task_acc:.4f}',
                'domain_acc': f'{domain_acc:.4f}',
                'alpha': f'{self.gradient_reversal.alpha:.3f}' if self.gradient_reversal else 'N/A'
            })
            
            # Log metrics
            if self.global_step % self.args.log_interval == 0:
                self._log_metrics({
                    'train_task_loss': total_task_loss / num_batches,
                    'train_domain_loss': total_domain_loss / num_batches,
                    'train_combined_loss': total_combined_loss / num_batches,
                    'train_task_accuracy': task_acc,
                    'train_domain_accuracy': domain_acc,
                    'gradient_reversal_alpha': self.gradient_reversal.alpha if self.gradient_reversal else 0,
                    'epoch': self.epoch,
                    'global_step': self.global_step
                })
        
        # Compute final epoch metrics
        epoch_task_acc = accuracy_score(task_labels, task_predictions) if task_labels else 0.0
        epoch_domain_acc = accuracy_score(domain_labels, domain_predictions) if domain_labels else 0.0
        
        epoch_metrics = {
            'train_task_loss': total_task_loss / num_batches if num_batches > 0 else 0.0,
            'train_domain_loss': total_domain_loss / num_batches if num_batches > 0 else 0.0,
            'train_combined_loss': total_combined_loss / num_batches if num_batches > 0 else 0.0,
            'train_task_accuracy': epoch_task_acc,
            'train_domain_accuracy': epoch_domain_acc,
            'gradient_reversal_alpha': self.gradient_reversal.alpha if self.gradient_reversal else 0
        }
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def _compute_task_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute task loss and predictions."""
        
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        
        task_loss = outputs['loss']
        predictions = torch.argmax(outputs['logits'], dim=-1)
        
        return task_loss, predictions
    
    def _compute_domain_loss(self, batch: Dict[str, Any], is_source: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute domain classification loss."""
        
        if not self.domain_classifier or not self.gradient_reversal:
            return torch.tensor(0.0, device=self.device), torch.tensor([])
        
        # Get features from the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            features = outputs['pooled_output'].detach()
        
        # Apply gradient reversal
        reversed_features = self.gradient_reversal(features)
        
        # Domain classification
        domain_logits = self.domain_classifier(reversed_features)
        domain_predictions = torch.argmax(domain_logits, dim=-1)
        
        # Domain loss
        domain_loss = nn.CrossEntropyLoss()(domain_logits, batch['domain_label'])
        
        return domain_loss, domain_predictions
    
    def evaluate_target_domain(self) -> Dict[str, float]:
        """Evaluate model on target domain."""
        
        self.model.eval()
        if self.domain_classifier:
            self.domain_classifier.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating Target"):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['label']
                )
                
                total_loss += outputs['loss'].item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        
        val_metrics = {
            'val_target_loss': total_loss / len(self.val_loader),
            'val_target_accuracy': accuracy,
            'val_target_precision': precision,
            'val_target_recall': recall,
            'val_target_f1': f1
        }
        
        # Add per-class metrics
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        for i, class_name in enumerate(class_names):
            if i < len(class_precision):
                val_metrics[f'val_target_{class_name.lower()}_precision'] = class_precision[i]
                val_metrics[f'val_target_{class_name.lower()}_recall'] = class_recall[i]
                val_metrics[f'val_target_{class_name.lower()}_f1'] = class_f1[i]
        
        return val_metrics
    
    def train(self):
        """Main training loop."""
        
        self.logger.info(f"Starting domain adaptation training for {self.args.epochs} epochs")
        
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.adaptation_metrics.append(train_metrics)
            
            # Validation on target domain
            if (epoch + 1) % self.args.eval_interval == 0:
                val_metrics = self.evaluate_target_domain()
                
                # Log validation metrics
                self._log_metrics(val_metrics)
                
                # Check for best model
                current_accuracy = val_metrics['val_target_accuracy']
                if current_accuracy > self.best_target_accuracy:
                    self.best_target_accuracy = current_accuracy
                    self._save_best_checkpoint(val_metrics)
                
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Task Loss={train_metrics['train_task_loss']:.4f}, "
                    f"Domain Loss={train_metrics['train_domain_loss']:.4f}, "
                    f"Task Acc={train_metrics['train_task_accuracy']:.4f}, "
                    f"Target Acc={current_accuracy:.4f}, "
                    f"Target F1={val_metrics['val_target_f1']:.4f}"
                )
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint()
        
        self.logger.info(f"Domain adaptation completed. Best target accuracy: {self.best_target_accuracy:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'model_scheduler_state_dict': self.model_scheduler.state_dict() if self.model_scheduler else None,
            'best_target_accuracy': self.best_target_accuracy,
            'adaptation_metrics': self.adaptation_metrics,
            'args': vars(self.args)
        }
        
        # Save domain classifier if available
        if self.domain_classifier:
            checkpoint['domain_classifier_state_dict'] = self.domain_classifier.state_dict()
            
        if self.domain_optimizer:
            checkpoint['domain_optimizer_state_dict'] = self.domain_optimizer.state_dict()
            
        if self.domain_scheduler:
            checkpoint['domain_scheduler_state_dict'] = self.domain_scheduler.state_dict()
        
        checkpoint_name = f"domain_adaptation_{self.args.target_domain}_epoch_{self.epoch}.pt"
        self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name)
    
    def _save_best_checkpoint(self, val_metrics: Dict[str, float]):
        """Save best domain-adapted model."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_target_accuracy': self.best_target_accuracy,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        # Save model using standard save_pretrained method
        best_model_dir = self.checkpoint_manager.savedir / f"best_adapted_{self.args.target_domain}"
        self.model.save_pretrained(str(best_model_dir))
        
        # Also save checkpoint for training resumption
        checkpoint_name = f"best_domain_adaptation_{self.args.target_domain}.pt"
        self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name)
        
        self.logger.info(f"Saved best domain-adapted model for {self.args.target_domain} with accuracy: {self.best_target_accuracy:.4f}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking systems."""
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(metrics, step=self.global_step)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            
            if checkpoint.get('model_scheduler_state_dict') and self.model_scheduler:
                self.model_scheduler.load_state_dict(checkpoint['model_scheduler_state_dict'])
            
            # Load domain components if available
            if 'domain_classifier_state_dict' in checkpoint and self.domain_classifier:
                self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])
            
            if 'domain_optimizer_state_dict' in checkpoint and self.domain_optimizer:
                self.domain_optimizer.load_state_dict(checkpoint['domain_optimizer_state_dict'])
            
            if 'domain_scheduler_state_dict' in checkpoint and self.domain_scheduler:
                self.domain_scheduler.load_state_dict(checkpoint['domain_scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_target_accuracy = checkpoint['best_target_accuracy']
            self.adaptation_metrics = checkpoint.get('adaptation_metrics', [])
            
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


def create_domain_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create source, target, and validation data loaders."""
    
    logger = get_logger("create_domain_data_loaders")
    
    # Map domain names to datasets
    domain_mapping = {
        'general': ('fever', 'train'),
        'political': ('liar', 'train'),
        'fever': ('fever', 'train'),
        'liar': ('liar', 'train')
    }
    
    # Source domain data
    source_dataset_name, source_split = domain_mapping.get(args.source_domain, ('fever', 'train'))
    
    if source_dataset_name == 'fever':
        source_dataset = FeverDataset(source_split)
    elif source_dataset_name == 'liar':
        source_dataset = LiarDataset(source_split)
    else:
        source_dataset = UnifiedFactDataset(source_split, use_both_datasets=True)
    
    # Target domain data
    target_dataset_name, target_split = domain_mapping.get(args.target_domain, ('liar', 'train'))
    
    if target_dataset_name == 'fever':
        target_dataset = FeverDataset(target_split)
    elif target_dataset_name == 'liar':
        target_dataset = LiarDataset(target_split)
    else:
        target_dataset = UnifiedFactDataset(target_split, use_both_datasets=True)
    
    # Few-shot sampling if specified
    if args.few_shot:
        target_indices = list(range(len(target_dataset)))
        random.shuffle(target_indices)
        
        # Use only specified number of target samples
        target_samples = min(args.target_samples, len(target_indices))
        target_subset_indices = target_indices[:target_samples]
        target_dataset = Subset(target_dataset, target_subset_indices)
        
        logger.info(f"Using {target_samples} samples for few-shot adaptation")
    
    # Validation data (from target domain)
    val_split = 'valid' if target_dataset_name == 'liar' else 'test'
    
    if target_dataset_name == 'fever':
        val_dataset = FeverDataset(val_split)
    elif target_dataset_name == 'liar':
        val_dataset = LiarDataset(val_split)
    else:
        val_dataset = UnifiedFactDataset(val_split, use_both_datasets=True)
    
    # Create data loaders
    loader_class = ChunkedDataLoader if args.use_chunked_loading else DataLoader
    
    source_loader = loader_class(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size if args.use_chunked_loading else None
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"Created domain data loaders:")
    logger.info(f"  Source ({args.source_domain}): {len(source_loader)} batches")
    logger.info(f"  Target ({args.target_domain}): {len(target_loader)} batches")
    logger.info(f"  Validation: {len(val_loader)} batches")
    
    return source_loader, target_loader, val_loader


def main(argv=None):
    """Main domain adaptation training function."""
    
    parser = argparse.ArgumentParser(description="Domain Adaptation for Fact Verification")
    
    # Domain arguments
    parser.add_argument("--source_domain", type=str, default="general", 
                       choices=["general", "political", "fever", "liar"], 
                       help="Source domain")
    parser.add_argument("--target_domain", type=str, default="political",
                       choices=["general", "political", "fever", "liar"],
                       help="Target domain")
    
    # Adaptation method
    parser.add_argument("--adaptation_method", type=str, default="feature",
                       choices=["feature", "adversarial"],
                       help="Domain adaptation method")
    
    # Few-shot learning
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot adaptation")
    parser.add_argument("--target_samples", type=int, default=100, help="Number of target samples for few-shot")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Learning rates
    parser.add_argument("--model_lr", type=float, default=1e-5, help="Model learning rate")
    parser.add_argument("--domain_lr", type=float, default=1e-4, help="Domain classifier learning rate")
    parser.add_argument("--classifier_lr_multiplier", type=float, default=10.0, help="Classifier LR multiplier")
    
    # Domain adaptation parameters
    parser.add_argument("--domain_lambda", type=float, default=0.1, help="Domain adaptation strength")
    parser.add_argument("--domain_weight", type=float, default=0.1, help="Domain loss weight")
    parser.add_argument("--dynamic_lambda", action="store_true", help="Use dynamic lambda scheduling")
    parser.add_argument("--domain_classifier_hidden", type=int, default=256, help="Domain classifier hidden size")
    
    # Training configuration
    parser.add_argument("--unsupervised", action="store_true", help="Unsupervised domain adaptation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Data arguments
    parser.add_argument("--use_chunked_loading", action="store_true", help="Use chunked data loading")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for data loading")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="fact_verification/checkpoints/domain_adaptation", 
                       help="Checkpoint directory")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum checkpoints to keep")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    
    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="factcheck-mm", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args(argv)
    
    # Setup logging
    logger = get_logger("domain_adaptation")
    logger.info(f"Starting domain adaptation with args: {args}")
    
    # Initialize wandb if specified
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name or f"domain_adapt_{args.source_domain}_to_{args.target_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Load pretrained model
    logger.info(f"Loading pretrained model from: {args.pretrained_model}")
    model = FactVerifier.from_pretrained(args.pretrained_model)
    
    # Create data loaders
    source_loader, target_loader, val_loader = create_domain_data_loaders(args)
    
    # Initialize trainer
    trainer = DomainAdaptationTrainer(
        model, source_loader, target_loader, val_loader, args, logger
    )
    
    # Load checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Run training or evaluation
    if args.eval_only:
        logger.info("Running evaluation only...")
        val_metrics = trainer.evaluate_target_domain()
        logger.info("Evaluation results:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    else:
        trainer.train()
    
    logger.info("Domain adaptation completed!")


if __name__ == "__main__":
    main()
