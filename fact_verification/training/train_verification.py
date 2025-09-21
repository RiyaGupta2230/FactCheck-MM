#!/usr/bin/env python3
"""
Training Script for Fact Verification Models

Trains fact verification models with curriculum learning, progressing from
gold evidence to noisy retrieved evidence for robust fact checking.

Example Usage:
    python train_verification.py --epochs 5 --batch_size 16 --curriculum_learning
    python train_verification.py --resume_from checkpoints/verifier_epoch_3.pt --eval_only
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
from fact_verification.models import FactVerifier, FactVerifierConfig, EvidenceRetriever
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
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FactVerificationTrainer:
    """Trainer for fact verification models with curriculum learning."""
    
    def __init__(
        self,
        model: FactVerifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
        evidence_retriever: Optional[EvidenceRetriever] = None,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.evidence_retriever = evidence_retriever
        self.logger = logger or get_logger("FactVerificationTrainer")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if self.evidence_retriever:
            self.evidence_retriever.to(self.device)
            self.evidence_retriever.eval()  # Keep in eval mode
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(args.checkpoint_dir),
            max_checkpoints=args.max_checkpoints
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.curriculum_stage = 0  # 0: gold evidence, 1: mixed, 2: retrieved only
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger.info(f"Initialized FactVerificationTrainer on device: {self.device}")
        self.logger.info(f"Curriculum learning: {args.curriculum_learning}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with layer-wise learning rates."""
        
        # Different learning rates for different parts of the model
        encoder_params = list(self.model.roberta.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        param_groups = [
            {'params': encoder_params, 'lr': self.args.learning_rate},
            {'params': classifier_params, 'lr': self.args.learning_rate * self.args.classifier_lr_multiplier}
        ]
        
        if self.args.optimizer == 'adamw':
            return optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            return optim.Adam(param_groups, weight_decay=self.args.weight_decay)
        else:
            return optim.SGD(param_groups, lr=self.args.learning_rate, momentum=0.9)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        
        if not TRANSFORMERS_AVAILABLE:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)
        
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        if self.args.scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
    
    def _update_curriculum_stage(self):
        """Update curriculum learning stage based on epoch."""
        
        if not self.args.curriculum_learning:
            return
        
        # Stage progression based on epochs
        if self.epoch < self.args.curriculum_epochs // 3:
            self.curriculum_stage = 0  # Gold evidence only
        elif self.epoch < 2 * self.args.curriculum_epochs // 3:
            self.curriculum_stage = 1  # Mixed evidence
        else:
            self.curriculum_stage = 2  # Retrieved evidence only
        
        self.logger.info(f"Curriculum stage: {self.curriculum_stage}")
    
    def _prepare_evidence(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare evidence based on curriculum stage."""
        
        if not self.args.curriculum_learning or self.curriculum_stage == 0:
            # Use gold evidence (from dataset)
            return batch
        
        elif self.curriculum_stage == 1:
            # Mix gold and retrieved evidence
            use_gold = torch.rand(1).item() > 0.5
            if use_gold:
                return batch
        
        # Use retrieved evidence (curriculum_stage == 2 or mixed decision)
        if self.evidence_retriever:
            return self._retrieve_evidence_for_batch(batch)
        else:
            return batch
    
    def _retrieve_evidence_for_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve evidence for claims in batch."""
        
        claims = batch.get('claim_text', [])
        if not claims:
            return batch
        
        # Retrieve evidence for each claim
        retrieved_evidence = []
        
        with torch.no_grad():
            for claim in claims:
                evidence_list = self.evidence_retriever.retrieve(claim, top_k=3)
                evidence_texts = [ev['text'] for ev in evidence_list] if evidence_list else ["No evidence found"]
                retrieved_evidence.append(evidence_texts)
        
        # Update batch with retrieved evidence
        batch_updated = batch.copy()
        batch_updated['evidence_text'] = retrieved_evidence
        
        # Re-process inputs with new evidence
        processed_inputs = []
        for i, claim in enumerate(claims):
            evidence = retrieved_evidence[i]
            combined_text = claim + " </s> " + " [SEP] ".join(evidence)
            
            # Tokenize (simplified - in practice would use text processor)
            inputs = self.model.text_processor.process_text(
                combined_text,
                max_length=self.model.config.max_sequence_length,
                truncation=True
            )
            processed_inputs.append(inputs)
        
        # Stack processed inputs
        if processed_inputs:
            batch_updated['input_ids'] = torch.stack([inp['input_ids'] for inp in processed_inputs])
            batch_updated['attention_mask'] = torch.stack([inp['attention_mask'] for inp in processed_inputs])
        
        return batch_updated
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        self._update_curriculum_stage()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} (Stage {self.curriculum_stage})")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare evidence based on curriculum
            batch = self._prepare_evidence(batch)
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['label'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            current_accuracy = accuracy_score(all_labels, all_predictions)
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{current_accuracy:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'stage': self.curriculum_stage
            })
            
            # Log training metrics
            if self.global_step % self.args.log_interval == 0:
                self._log_metrics({
                    'train_loss': avg_loss,
                    'train_accuracy': current_accuracy,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'curriculum_stage': self.curriculum_stage,
                    'epoch': self.epoch,
                    'global_step': self.global_step
                })
        
        # Compute final epoch metrics
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'train_accuracy': epoch_accuracy,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
            'curriculum_stage': self.curriculum_stage,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['label']
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        
        # Precision@k metrics
        precision_at_k = self._compute_precision_at_k(all_probabilities, all_labels)
        
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            **precision_at_k
        }
        
        # Add per-class metrics
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        for i, class_name in enumerate(class_names):
            val_metrics[f'val_{class_name.lower()}_precision'] = class_precision[i] if i < len(class_precision) else 0.0
            val_metrics[f'val_{class_name.lower()}_recall'] = class_recall[i] if i < len(class_recall) else 0.0
            val_metrics[f'val_{class_name.lower()}_f1'] = class_f1[i] if i < len(class_f1) else 0.0
        
        return val_metrics
    
    def _compute_precision_at_k(self, probabilities: List[np.ndarray], labels: List[int]) -> Dict[str, float]:
        """Compute Precision@k metrics."""
        
        probabilities = np.array(probabilities)
        labels = np.array(labels)
        
        # Get top-k predictions
        top_predictions = np.argsort(probabilities, axis=1)[:, ::-1]
        
        precision_at_k = {}
        
        for k in [1, 2, 3]:
            if k <= probabilities.shape[1]:
                correct = 0
                total = len(labels)
                
                for i in range(total):
                    if labels[i] in top_predictions[i, :k]:
                        correct += 1
                
                precision_at_k[f'precision@{k}'] = correct / total if total > 0 else 0.0
        
        return precision_at_k
    
    def train(self):
        """Main training loop."""
        
        self.logger.info(f"Starting training for {self.args.epochs} epochs")
        
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validation
            if (epoch + 1) % self.args.eval_interval == 0:
                val_metrics = self.evaluate()
                self.val_metrics.append(val_metrics)
                
                # Log validation metrics
                self._log_metrics(val_metrics)
                
                # Check for best model
                current_accuracy = val_metrics['val_accuracy']
                if current_accuracy > self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    self._save_best_checkpoint(val_metrics)
                
                self.logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                    f"Train Acc={train_metrics['train_accuracy']:.4f}, "
                    f"Val Loss={val_metrics['val_loss']:.4f}, "
                    f"Val Acc={current_accuracy:.4f}, "
                    f"Val F1={val_metrics['val_f1']:.4f}"
                )
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint()
        
        self.logger.info(f"Training completed. Best accuracy: {self.best_accuracy:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_accuracy': self.best_accuracy,
            'curriculum_stage': self.curriculum_stage,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'args': vars(self.args)
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint, f"verifier_epoch_{self.epoch}.pt")
    
    def _save_best_checkpoint(self, val_metrics: Dict[str, float]):
        """Save best model checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint, "best_verifier_model.pt")
        self.logger.info(f"Saved best model with accuracy: {self.best_accuracy:.4f}")
    
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
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_accuracy = checkpoint['best_accuracy']
            self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
            self.train_metrics = checkpoint.get('train_metrics', [])
            self.val_metrics = checkpoint.get('val_metrics', [])
            
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


def create_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    logger = get_logger("create_data_loaders")
    
    if args.use_unified_dataset:
        # Use unified dataset combining FEVER and LIAR
        logger.info("Loading unified dataset...")
        train_dataset = UnifiedFactDataset('train', use_both_datasets=True)
        val_dataset = UnifiedFactDataset('valid', use_both_datasets=True)
    else:
        # Use FEVER only
        logger.info("Loading FEVER dataset...")
        train_dataset = FeverDataset('train')
        val_dataset = FeverDataset('test')  # Use test for validation
    
    # Create data loaders
    train_loader_class = ChunkedDataLoader if args.use_chunked_loading else DataLoader
    
    train_loader = train_loader_class(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size if args.use_chunked_loading else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"Created data loaders: Train={len(train_loader)}, Val={len(val_loader)}")
    
    return train_loader, val_loader


def main(argv=None):
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train Fact Verification Models")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--classifier_lr_multiplier", type=float, default=10.0, help="LR multiplier for classifier")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Curriculum learning
    parser.add_argument("--curriculum_learning", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--curriculum_epochs", type=int, default=6, help="Epochs for curriculum progression")
    parser.add_argument("--retriever_checkpoint", type=str, help="Path to trained retriever checkpoint")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"])
    
    # Data arguments
    parser.add_argument("--use_unified_dataset", action="store_true", help="Use unified FEVER+LIAR dataset")
    parser.add_argument("--use_chunked_loading", action="store_true", help="Use chunked data loading")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for data loading")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="fact_verification/checkpoints", help="Checkpoint directory")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="Maximum checkpoints to keep")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    
    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="factcheck-mm", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args(argv)
    
    # Setup logging
    logger = get_logger("train_verification")
    logger.info(f"Starting fact verification training with args: {args}")
    
    # Initialize wandb if specified
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name or f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create model
    verifier_config = FactVerifierConfig(
        model_name=args.model_name,
        max_sequence_length=args.max_sequence_length,
        num_classes=args.num_classes
    )
    
    model = FactVerifier(verifier_config)
    
    # Load evidence retriever if curriculum learning is enabled
    evidence_retriever = None
    if args.curriculum_learning and args.retriever_checkpoint:
        logger.info(f"Loading evidence retriever from {args.retriever_checkpoint}")
        evidence_retriever = EvidenceRetriever.from_pretrained(args.retriever_checkpoint)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Initialize trainer
    trainer = FactVerificationTrainer(
        model, train_loader, val_loader, args, evidence_retriever, logger
    )
    
    # Load checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Run training or evaluation
    if args.eval_only:
        logger.info("Running evaluation only...")
        val_metrics = trainer.evaluate()
        logger.info("Evaluation results:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    else:
        trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
