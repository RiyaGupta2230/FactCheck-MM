#!/usr/bin/env python3
"""
Training Script for Evidence Retrieval Models

Trains dense retrieval models using contrastive learning with InfoNCE loss,
supporting hard negative sampling and evaluation on retrieval metrics.

Example Usage:
    python train_retrieval.py --epochs 10 --batch_size 16 --learning_rate 1e-5
    python train_retrieval.py --resume_from checkpoints/retrieval_epoch_5.pt --eval_only
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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
from fact_verification.models import EvidenceRetriever, EvidenceRetrieverConfig
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


class RetrievalTrainer:
    """Trainer for evidence retrieval models with contrastive learning."""
    
    def __init__(
        self,
        model: EvidenceRetriever,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger or get_logger("RetrievalTrainer")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
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
        self.best_recall = 0.0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger.info(f"Initialized RetrievalTrainer on device: {self.device}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for encoder and projection layers."""
        
        # Separate parameters for different learning rates
        encoder_params = []
        projection_params = []
        
        if self.model.dense_retriever:
            # Lower learning rate for pre-trained encoders
            encoder_params.extend(list(self.model.dense_retriever.query_encoder.parameters()))
            encoder_params.extend(list(self.model.dense_retriever.context_encoder.parameters()))
            
            # Higher learning rate for projection layers
            projection_params.extend(list(self.model.dense_retriever.query_projection.parameters()))
            projection_params.extend(list(self.model.dense_retriever.context_projection.parameters()))
        
        param_groups = [
            {'params': encoder_params, 'lr': self.args.learning_rate},
            {'params': projection_params, 'lr': self.args.learning_rate * self.args.projection_lr_multiplier}
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            loss = self._compute_contrastive_loss(batch)
            
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
            
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log training metrics
            if self.global_step % self.args.log_interval == 0:
                self._log_metrics({
                    'train_loss': avg_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': self.epoch,
                    'global_step': self.global_step
                })
        
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return epoch_metrics
    
    def _compute_contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        
        # Get claim and evidence inputs
        claim_input_ids = batch['input_ids']
        claim_attention_mask = batch['attention_mask']
        
        # For contrastive learning, we need positive and negative evidence
        # This assumes the batch contains evidence in the right format
        evidence_input_ids = batch.get('evidence_input_ids', claim_input_ids)
        evidence_attention_mask = batch.get('evidence_attention_mask', claim_attention_mask)
        
        # Forward pass through dense retriever
        outputs = self.model.forward(
            query_input_ids=claim_input_ids,
            query_attention_mask=claim_attention_mask,
            context_input_ids=evidence_input_ids,
            context_attention_mask=evidence_attention_mask
        )
        
        # Extract similarity scores
        similarity_scores = outputs['similarity_scores']
        
        # InfoNCE loss (assumes positive pairs are on the diagonal)
        batch_size = similarity_scores.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(similarity_scores, labels)
        
        return loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        
        self.model.eval()
        total_loss = 0.0
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                loss = self._compute_contrastive_loss(batch)
                total_loss += loss.item()
                
                # Collect predictions for retrieval metrics
                outputs = self.model.forward(
                    query_input_ids=batch['input_ids'],
                    query_attention_mask=batch['attention_mask'],
                    context_input_ids=batch.get('evidence_input_ids', batch['input_ids']),
                    context_attention_mask=batch.get('evidence_attention_mask', batch['attention_mask'])
                )
                
                similarity_scores = outputs['similarity_scores']
                batch_size = similarity_scores.size(0)
                labels = torch.arange(batch_size, device=self.device)
                
                all_scores.append(similarity_scores.cpu())
                all_labels.append(labels.cpu())
        
        # Compute retrieval metrics
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        retrieval_metrics = self._compute_retrieval_metrics(all_scores, all_labels)
        
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            **retrieval_metrics
        }
        
        return val_metrics
    
    def _compute_retrieval_metrics(
        self, 
        scores: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute retrieval metrics (Recall@k, MRR)."""
        
        # Get rankings
        _, rankings = torch.sort(scores, dim=-1, descending=True)
        
        metrics = {}
        
        # Recall@k
        for k in [1, 5, 10, 20]:
            if k <= scores.size(1):
                recall_at_k = 0.0
                for i in range(scores.size(0)):
                    if labels[i] in rankings[i, :k]:
                        recall_at_k += 1.0
                
                metrics[f'recall@{k}'] = recall_at_k / scores.size(0)
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i in range(scores.size(0)):
            # Find position of correct answer
            pos = (rankings[i] == labels[i]).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                mrr += 1.0 / (pos[0].item() + 1)
        
        metrics['mrr'] = mrr / scores.size(0)
        
        return metrics
    
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
                self._log_metrics(val_metrics, prefix='val')
                
                # Check for best model
                current_recall = val_metrics.get('recall@5', 0.0)
                if current_recall > self.best_recall:
                    self.best_recall = current_recall
                    self._save_best_checkpoint(val_metrics)
                
                self.logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                    f"Val Loss={val_metrics['val_loss']:.4f}, "
                    f"Recall@5={current_recall:.4f}"
                )
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint()
        
        self.logger.info(f"Training completed. Best Recall@5: {self.best_recall:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_recall': self.best_recall,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'args': vars(self.args)
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint, f"retrieval_epoch_{self.epoch}.pt")
    
    def _save_best_checkpoint(self, val_metrics: Dict[str, float]):
        """Save best model checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_recall': self.best_recall,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint, "best_retrieval_model.pt")
        self.logger.info(f"Saved best model with Recall@5: {self.best_recall:.4f}")
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to tracking systems."""
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run:
            log_dict = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(log_dict, step=self.global_step)
    
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
            self.best_recall = checkpoint['best_recall']
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
    
    # Load training data (FEVER)
    logger.info("Loading FEVER training data...")
    fever_train = FeverDataset('train')
    
    # Load validation data (LIAR for evaluation)
    logger.info("Loading LIAR validation data...")
    liar_val = LiarDataset('valid')
    
    # Create data loaders
    train_loader_class = ChunkedDataLoader if args.use_chunked_loading else DataLoader
    
    train_loader = train_loader_class(
        fever_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size if args.use_chunked_loading else None
    )
    
    val_loader = DataLoader(
        liar_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"Created data loaders: Train={len(train_loader)}, Val={len(val_loader)}")
    
    return train_loader, val_loader


def main(argv=None):
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train Evidence Retrieval Models")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name")
    parser.add_argument("--projection_dim", type=int, default=256, help="Projection dimension")
    parser.add_argument("--temperature", type=float, default=0.05, help="Contrastive temperature")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--projection_lr_multiplier", type=float, default=10.0, help="LR multiplier for projection layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"])
    
    # Data arguments
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
    logger = get_logger("train_retrieval")
    logger.info(f"Starting evidence retrieval training with args: {args}")
    
    # Initialize wandb if specified
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name or f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create model
    retriever_config = EvidenceRetrieverConfig(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        mode="dense"
    )
    
    model = EvidenceRetriever(retriever_config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Initialize trainer
    trainer = RetrievalTrainer(model, train_loader, val_loader, args, logger)
    
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
