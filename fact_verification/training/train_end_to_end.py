#!/usr/bin/env python3
"""
End-to-End Pipeline Training Script

Jointly trains evidence retrieval and fact verification components with alternating 
optimization strategy for comprehensive fact-checking pipeline performance.

Example Usage:
    python train_end_to_end.py --epochs 5 --batch_size 8 --joint_training
    python train_end_to_end.py --alternate_steps 2 --freeze_retriever --eval_only
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
from fact_verification.models import FactCheckPipeline, FactCheckPipelineConfig
from fact_verification.models import EvidenceRetriever, FactVerifier, StanceDetector
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


class EndToEndTrainer:
    """Trainer for end-to-end fact checking pipeline with joint optimization."""
    
    def __init__(
        self,
        pipeline: FactCheckPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
        logger: Optional[Any] = None
    ):
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger or get_logger("EndToEndTrainer")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline.to(self.device)
        
        # Initialize optimizers for different components
        self.retriever_optimizer = None
        self.verifier_optimizer = None
        self.stance_optimizer = None
        
        self.retriever_scheduler = None
        self.verifier_scheduler = None
        self.stance_scheduler = None
        
        self._setup_optimizers()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("fact_verification/checkpoints"),
            save_dir=checkpoint_dir,
            max_checkpoints=args.max_checkpoints
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.retriever_step = 0
        self.verifier_step = 0
        self.best_pipeline_accuracy = 0.0
        
        # Training phase tracking
        self.current_phase = "retriever"  # "retriever", "verifier", "joint"
        self.phase_step = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger.info(f"Initialized EndToEndTrainer on device: {self.device}")
        self.logger.info(f"Joint training: {args.joint_training}")
        self.logger.info(f"Alternating steps: {args.alternate_steps}")
    
    def _setup_optimizers(self):
        """Setup optimizers for different pipeline components."""
        
        # Retriever optimizer
        if self.pipeline.evidence_retriever and not self.args.freeze_retriever:
            retriever_params = list(self.pipeline.evidence_retriever.parameters())
            if retriever_params:
                self.retriever_optimizer = optim.AdamW(
                    retriever_params,
                    lr=self.args.retriever_lr,
                    weight_decay=self.args.weight_decay
                )
                
                if TRANSFORMERS_AVAILABLE:
                    total_steps = len(self.train_loader) * self.args.epochs
                    self.retriever_scheduler = get_cosine_schedule_with_warmup(
                        self.retriever_optimizer,
                        num_warmup_steps=int(total_steps * self.args.warmup_ratio),
                        num_training_steps=total_steps
                    )
        
        # Verifier optimizer
        if self.pipeline.fact_verifier and not self.args.freeze_verifier:
            # Separate learning rates for encoder and classifier
            encoder_params = list(self.pipeline.fact_verifier.roberta.parameters())
            classifier_params = list(self.pipeline.fact_verifier.classifier.parameters())
            
            param_groups = [
                {'params': encoder_params, 'lr': self.args.verifier_lr},
                {'params': classifier_params, 'lr': self.args.verifier_lr * self.args.classifier_lr_multiplier}
            ]
            
            self.verifier_optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay
            )
            
            if TRANSFORMERS_AVAILABLE:
                total_steps = len(self.train_loader) * self.args.epochs
                self.verifier_scheduler = get_cosine_schedule_with_warmup(
                    self.verifier_optimizer,
                    num_warmup_steps=int(total_steps * self.args.warmup_ratio),
                    num_training_steps=total_steps
                )
        
        # Stance detector optimizer (optional)
        if self.pipeline.stance_detector and not self.args.freeze_stance:
            stance_params = list(self.pipeline.stance_detector.parameters())
            if stance_params:
                self.stance_optimizer = optim.AdamW(
                    stance_params,
                    lr=self.args.stance_lr,
                    weight_decay=self.args.weight_decay
                )
    
    def _update_training_phase(self):
        """Update training phase for alternating optimization."""
        
        if self.args.joint_training:
            self.current_phase = "joint"
            return
        
        # Alternating optimization
        if self.phase_step >= self.args.alternate_steps:
            if self.current_phase == "retriever":
                self.current_phase = "verifier"
            else:
                self.current_phase = "retriever"
            self.phase_step = 0
        
        self.phase_step += 1
    
    def _compute_retrieval_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute retrieval loss for evidence retriever."""
        
        if not self.pipeline.evidence_retriever or self.args.freeze_retriever:
            return torch.tensor(0.0, device=self.device)
        
        # Get claim and evidence inputs
        claim_input_ids = batch['input_ids']
        claim_attention_mask = batch['attention_mask']
        evidence_input_ids = batch.get('evidence_input_ids', claim_input_ids)
        evidence_attention_mask = batch.get('evidence_attention_mask', claim_attention_mask)
        
        # Forward pass through retriever
        retriever_outputs = self.pipeline.evidence_retriever.forward(
            query_input_ids=claim_input_ids,
            query_attention_mask=claim_attention_mask,
            context_input_ids=evidence_input_ids,
            context_attention_mask=evidence_attention_mask
        )
        
        return retriever_outputs.get('loss', torch.tensor(0.0, device=self.device))
    
    def _compute_verification_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute verification loss for fact verifier."""
        
        if not self.pipeline.fact_verifier or self.args.freeze_verifier:
            return torch.tensor(0.0, device=self.device)
        
        # Forward pass through verifier
        verifier_outputs = self.pipeline.fact_verifier(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        
        return verifier_outputs.get('loss', torch.tensor(0.0, device=self.device))
    
    def _compute_stance_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute stance detection loss."""
        
        if not self.pipeline.stance_detector or self.args.freeze_stance:
            return torch.tensor(0.0, device=self.device)
        
        # For stance detection, we need claim-evidence pairs
        # This is simplified - in practice would need proper stance labels
        stance_labels = torch.randint(0, 3, (batch['input_ids'].size(0),), device=self.device)
        
        stance_outputs = self.pipeline.stance_detector(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=stance_labels
        )
        
        return stance_outputs.get('classification_loss', torch.tensor(0.0, device=self.device))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with alternating optimization."""
        
        self.pipeline.train()
        
        total_retrieval_loss = 0.0
        total_verification_loss = 0.0
        total_stance_loss = 0.0
        total_joint_loss = 0.0
        
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} ({self.current_phase})")
        
        for batch_idx, batch in enumerate(progress_bar):
            self._update_training_phase()
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Zero gradients
            if self.retriever_optimizer:
                self.retriever_optimizer.zero_grad()
            if self.verifier_optimizer:
                self.verifier_optimizer.zero_grad()
            if self.stance_optimizer:
                self.stance_optimizer.zero_grad()
            
            # Compute losses based on training phase
            retrieval_loss = torch.tensor(0.0, device=self.device)
            verification_loss = torch.tensor(0.0, device=self.device)
            stance_loss = torch.tensor(0.0, device=self.device)
            
            if self.current_phase == "retriever" or self.current_phase == "joint":
                retrieval_loss = self._compute_retrieval_loss(batch)
            
            if self.current_phase == "verifier" or self.current_phase == "joint":
                verification_loss = self._compute_verification_loss(batch)
                
                # Get predictions for metrics
                with torch.no_grad():
                    verifier_outputs = self.pipeline.fact_verifier(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    predictions = torch.argmax(verifier_outputs['logits'], dim=-1).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_labels.extend(batch['label'].cpu().numpy())
            
            if self.args.train_stance and (self.current_phase == "joint"):
                stance_loss = self._compute_stance_loss(batch)
            
            # Combine losses
            if self.args.joint_training:
                total_loss = (
                    self.args.retrieval_weight * retrieval_loss +
                    self.args.verification_weight * verification_loss +
                    self.args.stance_weight * stance_loss
                )
            else:
                if self.current_phase == "retriever":
                    total_loss = retrieval_loss
                elif self.current_phase == "verifier":
                    total_loss = verification_loss
                else:
                    total_loss = verification_loss + stance_loss
            
            # Gradient accumulation
            if self.args.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            total_loss.backward()
            
            # Gradient accumulation and optimization
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    if self.current_phase in ["retriever", "joint"] and self.retriever_optimizer:
                        torch.nn.utils.clip_grad_norm_(
                            self.pipeline.evidence_retriever.parameters(), 
                            self.args.max_grad_norm
                        )
                    
                    if self.current_phase in ["verifier", "joint"] and self.verifier_optimizer:
                        torch.nn.utils.clip_grad_norm_(
                            self.pipeline.fact_verifier.parameters(), 
                            self.args.max_grad_norm
                        )
                
                # Update optimizers based on phase
                if self.current_phase in ["retriever", "joint"] and self.retriever_optimizer:
                    self.retriever_optimizer.step()
                    if self.retriever_scheduler:
                        self.retriever_scheduler.step()
                    self.retriever_step += 1
                
                if self.current_phase in ["verifier", "joint"] and self.verifier_optimizer:
                    self.verifier_optimizer.step()
                    if self.verifier_scheduler:
                        self.verifier_scheduler.step()
                    self.verifier_step += 1
                
                if self.current_phase == "joint" and self.stance_optimizer:
                    self.stance_optimizer.step()
                
                self.global_step += 1
            
            # Track losses
            total_retrieval_loss += retrieval_loss.item()
            total_verification_loss += verification_loss.item()
            total_stance_loss += stance_loss.item()
            total_joint_loss += total_loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_accuracy = 0.0
            if all_labels:
                current_accuracy = accuracy_score(all_labels, all_predictions)
            
            progress_bar.set_postfix({
                'total_loss': f'{total_joint_loss / num_batches:.4f}',
                'ret_loss': f'{total_retrieval_loss / num_batches:.4f}',
                'ver_loss': f'{total_verification_loss / num_batches:.4f}',
                'acc': f'{current_accuracy:.4f}',
                'phase': self.current_phase
            })
            
            # Log training metrics
            if self.global_step % self.args.log_interval == 0:
                self._log_metrics({
                    'train_total_loss': total_joint_loss / num_batches,
                    'train_retrieval_loss': total_retrieval_loss / num_batches,
                    'train_verification_loss': total_verification_loss / num_batches,
                    'train_stance_loss': total_stance_loss / num_batches,
                    'train_accuracy': current_accuracy,
                    'training_phase': self.current_phase,
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'retriever_lr': self.retriever_scheduler.get_last_lr()[0] if self.retriever_scheduler else 0,
                    'verifier_lr': self.verifier_scheduler.get_last_lr()[0] if self.verifier_scheduler else 0
                })
        
        # Compute final epoch metrics
        epoch_accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
        
        epoch_metrics = {
            'train_total_loss': total_joint_loss / num_batches,
            'train_retrieval_loss': total_retrieval_loss / num_batches,
            'train_verification_loss': total_verification_loss / num_batches,
            'train_stance_loss': total_stance_loss / num_batches,
            'train_accuracy': epoch_accuracy,
            'retriever_steps': self.retriever_step,
            'verifier_steps': self.verifier_step
        }
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate end-to-end pipeline performance."""
        
        self.pipeline.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        pipeline_results = []
        
        total_verification_loss = 0.0
        processing_times = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                batch_start_time = time.time()
                
                # End-to-end pipeline evaluation
                claims = batch.get('claim_text', [])
                labels = batch['label'].cpu().numpy()
                
                for i, claim in enumerate(claims):
                    try:
                        # Run full pipeline
                        result = self.pipeline.check_fact(claim)
                        
                        # Map pipeline verdict to label ID
                        verdict_to_id = {
                            'SUPPORTS': 0,
                            'REFUTES': 1, 
                            'NOT_ENOUGH_INFO': 2
                        }
                        
                        predicted_id = verdict_to_id.get(result['verdict'], 2)
                        confidence = result['confidence']
                        
                        all_predictions.append(predicted_id)
                        all_confidences.append(confidence)
                        pipeline_results.append(result)
                        
                    except Exception as e:
                        # Handle pipeline failures
                        all_predictions.append(2)  # Default to NOT_ENOUGH_INFO
                        all_confidences.append(0.1)
                        pipeline_results.append({'verdict': 'ERROR', 'confidence': 0.1})
                
                all_labels.extend(labels)
                
                # Also evaluate verifier directly for comparison
                if self.pipeline.fact_verifier:
                    verifier_outputs = self.pipeline.fact_verifier(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['label']
                    )
                    total_verification_loss += verifier_outputs.get('loss', 0).item()
                
                processing_times.append(time.time() - batch_start_time)
        
        # Compute comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        
        # Pipeline-specific metrics
        pipeline_precision_at_k = self._compute_pipeline_precision_at_k(all_confidences, all_labels, all_predictions)
        
        val_metrics = {
            'val_pipeline_accuracy': accuracy,
            'val_pipeline_precision': precision,
            'val_pipeline_recall': recall,
            'val_pipeline_f1': f1,
            'val_verification_loss': total_verification_loss / len(self.val_loader) if self.pipeline.fact_verifier else 0,
            'val_avg_processing_time': np.mean(processing_times),
            'val_avg_confidence': np.mean(all_confidences),
            **pipeline_precision_at_k
        }
        
        # Add per-class metrics
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        for i, class_name in enumerate(class_names):
            if i < len(class_precision):
                val_metrics[f'val_{class_name.lower()}_precision'] = class_precision[i]
                val_metrics[f'val_{class_name.lower()}_recall'] = class_recall[i]
                val_metrics[f'val_{class_name.lower()}_f1'] = class_f1[i]
        
        return val_metrics
    
    def _compute_pipeline_precision_at_k(
        self, 
        confidences: List[float], 
        labels: List[int], 
        predictions: List[int]
    ) -> Dict[str, float]:
        """Compute Precision@k for pipeline predictions."""
        
        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        
        precision_at_k = {}
        
        for k_percent in [10, 20, 50]:  # Top k% most confident predictions
            k = max(1, int(len(sorted_indices) * k_percent / 100))
            top_k_indices = sorted_indices[:k]
            
            top_k_labels = [labels[i] for i in top_k_indices]
            top_k_predictions = [predictions[i] for i in top_k_indices]
            
            if top_k_labels:
                accuracy_at_k = accuracy_score(top_k_labels, top_k_predictions)
                precision_at_k[f'precision@{k_percent}'] = accuracy_at_k
        
        return precision_at_k
    
    def train(self):
        """Main training loop."""
        
        self.logger.info(f"Starting end-to-end training for {self.args.epochs} epochs")
        
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
                current_accuracy = val_metrics['val_pipeline_accuracy']
                if current_accuracy > self.best_pipeline_accuracy:
                    self.best_pipeline_accuracy = current_accuracy
                    self._save_best_checkpoint(val_metrics)
                
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['train_total_loss']:.4f}, "
                    f"Train Acc={train_metrics['train_accuracy']:.4f}, "
                    f"Val Pipeline Acc={current_accuracy:.4f}, "
                    f"Val F1={val_metrics['val_pipeline_f1']:.4f}, "
                    f"Avg Processing Time={val_metrics['val_avg_processing_time']:.3f}s"
                )
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint()
        
        self.logger.info(f"Training completed. Best pipeline accuracy: {self.best_pipeline_accuracy:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'retriever_step': self.retriever_step,
            'verifier_step': self.verifier_step,
            'current_phase': self.current_phase,
            'phase_step': self.phase_step,
            'best_pipeline_accuracy': self.best_pipeline_accuracy,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'args': vars(self.args)
        }
        
        # Save pipeline state
        try:
            pipeline_checkpoint = {
                'pipeline_state_dict': self.pipeline.state_dict(),
                **checkpoint
            }
        except:
            # Fallback: save individual components
            pipeline_checkpoint = checkpoint
            
            if self.pipeline.evidence_retriever:
                checkpoint['retriever_state_dict'] = self.pipeline.evidence_retriever.state_dict()
            if self.pipeline.fact_verifier:
                checkpoint['verifier_state_dict'] = self.pipeline.fact_verifier.state_dict()
            if self.pipeline.stance_detector:
                checkpoint['stance_state_dict'] = self.pipeline.stance_detector.state_dict()
        
        # Save optimizer states
        if self.retriever_optimizer:
            checkpoint['retriever_optimizer_state_dict'] = self.retriever_optimizer.state_dict()
            checkpoint['retriever_scheduler_state_dict'] = self.retriever_scheduler.state_dict() if self.retriever_scheduler else None
        
        if self.verifier_optimizer:
            checkpoint['verifier_optimizer_state_dict'] = self.verifier_optimizer.state_dict()
            checkpoint['verifier_scheduler_state_dict'] = self.verifier_scheduler.state_dict() if self.verifier_scheduler else None
        
        self.checkpoint_manager.save_checkpoint(checkpoint, f"pipeline_epoch_{self.epoch}.pt")
    
    def _save_best_checkpoint(self, val_metrics: Dict[str, float]):
        """Save best pipeline checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_pipeline_accuracy': self.best_pipeline_accuracy,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        # Save pipeline components separately for easier loading
        if self.pipeline.evidence_retriever:
            self.pipeline.evidence_retriever.save_pretrained(
                self.checkpoint_manager.save_dir / "best_retriever"
            )
        
        if self.pipeline.fact_verifier:
            self.pipeline.fact_verifier.save_pretrained(
                self.checkpoint_manager.save_dir / "best_verifier"
            )
        
        if self.pipeline.stance_detector:
            self.pipeline.stance_detector.save_pretrained(
                self.checkpoint_manager.save_dir / "best_stance_detector"
            )
        
        self.checkpoint_manager.save_checkpoint(checkpoint, "best_pipeline_model.pt")
        self.logger.info(f"Saved best pipeline with accuracy: {self.best_pipeline_accuracy:.4f}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking systems."""
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(metrics, step=self.global_step)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load pipeline state
            if 'pipeline_state_dict' in checkpoint:
                self.pipeline.load_state_dict(checkpoint['pipeline_state_dict'])
            else:
                # Load individual components
                if 'retriever_state_dict' in checkpoint and self.pipeline.evidence_retriever:
                    self.pipeline.evidence_retriever.load_state_dict(checkpoint['retriever_state_dict'])
                
                if 'verifier_state_dict' in checkpoint and self.pipeline.fact_verifier:
                    self.pipeline.fact_verifier.load_state_dict(checkpoint['verifier_state_dict'])
                
                if 'stance_state_dict' in checkpoint and self.pipeline.stance_detector:
                    self.pipeline.stance_detector.load_state_dict(checkpoint['stance_state_dict'])
            
            # Load optimizer states
            if 'retriever_optimizer_state_dict' in checkpoint and self.retriever_optimizer:
                self.retriever_optimizer.load_state_dict(checkpoint['retriever_optimizer_state_dict'])
            
            if 'retriever_scheduler_state_dict' in checkpoint and self.retriever_scheduler:
                self.retriever_scheduler.load_state_dict(checkpoint['retriever_scheduler_state_dict'])
            
            if 'verifier_optimizer_state_dict' in checkpoint and self.verifier_optimizer:
                self.verifier_optimizer.load_state_dict(checkpoint['verifier_optimizer_state_dict'])
            
            if 'verifier_scheduler_state_dict' in checkpoint and self.verifier_scheduler:
                self.verifier_scheduler.load_state_dict(checkpoint['verifier_scheduler_state_dict'])
            
            # Load training state
            self.epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.retriever_step = checkpoint.get('retriever_step', 0)
            self.verifier_step = checkpoint.get('verifier_step', 0)
            self.current_phase = checkpoint.get('current_phase', 'verifier')
            self.phase_step = checkpoint.get('phase_step', 0)
            self.best_pipeline_accuracy = checkpoint['best_pipeline_accuracy']
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
    
    # Use unified dataset for end-to-end training
    logger.info("Loading unified dataset for end-to-end training...")
    train_dataset = UnifiedFactDataset('train', use_both_datasets=True, balance_datasets=True)
    val_dataset = UnifiedFactDataset('valid', use_both_datasets=True, balance_datasets=True)
    
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
    
    parser = argparse.ArgumentParser(description="Train End-to-End Fact Checking Pipeline")
    
    # Pipeline arguments
    parser.add_argument("--enable_claim_detection", action="store_true", help="Enable claim detection")
    parser.add_argument("--enable_evidence_retrieval", action="store_true", default=True, help="Enable evidence retrieval")
    parser.add_argument("--enable_stance_detection", action="store_true", help="Enable stance detection")
    parser.add_argument("--enable_fact_verification", action="store_true", default=True, help="Enable fact verification")
    
    # Training strategy
    parser.add_argument("--joint_training", action="store_true", help="Joint training of all components")
    parser.add_argument("--alternate_steps", type=int, default=1, help="Steps before alternating components")
    
    # Component freezing
    parser.add_argument("--freeze_retriever", action="store_true", help="Freeze retriever parameters")
    parser.add_argument("--freeze_verifier", action="store_true", help="Freeze verifier parameters") 
    parser.add_argument("--freeze_stance", action="store_true", help="Freeze stance detector parameters")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    
    # Learning rates
    parser.add_argument("--retriever_lr", type=float, default=1e-5, help="Retriever learning rate")
    parser.add_argument("--verifier_lr", type=float, default=2e-5, help="Verifier learning rate")
    parser.add_argument("--stance_lr", type=float, default=2e-5, help="Stance detector learning rate")
    parser.add_argument("--classifier_lr_multiplier", type=float, default=10.0, help="Classifier LR multiplier")
    
    # Loss weights
    parser.add_argument("--retrieval_weight", type=float, default=0.3, help="Retrieval loss weight")
    parser.add_argument("--verification_weight", type=float, default=0.6, help="Verification loss weight")
    parser.add_argument("--stance_weight", type=float, default=0.1, help="Stance loss weight")
    
    # Optimization
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Component training
    parser.add_argument("--train_stance", action="store_true", help="Include stance detection in training")
    
    # Data arguments
    parser.add_argument("--use_chunked_loading", action="store_true", help="Use chunked data loading")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for data loading")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="fact_verification/checkpoints", help="Checkpoint directory")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum checkpoints to keep")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    
    # Pre-trained component paths
    parser.add_argument("--retriever_checkpoint", type=str, help="Pre-trained retriever checkpoint")
    parser.add_argument("--verifier_checkpoint", type=str, help="Pre-trained verifier checkpoint")
    parser.add_argument("--stance_checkpoint", type=str, help="Pre-trained stance detector checkpoint")
    
    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="factcheck-mm", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args(argv)
    
    # Setup logging
    logger = get_logger("train_end_to_end")
    logger.info(f"Starting end-to-end pipeline training with args: {args}")
    
    # Initialize wandb if specified
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name or f"end_to_end_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create pipeline configuration
    pipeline_config = FactCheckPipelineConfig(
        enable_claim_detection=args.enable_claim_detection,
        enable_evidence_retrieval=args.enable_evidence_retrieval,
        enable_stance_detection=args.enable_stance_detection,
        enable_fact_verification=args.enable_fact_verification,
        return_evidence_details=True,
        detailed_explanations=True,
        claim_detector_path=None,
        evidence_retriever_path=args.retriever_checkpoint,
        stance_detector_path=args.stance_checkpoint,
        fact_verifier_path=args.verifier_checkpoint
    )
    
    # Create pipeline
    pipeline = FactCheckPipeline(pipeline_config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Initialize trainer
    trainer = EndToEndTrainer(pipeline, train_loader, val_loader, args, logger)
    
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
