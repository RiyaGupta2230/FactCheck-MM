# sarcasm_detection/training/train_text.py
"""
Text-Only Sarcasm Detection Training
Trainer for RoBERTa and LSTM text models with comprehensive training features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import gc

from shared.utils import (
    get_logger, CheckpointManager, MetricsComputer,
    ExperimentLogger
)
from shared.datasets import create_hardware_aware_dataloader, MultimodalCollator
from ..models import TextSarcasmModel, RobertaSarcasmModel, LSTMSarcasmModel
from ..utils import SarcasmMetrics


@dataclass
class TextTrainingConfig:
    """Configuration for text model training."""
    
    # Model settings
    model_name: str = "roberta-large"
    num_classes: int = 2
    max_length: int = 512
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "linear"  # linear, cosine, constant
    use_mixed_precision: bool = True
    accumulation_steps: int = 1
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    early_stopping_patience: int = 3
    
    # Logging and checkpointing
    save_every: int = 500
    eval_every: int = 200
    log_every: int = 50
    
    # Hardware
    device: str = "auto"
    max_memory_gb: float = 7.0
    use_chunked_loading: bool = False


class TextSarcasmTrainer:
    """Base trainer for text-based sarcasm detection models."""
    
    def __init__(
        self,
        model: TextSarcasmModel,
        config: Union[TextTrainingConfig, Dict[str, Any]],
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        """
        Initialize text sarcasm trainer.
        
        Args:
            model: Text sarcasm model
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if isinstance(config, dict):
            config = TextTrainingConfig(**config)
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger = get_logger("TextSarcasmTrainer")
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Initialize training components
        self._setup_optimization()
        self._setup_loss_function()
        self._setup_metrics()
        self._setup_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        self.logger.info(f"Initialized text trainer on {self.device}")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        
        # Optimizer
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Learning rate scheduler
        if self.train_dataset:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            
            if self.config.scheduler.lower() == "linear":
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=self.config.warmup_steps
                )
            elif self.config.scheduler.lower() == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps
                )
            else:
                self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_loss_function(self):
        """Setup loss function."""
        if self.config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def _setup_metrics(self):
        """Setup metrics computation."""
        self.metrics_computer = SarcasmMetrics(
            task_name="sarcasm_detection",
            num_classes=self.config.num_classes
        )
    
    def _setup_data_loaders(self):
        """Setup data loaders."""
        # Text processor for collation
        from shared.preprocessing import TextProcessor
        text_processor = TextProcessor(
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
        
        # Collator
        collate_fn = MultimodalCollator(
            text_processor=text_processor.tokenizer,
            max_length=self.config.max_length
        )
        
        # Training data loader
        if self.train_dataset:
            if self.config.use_chunked_loading:
                from shared.datasets import create_chunked_dataloader
                self.train_dataloader = create_chunked_dataloader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    max_memory_gb=self.config.max_memory_gb
                )
            else:
                self.train_dataloader = create_hardware_aware_dataloader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    collate_fn=collate_fn
                )
        
        # Validation data loader
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        # Test data loader
        if self.test_dataset:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Extract text inputs
                text_inputs = {}
                for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    if key in batch:
                        text_inputs[key] = batch[key]
                
                # Model forward
                logits = self.model(**text_inputs)
                
                # Compute loss
                loss = self.criterion(logits, batch['labels'])
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate model on given dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Extract text inputs
                text_inputs = {}
                for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    if key in batch:
                        text_inputs[key] = batch[key]
                
                # Forward pass
                logits = self.model(**text_inputs)
                
                # Compute loss
                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = self.metrics_computer.compute_classification_metrics(
            predictions=all_predictions,
            labels=all_labels,
            mode=split_name
        )
        
        metrics[f'{split_name}_loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        resume_from_checkpoint: Optional[str] = None,
        experiment_name: str = "text_sarcasm_training"
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            experiment_name: Name for experiment logging
            
        Returns:
            Training results
        """
        if not self.train_dataset:
            raise ValueError("Training dataset not provided")
        
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint_manager = CheckpointManager(
                save_dir=checkpoint_dir,
                max_checkpoints=3,
                monitor_metric="val_f1",
                mode="max"
            )
        
        # Setup experiment logging
        self.experiment_logger = ExperimentLogger(
            log_dir=checkpoint_dir or Path("logs"),
            project_name="sarcasm_detection",
            experiment_name=experiment_name,
            config=self.config.__dict__
        )
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        # Log model info
        self.experiment_logger.log_model_info(self.model)
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Early stopping
        best_val_score = self.best_val_score
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_dataset and epoch % (self.config.eval_every // len(self.train_dataloader) + 1) == 0:
                val_metrics = self.evaluate(self.val_dataloader, "val")
                
                # Check for improvement
                current_val_score = val_metrics.get('val_f1', 0.0)
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    patience_counter = 0
                    
                    # Save best checkpoint
                    if hasattr(self, 'checkpoint_manager'):
                        self._save_checkpoint(val_metrics, is_best=True)
                else:
                    patience_counter += 1
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in epoch_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            # Log metrics
            self.experiment_logger.log_metrics(epoch_metrics, step=epoch)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics.get('train_loss', 0):.4f} - "
                f"Val F1: {val_metrics.get('val_f1', 0):.4f}"
            )
            
            # Save regular checkpoint
            if hasattr(self, 'checkpoint_manager') and epoch % (self.config.save_every // len(self.train_dataloader) + 1) == 0:
                self._save_checkpoint(epoch_metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Final evaluation
        final_results = {'training_history': self.training_history}
        
        if self.val_dataset:
            final_val_metrics = self.evaluate(self.val_dataloader, "final_val")
            final_results['final_validation'] = final_val_metrics
        
        if self.test_dataset:
            final_test_metrics = self.evaluate(self.test_dataloader, "test")
            final_results['final_test'] = final_test_metrics
        
        self.experiment_logger.close()
        self.logger.info("Training completed")
        
        return final_results
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if not hasattr(self, 'checkpoint_manager'):
            return
        
        from shared.utils import ModelState
        
        model_state = ModelState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            epoch=self.current_epoch,
            step=self.global_step,
            best_metric=self.best_val_score,
            config=self.config.__dict__,
            metadata={'model_type': 'text_sarcasm'}
        )
        
        self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics=metrics,
            is_best=is_best
        )
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('step', 0)
        self.best_val_score = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f"Resumed training from epoch {self.current_epoch}")


class RobertaTrainer(TextSarcasmTrainer):
    """Specialized trainer for RoBERTa sarcasm models."""
    
    def __init__(self, model: RobertaSarcasmModel, config: Union[TextTrainingConfig, Dict[str, Any]], **kwargs):
        super().__init__(model, config, **kwargs)
        
        # RoBERTa-specific optimizations
        if isinstance(self.config, TextTrainingConfig):
            # Adjust learning rate for different layers
            self._setup_layer_wise_lr()
    
    def _setup_layer_wise_lr(self):
        """Setup different learning rates for different layers."""
        # Lower learning rate for pretrained layers
        pretrained_params = []
        task_specific_params = []
        
        for name, param in self.model.named_parameters():
            if 'roberta' in name:
                pretrained_params.append(param)
            else:
                task_specific_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': pretrained_params, 'lr': self.config.learning_rate * 0.1},
            {'params': task_specific_params, 'lr': self.config.learning_rate}
        ]
        
        # Recreate optimizer with parameter groups
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay
            )


class LSTMTrainer(TextSarcasmTrainer):
    """Specialized trainer for LSTM sarcasm models."""
    
    def __init__(self, model: LSTMSarcasmModel, config: Union[TextTrainingConfig, Dict[str, Any]], **kwargs):
        super().__init__(model, config, **kwargs)
        
        # LSTM-specific optimizations
        if isinstance(self.config, TextTrainingConfig):
            # Higher learning rate for LSTM
            self.config.learning_rate = max(self.config.learning_rate, 1e-3)
            # Different scheduler
            self.config.scheduler = "cosine"
