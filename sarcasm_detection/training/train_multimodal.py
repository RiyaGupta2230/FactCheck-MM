# sarcasm_detection/training/train_multimodal.py
"""
Multimodal Sarcasm Detection Training
Trainer for full multimodal models with text, audio, image, and video.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import gc

from shared.utils import (
    get_logger, CheckpointManager, MetricsComputer,
    ExperimentLogger
)
from shared.datasets import create_hardware_aware_dataloader, MultimodalCollator
from ..models import MultimodalSarcasmModel
from ..utils import SarcasmMetrics
from .train_text import TextSarcasmTrainer


@dataclass 
class MultimodalTrainingConfig:
    """Configuration for multimodal training."""
    
    # Model settings
    fusion_strategy: str = "cross_modal_attention"
    modalities: List[str] = field(default_factory=lambda: ['text', 'audio', 'image', 'video'])
    feature_dims: Dict[str, int] = field(default_factory=lambda: {
        'text': 1024, 'audio': 768, 'image': 768, 'video': 768
    })
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 8  # Smaller due to multimodal complexity
    num_epochs: int = 15
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    use_mixed_precision: bool = True
    accumulation_steps: int = 2  # Higher accumulation for stability
    
    # Modality-specific settings
    modality_dropout: float = 0.1  # Randomly drop modalities during training
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'text': 1.0, 'audio': 0.8, 'image': 0.6, 'video': 0.4
    })
    progressive_unfreezing: bool = True  # Progressively unfreeze modalities
    
    # Regularization
    dropout_rate: float = 0.2
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5
    
    # Logging and checkpointing
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    
    # Hardware
    device: str = "auto"
    max_memory_gb: float = 12.0
    use_chunked_loading: bool = True


class MultimodalSarcasmTrainer:
    """Comprehensive trainer for multimodal sarcasm detection."""
    
    def __init__(
        self,
        model: MultimodalSarcasmModel,
        config: Union[MultimodalTrainingConfig, Dict[str, Any]],
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        """
        Initialize multimodal trainer.
        
        Args:
            model: Multimodal sarcasm model
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if isinstance(config, dict):
            config = MultimodalTrainingConfig(**config)
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger = get_logger("MultimodalSarcasmTrainer")
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Initialize training components
        self._setup_modality_management()
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
            'val_f1': [],
            'modality_usage': []
        }
        
        self.logger.info(
            f"Initialized multimodal trainer on {self.device} "
            f"with modalities: {self.config.modalities}"
        )
    
    def _setup_modality_management(self):
        """Setup modality management for progressive training."""
        self.available_modalities = set(self.config.modalities)
        self.current_modalities = set(['text'])  # Start with text only
        
        if self.config.progressive_unfreezing:
            # Schedule for progressively adding modalities
            self.modality_schedule = {
                0: ['text'],
                2: ['text', 'image'],
                5: ['text', 'image', 'audio'],
                8: ['text', 'image', 'audio', 'video']
            }
        else:
            self.modality_schedule = {0: self.config.modalities}
        
        self.logger.info(f"Modality schedule: {self.modality_schedule}")
    
    def _setup_optimization(self):
        """Setup multimodal-aware optimization."""
        
        # Create parameter groups for different modalities
        param_groups = []
        
        # Text encoder parameters
        text_params = []
        audio_params = []
        image_params = []
        video_params = []
        fusion_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'text' in name.lower() or 'roberta' in name.lower():
                text_params.append(param)
            elif 'audio' in name.lower() or 'wav2vec' in name.lower():
                audio_params.append(param)
            elif 'image' in name.lower() or 'vit' in name.lower() or 'vision' in name.lower():
                image_params.append(param)
            elif 'video' in name.lower():
                video_params.append(param)
            elif 'fusion' in name.lower() or 'attention' in name.lower():
                fusion_params.append(param)
            else:
                classifier_params.append(param)
        
        # Different learning rates for different components
        base_lr = self.config.learning_rate
        
        if text_params:
            param_groups.append({'params': text_params, 'lr': base_lr * 0.1, 'name': 'text'})
        if audio_params:
            param_groups.append({'params': audio_params, 'lr': base_lr * 0.5, 'name': 'audio'})
        if image_params:
            param_groups.append({'params': image_params, 'lr': base_lr * 0.5, 'name': 'image'})
        if video_params:
            param_groups.append({'params': video_params, 'lr': base_lr * 0.5, 'name': 'video'})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': base_lr, 'name': 'fusion'})
        if classifier_params:
            param_groups.append({'params': classifier_params, 'lr': base_lr, 'name': 'classifier'})
        
        # Optimizer
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(param_groups, weight_decay=self.config.weight_decay)
        
        # Scheduler
        if self.train_dataset:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            
            if self.config.scheduler.lower() == "linear":
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=self.config.warmup_steps
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps
                )
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_loss_function(self):
        """Setup loss function with modality weighting."""
        if self.config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def _setup_metrics(self):
        """Setup metrics computation."""
        self.metrics_computer = SarcasmMetrics(
            task_name="multimodal_sarcasm_detection",
            num_classes=2
        )
    
    def _setup_data_loaders(self):
        """Setup multimodal data loaders."""
        from shared.preprocessing import TextProcessor, AudioProcessor, ImageProcessor, VideoProcessor
        
        # Initialize processors
        processors = {
            'text': TextProcessor(max_length=512),
            'audio': AudioProcessor(sample_rate=16000),
            'image': ImageProcessor(image_size=224),
            'video': VideoProcessor(max_frames=16, frame_size=(224, 224))
        }
        
        # Multimodal collator
        collate_fn = MultimodalCollator(
            text_processor=processors['text'].tokenizer,
            max_length=512
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
        
        # Validation and test data loaders
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        if self.test_dataset:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
    
    def _update_active_modalities(self, epoch: int):
        """Update which modalities are active based on schedule."""
        if not self.config.progressive_unfreezing:
            return
        
        # Find current modalities based on epoch
        active_modalities = ['text']  # Always include text
        for schedule_epoch, modalities in sorted(self.modality_schedule.items()):
            if epoch >= schedule_epoch:
                active_modalities = modalities
        
        # Update current modalities
        old_modalities = self.current_modalities.copy()
        self.current_modalities = set(active_modalities)
        
        if old_modalities != self.current_modalities:
            self.logger.info(f"Updated active modalities: {self.current_modalities}")
            
            # Freeze/unfreeze parameters
            for name, param in self.model.named_parameters():
                should_train = False
                
                for modality in self.current_modalities:
                    if modality.lower() in name.lower():
                        should_train = True
                        break
                
                # Always train fusion and classifier layers
                if 'fusion' in name.lower() or 'classifier' in name.lower():
                    should_train = True
                
                param.requires_grad = should_train
    
    def _apply_modality_dropout(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply modality dropout during training."""
        if not self.training or self.config.modality_dropout <= 0:
            return batch, list(self.current_modalities)
        
        available_in_batch = []
        for modality in self.current_modalities:
            if modality in batch and batch[modality] is not None:
                available_in_batch.append(modality)
        
        # Always keep at least one modality (text preferred)
        if len(available_in_batch) <= 1:
            return batch, available_in_batch
        
        # Randomly drop modalities
        kept_modalities = []
        for modality in available_in_batch:
            if modality == 'text' or np.random.random() > self.config.modality_dropout:
                kept_modalities.append(modality)
        
        # Ensure at least text is kept
        if not kept_modalities:
            kept_modalities = ['text']
        
        # Zero out dropped modalities
        modified_batch = batch.copy()
        for modality in self.current_modalities:
            if modality not in kept_modalities:
                modified_batch[modality] = None
        
        return modified_batch, kept_modalities
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with multimodal handling."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        modality_usage_count = {mod: 0 for mod in self.config.modalities}
        
        # Update active modalities
        self._update_active_modalities(self.current_epoch)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Apply modality dropout
            batch, used_modalities = self._apply_modality_dropout(batch)
            
            # Count modality usage
            for modality in used_modalities:
                modality_usage_count[modality] += 1
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Extract multimodal inputs
                text_input = batch.get('text')
                audio_input = batch.get('audio_features')
                image_input = batch.get('image_features') 
                video_input = batch.get('video_features')
                
                # Model forward with available modalities
                logits = self.model(
                    text=text_input,
                    audio=audio_input,
                    image=image_input,
                    video=video_input,
                    available_modalities=used_modalities
                )
                
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
                    'lr': f'{current_lr:.2e}',
                    'mods': f'{len(used_modalities)}'
                })
            
            # Memory cleanup
            if batch_idx % 50 == 0:  # More frequent cleanup for multimodal
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate modality usage statistics
        total_batches = num_batches
        modality_usage_stats = {
            f"usage_{mod}": count / total_batches 
            for mod, count in modality_usage_count.items()
        }
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            **modality_usage_stats
        }
    
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate multimodal model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        modality_availability = {mod: 0 for mod in self.config.modalities}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                batch = self._move_batch_to_device(batch)
                
                # Track modality availability
                available_modalities = []
                for modality in self.config.modalities:
                    if modality in batch and batch[modality] is not None:
                        available_modalities.append(modality)
                        modality_availability[modality] += 1
                
                # Extract multimodal inputs
                text_input = batch.get('text')
                audio_input = batch.get('audio_features')
                image_input = batch.get('image_features')
                video_input = batch.get('video_features')
                
                # Forward pass
                logits = self.model(
                    text=text_input,
                    audio=audio_input,
                    image=image_input,
                    video=video_input,
                    available_modalities=available_modalities
                )
                
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
        
        # Add modality availability statistics
        for modality, count in modality_availability.items():
            metrics[f'{split_name}_availability_{modality}'] = count / num_batches
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device, handling multimodal inputs."""
        device_batch = {}
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dictionaries (e.g., text tokens)
                device_batch[key] = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch
    
    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        resume_from_checkpoint: Optional[str] = None,
        experiment_name: str = "multimodal_sarcasm_training"
    ) -> Dict[str, Any]:
        """
        Main multimodal training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            experiment_name: Name for experiment logging
            
        Returns:
            Training results
        """
        if not self.train_dataset:
            raise ValueError("Training dataset not provided")
        
        # Setup checkpointing and logging (similar to TextSarcasmTrainer)
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
            project_name="multimodal_sarcasm_detection",
            experiment_name=experiment_name,
            config=self.config.__dict__
        )
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        # Log model info
        self.experiment_logger.log_model_info(self.model)
        
        self.logger.info(f"Starting multimodal training for {self.config.num_epochs} epochs")
        
        # Training loop (similar structure to text trainer)
        best_val_score = self.best_val_score
        patience_counter = 0
        
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
            
            # Combine and log metrics
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
                f"Val F1: {val_metrics.get('val_f1', 0):.4f} - "
                f"Active Modalities: {self.current_modalities}"
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
        self.logger.info("Multimodal training completed")
        
        return final_results
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save multimodal model checkpoint."""
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
            metadata={
                'model_type': 'multimodal_sarcasm',
                'active_modalities': list(self.current_modalities),
                'fusion_strategy': self.config.fusion_strategy
            }
        )
        
        self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics=metrics,
            is_best=is_best
        )
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume multimodal training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('step', 0)
        self.best_val_score = checkpoint.get('best_metric', 0.0)
        
        # Restore active modalities if available
        if 'metadata' in checkpoint and 'active_modalities' in checkpoint['metadata']:
            self.current_modalities = set(checkpoint['metadata']['active_modalities'])
        
        self.logger.info(f"Resumed multimodal training from epoch {self.current_epoch}")
