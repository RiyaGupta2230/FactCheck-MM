# sarcasm_detection/training/train_ensemble.py
"""
Ensemble Training for Sarcasm Detection
Trainer for ensemble models including voting, weighted, and stacking ensembles.
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
import pickle
import joblib

from shared.utils import (
    get_logger, CheckpointManager, MetricsComputer,
    ExperimentLogger
)
from shared.datasets import create_hardware_aware_dataloader, MultimodalCollator
from ..models import (
    EnsembleSarcasmModel, VotingEnsemble, WeightedEnsemble, StackingEnsemble,
    RobertaSarcasmModel, LSTMSarcasmModel, MultimodalSarcasmModel
)
from ..utils import SarcasmMetrics
from .train_text import TextSarcasmTrainer
from .train_multimodal import MultimodalSarcasmTrainer


@dataclass
class EnsembleTrainingConfig:
    """Configuration for ensemble training."""
    
    # Model configurations for individual models
    base_model_configs: List[Dict[str, Any]] = field(default_factory=list)
    base_model_checkpoints: List[str] = field(default_factory=list)
    
    # Ensemble settings
    ensemble_method: str = "weighted"  # voting, weighted, stacking
    voting_strategy: str = "soft"  # hard, soft (for voting ensemble)
    
    # Meta-learner training (for stacking)
    meta_learner_epochs: int = 20
    meta_learner_lr: float = 0.001
    cross_validation_folds: int = 5
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 10  # Shorter since base models are pre-trained
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    
    # Logging and checkpointing
    save_every: int = 1
    eval_every: int = 1
    log_every: int = 100
    
    # Hardware
    device: str = "auto"
    use_mixed_precision: bool = False  # Disabled for ensemble stability


class EnsembleTrainer:
    """Comprehensive trainer for ensemble sarcasm detection models."""
    
    def __init__(
        self,
        ensemble_model: Optional[EnsembleSarcasmModel] = None,
        config: Union[EnsembleTrainingConfig, Dict[str, Any]] = None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble_model: Pre-built ensemble model (optional)
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if isinstance(config, dict):
            config = EnsembleTrainingConfig(**config)
        
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger = get_logger("EnsembleTrainer")
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize or build ensemble model
        if ensemble_model is not None:
            self.model = ensemble_model
        else:
            self.model = self._build_ensemble_model()
        
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
            'val_f1': [],
            'individual_model_scores': []
        }
        
        self.logger.info(
            f"Initialized ensemble trainer with {self.model.num_models} models "
            f"using {self.config.ensemble_method} method on {self.device}"
        )
    
    def _build_ensemble_model(self) -> EnsembleSarcasmModel:
        """Build ensemble model from configuration."""
        
        if not self.config.base_model_configs and not self.config.base_model_checkpoints:
            raise ValueError("Must provide either base_model_configs or base_model_checkpoints")
        
        # Load base models
        base_models = []
        
        # From checkpoints
        if self.config.base_model_checkpoints:
            for checkpoint_path in self.config.base_model_checkpoints:
                model = self._load_model_from_checkpoint(checkpoint_path)
                if model is not None:
                    base_models.append(model)
        
        # From configurations
        if self.config.base_model_configs:
            for model_config in self.config.base_model_configs:
                model = self._create_model_from_config(model_config)
                if model is not None:
                    base_models.append(model)
        
        if not base_models:
            raise ValueError("No base models could be loaded")
        
        # Create ensemble
        ensemble_config = {
            'ensemble_method': self.config.ensemble_method,
            'voting_strategy': self.config.voting_strategy
        }
        
        if self.config.ensemble_method == "voting":
            return VotingEnsemble(
                config=ensemble_config,
                models=base_models,
                voting_strategy=self.config.voting_strategy
            )
        elif self.config.ensemble_method == "weighted":
            return WeightedEnsemble(
                config=ensemble_config,
                models=base_models,
                learn_weights=True
            )
        elif self.config.ensemble_method == "stacking":
            return StackingEnsemble(
                config=ensemble_config,
                models=base_models,
                meta_learner_type="mlp"
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
    
    def _load_model_from_checkpoint(self, checkpoint_path: str) -> Optional[nn.Module]:
        """Load model from checkpoint."""
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Determine model type from metadata
            metadata = checkpoint.get('metadata', {})
            model_type = metadata.get('model_type', 'roberta_sarcasm')
            
            # Create model based on type
            if model_type == 'roberta_sarcasm':
                model = RobertaSarcasmModel(
                    config=checkpoint.get('config', {}),
                    num_classes=2
                )
            elif model_type == 'lstm_sarcasm':
                model = LSTMSarcasmModel(
                    config=checkpoint.get('config', {}),
                    num_classes=2
                )
            elif model_type == 'multimodal_sarcasm':
                model = MultimodalSarcasmModel(
                    config=checkpoint.get('config', {}),
                    num_classes=2
                )
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to eval mode for ensemble
            
            self.logger.info(f"Loaded model from {checkpoint_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            return None
    
    def _create_model_from_config(self, model_config: Dict[str, Any]) -> Optional[nn.Module]:
        """Create and optionally train model from configuration."""
        try:
            model_type = model_config.get('model_type', 'roberta')
            
            if model_type == 'roberta':
                model = RobertaSarcasmModel(model_config, num_classes=2)
            elif model_type == 'lstm':
                model = LSTMSarcasmModel(model_config, num_classes=2)
            elif model_type == 'multimodal':
                model = MultimodalSarcasmModel(model_config, num_classes=2)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
            
            # If pre-training is requested
            if model_config.get('pretrain', False) and self.train_dataset:
                self.logger.info(f"Pre-training {model_type} model for ensemble")
                self._pretrain_base_model(model, model_config)
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create model from config: {e}")
            return None
    
    def _pretrain_base_model(self, model: nn.Module, model_config: Dict[str, Any]):
        """Pre-train a base model for the ensemble."""
        # Create appropriate trainer
        if isinstance(model, (RobertaSarcasmModel, LSTMSarcasmModel)):
            trainer = TextSarcasmTrainer(
                model=model,
                config=model_config,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset
            )
        elif isinstance(model, MultimodalSarcasmModel):
            trainer = MultimodalSarcasmTrainer(
                model=model,
                config=model_config,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset
            )
        else:
            self.logger.warning(f"Cannot pre-train unknown model type: {type(model)}")
            return
        
        # Train the model
        trainer.train(
            checkpoint_dir=Path("temp_pretrain_checkpoints"),
            experiment_name=f"pretrain_{model_config.get('model_type', 'unknown')}"
        )
    
    def _setup_optimization(self):
        """Setup optimization for ensemble training."""
        # Only optimize trainable parameters (for weighted/stacking ensembles)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            self.optimizer = None
            self.scheduler = None
            self.logger.info("No trainable parameters found - using pre-trained ensemble")
            return
        
        # Optimizer
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        # Scheduler
        if self.train_dataset and self.optimizer:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            
            if self.config.scheduler.lower() == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps
                )
            else:
                self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)
    
    def _setup_loss_function(self):
        """Setup loss function for ensemble training."""
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_metrics(self):
        """Setup metrics computation."""
        self.metrics_computer = SarcasmMetrics(
            task_name="ensemble_sarcasm_detection",
            num_classes=2
        )
    
    def _setup_data_loaders(self):
        """Setup data loaders for ensemble training."""
        from shared.preprocessing import TextProcessor
        text_processor = TextProcessor(max_length=512)
        
        collate_fn = MultimodalCollator(
            text_processor=text_processor.tokenizer,
            max_length=512
        )
        
        # Training data loader
        if self.train_dataset:
            self.train_dataloader = create_hardware_aware_dataloader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_fn,
                device_type="auto"
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train ensemble for one epoch."""
        if self.optimizer is None:
            self.logger.info("No trainable parameters - skipping training")
            return {'train_loss': 0.0}
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Ensemble Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if isinstance(self.model, StackingEnsemble):
                # For stacking ensemble, we need to pass the full batch
                logits = self.model(**batch)
            else:
                # For voting/weighted ensembles, extract inputs appropriately
                # This depends on the base models in the ensemble
                logits = self._ensemble_forward(batch)
            
            # Compute loss
            loss = self.criterion(logits, batch['labels'])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def _ensemble_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through ensemble with appropriate input handling."""
        # Extract different types of inputs for different base models
        inputs_list = []
        
        # Text inputs (for text-based models)
        text_inputs = {}
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if key in batch:
                text_inputs[key] = batch[key]
        
        if text_inputs:
            inputs_list.append(text_inputs)
        
        # Multimodal inputs (for multimodal models)
        multimodal_inputs = {
            'text': text_inputs if text_inputs else None,
            'audio': batch.get('audio_features'),
            'image': batch.get('image_features'),
            'video': batch.get('video_features')
        }
        
        # The ensemble model should handle input routing to appropriate base models
        return self.model(**multimodal_inputs)
    
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate ensemble model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        individual_predictions = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass with individual predictions
                if hasattr(self.model, 'forward') and 'return_individual_predictions' in self.model.forward.__code__.co_varnames:
                    result = self.model(**batch, return_individual_predictions=True)
                    if isinstance(result, dict):
                        logits = result['ensemble_logits']
                        individual_preds = result.get('individual_predictions', [])
                        individual_predictions.extend(individual_preds)
                    else:
                        logits = result
                else:
                    logits = self._ensemble_forward(batch)
                
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
        
        # Add individual model performance if available
        if individual_predictions:
            metrics[f'{split_name}_num_base_models'] = len(individual_predictions[0]) if individual_predictions else 0
        
        return metrics
    
    def train_stacking_meta_learner(self):
        """Train meta-learner for stacking ensemble using cross-validation."""
        if not isinstance(self.model, StackingEnsemble):
            self.logger.info("Meta-learner training only applies to stacking ensembles")
            return
        
        if not self.train_dataset:
            self.logger.error("Training dataset required for meta-learner training")
            return
        
        self.logger.info("Training stacking meta-learner with cross-validation")
        
        # This would implement k-fold cross-validation to generate
        # out-of-fold predictions for meta-learner training
        # For brevity, we'll use the existing train/val split
        
        self.model.train_meta_learner(
            train_loader=self.train_dataloader,
            val_loader=self.val_dataloader,
            num_epochs=self.config.meta_learner_epochs,
            lr=self.config.meta_learner_lr
        )
        
        self.logger.info("Meta-learner training completed")
    
    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        resume_from_checkpoint: Optional[str] = None,
        experiment_name: str = "ensemble_sarcasm_training"
    ) -> Dict[str, Any]:
        """
        Main ensemble training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            experiment_name: Name for experiment logging
            
        Returns:
            Training results
        """
        # Setup checkpointing and logging
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint_manager = CheckpointManager(
                save_dir=checkpoint_dir,
                max_checkpoints=3,
                monitor_metric="val_f1",
                mode="max"
            )
        
        self.experiment_logger = ExperimentLogger(
            log_dir=checkpoint_dir or Path("logs"),
            project_name="ensemble_sarcasm_detection",
            experiment_name=experiment_name,
            config=self.config.__dict__
        )
        
        # Train stacking meta-learner if applicable
        if self.config.ensemble_method == "stacking":
            self.train_stacking_meta_learner()
        
        # Main training loop (if ensemble has trainable parameters)
        if self.optimizer is not None and self.train_dataset:
            self.logger.info(f"Starting ensemble training for {self.config.num_epochs} epochs")
            
            best_val_score = self.best_val_score
            patience_counter = 0
            
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = {}
                if self.val_dataset:
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
                    f"Val F1: {val_metrics.get('val_f1', 0):.4f}"
                )
                
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
        
        # Save ensemble model
        if checkpoint_dir:
            ensemble_path = checkpoint_dir / "final_ensemble_model.pkl"
            self._save_ensemble_model(ensemble_path)
        
        self.experiment_logger.close()
        self.logger.info("Ensemble training completed")
        
        return final_results
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save ensemble checkpoint."""
        if not hasattr(self, 'checkpoint_manager'):
            return
        
        from shared.utils import ModelState
        
        # Only save trainable parameters
        trainable_state_dict = {
            name: param for name, param in self.model.state_dict().items()
            if param.requires_grad
        }
        
        model_state = ModelState(
            model_state_dict=trainable_state_dict,
            optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            epoch=self.current_epoch,
            step=self.global_step,
            best_metric=self.best_val_score,
            config=self.config.__dict__,
            metadata={
                'model_type': 'ensemble_sarcasm',
                'ensemble_method': self.config.ensemble_method,
                'num_base_models': self.model.num_models
            }
        )
        
        self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics=metrics,
            is_best=is_best
        )
    
    def _save_ensemble_model(self, save_path: Path):
        """Save complete ensemble model."""
        try:
            # Save using pickle for complete model preservation
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Also save configuration
            config_path = save_path.with_suffix('.json')
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            self.logger.info(f"Saved ensemble model to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble model: {e}")
