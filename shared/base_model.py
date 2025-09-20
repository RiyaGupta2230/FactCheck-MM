"""
Abstract Base Model for FactCheck-MM
Defines the common interface for all multimodal models in the pipeline.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from .utils import get_logger, CheckpointManager, MetricsComputer
from config import BaseConfig


class BaseMultimodalModel(nn.Module, ABC):
    """
    Abstract base class for all multimodal models in FactCheck-MM.
    
    Provides common functionality for:
    - Forward pass interface
    - Loss computation
    - Training step execution
    - Evaluation hooks
    - Checkpoint management
    - Logging integration
    """
    
    def __init__(
        self,
        config: BaseConfig,
        model_name: str,
        task_name: str,
        num_classes: int,
        supported_modalities: List[str]
    ):
        """
        Initialize base multimodal model.
        
        Args:
            config: Base configuration object
            model_name: Name of the model (e.g., "sarcasm_detector")
            task_name: Task name (e.g., "sarcasm_detection")
            num_classes: Number of output classes
            supported_modalities: List of supported modalities ['text', 'audio', 'image', 'video']
        """
        super().__init__()
        
        self.config = config
        self.model_name = model_name
        self.task_name = task_name
        self.num_classes = num_classes
        self.supported_modalities = supported_modalities
        
        # Setup logging
        self.logger = get_logger(f"{model_name}")
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.checkpoint_dir / model_name,
            max_checkpoints=5
        )
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(task_name=task_name)
        
        # Training state
        self.training_step_count = 0
        self.eval_step_count = 0
        self.current_epoch = 0
        
        # Loss tracking
        self.train_losses = []
        self.eval_losses = []
        
        # Model compilation for PyTorch 2.0
        self._compiled = False
        
        self.logger.info(f"Initialized {model_name} for {task_name}")
        self.logger.info(f"Supported modalities: {supported_modalities}")
    
    @abstractmethod
    def forward(
        self,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        video_inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            text_inputs: Tokenized text inputs {input_ids, attention_mask, ...}
            audio_inputs: Audio feature tensor [batch_size, seq_len, features]
            image_inputs: Image tensor [batch_size, channels, height, width]
            video_inputs: Video tensor [batch_size, frames, channels, height, width]
            labels: Ground truth labels [batch_size] or [batch_size, seq_len]
            **kwargs: Additional task-specific inputs
            
        Returns:
            Dictionary containing:
                - logits: Model predictions
                - loss: Computed loss (if labels provided)
                - hidden_states: Intermediate representations (optional)
                - attention_weights: Attention matrices (optional)
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            **kwargs: Additional loss computation parameters
            
        Returns:
            Computed loss tensor
        """
        pass
    
    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute single training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Batch index
            
        Returns:
            Dictionary containing loss and metrics
        """
        self.train()
        
        # Forward pass
        outputs = self(**batch)
        
        # Extract loss
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model forward() must return 'loss' when labels are provided")
        
        # Compute metrics
        predictions = outputs.get("logits")
        labels = batch.get("labels")
        
        metrics = {}
        if predictions is not None and labels is not None:
            metrics = self.metrics_computer.compute_metrics(
                predictions=predictions,
                labels=labels,
                mode="train"
            )
        
        # Update training state
        self.training_step_count += 1
        self.train_losses.append(loss.item())
        
        # Log metrics
        if self.training_step_count % 100 == 0:
            self.logger.info(
                f"Step {self.training_step_count}: loss={loss.item():.4f}, "
                f"metrics={metrics}"
            )
        
        return {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            **metrics
        }
    
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute single validation step.
        
        Args:
            batch: Batch of validation data  
            batch_idx: Batch index
            
        Returns:
            Dictionary containing loss and metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self(**batch)
            
            # Extract loss and predictions
            loss = outputs.get("loss", torch.tensor(0.0))
            predictions = outputs.get("logits")
            labels = batch.get("labels")
            
            # Compute metrics
            metrics = {}
            if predictions is not None and labels is not None:
                metrics = self.metrics_computer.compute_metrics(
                    predictions=predictions,
                    labels=labels,
                    mode="eval"
                )
            
            # Update eval state
            self.eval_step_count += 1
            self.eval_losses.append(loss.item())
            
            return {
                "loss": loss,
                "predictions": predictions,
                "labels": labels,
                **metrics
            }
    
    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute prediction step (inference only).
        
        Args:
            batch: Batch of input data
            batch_idx: Batch index
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass without labels
            batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
            outputs = self(**batch_no_labels)
            
            logits = outputs.get("logits")
            
            if logits is None:
                raise ValueError("Model must return 'logits' for prediction")
            
            # Compute probabilities and predictions
            if self.task_name == "paraphrasing":
                # For generation tasks, return raw logits
                predictions = logits
                probabilities = None
            else:
                # For classification tasks
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "logits": logits
            }
    
    def configure_optimizers(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw"
    ) -> Dict[str, Any]:
        """
        Configure optimizers and schedulers.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            optimizer_name: Optimizer type
            
        Returns:
            Dictionary with optimizer and scheduler
        """
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return {"optimizer": optimizer}
    
    def compile_model(self) -> None:
        """Compile model for PyTorch 2.0 optimization."""
        if self.config.compile_model and not self._compiled:
            try:
                self.logger.info("Compiling model with PyTorch 2.0...")
                # Note: Actual compilation would be done at the trainer level
                # This is a placeholder for model preparation
                self._compiled = True
                self.logger.info("Model compilation prepared")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
    
    def freeze_modality_encoder(self, modality: str) -> None:
        """
        Freeze parameters of specific modality encoder.
        
        Args:
            modality: Modality to freeze ('text', 'audio', 'image', 'video')
        """
        if modality not in self.supported_modalities:
            raise ValueError(f"Modality {modality} not supported by this model")
        
        # This should be implemented by subclasses
        encoder_attr = f"{modality}_encoder"
        if hasattr(self, encoder_attr):
            encoder = getattr(self, encoder_attr)
            for param in encoder.parameters():
                param.requires_grad = False
            self.logger.info(f"Frozen {modality} encoder parameters")
        else:
            self.logger.warning(f"No {modality} encoder found to freeze")
    
    def unfreeze_modality_encoder(self, modality: str) -> None:
        """
        Unfreeze parameters of specific modality encoder.
        
        Args:
            modality: Modality to unfreeze ('text', 'audio', 'image', 'video')
        """
        if modality not in self.supported_modalities:
            raise ValueError(f"Modality {modality} not supported by this model")
        
        encoder_attr = f"{modality}_encoder"
        if hasattr(self, encoder_attr):
            encoder = getattr(self, encoder_attr)
            for param in encoder.parameters():
                param.requires_grad = True
            self.logger.info(f"Unfrozen {modality} encoder parameters")
        else:
            self.logger.warning(f"No {modality} encoder found to unfreeze")
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size information.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params
        }
    
    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metadata: Additional metadata
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "config": self.config.to_dict(),
            "model_size": self.get_model_size(),
            "training_step_count": self.training_step_count
        }
        
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scheduler_state:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if metadata:
            checkpoint["metadata"] = metadata
        
        self.checkpoint_manager.save_checkpoint(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Loaded checkpoint data
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(filepath)
        
        # Load model state
        self.load_state_dict(checkpoint["model_state_dict"])
        
        # Update internal state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.training_step_count = checkpoint.get("training_step_count", 0)
        
        self.logger.info(
            f"Loaded checkpoint from {filepath} "
            f"(epoch {self.current_epoch}, step {self.training_step_count})"
        )
        
        return checkpoint
    
    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get attention weights from the last forward pass.
        
        Returns:
            Dictionary of attention weights by layer/head
        """
        # This should be implemented by subclasses that support attention visualization
        return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        size_info = self.get_model_size()
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_name='{self.model_name}',\n"
            f"  task_name='{self.task_name}',\n"
            f"  num_classes={self.num_classes},\n"
            f"  modalities={self.supported_modalities},\n"
            f"  parameters={size_info['total_parameters']:,},\n"
            f"  trainable={size_info['trainable_parameters']:,}\n"
            f")"
        )
