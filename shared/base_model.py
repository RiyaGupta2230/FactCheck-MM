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
from Config.base_config import BaseConfig


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
        """Save model checkpoint with comprehensive metadata."""
        
        checkpoint = {
            # Model state
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "task_name": self.task_name,
            "num_classes": self.num_classes,
            "supported_modalities": self.supported_modalities,
            
            # Training state
            "epoch": epoch,
            "training_step_count": self.training_step_count,
            
            # Config (CRITICAL for reconstruction)
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config.__dict__,
            
            # Optimizer/Scheduler (for training resumption)
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            
            # Model architecture info
            "model_size": self.get_model_size(),
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
            
            # Reproducibility
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "random_seed": getattr(self.config, 'seed', None),
            
            # Custom metadata
            "metadata": metadata or {}
        }
        
        # Validate checkpoint before saving
        self._validate_checkpoint(checkpoint)
        
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

        # ==================== INFERENCE API METHODS ====================
    
    @classmethod
    def _build_from_config(cls, config):
        """
        Subclasses must implement this method to construct the model
        using their own __init__ signature.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _build_from_config(config)"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        strict: bool = True,
        config_override: Optional[Dict] = None
    ) -> 'BaseMultimodalModel':
        """
        Load model from checkpoint for INFERENCE ONLY.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on (None = auto-detect)
            strict: Strict state_dict loading
            config_override: Override saved config values
        
        Returns:
            Model instance in eval mode
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or config_class missing
            KeyError: If required checkpoint fields missing
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger = get_logger(f"{cls.__name__}.from_checkpoint")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate required keys
        required_keys = ['model_state_dict', 'config']
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Reconstruct config
        config_dict = checkpoint['config']
        if config_override:
            config_dict.update(config_override)
        
        # CRITICAL: Subclass must declare _config_class
        if not hasattr(cls, '_config_class'):
            raise ValueError(
                f"{cls.__name__} must declare _config_class attribute. "
                f"Example: _config_class = {cls.__name__}Config"
            )
        
        config_class = cls._config_class
        
        # Reconstruct config object
        if hasattr(config_class, 'from_dict'):
            config = config_class.from_dict(config_dict)
        elif hasattr(config_class, '__call__'):
            config = config_class(**config_dict)
        else:
            raise ValueError(f"Config class {config_class} must support from_dict() or **kwargs")
        
        # Call subclass-specific builder
        logger.info(f"Building {cls.__name__} from config")
        model = cls._build_from_config(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Move to device and set eval mode
        model = model.to(device)
        model.eval()
        
        # Log success with guarded access
        epoch = checkpoint.get('epoch', 'unknown')
        total_params = checkpoint.get('model_size', {}).get('total_parameters', 'unknown')
        logger.info(f"✅ Loaded {cls.__name__} (epoch={epoch}, params={total_params}, device={device})")
        
        return model

    @staticmethod
    def _validate_checkpoint_structure(checkpoint: Dict) -> None:
        """Validate checkpoint has required fields."""
        required_fields = [
            'model_state_dict', 'model_name', 'task_name', 
            'config', 'num_classes', 'supported_modalities'
        ]
        missing = [f for f in required_fields if f not in checkpoint]
        if missing:
            raise ValueError(f"Invalid checkpoint: missing fields {missing}")

    
    def predict_text(
        self,
        text_input: Union[str, List[str]],
        return_probabilities: bool = True,
        return_raw_outputs: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on text input(s).
        
        Args:
            text_input: Single text string or list of text strings
            return_probabilities: Whether to return class probabilities
            return_raw_outputs: Whether to return raw model outputs (logits, hidden states)
            
        Returns:
            Dictionary containing:
                - predictions: Predicted class indices or generated text
                - probabilities: Class probabilities (if return_probabilities=True)
                - labels: Human-readable label names (if available)
                - raw_outputs: Raw model outputs (if return_raw_outputs=True)
                
        Example:
            >>> model = MultimodalSarcasmModel.from_checkpoint('model.pt')
            >>> result = model.predict_text("This is totally not sarcastic at all!")
            >>> print(result['predictions'])  # 1 (sarcastic)
            >>> print(result['probabilities'])  # [0.15, 0.85]
        """
        self.eval()
        
        # Convert single string to list
        is_single = isinstance(text_input, str)
        if is_single:
            text_input = [text_input]
        
        # Check if model has a text processor/tokenizer
        if hasattr(self, 'text_processor'):
            processor = self.text_processor
        elif hasattr(self, 'tokenizer'):
            processor = self.tokenizer
        else:
            raise AttributeError(
                "Model must have 'text_processor' or 'tokenizer' attribute for text inference. "
                "Please set this attribute during model initialization."
            )
        
        # Tokenize inputs
        try:
            if hasattr(processor, 'process_text'):
                # Custom TextProcessor
                encoded = processor.process_text(
                    text_input,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                # HuggingFace tokenizer
                encoded = processor(
                    text_input,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize input: {e}")
        
        # Move to model device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self(text_inputs=encoded)
        
        logits = outputs.get('logits')
        if logits is None:
            raise ValueError("Model forward() must return 'logits' for prediction")
        
        # Process outputs based on task type
        results = {}
        
        if self.task_name == 'paraphrasing':
            # For generation tasks
            results['predictions'] = logits  # Raw token predictions
            if return_probabilities:
                results['probabilities'] = F.softmax(logits, dim=-1)
        else:
            # For classification tasks
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            results['predictions'] = predictions.cpu().tolist()
            
            if return_probabilities:
                results['probabilities'] = probabilities.cpu().tolist()
            
            # Add human-readable labels if available
            if hasattr(self, 'id2label'):
                results['labels'] = [self.id2label[pred] for pred in results['predictions']]
        
        if return_raw_outputs:
            results['raw_outputs'] = {
                'logits': logits.cpu(),
                'hidden_states': outputs.get('hidden_states'),
                'attention_weights': outputs.get('attention_weights')
            }
        
        # If single input, return single output (not list)
        if is_single:
            for key in ['predictions', 'probabilities', 'labels']:
                if key in results and isinstance(results[key], list):
                    results[key] = results[key][0]
        
        return results
    
    def predict_batch(
        self,
        batch_input: Dict[str, Any],
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch_input: Dictionary containing batched inputs (text, audio, image, video)
            batch_size: Batch size for processing (if re-batching is needed)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries, one per sample
            
        Example:
            >>> batch = {
            ...     'text_inputs': {'input_ids': tensor(...), 'attention_mask': tensor(...)},
            ...     'image_inputs': tensor(...)
            ... }
            >>> results = model.predict_batch(batch)
            >>> for result in results:
            ...     print(result['predictions'])
        """
        self.eval()
        
        # Move inputs to device
        device = next(self.parameters()).device
        batch_input = self._move_batch_to_device(batch_input, device)
        
        # Run inference
        with torch.no_grad():
            outputs = self(**batch_input)
        
        logits = outputs.get('logits')
        if logits is None:
            raise ValueError("Model forward() must return 'logits' for prediction")
        
        # Process outputs
        results = []
        batch_len = logits.shape[0]
        
        if self.task_name == 'paraphrasing':
            # For generation tasks
            for i in range(batch_len):
                result = {'predictions': logits[i]}
                if return_probabilities:
                    result['probabilities'] = F.softmax(logits[i], dim=-1)
                results.append(result)
        else:
            # For classification tasks
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(batch_len):
                result = {
                    'predictions': predictions[i].item()
                }
                
                if return_probabilities:
                    result['probabilities'] = probabilities[i].cpu().tolist()
                
                if hasattr(self, 'id2label'):
                    result['label'] = self.id2label[result['predictions']]
                
                results.append(result)
        
        return results
    
    def predict_file(
        self,
        file_path: Union[str, Path],
        input_format: str = 'auto',
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on inputs from a file.
        
        Args:
            file_path: Path to input file (txt, csv, json, jsonl)
            input_format: Format of input file ('auto', 'txt', 'csv', 'json', 'jsonl')
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries
            
        Example:
            >>> results = model.predict_file('test_data.txt')
            >>> model.save_predictions(results, 'predictions.json', format='json')
        """
        import json
        import csv
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Auto-detect format
        if input_format == 'auto':
            input_format = file_path.suffix.lstrip('.')
        
        # Read inputs from file
        inputs = []
        
        if input_format == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                inputs = [line.strip() for line in f if line.strip()]
        
        elif input_format == 'csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Assume first column or 'text' column contains input
                    if 'text' in row:
                        inputs.append(row['text'])
                    else:
                        inputs.append(list(row.values())[0])
        
        elif input_format in ['json', 'jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                if input_format == 'jsonl':
                    for line in f:
                        data = json.loads(line)
                        inputs.append(data.get('text', str(data)))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        inputs = [item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in data]
                    else:
                        inputs = [data.get('text', str(data))]
        
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        if not inputs:
            raise ValueError(f"No valid inputs found in {file_path}")
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(inputs), batch_size):
            batch_texts = inputs[i:i + batch_size]
            
            # Use predict_text for each batch
            for text in batch_texts:
                result = self.predict_text(text, return_probabilities=return_probabilities)
                result['input_text'] = text  # Include original input
                all_results.append(result)
        
        self.logger.info(f"Processed {len(all_results)} samples from {file_path}")
        
        return all_results
    
    def save_predictions(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Save predictions to file.
        
        Args:
            results: Prediction results (single dict or list of dicts)
            output_path: Path to output file
            format: Output format ('json', 'csv', 'txt')
            
        Example:
            >>> results = model.predict_file('input.txt')
            >>> model.save_predictions(results, 'output.json', format='json')
        """
        import json
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure results is a list
        if isinstance(results, dict):
            results = [results]
        
        if format == 'json':
            # Convert tensors to lists for JSON serialization
            serializable_results = []
            for result in results:
                serializable = {}
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        serializable[key] = value.tolist()
                    else:
                        serializable[key] = value
                serializable_results.append(serializable)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            # Flatten nested structures for CSV
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                if not results:
                    return
                
                # Determine fieldnames
                fieldnames = set()
                for result in results:
                    for key, value in result.items():
                        if not isinstance(value, (dict, list, torch.Tensor)):
                            fieldnames.add(key)
                        elif key == 'probabilities' and isinstance(value, list):
                            fieldnames.add('max_probability')
                
                fieldnames = sorted(fieldnames)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {}
                    for key in fieldnames:
                        if key == 'max_probability' and 'probabilities' in result:
                            probs = result['probabilities']
                            if isinstance(probs, list):
                                row[key] = max(probs)
                        elif key in result:
                            value = result[key]
                            if isinstance(value, torch.Tensor):
                                row[key] = value.item()
                            else:
                                row[key] = value
                    writer.writerow(row)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    f.write(f"Sample {i + 1}:\n")
                    
                    if 'input_text' in result:
                        f.write(f"  Input: {result['input_text']}\n")
                    
                    if 'predictions' in result:
                        f.write(f"  Prediction: {result['predictions']}\n")
                    
                    if 'label' in result:
                        f.write(f"  Label: {result['label']}\n")
                    
                    if 'probabilities' in result:
                        probs = result['probabilities']
                        if isinstance(probs, list):
                            f.write(f"  Probabilities: {probs}\n")
                    
                    f.write("\n")
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        self.logger.info(f"Saved {len(results)} predictions to {output_path}")
    
    def _move_batch_to_device(
        self,
        batch: Dict[str, Any],
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Move batch tensors to specified device.
        
        Args:
            batch: Batch dictionary
            device: Target device
            
        Returns:
            Batch with tensors moved to device
        """
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            elif isinstance(value, dict):
                moved_batch[key] = self._move_batch_to_device(value, device)
            else:
                moved_batch[key] = value
        return moved_batch
