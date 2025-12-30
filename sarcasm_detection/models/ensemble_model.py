# sarcasm_detection/models/ensemble_model.py
"""
Ensemble Models for Sarcasm Detection
Voting and weighted ensemble wrappers for combining trained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from shared.base_model import BaseMultimodalModel

from shared.utils import get_logger
from .text_sarcasm_model import RobertaSarcasmModel, LSTMSarcasmModel
from .multimodal_sarcasm import MultimodalSarcasmModel


class EnsembleSarcasmModel(BaseMultimodalModel):
    """Base class for ensemble sarcasm detection models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        models: Optional[List[nn.Module]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        ensemble_method: str = "voting"
    ):
        """
        Initialize ensemble model.
        
        Args:
            config: Model configuration
            models: List of pre-trained models
            model_configs: List of model configurations for loading
            ensemble_method: Ensemble method ('voting', 'weighted', 'stacking')
        """
        super().__init__(config, f"ensemble_sarcasm_{ensemble_method}")
        
        self.ensemble_method = ensemble_method
        self.models = models or []
        self.model_configs = model_configs or []
        
        self.logger = get_logger("EnsembleSarcasmModel")
        
        # Load models if configs provided
        if model_configs and not models:
            self._load_models_from_configs()
        
        if not self.models:
            raise ValueError("No models provided for ensemble")
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
        
        self.num_models = len(self.models)
        self.logger.info(f"Initialized ensemble with {self.num_models} models using {ensemble_method} method")
    
    def _load_models_from_configs(self):
        """Load models from configuration."""
        for i, model_config in enumerate(self.model_configs):
            try:
                model_type = model_config.get('model_type', 'roberta')
                checkpoint_path = model_config.get('checkpoint_path')
                
                # Create model based on type
                if model_type == 'roberta':
                    model = RobertaSarcasmModel(model_config)
                elif model_type == 'lstm':
                    model = LSTMSarcasmModel(model_config)
                elif model_type == 'multimodal':
                    model = MultimodalSarcasmModel(model_config)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Load checkpoint if provided
                if checkpoint_path and Path(checkpoint_path).exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    self.logger.info(f"Loaded model {i} from {checkpoint_path}")
                
                self.models.append(model)
                
            except Exception as e:
                self.logger.error(f"Failed to load model {i}: {e}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_model_predictions(
        self,
        *args,
        return_probabilities: bool = True,
        **kwargs
    ) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get predictions from all models in the ensemble.
        
        Args:
            return_probabilities: Whether to return probabilities instead of logits
            *args, **kwargs: Input arguments for models
            
        Returns:
            List of model predictions
        """
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                with torch.no_grad():
                    output = model(*args, **kwargs)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    if return_probabilities:
                        probs = F.softmax(logits, dim=-1)
                        predictions.append(probs)
                    else:
                        predictions.append(logits)
                        
            except Exception as e:
                self.logger.warning(f"Model {i} failed to predict: {e}")
                # Create dummy prediction
                batch_size = args[0].shape[0] if args else 1
                dummy = torch.ones(batch_size, 2) * 0.5  # Neutral prediction
                predictions.append(dummy)
        
        return predictions


class VotingEnsemble(EnsembleSarcasmModel):
    """Simple voting ensemble for sarcasm detection."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        models: Optional[List[nn.Module]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        voting_strategy: str = "soft"  # 'hard' or 'soft'
    ):
        """
        Initialize voting ensemble.
        
        Args:
            config: Model configuration
            models: List of pre-trained models
            model_configs: List of model configurations
            voting_strategy: Voting strategy ('hard' or 'soft')
        """
        super().__init__(config, models, model_configs, "voting")
        
        self.voting_strategy = voting_strategy
        self.logger.info(f"Initialized voting ensemble with {voting_strategy} voting")
    
    def forward(
        self,
        *args,
        return_individual_predictions: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through voting ensemble.
        
        Args:
            return_individual_predictions: Whether to return individual model predictions
            *args, **kwargs: Input arguments for models
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        model_predictions = self.get_model_predictions(*args, **kwargs)
        
        if self.voting_strategy == "soft":
            # Average probabilities
            ensemble_probs = torch.mean(torch.stack(model_predictions), dim=0)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)  # Convert back to logits
            
        else:  # hard voting
            # Get hard predictions and vote
            hard_predictions = []
            for probs in model_predictions:
                hard_pred = torch.argmax(probs, dim=-1)
                hard_predictions.append(hard_pred)
            
            # Stack and vote
            stacked_predictions = torch.stack(hard_predictions)  # [num_models, batch_size]
            
            # Count votes for each class
            batch_size = stacked_predictions.shape[1]
            ensemble_predictions = torch.zeros(batch_size, dtype=torch.long)
            
            for i in range(batch_size):
                votes = stacked_predictions[:, i]
                # Get most common vote
                unique_votes, counts = torch.unique(votes, return_counts=True)
                winner = unique_votes[torch.argmax(counts)]
                ensemble_predictions[i] = winner
            
            # Convert to logits (one-hot style)
            ensemble_logits = torch.zeros(batch_size, 2)
            ensemble_logits[range(batch_size), ensemble_predictions] = 1.0
            ensemble_logits = torch.log(ensemble_logits + 1e-8)
        
        if return_individual_predictions:
            return {
                'ensemble_logits': ensemble_logits,
                'individual_predictions': model_predictions,
                'voting_strategy': self.voting_strategy
            }
        else:
            return ensemble_logits


class WeightedEnsemble(EnsembleSarcasmModel):
    """Weighted ensemble that learns optimal model weights."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        models: Optional[List[nn.Module]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        model_weights: Optional[List[float]] = None,
        learn_weights: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize weighted ensemble.
        
        Args:
            config: Model configuration
            models: List of pre-trained models
            model_configs: List of model configurations
            model_weights: Initial model weights
            learn_weights: Whether to learn optimal weights
            temperature: Temperature for weight softmax
        """
        super().__init__(config, models, model_configs, "weighted")
        
        self.temperature = temperature
        self.learn_weights = learn_weights
        
        # Initialize model weights
        if model_weights is not None:
            initial_weights = torch.tensor(model_weights, dtype=torch.float32)
        else:
            initial_weights = torch.ones(self.num_models, dtype=torch.float32)
        
        if learn_weights:
            self.model_weights = nn.Parameter(initial_weights)
        else:
            self.register_buffer('model_weights', initial_weights)
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.num_models, self.num_models * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_models * 2, self.num_models),
            nn.Sigmoid()
        )
        
        self.logger.info(f"Initialized weighted ensemble with learnable weights: {learn_weights}")
    
    def get_model_weights(self, model_confidences: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get normalized model weights.
        
        Args:
            model_confidences: Confidence scores for each model
            
        Returns:
            Normalized model weights
        """
        base_weights = F.softmax(self.model_weights / self.temperature, dim=0)
        
        if model_confidences is not None:
            # Adjust weights based on model confidence
            confidence_weights = self.confidence_estimator(model_confidences)
            combined_weights = base_weights * confidence_weights
            return F.softmax(combined_weights, dim=0)
        else:
            return base_weights
    
    def forward(
        self,
        *args,
        return_individual_predictions: bool = False,
        return_weights: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through weighted ensemble.
        
        Args:
            return_individual_predictions: Whether to return individual predictions
            return_weights: Whether to return model weights
            *args, **kwargs: Input arguments for models
            
        Returns:
            Weighted ensemble predictions
        """
        # Get predictions from all models
        model_predictions = self.get_model_predictions(*args, **kwargs)
        
        # Calculate model confidences
        model_confidences = []
        for probs in model_predictions:
            confidence = torch.max(probs, dim=-1)[0]  # Max probability as confidence
            model_confidences.append(confidence.mean())  # Average over batch
        
        model_confidences = torch.stack(model_confidences)
        
        # Get adaptive weights
        weights = self.get_model_weights(model_confidences)
        
        # Weighted combination
        weighted_predictions = []
        for i, (prediction, weight) in enumerate(zip(model_predictions, weights)):
            weighted_predictions.append(prediction * weight)
        
        ensemble_probs = torch.sum(torch.stack(weighted_predictions), dim=0)
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        if return_individual_predictions or return_weights:
            output = {
                'ensemble_logits': ensemble_logits,
                'individual_predictions': model_predictions,
                'model_weights': weights,
                'model_confidences': model_confidences
            }
            return output
        else:
            return ensemble_logits
    
    def update_weights_from_validation(
        self,
        validation_scores: List[float]
    ):
        """
        Update model weights based on validation performance.
        
        Args:
            validation_scores: Validation scores for each model
        """
        if not self.learn_weights:
            # Update fixed weights based on performance
            performance_weights = torch.tensor(validation_scores, dtype=torch.float32)
            performance_weights = performance_weights / performance_weights.sum()
            self.model_weights.data = performance_weights
            
            self.logger.info(f"Updated model weights based on validation: {performance_weights.tolist()}")


class StackingEnsemble(EnsembleSarcasmModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        models: Optional[List[nn.Module]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        meta_learner_type: str = "logistic"  # 'logistic', 'mlp', 'xgb'
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            config: Model configuration
            models: List of pre-trained models
            model_configs: List of model configurations
            meta_learner_type: Type of meta-learner
        """
        super().__init__(config, models, model_configs, "stacking")
        
        self.meta_learner_type = meta_learner_type
        
        # Meta-learner
        if meta_learner_type == "logistic":
            self.meta_learner = nn.Linear(self.num_models * 2, 2)  # *2 for probabilities of both classes
        elif meta_learner_type == "mlp":
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * 2, self.num_models * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.num_models * 4, self.num_models),
                nn.ReLU(),
                nn.Linear(self.num_models, 2)
            )
        else:
            raise ValueError(f"Unsupported meta-learner type: {meta_learner_type}")
        
        self.logger.info(f"Initialized stacking ensemble with {meta_learner_type} meta-learner")
    
    def forward(
        self,
        *args,
        return_individual_predictions: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through stacking ensemble.
        
        Args:
            return_individual_predictions: Whether to return individual predictions
            *args, **kwargs: Input arguments for models
            
        Returns:
            Meta-learner predictions
        """
        # Get predictions from all models
        model_predictions = self.get_model_predictions(*args, **kwargs)
        
        # Concatenate all model predictions as meta-features
        meta_features = torch.cat(model_predictions, dim=-1)  # [batch_size, num_models * 2]
        
        # Meta-learner prediction
        ensemble_logits = self.meta_learner(meta_features)
        
        if return_individual_predictions:
            return {
                'ensemble_logits': ensemble_logits,
                'individual_predictions': model_predictions,
                'meta_features': meta_features
            }
        else:
            return ensemble_logits
    
    def train_meta_learner(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        lr: float = 0.001
    ):
        """
        Train the meta-learner using cross-validation predictions.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.meta_learner.train()
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Get base model predictions (these should be from cross-validation)
                model_predictions = self.get_model_predictions(**batch)
                meta_features = torch.cat(model_predictions, dim=-1)
                
                # Meta-learner forward pass
                meta_logits = self.meta_learner(meta_features)
                loss = criterion(meta_logits, batch['labels'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            # Validation
            if val_loader:
                val_loss = 0.0
                val_accuracy = 0.0
                val_batches = 0
                
                self.meta_learner.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        model_predictions = self.get_model_predictions(**batch)
                        meta_features = torch.cat(model_predictions, dim=-1)
                        meta_logits = self.meta_learner(meta_features)
                        
                        loss = criterion(meta_logits, batch['labels'])
                        val_loss += loss.item()
                        
                        predictions = torch.argmax(meta_logits, dim=-1)
                        accuracy = (predictions == batch['labels']).float().mean()
                        val_accuracy += accuracy.item()
                        val_batches += 1
                
                self.meta_learner.train()
                
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {train_loss/num_batches:.4f}, "
                    f"Val Loss: {val_loss/val_batches:.4f}, "
                    f"Val Accuracy: {val_accuracy/val_batches:.4f}"
                )
