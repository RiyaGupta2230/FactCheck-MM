# sarcasm_detection/models/text_sarcasm_model.py
"""
Text-based Sarcasm Detection Models
RoBERTa-based transformer and lightweight LSTM variants for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from transformers import (
    RobertaModel, RobertaTokenizer, RobertaConfig,
    AutoModel, AutoTokenizer
)
import numpy as np

from shared.base_model import BaseModel
from shared.utils import get_logger


class TextSarcasmModel(BaseModel):
    """Abstract base class for text-based sarcasm detection models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = 2,
        model_name: str = "text_sarcasm"
    ):
        """
        Initialize text sarcasm model.
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
            model_name: Model name for logging
        """
        super().__init__(config, model_name)
        self.num_classes = num_classes
        self.logger = get_logger(f"TextSarcasmModel_{model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'text_sarcasm',
            'num_classes': self.num_classes,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_num_trainable_parameters()
        }


class RobertaSarcasmModel(TextSarcasmModel):
    """
    RoBERTa-based sarcasm detection model with task-specific head.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = 2,
        model_name: str = "roberta-large",
        freeze_base: bool = False,
        add_pooling_layers: bool = True,
        dropout_rate: float = 0.1,
        use_attention_pooling: bool = True
    ):
        """
        Initialize RoBERTa sarcasm detection model.
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
            model_name: Pretrained model name
            freeze_base: Whether to freeze base model parameters
            add_pooling_layers: Whether to add additional pooling layers
            dropout_rate: Dropout rate for classification layers
            use_attention_pooling: Whether to use attention-based pooling
        """
        super().__init__(config, num_classes, "roberta_sarcasm")
        
        self.model_name = model_name
        self.freeze_base = freeze_base
        self.dropout_rate = dropout_rate
        self.use_attention_pooling = use_attention_pooling
        
        # Load pretrained RoBERTa
        try:
            self.roberta = RobertaModel.from_pretrained(
                model_name,
                cache_dir=config.get('cache_dir')
            )
            self.hidden_dim = self.roberta.config.hidden_size
        except Exception as e:
            self.logger.error(f"Failed to load RoBERTa model {model_name}: {e}")
            raise
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.logger.info("Frozen RoBERTa base model parameters")
        
        # Attention pooling layer
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(self.hidden_dim, 1)
        
        # Additional pooling layers
        if add_pooling_layers:
            self.pooling_layers = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            classifier_input_dim = self.hidden_dim // 4
        else:
            self.pooling_layers = None
            classifier_input_dim = self.hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        # Sarcasm-specific features
        self.sarcasm_feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"Initialized RoBERTa sarcasm model: {self.get_num_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize classification layer weights."""
        for module in [self.classifier, self.sarcasm_feature_extractor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        
        if self.use_attention_pooling:
            torch.nn.init.xavier_uniform_(self.attention_pooling.weight)
            torch.nn.init.zeros_(self.attention_pooling.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through RoBERTa sarcasm model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            return_features: Whether to return intermediate features
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Logits tensor or dictionary with logits and features
        """
        # RoBERTa forward pass
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention_weights
        )
        
        sequence_output = roberta_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled_output = roberta_outputs.pooler_output  # [batch_size, hidden_dim]
        
        # Apply attention pooling if enabled
        if self.use_attention_pooling:
            attention_scores = self.attention_pooling(sequence_output)  # [batch_size, seq_len, 1]
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            attention_weights = F.softmax(attention_scores, dim=1)
            attended_output = torch.sum(attention_weights * sequence_output, dim=1)  # [batch_size, hidden_dim]
            
            # Combine pooled and attended outputs
            combined_output = (pooled_output + attended_output) / 2
        else:
            combined_output = pooled_output
        
        # Apply additional pooling layers
        if self.pooling_layers:
            for layer in self.pooling_layers:
                combined_output = layer(combined_output)
        
        # Extract sarcasm-specific features
        sarcasm_features = self.sarcasm_feature_extractor(pooled_output)
        
        # Classification
        logits = self.classifier(combined_output)
        
        if return_features or return_attention_weights:
            output = {
                'logits': logits,
                'pooled_output': pooled_output,
                'sarcasm_features': sarcasm_features,
                'combined_features': combined_output
            }
            
            if return_attention_weights:
                output['roberta_attentions'] = roberta_outputs.attentions
                if self.use_attention_pooling:
                    output['pooling_attention_weights'] = attention_weights
            
            return output
        else:
            return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get text embeddings for similarity computation."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=True
            )
            return outputs['sarcasm_features']


class LSTMSarcasmModel(TextSarcasmModel):
    """
    Lightweight LSTM-based sarcasm detection model for baseline comparison.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = 2,
        vocab_size: int = 50000,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        use_pretrained_embeddings: bool = True,
        max_seq_length: int = 128
    ):
        """
        Initialize LSTM sarcasm detection model.
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate
            use_pretrained_embeddings: Whether to use pretrained embeddings
            max_seq_length: Maximum sequence length
        """
        super().__init__(config, num_classes, "lstm_sarcasm")
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if specified
        if use_pretrained_embeddings:
            self._init_pretrained_embeddings()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim // 4, lstm_output_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 8, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"Initialized LSTM sarcasm model: {self.get_num_parameters():,} parameters")
    
    def _init_pretrained_embeddings(self):
        """Initialize embeddings with pretrained vectors (placeholder implementation)."""
        # In practice, you would load GloVe, FastText, or similar embeddings
        # For now, we'll use Xavier initialization
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding token embedding to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize other layers
        for module in [self.attention, self.feature_extractor, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through LSTM sarcasm model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_features: Whether to return intermediate features
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Logits tensor or dictionary with logits and features
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply dropout to embeddings
        embedded = F.dropout(embedded, p=self.dropout_rate, training=self.training)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(embedded)  # [batch_size, seq_len, lstm_output_dim]
        
        # Apply attention mechanism
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        
        # Mask attention scores if mask is provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf')
            )
        
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Weighted sum of LSTM outputs
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, lstm_output_dim]
        
        # Feature extraction
        features = self.feature_extractor(attended_output)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features or return_attention_weights:
            output = {
                'logits': logits,
                'lstm_output': lstm_output,
                'attended_output': attended_output,
                'features': features
            }
            
            if return_attention_weights:
                output['attention_weights'] = attention_weights
            
            return output
        else:
            return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get text embeddings for similarity computation."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=True
            )
            return outputs['features']
