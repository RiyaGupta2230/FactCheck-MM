"""
Model utilities and architectures for milestone1 sarcasm detection
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    RobertaModel, BertModel
)
import os
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MultiModalFusionLayer(nn.Module):
    """Advanced fusion layer for multimodal features"""
    
    def __init__(self, text_dim, feature_dim, fusion_dim, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.text_projection = nn.Linear(text_dim, fusion_dim)
            self.feature_projection = nn.Linear(feature_dim, fusion_dim)
            self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
            self.output_projection = nn.Linear(fusion_dim, fusion_dim)
            
        elif fusion_type == 'gated':
            self.text_gate = nn.Linear(text_dim + feature_dim, text_dim)
            self.feature_gate = nn.Linear(text_dim + feature_dim, feature_dim)
            self.fusion_layer = nn.Linear(text_dim + feature_dim, fusion_dim)
            
        elif fusion_type == 'concat':
            self.fusion_layer = nn.Linear(text_dim + feature_dim, fusion_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, text_features, linguistic_features):
        if self.fusion_type == 'attention':
            # Project features
            text_proj = self.text_projection(text_features).unsqueeze(1)  # Add sequence dimension
            feature_proj = self.feature_projection(linguistic_features).unsqueeze(1)
            
            # Concatenate for attention
            combined = torch.cat([text_proj, feature_proj], dim=1)  # [batch, 2, fusion_dim]
            
            # Self-attention
            attended, _ = self.attention(combined, combined, combined)
            
            # Pool and project
            pooled = attended.mean(dim=1)  # [batch, fusion_dim]
            output = self.output_projection(pooled)
            
        elif self.fusion_type == 'gated':
            # Concatenate features
            combined = torch.cat([text_features, linguistic_features], dim=1)
            
            # Gating mechanism
            text_gate = torch.sigmoid(self.text_gate(combined))
            feature_gate = torch.sigmoid(self.feature_gate(combined))
            
            # Apply gates
            gated_text = text_features * text_gate
            gated_features = linguistic_features * feature_gate
            
            # Fuse
            fused = torch.cat([gated_text, gated_features], dim=1)
            output = self.fusion_layer(fused)
            
        else:  # concat
            combined = torch.cat([text_features, linguistic_features], dim=1)
            output = self.fusion_layer(combined)
        
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output

class AdvancedSarcasmClassifier(nn.Module):
    """Advanced sarcasm detection model with multiple fusion strategies"""
    
    def __init__(self, 
                 model_name: str,
                 num_features: int,
                 num_labels: int = 2,
                 fusion_type: str = 'attention',
                 dropout: float = 0.15,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.num_features = num_features
        self.num_labels = num_labels
        self.fusion_type = fusion_type
        
        # Load text encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.config.hidden_size
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # Fusion layer
        self.fusion = MultiModalFusionLayer(
            text_dim=hidden_size,
            feature_dim=hidden_size // 4,
            fusion_dim=hidden_size,
            fusion_type=fusion_type
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in [self.feature_processor, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, features, labels=None):
        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output
        
        # Feature processing
        processed_features = self.feature_processor(features)
        
        # Fusion
        fused_features = self.fusion(text_features, processed_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fn(logits, labels)
        
        return outputs

class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Ensure weights sum to 1
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def forward(self, input_ids, attention_mask, features, labels=None):
        all_logits = []
        total_loss = 0
        
        for i, model in enumerate(self.models):
            model_outputs = model(input_ids, attention_mask, features, labels)
            all_logits.append(model_outputs['logits'] * self.weights[i])
            
            if labels is not None and 'loss' in model_outputs:
                total_loss += model_outputs['loss'] * self.weights[i]
        
        # Weighted average of logits
        ensemble_logits = torch.stack(all_logits).sum(dim=0)
        
        outputs = {'logits': ensemble_logits}
        
        if labels is not None:
            outputs['loss'] = total_loss
        
        return outputs

def create_model(config: Dict):
    """Factory function to create model from config"""
    
    model_type = config.get('model_type', 'advanced')
    
    if model_type == 'advanced':
        model = AdvancedSarcasmClassifier(
            model_name=config['model']['name'],
            num_features=config['num_features'],
            num_labels=config['model'].get('num_labels', 2),
            fusion_type=config['model'].get('fusion_type', 'attention'),
            dropout=config['model'].get('dropout', 0.15),
            freeze_encoder=config['model'].get('freeze_encoder', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def save_model_with_config(model, tokenizer, config, save_path):
    """Save model with configuration"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save config
    model_config = {
        'model_name': config['model']['name'],
        'num_features': getattr(model, 'num_features', 0),
        'num_labels': config['model'].get('num_labels', 2),
        'fusion_type': config['model'].get('fusion_type', 'attention'),
        'dropout': config['model'].get('dropout', 0.15)
    }
    
    with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Model saved to: {save_path}")

def load_model_with_config(model_path):
    """Load model with configuration"""
    
    # Load config
    config_path = os.path.join(model_path, 'model_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create model
    model = AdvancedSarcasmClassifier(
        model_name=model_config['model_name'],
        num_features=model_config['num_features'],
        num_labels=model_config['num_labels'],
        fusion_type=model_config['fusion_type'],
        dropout=model_config['dropout']
    )
    
    # Load state dict
    state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, model_config

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }
