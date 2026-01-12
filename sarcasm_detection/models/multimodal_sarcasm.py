# sarcasm_detection/models/multimodal_sarcasm.py
"""
Multimodal Sarcasm Detection Model
Full multimodal architecture combining text, audio, image, and video.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import numpy as np

from shared.base_model import BaseMultimodalModel

from shared.multimodal_encoder import MultimodalEncoder
from shared.fusion_layers import FusionLayerFactory
from .fusion_strategies import FusionFactory
from shared.utils import get_logger


class MultimodalSarcasmModel(BaseMultimodalModel):
    """
    Multimodal sarcasm detection model combining text, audio, image, and video modalities.
    """
    _config_class = MultimodalSarcasmConfig
    def __init__(
        self,
        config: MultimodalSarcasmConfig,
        num_classes: int = 2,
        fusion_strategy: str = "cross_modal_attention",
        modalities: Optional[List[str]] = None,
        feature_dims: Optional[Dict[str, int]] = None
    ):
        """
        Initialize multimodal sarcasm detection model.
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
            fusion_strategy: Fusion strategy for combining modalities
            modalities: List of modalities to use
            feature_dims: Feature dimensions for each modality
        """
        super().__init__(
            config=config,
            model_name="multimodal_sarcasm",
            task_name="sarcasm_detection",
            num_classes=config.num_classes if hasattr(config, "num_classes") else 2,
            supported_modalities=["text", "image"]
        )

        
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.modalities = modalities or ['text', 'audio', 'image', 'video']
        
        # Default feature dimensions
        self.feature_dims = feature_dims or {
            'text': config.get('text_hidden_dim', 1024),
            'audio': config.get('audio_hidden_dim', 768),
            'image': config.get('image_hidden_dim', 768),
            'video': config.get('video_hidden_dim', 768)
        }
        
        self.logger = get_logger("MultimodalSarcasmModel")
        
        # Initialize multimodal encoder
        self.encoder = MultimodalEncoder(config)
        
        # Get actual feature dimensions from encoder
        encoder_dims = self.encoder.get_output_dimensions()
        for modality in self.modalities:
            if modality in encoder_dims:
                self.feature_dims[modality] = encoder_dims[modality]
        
        # Initialize fusion layer
        self.fusion_layer = FusionFactory.create_fusion_layer(
            fusion_type=fusion_strategy,
            input_dims={mod: self.feature_dims[mod] for mod in self.modalities},
            output_dim=config.get('fusion_output_dim', 512),
            **config.get('fusion_config', {})
        )
        
        # Task-specific layers
        self.task_head = MultimodalSarcasmClassifier(
            input_dim=config.get('fusion_output_dim', 512),
            hidden_dim=config.get('classifier_hidden_dim', 256),
            num_classes=num_classes,
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        
        # Modality-specific attention
        self.modality_attention = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.feature_dims[modality], 1),
                nn.Sigmoid()
            ) for modality in self.modalities
        })
        
        # Cross-modal interaction layers
        self.cross_modal_layers = nn.ModuleDict()
        for i, mod1 in enumerate(self.modalities):
            for mod2 in self.modalities[i+1:]:
                layer_name = f"{mod1}_{mod2}_interaction"
                self.cross_modal_layers[layer_name] = nn.MultiheadAttention(
                    embed_dim=min(self.feature_dims[mod1], self.feature_dims[mod2]),
                    num_heads=8,
                    dropout=config.get('dropout_rate', 0.1),
                    batch_first=True
                )
        
        self.logger.info(
            f"Initialized multimodal sarcasm model with {len(self.modalities)} modalities: "
            f"{self.get_num_parameters():,} parameters"
        )
    
    def forward(
        self,
        text: Optional[Dict[str, torch.Tensor]] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention_weights: bool = False,
        available_modalities: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through multimodal sarcasm model.
        
        Args:
            text: Text input dictionary
            audio: Audio tensor [batch_size, audio_len] or [batch_size, audio_features]
            image: Image tensor [batch_size, channels, height, width]
            video: Video tensor [batch_size, frames, channels, height, width]
            return_features: Whether to return intermediate features
            return_attention_weights: Whether to return attention weights
            available_modalities: List of available modalities for this batch
            
        Returns:
            Logits tensor or dictionary with logits and features
        """
        batch_size = self._get_batch_size(text, audio, image, video)
        
        # Determine which modalities are available
        if available_modalities is None:
            available_modalities = []
            if text is not None:
                available_modalities.append('text')
            if audio is not None:
                available_modalities.append('audio')
            if image is not None:
                available_modalities.append('image')
            if video is not None:
                available_modalities.append('video')
        
        # Encode each modality
        modality_features = {}
        modality_attentions = {}
        
        # Text encoding
        if 'text' in available_modalities and text is not None:
            text_features = self.encoder.encode_text(text)
            text_attention = self.modality_attention['text'](text_features)
            modality_features['text'] = text_features * text_attention
            modality_attentions['text'] = text_attention
        
        # Audio encoding
        if 'audio' in available_modalities and audio is not None:
            audio_features = self.encoder.encode_audio(audio)
            audio_attention = self.modality_attention['audio'](audio_features)
            modality_features['audio'] = audio_features * audio_attention
            modality_attentions['audio'] = audio_attention
        
        # Image encoding
        if 'image' in available_modalities and image is not None:
            image_features = self.encoder.encode_image(image)
            image_attention = self.modality_attention['image'](image_features)
            modality_features['image'] = image_features * image_attention
            modality_attentions['image'] = image_attention
        
        # Video encoding
        if 'video' in available_modalities and video is not None:
            video_features = self.encoder.encode_video(video)
            video_attention = self.modality_attention['video'](video_features)
            modality_features['video'] = video_features * video_attention
            modality_attentions['video'] = video_attention
        
        # Handle missing modalities with zero tensors
        for modality in self.modalities:
            if modality not in modality_features:
                modality_features[modality] = torch.zeros(
                    batch_size, self.feature_dims[modality],
                    device=next(self.parameters()).device
                )
        
        # Cross-modal interactions
        cross_modal_features = {}
        cross_modal_attention_weights = {}
        
        for layer_name, interaction_layer in self.cross_modal_layers.items():
            mod1, mod2 = layer_name.replace('_interaction', '').split('_')
            
            if mod1 in modality_features and mod2 in modality_features:
                feat1 = modality_features[mod1].unsqueeze(1)  # [batch_size, 1, feat_dim]
                feat2 = modality_features[mod2].unsqueeze(1)  # [batch_size, 1, feat_dim]
                
                # Ensure feature dimensions match for attention
                min_dim = min(feat1.shape[-1], feat2.shape[-1])
                feat1 = feat1[..., :min_dim]
                feat2 = feat2[..., :min_dim]
                
                # Cross-modal attention
                attended_feat, attention_weights = interaction_layer(feat1, feat2, feat2)
                cross_modal_features[layer_name] = attended_feat.squeeze(1)
                
                if return_attention_weights:
                    cross_modal_attention_weights[layer_name] = attention_weights
        
        # Fusion
        fused_features = self.fusion_layer(modality_features)
        
        # Classification
        logits = self.task_head(fused_features)
        
        if return_features or return_attention_weights:
            output = {
                'logits': logits,
                'fused_features': fused_features,
                'modality_features': modality_features,
                'cross_modal_features': cross_modal_features,
                'available_modalities': available_modalities
            }
            
            if return_attention_weights:
                output['modality_attentions'] = modality_attentions
                output['cross_modal_attentions'] = cross_modal_attention_weights
            
            return output
        else:
            return logits
    
    def _get_batch_size(self, text, audio, image, video) -> int:
        """Get batch size from available inputs."""
        if text is not None:
            return text['input_ids'].shape[0]
        elif audio is not None:
            return audio.shape[0]
        elif image is not None:
            return image.shape[0]
        elif video is not None:
            return video.shape[0]
        else:
            raise ValueError("At least one modality must be provided")
    
    def get_embeddings(
        self,
        text: Optional[Dict[str, torch.Tensor]] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get multimodal embeddings for similarity computation."""
        with torch.no_grad():
            outputs = self.forward(
                text=text, audio=audio, image=image, video=video,
                return_features=True
            )
            return outputs['fused_features']
    
    def predict_with_confidence(
        self,
        text: Optional[Dict[str, torch.Tensor]] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Get predictions with confidence scores."""
        with torch.no_grad():
            logits = self.forward(text=text, audio=audio, image=image, video=video)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'logits': logits
            }
        
    def compute_loss(self, logits, labels, **kwargs):
        """
        Concrete implementation required by BaseMultimodalModel.
        Uses standard cross-entropy for sarcasm classification.
        """
        import torch.nn.functional as F

        # logits: [batch_size, num_classes]
        # labels: [batch_size]
        return F.cross_entropy(logits, labels)


    @classmethod
    def _build_from_config(cls, config: MultimodalSarcasmConfig):
        return cls(
            config=config,
            num_classes=config.num_classes,
            modalities=config.modalities,
            fusion_strategy=config.fusion_strategy,
        )


class MultimodalSarcasmClassifier(nn.Module):
    """Classification head for multimodal sarcasm detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_layer_norm = use_layer_norm
        
        # Classification layers
        layers = []
        
        # First hidden layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        # Second hidden layer
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 2, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier."""
        return self.classifier(features)
