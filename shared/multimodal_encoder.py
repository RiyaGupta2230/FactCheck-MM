"""
Unified Multimodal Encoder for FactCheck-MM
Handles text, audio, image, and video encoding with standardized output format.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor,
    ViTModel, ViTImageProcessor
)
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .utils import get_logger


class TextEncoder(nn.Module):
    """RoBERTa/DeBERTa-based text encoder."""
    
    def __init__(
        self,
        model_name: str = "roberta-large",
        hidden_size: int = 1024,
        dropout: float = 0.1,
        freeze: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Projection layer to standardized size
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_hidden_size, hidden_size)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Freeze if requested
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features and optional attention
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention
        )
        
        # Get pooled representation (CLS token)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Project to standardized size
        encoded = self.projection(pooled_output)  # [batch_size, target_hidden_size]
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        result = {
            "encoded_features": encoded,
            "sequence_output": outputs.last_hidden_state,
            "pooled_output": pooled_output
        }
        
        if return_attention:
            result["attention_weights"] = outputs.attentions
        
        return result


class AudioEncoder(nn.Module):
    """Wav2Vec2-based audio encoder."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        hidden_size: int = 1024,
        dropout: float = 0.1,
        freeze: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Load pre-trained model
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Projection layer
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_hidden_size, hidden_size)
        
        # Pooling layer (mean pooling over time)
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""  
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through audio encoder.
        
        Args:
            audio_features: Raw audio waveform [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features
        """
        outputs = self.encoder(
            input_values=audio_features,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Get sequence output [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # Temporal pooling to get fixed-size representation
        pooled_output = self.temporal_pooling(sequence_output.transpose(1, 2))  # [batch_size, hidden_size, 1]
        pooled_output = pooled_output.squeeze(-1)  # [batch_size, hidden_size]
        
        # Project to standardized size
        encoded = self.projection(pooled_output)
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        result = {
            "encoded_features": encoded,
            "sequence_output": sequence_output,
            "pooled_output": pooled_output
        }
        
        if return_attention:
            result["attention_weights"] = outputs.attentions
        
        return result


class VisionEncoder(nn.Module):
    """ViT-based image encoder."""
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        hidden_size: int = 1024,
        dropout: float = 0.1,
        freeze: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Load pre-trained model
        self.encoder = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Projection layer
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_hidden_size, hidden_size)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.
        
        Args:
            pixel_values: Image pixels [batch_size, channels, height, width]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features
        """
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=return_attention
        )
        
        # Get pooled representation (CLS token)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Project to standardized size
        encoded = self.projection(pooled_output)
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        result = {
            "encoded_features": encoded,
            "sequence_output": outputs.last_hidden_state,
            "pooled_output": pooled_output
        }
        
        if return_attention:
            result["attention_weights"] = outputs.attentions
        
        return result


class VideoEncoder(nn.Module):
    """Video encoder using frame-level ViT + temporal aggregation."""
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        hidden_size: int = 1024,
        max_frames: int = 16,
        temporal_strategy: str = "attention",  # "mean", "max", "attention"
        dropout: float = 0.1,
        freeze: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.temporal_strategy = temporal_strategy
        
        # Frame-level encoder (shared ViT)
        self.frame_encoder = VisionEncoder(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze=freeze
        )
        
        # Temporal aggregation
        if temporal_strategy == "attention":
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        video_frames: torch.Tensor,  # [batch_size, num_frames, channels, height, width]
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through video encoder.
        
        Args:
            video_frames: Video frames [batch_size, num_frames, channels, height, width]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features
        """
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        # Encode each frame independently
        frame_features = []
        for i in range(num_frames):
            frame = video_frames[:, i]  # [batch_size, channels, height, width]
            frame_output = self.frame_encoder(frame, return_attention=False)
            frame_features.append(frame_output["encoded_features"])
        
        # Stack frame features [batch_size, num_frames, hidden_size]
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal aggregation
        if self.temporal_strategy == "mean":
            encoded = torch.mean(frame_features, dim=1)
        elif self.temporal_strategy == "max":
            encoded = torch.max(frame_features, dim=1)[0]
        elif self.temporal_strategy == "attention":
            # Self-attention over frames
            encoded, attention_weights = self.temporal_attention(
                frame_features, frame_features, frame_features
            )
            encoded = torch.mean(encoded, dim=1)  # Average over temporal dimension
        else:
            raise ValueError(f"Unknown temporal strategy: {self.temporal_strategy}")
        
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        result = {
            "encoded_features": encoded,
            "frame_features": frame_features,
            "pooled_output": encoded
        }
        
        if return_attention and self.temporal_strategy == "attention":
            result["temporal_attention_weights"] = attention_weights
        
        return result


class MultimodalEncoder(nn.Module):
    """
    Unified multimodal encoder that handles text, audio, image, and video inputs.
    Returns standardized embeddings for fusion layers.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        modalities: List[str] = ["text", "audio", "image", "video"],
        hidden_size: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize multimodal encoder.
        
        Args:
            config: Configuration dictionary with model names
            modalities: List of modalities to support
            hidden_size: Standardized hidden size for all modalities
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.logger = get_logger("MultimodalEncoder")
        
        # Initialize encoders for each modality
        self.encoders = nn.ModuleDict()
        
        if "text" in modalities:
            self.encoders["text"] = TextEncoder(
                model_name=config.get("text_model_name", "roberta-large"),
                hidden_size=hidden_size,
                dropout=dropout
            )
            self.logger.info(f"Initialized text encoder: {config.get('text_model_name', 'roberta-large')}")
        
        if "audio" in modalities:
            self.encoders["audio"] = AudioEncoder(
                model_name=config.get("audio_model_name", "facebook/wav2vec2-large-960h"),
                hidden_size=hidden_size,
                dropout=dropout
            )
            self.logger.info(f"Initialized audio encoder: {config.get('audio_model_name', 'facebook/wav2vec2-large-960h')}")
        
        if "image" in modalities:
            self.encoders["image"] = VisionEncoder(
                model_name=config.get("vision_model_name", "google/vit-large-patch16-224"),
                hidden_size=hidden_size,
                dropout=dropout
            )
            self.logger.info(f"Initialized image encoder: {config.get('vision_model_name', 'google/vit-large-patch16-224')}")
        
        if "video" in modalities:
            self.encoders["video"] = VideoEncoder(
                model_name=config.get("vision_model_name", "google/vit-large-patch16-224"),
                hidden_size=hidden_size,
                dropout=dropout
            )
            self.logger.info(f"Initialized video encoder: {config.get('vision_model_name', 'google/vit-large-patch16-224')}")
    
    def forward(
        self,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        video_inputs: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all available encoders.
        
        Args:
            text_inputs: Text tokenizer outputs
            audio_inputs: Audio waveform tensor
            image_inputs: Image pixel tensor  
            video_inputs: Video frames tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features for each modality
        """
        encoded_features = {}
        attention_weights = {}
        
        # Encode text
        if "text" in self.modalities and text_inputs is not None:
            text_output = self.encoders["text"](
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                token_type_ids=text_inputs.get("token_type_ids"),
                return_attention=return_attention
            )
            encoded_features["text"] = text_output["encoded_features"]
            if return_attention:
                attention_weights["text"] = text_output.get("attention_weights")
        
        # Encode audio
        if "audio" in self.modalities and audio_inputs is not None:
            audio_output = self.encoders["audio"](
                audio_features=audio_inputs,
                return_attention=return_attention
            )
            encoded_features["audio"] = audio_output["encoded_features"]
            if return_attention:
                attention_weights["audio"] = audio_output.get("attention_weights")
        
        # Encode image
        if "image" in self.modalities and image_inputs is not None:
            image_output = self.encoders["image"](
                pixel_values=image_inputs,
                return_attention=return_attention
            )
            encoded_features["image"] = image_output["encoded_features"]
            if return_attention:
                attention_weights["image"] = image_output.get("attention_weights")
        
        # Encode video  
        if "video" in self.modalities and video_inputs is not None:
            video_output = self.encoders["video"](
                video_frames=video_inputs,
                return_attention=return_attention
            )
            encoded_features["video"] = video_output["encoded_features"]
            if return_attention:
                attention_weights["video"] = video_output.get("attention_weights")
        
        result = {"encoded_features": encoded_features}
        if return_attention:
            result["attention_weights"] = attention_weights
        
        return result
    
    def freeze_modality(self, modality: str):
        """Freeze specific modality encoder."""
        if modality in self.encoders:
            self.encoders[modality].freeze_encoder()
            self.logger.info(f"Frozen {modality} encoder")
        else:
            self.logger.warning(f"Modality {modality} not found")
    
    def unfreeze_modality(self, modality: str):
        """Unfreeze specific modality encoder."""
        if modality in self.encoders:
            self.encoders[modality].unfreeze_encoder()
            self.logger.info(f"Unfrozen {modality} encoder")
        else:
            self.logger.warning(f"Modality {modality} not found")
    
    def get_encoding_dim(self) -> int:
        """Get the encoding dimension."""
        return self.hidden_size
    
    def get_supported_modalities(self) -> List[str]:
        """Get list of supported modalities."""
        return self.modalities
