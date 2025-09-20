"""
Multimodal Fusion Strategies for FactCheck-MM
Implements concatenation, self-attention, and cross-modal attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import math

from .utils import get_logger


class FusionStrategy(nn.Module, ABC):
    """Abstract base class for fusion strategies."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize fusion strategy.
        
        Args:
            input_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output dimension after fusion
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.modalities = list(input_dims.keys())
        
        self.dropout = nn.Dropout(dropout)
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse features from multiple modalities.
        
        Args:
            modality_features: Dictionary mapping modality names to feature tensors
            
        Returns:
            Fused feature tensor [batch_size, output_dim]
        """
        pass
    
    def _validate_inputs(self, modality_features: Dict[str, torch.Tensor]) -> None:
        """Validate input features."""
        for modality, features in modality_features.items():
            if modality not in self.modalities:
                self.logger.warning(f"Unknown modality: {modality}")
            
            expected_dim = self.input_dims.get(modality)
            if expected_dim and features.shape[-1] != expected_dim:
                raise ValueError(
                    f"Expected {modality} features to have dimension {expected_dim}, "
                    f"got {features.shape[-1]}"
                )


class ConcatenationFusion(FusionStrategy):
    """Simple concatenation-based fusion with projection."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__(input_dims, output_dim, dropout)
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Calculate total concatenated dimension
        self.concat_dim = sum(input_dims.values())
        
        # Projection layers
        self.projection = nn.Linear(self.concat_dim, output_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        
        # Residual connection (if dimensions match)
        if use_residual and len(set(input_dims.values())) == 1:
            # All modalities have same dimension
            single_dim = list(input_dims.values())[0]
            if single_dim == output_dim:
                self.residual_projection = None
            else:
                self.residual_projection = nn.Linear(single_dim, output_dim)
        else:
            self.use_residual = False
        
        self.logger.info(
            f"Initialized concatenation fusion: {self.concat_dim} -> {output_dim}"
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate and project modality features.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        self._validate_inputs(modality_features)
        
        # Get features in consistent order
        feature_list = []
        for modality in self.modalities:
            if modality in modality_features:
                features = modality_features[modality]
                feature_list.append(features)
        
        if not feature_list:
            raise ValueError("No valid modality features provided")
        
        # Concatenate features
        concatenated = torch.cat(feature_list, dim=-1)  # [batch_size, concat_dim]
        
        # Project to output dimension
        fused = self.projection(concatenated)  # [batch_size, output_dim]
        
        # Apply residual connection if enabled
        if self.use_residual and len(feature_list) == 1:
            residual = feature_list[0]
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            fused = fused + residual
        
        # Apply layer normalization
        if self.use_layer_norm:
            fused = self.layer_norm(fused)
        
        # Apply dropout
        fused = self.dropout(fused)
        
        return fused


class SelfAttentionFusion(FusionStrategy):
    """Self-attention based fusion across modalities."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__(input_dims, output_dim, dropout)
        
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        
        # Check that all input dimensions are the same for attention
        unique_dims = set(input_dims.values())
        if len(unique_dims) > 1:
            # Project all modalities to the same dimension
            self.input_projections = nn.ModuleDict()
            self.common_dim = output_dim
            for modality, dim in input_dims.items():
                self.input_projections[modality] = nn.Linear(dim, self.common_dim)
        else:
            self.common_dim = list(unique_dims)[0]
            self.input_projections = None
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        if self.common_dim != output_dim:
            self.output_projection = nn.Linear(self.common_dim, output_dim)
        else:
            self.output_projection = None
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        
        self.logger.info(
            f"Initialized self-attention fusion: {len(input_dims)} modalities, "
            f"{num_heads} heads, {self.common_dim} -> {output_dim}"
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply self-attention across modality features.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        self._validate_inputs(modality_features)
        
        # Project modalities to common dimension if needed
        projected_features = []
        for modality in self.modalities:
            if modality in modality_features:
                features = modality_features[modality]
                
                if self.input_projections is not None:
                    features = self.input_projections[modality](features)
                
                projected_features.append(features)
        
        if not projected_features:
            raise ValueError("No valid modality features provided")
        
        # Stack features for attention [batch_size, num_modalities, common_dim]
        stacked_features = torch.stack(projected_features, dim=1)
        
        # Apply self-attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Average across modalities (or use first token as in BERT)
        fused = torch.mean(attended_features, dim=1)  # [batch_size, common_dim]
        
        # Output projection if needed
        if self.output_projection is not None:
            fused = self.output_projection(fused)
        
        # Layer normalization
        if self.use_layer_norm:
            fused = self.layer_norm(fused)
        
        # Dropout
        fused = self.dropout(fused)
        
        return fused


class CrossModalAttentionFusion(FusionStrategy):
    """Cross-modal attention fusion with query-key-value mechanism."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__(input_dims, output_dim, dropout)
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Project all modalities to common dimension
        self.modality_projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.modality_projections[modality] = nn.Linear(dim, output_dim)
        
        # Cross-modal attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict()
            
            # Attention mechanisms for each modality pair
            for mod1 in self.modalities:
                layer[f"{mod1}_attention"] = nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            
            if use_layer_norm:
                layer["layer_norm"] = nn.LayerNorm(output_dim)
            
            # Feed-forward network
            layer["ffn"] = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(dropout)
            )
            
            if use_layer_norm:
                layer["ffn_layer_norm"] = nn.LayerNorm(output_dim)
            
            self.attention_layers.append(layer)
        
        # Final fusion layer
        self.final_fusion = nn.Linear(output_dim * len(self.modalities), output_dim)
        
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(output_dim)
        
        self.logger.info(
            f"Initialized cross-modal attention fusion: {len(input_dims)} modalities, "
            f"{num_heads} heads, {num_layers} layers"
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply cross-modal attention fusion.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        self._validate_inputs(modality_features)
        
        # Project all modalities to common dimension
        projected_features = {}
        for modality in self.modalities:
            if modality in modality_features:
                features = modality_features[modality]
                projected = self.modality_projections[modality](features)
                projected_features[modality] = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
        
        if not projected_features:
            raise ValueError("No valid modality features provided")
        
        # Apply cross-modal attention layers
        current_features = projected_features.copy()
        
        for layer in self.attention_layers:
            new_features = {}
            
            # Cross-modal attention for each modality
            for modality in current_features.keys():
                query = current_features[modality]
                
                # Use other modalities as key and value
                other_modalities = [m for m in current_features.keys() if m != modality]
                if other_modalities:
                    # Concatenate other modalities
                    key_value = torch.cat([current_features[m] for m in other_modalities], dim=1)
                    
                    # Apply attention
                    attended, _ = layer[f"{modality}_attention"](
                        query, key_value, key_value
                    )
                    
                    # Residual connection
                    if self.use_residual:
                        attended = attended + query
                    
                    # Layer normalization
                    if self.use_layer_norm:
                        attended = layer["layer_norm"](attended)
                    
                    new_features[modality] = attended
                else:
                    new_features[modality] = query
            
            # Apply feed-forward network
            for modality in new_features.keys():
                features = new_features[modality]
                ffn_output = layer["ffn"](features)
                
                # Residual connection
                if self.use_residual:
                    ffn_output = ffn_output + features
                
                # Layer normalization
                if self.use_layer_norm:
                    ffn_output = layer["ffn_layer_norm"](ffn_output)
                
                new_features[modality] = ffn_output
            
            current_features = new_features
        
        # Final fusion
        final_features = [features.squeeze(1) for features in current_features.values()]
        concatenated = torch.cat(final_features, dim=-1)
        
        fused = self.final_fusion(concatenated)
        
        if self.use_layer_norm:
            fused = self.final_layer_norm(fused)
        
        fused = self.dropout(fused)
        
        return fused


class FusionLayerFactory:
    """Factory for creating fusion layers from configuration."""
    
    @staticmethod
    def create_fusion_layer(
        fusion_type: str,
        input_dims: Dict[str, int],
        output_dim: int,
        **kwargs
    ) -> FusionStrategy:
        """
        Create fusion layer from configuration.
        
        Args:
            fusion_type: Type of fusion ("concat", "self_attention", "cross_modal_attention")
            input_dims: Dictionary mapping modality names to dimensions
            output_dim: Output dimension
            **kwargs: Additional configuration parameters
            
        Returns:
            Fusion layer instance
        """
        fusion_type = fusion_type.lower().replace("-", "_").replace(" ", "_")
        
        if fusion_type in ["concat", "concatenation", "concatenation_fusion"]:
            return ConcatenationFusion(input_dims, output_dim, **kwargs)
        
        elif fusion_type in ["self_attention", "self_attention_fusion"]:
            return SelfAttentionFusion(input_dims, output_dim, **kwargs)
        
        elif fusion_type in ["cross_modal_attention", "cross_modal", "cross_attention"]:
            return CrossModalAttentionFusion(input_dims, output_dim, **kwargs)
        
        else:
            raise ValueError(
                f"Unknown fusion type: {fusion_type}. "
                f"Available: concat, self_attention, cross_modal_attention"
            )
    
    @staticmethod
    def get_available_fusion_types() -> List[str]:
        """Get list of available fusion types."""
        return [
            "concatenation",
            "self_attention", 
            "cross_modal_attention"
        ]
