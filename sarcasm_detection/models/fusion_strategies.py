# sarcasm_detection/models/fusion_strategies.py
"""
Fusion Strategies for Multimodal Sarcasm Detection
Task-specific fusion modules optimized for sarcasm detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

from shared.utils import get_logger


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion with learned weighting."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout_rate: float = 0.1,
        use_modality_weights: bool = True,
        normalize_features: bool = True
    ):
        """
        Initialize concatenation fusion.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            output_dim: Output dimension
            dropout_rate: Dropout rate
            use_modality_weights: Whether to learn modality importance weights
            normalize_features: Whether to normalize input features
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.modalities = list(input_dims.keys())
        self.use_modality_weights = use_modality_weights
        self.normalize_features = normalize_features
        
        self.logger = get_logger("ConcatenationFusion")
        
        # Modality projection layers
        self.projections = nn.ModuleDict()
        total_projected_dim = 0
        
        for modality, input_dim in input_dims.items():
            projected_dim = min(input_dim, output_dim // len(input_dims))
            self.projections[modality] = nn.Sequential(
                nn.Linear(input_dim, projected_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            total_projected_dim += projected_dim
        
        # Modality importance weights
        if use_modality_weights:
            self.modality_weights = nn.Parameter(torch.ones(len(self.modalities)))
        
        # Feature normalization
        if normalize_features:
            self.feature_norms = nn.ModuleDict({
                modality: nn.LayerNorm(self.projections[modality][0].out_features)
                for modality in self.modalities
            })
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(total_projected_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.logger.info(f"Initialized concatenation fusion: {total_projected_dim} -> {output_dim}")
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through concatenation fusion.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features
        """
        projected_features = []
        
        for i, modality in enumerate(self.modalities):
            if modality in modality_features:
                # Project features
                projected = self.projections[modality](modality_features[modality])
                
                # Normalize if enabled
                if self.normalize_features:
                    projected = self.feature_norms[modality](projected)
                
                # Apply modality weights
                if self.use_modality_weights:
                    weight = F.softmax(self.modality_weights, dim=0)[i]
                    projected = projected * weight
                
                projected_features.append(projected)
            else:
                # Handle missing modality with zeros
                projected_dim = self.projections[modality][0].out_features
                zero_features = torch.zeros(
                    modality_features[list(modality_features.keys())[0]].shape[0],
                    projected_dim,
                    device=modality_features[list(modality_features.keys())[0]].device
                )
                projected_features.append(zero_features)
        
        # Concatenate all projected features
        concatenated = torch.cat(projected_features, dim=-1)
        
        # Final projection
        fused = self.final_projection(concatenated)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for multimodal sarcasm detection."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize cross-attention fusion.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            output_dim: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout_rate: Dropout rate
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.modalities = list(input_dims.keys())
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        self.logger = get_logger("CrossAttentionFusion")
        
        # Project all modalities to same dimension
        self.modality_projections = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.modality_projections[modality] = nn.Linear(input_dim, output_dim)
        
        # Cross-attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_dict = nn.ModuleDict()
            
            # Attention between each pair of modalities
            for i, mod1 in enumerate(self.modalities):
                for j, mod2 in enumerate(self.modalities):
                    if i != j:  # Don't create self-attention (that's handled separately)
                        layer_dict[f"{mod1}_to_{mod2}"] = nn.MultiheadAttention(
                            embed_dim=output_dim,
                            num_heads=num_heads,
                            dropout=dropout_rate,
                            batch_first=True
                        )
            
            # Layer normalization
            layer_dict["layer_norm"] = nn.LayerNorm(output_dim)
            
            # Feed-forward network
            layer_dict["ffn"] = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(dropout_rate)
            )
            
            layer_dict["ffn_norm"] = nn.LayerNorm(output_dim)
            
            self.attention_layers.append(layer_dict)
        
        # Final aggregation
        self.final_aggregation = nn.Sequential(
            nn.Linear(output_dim * len(self.modalities), output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.logger.info(
            f"Initialized cross-attention fusion: {len(self.modalities)} modalities, "
            f"{num_heads} heads, {num_layers} layers"
        )
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through cross-attention fusion.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project all modalities to same dimension
        projected_features = {}
        for modality in self.modalities:
            if modality in modality_features:
                projected = self.modality_projections[modality](modality_features[modality])
                projected_features[modality] = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
            else:
                # Handle missing modality
                projected_features[modality] = torch.zeros(
                    batch_size, 1, self.output_dim,
                    device=next(iter(modality_features.values())).device
                )
        
        # Apply cross-attention layers
        current_features = projected_features.copy()
        
        for layer in self.attention_layers:
            new_features = {}
            
            # Apply cross-attention between modalities
            for modality in self.modalities:
                modality_output = current_features[modality]
                
                # Attend to all other modalities
                attended_outputs = []
                for other_modality in self.modalities:
                    if other_modality != modality:
                        attention_name = f"{modality}_to_{other_modality}"
                        if attention_name in layer:
                            attended, _ = layer[attention_name](
                                current_features[modality],  # Query
                                current_features[other_modality],  # Key
                                current_features[other_modality]   # Value
                            )
                            attended_outputs.append(attended)
                
                # Combine attended outputs
                if attended_outputs:
                    combined_attended = torch.mean(torch.stack(attended_outputs), dim=0)
                    
                    # Residual connection
                    if self.use_residual:
                        modality_output = modality_output + combined_attended
                    else:
                        modality_output = combined_attended
                    
                    # Layer normalization
                    modality_output = layer["layer_norm"](modality_output)
                
                # Feed-forward network
                ffn_output = layer["ffn"](modality_output)
                
                # Residual connection
                if self.use_residual:
                    ffn_output = ffn_output + modality_output
                
                # Layer normalization
                new_features[modality] = layer["ffn_norm"](ffn_output)
            
            current_features = new_features
        
        # Final aggregation
        final_features = [features.squeeze(1) for features in current_features.values()]
        concatenated = torch.cat(final_features, dim=-1)
        
        fused = self.final_aggregation(concatenated)
        
        return fused


class SarcasmSpecificFusion(nn.Module):
    """Sarcasm-specific fusion that focuses on contradictory patterns."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        """
        Initialize sarcasm-specific fusion.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            output_dim: Output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.modalities = list(input_dims.keys())
        
        self.logger = get_logger("SarcasmSpecificFusion")
        
        # Project modalities to common dimension
        common_dim = output_dim // 2
        self.projections = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.projections[modality] = nn.Linear(input_dim, common_dim)
        
        # Contradiction detection between modalities
        self.contradiction_detectors = nn.ModuleDict()
        for i, mod1 in enumerate(self.modalities):
            for mod2 in self.modalities[i+1:]:
                detector_name = f"{mod1}_{mod2}_contradiction"
                self.contradiction_detectors[detector_name] = nn.Sequential(
                    nn.Linear(common_dim * 2, common_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(common_dim, 1),
                    nn.Sigmoid()
                )
        
        # Sarcasm pattern detector
        self.sarcasm_pattern_detector = nn.Sequential(
            nn.Linear(common_dim * len(self.modalities), output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        self.logger.info(f"Initialized sarcasm-specific fusion with {len(self.contradiction_detectors)} contradiction detectors")
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through sarcasm-specific fusion.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features with sarcasm-specific patterns
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project modalities
        projected = {}
        for modality in self.modalities:
            if modality in modality_features:
                projected[modality] = self.projections[modality](modality_features[modality])
            else:
                projected[modality] = torch.zeros(
                    batch_size, self.projections[modality].out_features,
                    device=next(iter(modality_features.values())).device
                )
        
        # Detect contradictions between modalities
        contradiction_scores = []
        for detector_name, detector in self.contradiction_detectors.items():
            mod1, mod2 = detector_name.replace('_contradiction', '').split('_')
            
            # Concatenate modality features
            combined = torch.cat([projected[mod1], projected[mod2]], dim=-1)
            
            # Detect contradiction
            contradiction_score = detector(combined)
            contradiction_scores.append(contradiction_score)
        
        # Weight modality features by contradiction scores
        if contradiction_scores:
            avg_contradiction = torch.mean(torch.cat(contradiction_scores, dim=-1), dim=-1, keepdim=True)
            
            # Apply contradiction weighting to modality features
            weighted_features = []
            for modality in self.modalities:
                weighted = projected[modality] * (1 + avg_contradiction)  # Amplify when contradictions are detected
                weighted_features.append(weighted)
        else:
            weighted_features = list(projected.values())
        
        # Combine all features
        combined_features = torch.cat(weighted_features, dim=-1)
        
        # Detect sarcasm patterns
        fused = self.sarcasm_pattern_detector(combined_features)
        
        return fused


class FusionFactory:
    """Factory for creating sarcasm-specific fusion layers."""
    
    @staticmethod
    def create_fusion_layer(
        fusion_type: str,
        input_dims: Dict[str, int],
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """
        Create fusion layer for sarcasm detection.
        
        Args:
            fusion_type: Type of fusion
            input_dims: Input dimensions
            output_dim: Output dimension
            **kwargs: Additional arguments
            
        Returns:
            Fusion layer
        """
        fusion_type = fusion_type.lower().replace('-', '_')
        
        if fusion_type in ['concat', 'concatenation']:
            return ConcatenationFusion(input_dims, output_dim, **kwargs)
        elif fusion_type in ['cross_attention', 'cross_modal_attention']:
            return CrossAttentionFusion(input_dims, output_dim, **kwargs)
        elif fusion_type in ['sarcasm_specific', 'contradiction_aware']:
            return SarcasmSpecificFusion(input_dims, output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    @staticmethod
    def get_available_fusion_types() -> List[str]:
        """Get list of available fusion types."""
        return ['concatenation', 'cross_attention', 'sarcasm_specific']
