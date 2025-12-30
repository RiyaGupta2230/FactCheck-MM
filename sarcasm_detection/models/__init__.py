# sarcasm_detection/models/__init__.py
"""
Sarcasm Detection Model Architectures
Comprehensive model implementations for text and multimodal sarcasm detection.
"""

from .text_sarcasm_model import (
    TextSarcasmModel,
    RobertaSarcasmModel,
    LSTMSarcasmModel
)
from .multimodal_sarcasm import (
    MultimodalSarcasmModel,
    MultimodalSarcasmClassifier
)
from .fusion_strategies import (
    ConcatenationFusion,
    CrossAttentionFusion,
    FusionFactory
)
from .ensemble_model import (
    EnsembleSarcasmModel,
    VotingEnsemble,
    WeightedEnsemble,
    StackingEnsemble
)

__all__ = [
    # Text models
    "TextSarcasmModel",
    "RobertaSarcasmModel", 
    "LSTMSarcasmModel",
    
    # Multimodal models
    "MultimodalSarcasmModel",
    "MultimodalSarcasmClassifier",
    
    # Fusion strategies
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "FusionFactory",
    
    # Ensemble models
    "EnsembleSarcasmModel",
    "VotingEnsemble",
    "StackingEnsemble",
    "WeightedEnsemble"
]

def create_sarcasm_model(
    model_type: str,
    config: dict,
    num_classes: int = 2,
    **kwargs
):
    """
    Factory function to create sarcasm detection models.
    
    Args:
        model_type: Type of model ('roberta', 'lstm', 'multimodal', 'ensemble')
        config: Model configuration
        num_classes: Number of classes (default: 2 for binary sarcasm)
        **kwargs: Additional arguments
        
    Returns:
        Model instance
    """
    if model_type == "roberta":
        return RobertaSarcasmModel(config, num_classes, **kwargs)
    elif model_type == "lstm":
        return LSTMSarcasmModel(config, num_classes, **kwargs)
    elif model_type == "multimodal":
        return MultimodalSarcasmModel(config, num_classes, **kwargs)
    elif model_type == "ensemble":
        return EnsembleSarcasmModel(config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
