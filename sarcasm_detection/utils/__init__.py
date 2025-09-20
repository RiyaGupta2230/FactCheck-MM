# sarcasm_detection/utils/__init__.py
"""
Sarcasm Detection Utilities
Specialized metrics, data augmentation, and helper functions for sarcasm detection.
"""

from .sarcasm_metrics import (
    SarcasmMetrics,
    SarcasmEvaluator,
    IronyDetectionMetrics,
    MultimodalSarcasmMetrics,
    ClassImbalanceMetrics
)

from .data_augmentation import (
    SarcasmDataAugmenter,
    TextAugmenter,
    AudioAugmenter,
    ImageAugmenter,
    VideoAugmenter,
    MultimodalAugmenter,
    ContextualAugmenter,
    BackTranslationAugmenter,
    SyntacticAugmenter
)

__all__ = [
    # Sarcasm-specific metrics
    "SarcasmMetrics",
    "SarcasmEvaluator", 
    "IronyDetectionMetrics",
    "MultimodalSarcasmMetrics",
    "ClassImbalanceMetrics",
    
    # Data augmentation
    "SarcasmDataAugmenter",
    "TextAugmenter",
    "AudioAugmenter", 
    "ImageAugmenter",
    "VideoAugmenter",
    "MultimodalAugmenter",
    "ContextualAugmenter",
    "BackTranslationAugmenter",
    "SyntacticAugmenter"
]

# Convenience functions for quick access
def compute_sarcasm_metrics(predictions, labels, probabilities=None):
    """Compute comprehensive sarcasm detection metrics."""
    metrics = SarcasmMetrics("sarcasm_detection", num_classes=2)
    return metrics.compute_classification_metrics(predictions, labels, probabilities)

def augment_sarcasm_data(dataset, augmentation_config):
    """Apply data augmentation to sarcasm dataset."""
    augmenter = SarcasmDataAugmenter(augmentation_config)
    return augmenter.augment_dataset(dataset)

def create_balanced_sarcasm_dataset(dataset, target_ratio=0.5):
    """Create balanced sarcasm dataset."""
    metrics = ClassImbalanceMetrics()
    return metrics.balance_dataset(dataset, target_ratio)
