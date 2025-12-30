"""
FactCheck-MM Shared Components
Shared backbone modules for multimodal fact-checking pipeline.
"""

from .base_model import BaseMultimodalModel
from .multimodal_encoder import MultimodalEncoder
from .fusion_layers import (
    FusionStrategy,
    ConcatenationFusion,
    SelfAttentionFusion,
    CrossModalAttentionFusion,
    FusionLayerFactory
)

# Preprocessing modules
from .preprocessing import (
    TextProcessor,
    AudioProcessor,
    ImageProcessor,
    VideoProcessor
)

# Utility modules
from .utils import (
    get_logger,
    setup_logging,
    CheckpointManager,
    MetricsComputer,
    Visualizer
)

# Dataset modules
from .datasets import (
    BaseDataset,
    MultimodalDataset,
    create_dataloader,
    ChunkedDataLoader,
    create_hardware_aware_dataloader
)

__version__ = "1.0.0"

__all__ = [
    # Core models
    "BaseMultimodalModel",
    "MultimodalEncoder",
    
    # Fusion strategies
    "FusionStrategy",
    "ConcatenationFusion", 
    "SelfAttentionFusion",
    "CrossModalAttentionFusion",
    "FusionLayerFactory",
    
    # Preprocessing
    "TextProcessor",
    "AudioProcessor", 
    "ImageProcessor",
    "VideoProcessor",
    
    # Utilities
    "get_logger",
    "setup_logging",
    "CheckpointManager",
    "MetricsComputer",
    "Visualizer",
    
    # Datasets
    "BaseDataset",
    "MultimodalDataset",
    "create_dataloader",
    "ChunkedDataLoader",
    "create_hardware_aware_dataloader"
]

def get_shared_config():
    """Get default shared configuration."""
    from Config import get_config
    return get_config()["base"]
