# sarcasm_detection/data/__init__.py
"""
Sarcasm Detection Dataset Loaders
Comprehensive dataset loaders for all 7 sarcasm datasets with multimodal support.
"""

from .mustard_loader import MustardDataset
from .mmsd2_loader import MMSD2Dataset
from .sarcnet_loader import SarcNetDataset
from .sarc_loader import SARCDataset
from .headlines_loader import HeadlinesDataset
from .spanish_loader import SpanishSarcasmDataset
from .funny_loader import URFunnyDataset
from .unified_loader import UnifiedSarcasmDataset

__all__ = [
    "MustardDataset",
    "MMSD2Dataset", 
    "SarcNetDataset",
    "SARCDataset",
    "HeadlinesDataset",
    "SpanishSarcasmDataset",
    "URFunnyDataset",
    "UnifiedSarcasmDataset"
]

# Dataset registry for easy access
SARCASM_DATASETS = {
    'mustard': MustardDataset,
    'mmsd2': MMSD2Dataset,
    'sarcnet': SarcNetDataset,
    'sarc': SARCDataset,
    'headlines': HeadlinesDataset,
    'spanish': SpanishSarcasmDataset,
    'ur_funny': URFunnyDataset
}

def create_sarcasm_dataset(
    dataset_name: str,
    data_dir: str,
    split: str = "train",
    **kwargs
):
    """
    Factory function to create sarcasm datasets.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Data directory path
        split: Dataset split
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if dataset_name not in SARCASM_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(SARCASM_DATASETS.keys())}")
    
    dataset_class = SARCASM_DATASETS[dataset_name]
    return dataset_class(data_dir=data_dir, split=split, **kwargs)
