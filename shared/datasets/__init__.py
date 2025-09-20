"""
FactCheck-MM Dataset Components
Unified dataset loaders for multimodal fact-checking pipeline.
"""

from .base_dataset import BaseDataset, DatasetConfig
from .multimodal_dataset import (
    MultimodalDataset,
    SarcasmDataset,
    ParaphraseDataset,
    FactVerificationDataset
)
from .data_loaders import (
    create_dataloader,
    ChunkedDataLoader,
    MultimodalCollator,
    create_chunked_dataloader,
    get_optimal_batch_size
)

__all__ = [
    # Base classes
    "BaseDataset",
    "DatasetConfig",
    
    # Multimodal datasets
    "MultimodalDataset",
    "SarcasmDataset", 
    "ParaphraseDataset",
    "FactVerificationDataset",
    
    # Data loaders
    "create_dataloader",
    "ChunkedDataLoader",
    "MultimodalCollator",
    "create_chunked_dataloader",
    "get_optimal_batch_size"
]

def create_dataset(
    task: str,
    dataset_names: List[str],
    data_dir: Path,
    split: str = "train",
    **kwargs
):
    """
    Factory function to create task-specific datasets.
    
    Args:
        task: Task name ('sarcasm_detection', 'paraphrasing', 'fact_verification')
        dataset_names: List of dataset names to use
        data_dir: Data directory path
        split: Dataset split
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if task == "sarcasm_detection":
        return SarcasmDataset(dataset_names, data_dir, split, **kwargs)
    elif task == "paraphrasing":
        return ParaphraseDataset(dataset_names, data_dir, split, **kwargs)
    elif task == "fact_verification":
        return FactVerificationDataset(dataset_names, data_dir, split, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
