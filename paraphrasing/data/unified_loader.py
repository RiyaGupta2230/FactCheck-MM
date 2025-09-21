#!/usr/bin/env python3
"""
Unified Paraphrasing Dataset Loader

Provides a unified interface for loading and combining multiple paraphrasing datasets
including ParaNMT-5M, MRPC, and Quora with standardized output format.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import logging
from collections import Counter

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .paranmt_loader import ParaNMTLoader, ParaNMTConfig
from .mrpc_loader import MRPCDataset, MRPCConfig
from .quora_loader import QuoraDataset, QuoraConfig
from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class UnifiedParaphraseConfig:
    """Configuration for unified paraphrase dataset loading."""
    
    # Dataset selection flags
    use_paranmt: bool = True
    use_mrpc: bool = True
    use_quora: bool = True
    
    # Dataset-specific configurations
    paranmt_config: Optional[ParaNMTConfig] = field(default_factory=lambda: ParaNMTConfig())
    mrpc_config: Optional[MRPCConfig] = field(default_factory=lambda: MRPCConfig())
    quora_config: Optional[QuoraConfig] = field(default_factory=lambda: QuoraConfig())
    
    # Unified processing parameters
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    
    # Dataset balancing
    balance_datasets: bool = True
    dataset_weights: Optional[Dict[str, float]] = None
    
    # Sampling configuration
    max_samples_per_dataset: Optional[Dict[str, int]] = None
    total_max_samples: Optional[int] = None
    
    # Output standardization
    standardize_outputs: bool = True
    include_dataset_id: bool = True
    
    # Task configuration
    task_type: str = "classification"  # "classification" or "generation"


class UnifiedParaphraseDataset(Dataset):
    """
    Unified dataset combining multiple paraphrasing datasets.
    
    Provides standardized interface and output format for training
    paraphrase models across different datasets.
    """
    
    def __init__(
        self,
        config: UnifiedParaphraseConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize unified paraphrase dataset.
        
        Args:
            config: Unified dataset configuration
            split: Data split (train/val/test)
            transform: Optional data transformation function
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Setup logging
        self.logger = get_logger("UnifiedParaphraseDataset")
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length
        )
        
        # Load individual datasets
        self.datasets = {}
        self.dataset_indices = {}
        self._load_datasets()
        
        # Create unified mapping
        self._create_unified_mapping()
        
        # Setup balanced sampling if needed
        self.sampler = None
        if config.balance_datasets and split == "train":
            self.sampler = self._create_balanced_sampler()
        
        total_samples = sum(len(dataset) for dataset in self.datasets.values())
        self.logger.info(f"Created unified dataset with {total_samples} total samples from {len(self.datasets)} datasets")
    
    def _load_datasets(self):
        """Load individual datasets based on configuration."""
        
        dataset_configs = [
            ("paranmt", self.config.use_paranmt, self.config.paranmt_config, ParaNMTLoader),
            ("mrpc", self.config.use_mrpc, self.config.mrpc_config, MRPCDataset),
            ("quora", self.config.use_quora, self.config.quora_config, QuoraDataset)
        ]
        
        for name, use_dataset, dataset_config, dataset_class in dataset_configs:
            if not use_dataset:
                continue
            
            try:
                # Apply max samples limit if specified
                if (self.config.max_samples_per_dataset and 
                    name in self.config.max_samples_per_dataset):
                    max_samples = self.config.max_samples_per_dataset[name]
                    
                    # Update dataset config
                    if hasattr(dataset_config, 'max_samples'):
                        dataset_config.max_samples = max_samples
                
                # Special handling for ParaNMT split (no official splits)
                if name == "paranmt" and self.split in ["val", "test"]:
                    # Skip ParaNMT for val/test or use a portion
                    if self.split == "val":
                        dataset_config.max_samples = min(1000, dataset_config.max_samples or 1000)
                    else:  # test
                        continue
                
                dataset = dataset_class(dataset_config, self.split)
                self.datasets[name] = dataset
                
                self.logger.info(f"Loaded {name} dataset: {len(dataset)} samples")
                
            except Exception as e:
                self.logger.warning(f"Failed to load {name} dataset: {e}")
                continue
    
    def _create_unified_mapping(self):
        """Create mapping from unified indices to dataset indices."""
        
        current_index = 0
        
        for dataset_name, dataset in self.datasets.items():
            dataset_size = len(dataset)
            self.dataset_indices[dataset_name] = {
                'start': current_index,
                'end': current_index + dataset_size,
                'size': dataset_size
            }
            current_index += dataset_size
        
        self.total_size = current_index
        
        # Apply total sample limit
        if self.config.total_max_samples and self.total_size > self.config.total_max_samples:
            # Calculate proportional limits for each dataset
            scale_factor = self.config.total_max_samples / self.total_size
            
            new_total = 0
            for dataset_name, indices in self.dataset_indices.items():
                new_size = int(indices['size'] * scale_factor)
                self.dataset_indices[dataset_name]['size'] = new_size
                self.dataset_indices[dataset_name]['end'] = new_total + new_size
                new_total += new_size
            
            self.total_size = new_total
            self.logger.info(f"Applied total sample limit: {self.total_size} samples")
    
    def _create_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Create balanced sampler across datasets."""
        
        if not self.config.balance_datasets:
            return None
        
        # Calculate dataset weights
        if self.config.dataset_weights:
            weights = self.config.dataset_weights
        else:
            # Equal weight to each dataset regardless of size
            num_datasets = len(self.datasets)
            weights = {name: 1.0/num_datasets for name in self.datasets.keys()}
        
        # Create sample weights
        sample_weights = []
        
        for dataset_name, dataset in self.datasets.items():
            dataset_weight = weights.get(dataset_name, 1.0)
            dataset_size = self.dataset_indices[dataset_name]['size']
            
            # Weight each sample in the dataset
            sample_weight = dataset_weight / dataset_size
            sample_weights.extend([sample_weight] * dataset_size)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights[:self.total_size],
            num_samples=self.total_size,
            replacement=True
        )
        
        self.logger.info(f"Created balanced sampler with dataset weights: {weights}")
        
        return sampler
    
    def _find_dataset_and_index(self, unified_idx: int) -> Tuple[str, int]:
        """Find which dataset and local index for a unified index."""
        
        for dataset_name, indices in self.dataset_indices.items():
            if indices['start'] <= unified_idx < indices['end']:
                local_idx = unified_idx - indices['start']
                
                # Ensure we don't exceed the actual dataset size
                actual_size = min(indices['size'], len(self.datasets[dataset_name]))
                if local_idx >= actual_size:
                    local_idx = local_idx % actual_size
                
                return dataset_name, local_idx
        
        raise IndexError(f"Unified index {unified_idx} out of range")
    
    def __len__(self) -> int:
        """Return total dataset size."""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the unified dataset.
        
        Args:
            idx: Unified sample index
            
        Returns:
            Dictionary containing standardized sample data
        """
        if idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        # Find source dataset and local index
        dataset_name, local_idx = self._find_dataset_and_index(idx)
        dataset = self.datasets[dataset_name]
        
        # Get sample from source dataset
        sample = dataset[local_idx]
        
        # Standardize output format
        if self.config.standardize_outputs:
            sample = self._standardize_sample(sample, dataset_name)
        
        # Add dataset identifier if requested
        if self.config.include_dataset_id:
            dataset_id = list(self.datasets.keys()).index(dataset_name)
            sample['dataset_id'] = torch.tensor(dataset_id, dtype=torch.long)
            sample['dataset_name'] = dataset_name
        
        # Apply optional transform
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _standardize_sample(self, sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Standardize sample format across datasets."""
        
        # Standard output format
        standardized = {
            'input_ids': sample.get('input_ids'),
            'attention_mask': sample.get('attention_mask'),
            'labels': sample.get('labels', torch.tensor(1, dtype=torch.long)),  # Default to positive
        }
        
        # Add token type ids if available
        if 'token_type_ids' in sample:
            standardized['token_type_ids'] = sample['token_type_ids']
        
        # Standardize text fields
        if dataset_name == "paranmt":
            standardized['text1'] = sample.get('reference_text', '')
            standardized['text2'] = sample.get('paraphrase_text', '')
        elif dataset_name == "mrpc":
            standardized['text1'] = sample.get('sentence1_text', '')
            standardized['text2'] = sample.get('sentence2_text', '')
        elif dataset_name == "quora":
            standardized['text1'] = sample.get('question1_text', '')
            standardized['text2'] = sample.get('question2_text', '')
        
        # Add generation targets if needed
        if self.config.task_type == "generation":
            standardized['target_ids'] = sample.get('target_ids', standardized.get('input_ids'))
        
        # Add metadata
        standardized['sample_id'] = sample.get('sample_id', 0)
        
        # Copy additional fields
        additional_fields = ['quality_score', 'score']
        for field in additional_fields:
            if field in sample:
                standardized[field] = sample[field]
        
        return standardized
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about constituent datasets."""
        
        info = {}
        
        for dataset_name, dataset in self.datasets.items():
            indices = self.dataset_indices[dataset_name]
            
            info[dataset_name] = {
                'size': len(dataset),
                'unified_start': indices['start'],
                'unified_end': indices['end'],
                'unified_size': indices['size']
            }
            
            # Add dataset-specific info
            if hasattr(dataset, 'get_label_distribution'):
                info[dataset_name]['label_distribution'] = dataset.get_label_distribution()
        
        return info
    
    def get_text_pairs_by_dataset(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get text pairs organized by dataset."""
        
        pairs_by_dataset = {}
        
        for dataset_name, dataset in self.datasets.items():
            if hasattr(dataset, 'get_text_pairs'):
                pairs_by_dataset[dataset_name] = dataset.get_text_pairs()
        
        return pairs_by_dataset
    
    def get_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get the balanced sampler if created."""
        return self.sampler


def create_unified_dataloader(
    config: UnifiedParaphraseConfig,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader for unified paraphrase dataset.
    
    Args:
        config: Unified dataset configuration
        split: Data split
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        PyTorch DataLoader
    """
    dataset = UnifiedParaphraseDataset(config, split)
    
    # Use balanced sampler for training if available
    sampler = dataset.get_balanced_sampler() if split == "train" else None
    if sampler:
        shuffle = False  # Don't shuffle when using sampler
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_unified_batch,
        **kwargs
    )


def _collate_unified_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for unified paraphrase batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Initialize batch dictionary
    batched = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'text1': [],
        'text2': []
    }
    
    # Add optional fields
    optional_fields = ['token_type_ids', 'target_ids', 'dataset_id', 'dataset_name', 'quality_score']
    for field in optional_fields:
        if any(field in sample for sample in batch):
            batched[field] = []
    
    # Collect all samples
    for sample in batch:
        for key in batched.keys():
            if key in sample:
                batched[key].append(sample[key])
            elif key in ['text1', 'text2']:
                batched[key].append('')  # Default empty string
    
    # Convert to tensors where appropriate
    tensor_keys = ['input_ids', 'attention_mask', 'labels']
    tensor_keys.extend([k for k in optional_fields if k in batched and k not in ['dataset_name']])
    
    for key in tensor_keys:
        if key in batched and batched[key] and torch.is_tensor(batched[key][0]):
            batched[key] = torch.stack(batched[key])
    
    return batched


def main():
    """Example usage of unified paraphrase loader."""
    
    # Configuration with selective dataset loading
    config = UnifiedParaphraseConfig(
        use_paranmt=True,
        use_mrpc=True,
        use_quora=True,
        balance_datasets=True,
        max_samples_per_dataset={
            'paranmt': 1000,
            'mrpc': None,  # Use all available
            'quora': 500
        },
        total_max_samples=2000
    )
    
    # Update individual dataset configs for testing
    config.paranmt_config.max_samples = 1000
    config.paranmt_config.use_chunked_loading = False
    config.quora_config.max_samples = 500
    
    # Test different splits
    for split in ["train", "val"]:
        try:
            dataset = UnifiedParaphraseDataset(config, split)
            print(f"\n{split.upper()} Split:")
            print(f"  Total dataset size: {len(dataset)}")
            
            # Show dataset composition
            dataset_info = dataset.get_dataset_info()
            print("  Dataset composition:")
            for name, info in dataset_info.items():
                print(f"    {name}: {info['unified_size']} samples")
            
            # Test sample access
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Text 1: {sample['text1'][:50]}...")
                print(f"  Text 2: {sample['text2'][:50]}...")
                print(f"  Labels: {sample['labels']}")
                if 'dataset_name' in sample:
                    print(f"  Dataset: {sample['dataset_name']}")
            
        except Exception as e:
            print(f"\n{split.upper()} Split: Error - {e}")
    
    # Create dataloader with balanced sampling
    try:
        dataloader = create_unified_dataloader(config, "train", batch_size=4)
        
        # Test batch loading
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            if 'dataset_id' in batch:
                unique_datasets = torch.unique(batch['dataset_id'])
                print(f"  Datasets in batch: {unique_datasets.tolist()}")
            break
            
    except Exception as e:
        print(f"DataLoader test failed: {e}")


if __name__ == "__main__":
    main()
