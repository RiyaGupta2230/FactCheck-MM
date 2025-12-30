# sarcasm_detection/data/unified_loader.py

"""
Unified Sarcasm Dataset Loader
Combined wrapper to sample uniformly across all sarcasm datasets.
Research-grade balancing to ensure no single dataset dominates.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset

from .mustard_loader import MustardDataset
from .mmsd2_loader import MMSD2Dataset
from .sarcnet_loader import SarcNetDataset
from .sarc_loader import SARCDataset
from .headlines_loader import HeadlinesDataset
from shared.utils import get_logger


class UnifiedSarcasmDataset(Dataset):
    """
    Unified Sarcasm Dataset that combines all available sarcasm datasets.
    Implements research-grade balancing strategies to prevent task domination.
    
    Available datasets:
    - MUStARD: 690 samples (perfectly balanced, multimodal)
    - MMSD2: ~24k samples (text + image)
    - SarcNet: ~3.3k samples (text + image, multi-label)
    - SARC: 1.3M samples (capped to 50k for balance)
    - Headlines: ~28k samples (text-only)
    
    Sampling strategies:
    - 'uniform': Equal samples from each dataset
    - 'proportional': Samples proportional to (capped) dataset size
    - 'balanced': Balanced classes across all datasets
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_names: Optional[List[str]] = None,
        split: str = "train",
        sampling_strategy: str = "uniform",
        max_samples_per_dataset: Optional[int] = None,
        total_max_samples: Optional[int] = None,
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize Unified Sarcasm Dataset.
        
        Args:
            data_dir: Path to datasets directory
            dataset_names: List of dataset names to include
            split: Dataset split ('train', 'val', 'test')
            sampling_strategy: How to sample across datasets
            max_samples_per_dataset: Maximum samples per individual dataset
            total_max_samples: Maximum total samples
            processors: Dictionary of processors for each modality
            cache_data: Whether to cache processed data
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_names = dataset_names or [
            'mustard', 'mmsd2', 'sarcnet', 'sarc', 'sarcasm_headlines'
        ]
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.max_samples_per_dataset = max_samples_per_dataset
        self.total_max_samples = total_max_samples
        self.processors = processors
        self.cache_data = cache_data
        self.random_seed = random_seed
        self.logger = get_logger("UnifiedSarcasmDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Dataset registry (only 5 datasets as per specification)
        self.dataset_classes = {
            'mustard': MustardDataset,
            'mmsd2': MMSD2Dataset,
            'sarcnet': SarcNetDataset,
            'sarc': SARCDataset,
            'sarcasm_headlines': HeadlinesDataset
        }
        
        self.datasets = {}
        self.dataset_sizes = {}
        self._load_datasets()
        
        self.unified_indices = self._create_unified_indices()
        
        self.logger.info(
            f"Unified dataset initialized: {len(self.unified_indices)} total samples "
            f"from {len(self.datasets)} datasets (strategy: {sampling_strategy})"
        )
    
    def _load_datasets(self):
        """Load all requested datasets with correct paths."""
        for dataset_name in self.dataset_names:
            if dataset_name not in self.dataset_classes:
                self.logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            try:
                dataset_class = self.dataset_classes[dataset_name]
                
                # Resolve correct paths as per PDF specification
                if dataset_name == 'mustard':
                    dataset_path = self.data_dir / 'mustard_repo'
                elif dataset_name == 'sarcnet':
                    dataset_path = self.data_dir / 'sarcnet' / 'SarcNet Image-Text'
                elif dataset_name == 'sarcasm_headlines':
                    dataset_path = self.data_dir / 'Sarcasm Headlines'
                else:
                    dataset_path = self.data_dir / dataset_name
                
                dataset = dataset_class(
                    data_dir=dataset_path,
                    split=self.split,
                    max_samples=self.max_samples_per_dataset,
                    processors=self.processors,
                    cache_data=self.cache_data,
                    random_seed=self.random_seed
                )
                
                if len(dataset) > 0:
                    self.datasets[dataset_name] = dataset
                    self.dataset_sizes[dataset_name] = len(dataset)
                    self.logger.info(f"Loaded {dataset_name}: {len(dataset)} samples")
                else:
                    self.logger.warning(f"Dataset {dataset_name} is empty")
            
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                raise
    
    def _create_unified_indices(self) -> List[Tuple[str, int]]:
        """Create unified indices for sampling across datasets with research-grade balancing."""
        if not self.datasets:
            return []
        
        unified_indices = []
        
        if self.sampling_strategy == "uniform":
            # Equal samples from each dataset (prevents domination)
            min_size = min(self.dataset_sizes.values())
            samples_per_dataset = min_size
            
            self.logger.info(f"Uniform sampling: {samples_per_dataset} samples per dataset")
            
            for dataset_name in self.datasets.keys():
                dataset_indices = np.random.choice(
                    self.dataset_sizes[dataset_name],
                    size=samples_per_dataset,
                    replace=False
                )
                for idx in dataset_indices:
                    unified_indices.append((dataset_name, int(idx)))
        
        elif self.sampling_strategy == "proportional":
            # Samples proportional to dataset size (after capping)
            for dataset_name, dataset_size in self.dataset_sizes.items():
                for idx in range(dataset_size):
                    unified_indices.append((dataset_name, idx))
        
        elif self.sampling_strategy == "balanced":
            # Balance classes across all datasets
            sarcastic_indices = []
            non_sarcastic_indices = []
            
            for dataset_name, dataset in self.datasets.items():
                for idx in range(len(dataset)):
                    sample = dataset[idx]
                    label = sample['label']
                    
                    if torch.is_tensor(label):
                        label = label.item()
                    
                    if label == 1:
                        sarcastic_indices.append((dataset_name, idx))
                    else:
                        non_sarcastic_indices.append((dataset_name, idx))
            
            # Balance to minority class
            min_class_size = min(len(sarcastic_indices), len(non_sarcastic_indices))
            np.random.shuffle(sarcastic_indices)
            np.random.shuffle(non_sarcastic_indices)
            
            unified_indices = (
                sarcastic_indices[:min_class_size] +
                non_sarcastic_indices[:min_class_size]
            )
            
            self.logger.info(
                f"Balanced sampling: {min_class_size} per class, "
                f"total {len(unified_indices)} samples"
            )
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Shuffle unified indices
        np.random.shuffle(unified_indices)
        
        # Apply total max samples limit
        if self.total_max_samples and len(unified_indices) > self.total_max_samples:
            unified_indices = unified_indices[:self.total_max_samples]
            self.logger.info(f"Capped total samples to {self.total_max_samples}")
        
        return unified_indices
    
    def __len__(self) -> int:
        return len(self.unified_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.unified_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        dataset_name, dataset_idx = self.unified_indices[idx]
        dataset = self.datasets[dataset_name]
        
        # Get sample from underlying dataset
        sample = dataset[dataset_idx]
        
        # Add unified metadata (do not mutate original sample structure)
        if 'metadata' not in sample:
            sample['metadata'] = {}
        
        sample['metadata']['unified_idx'] = idx
        sample['metadata']['source_dataset'] = dataset_name
        sample['metadata']['source_idx'] = dataset_idx
        
        return sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        total_samples = len(self.unified_indices)
        
        sarcastic_count = 0
        non_sarcastic_count = 0
        modality_counts = {'text': 0, 'audio': 0, 'image': 0, 'video': 0}
        dataset_distribution = {name: 0 for name in self.datasets.keys()}
        
        for dataset_name, dataset_idx in self.unified_indices:
            sample = self.datasets[dataset_name][dataset_idx]
            
            # Count labels
            label = sample['label']
            if torch.is_tensor(label):
                label = label.item()
            
            if label == 1:
                sarcastic_count += 1
            else:
                non_sarcastic_count += 1
            
            # Count modalities (detect presence, not assume keys)
            if sample.get('text') is not None:
                modality_counts['text'] += 1
            if sample.get('audio') is not None:
                modality_counts['audio'] += 1
            if sample.get('image') is not None:
                modality_counts['image'] += 1
            if sample.get('video') is not None:
                modality_counts['video'] += 1
            
            # Dataset distribution
            dataset_distribution[dataset_name] += 1
        
        # Individual dataset statistics
        dataset_stats = {}
        for dataset_name, dataset in self.datasets.items():
            dataset_stats[dataset_name] = dataset.get_statistics()
        
        return {
            'total_samples': total_samples,
            'sarcastic_samples': sarcastic_count,
            'non_sarcastic_samples': non_sarcastic_count,
            'class_balance': sarcastic_count / total_samples if total_samples > 0 else 0,
            'sampling_strategy': self.sampling_strategy,
            'dataset_distribution': dataset_distribution,
            'modality_coverage': modality_counts,
            'available_datasets': list(self.datasets.keys()),
            'dataset_sizes': self.dataset_sizes,
            'individual_stats': dataset_stats
        }
    
    def get_dataset_by_name(self, dataset_name: str) -> Optional[Dataset]:
        """Get individual dataset by name."""
        return self.datasets.get(dataset_name)
    
    def filter_by_modalities(self, required_modalities: List[str]) -> List[int]:
        """Get indices of samples that have all required modalities."""
        filtered_indices = []
        
        for idx, (dataset_name, dataset_idx) in enumerate(self.unified_indices):
            sample = self.datasets[dataset_name][dataset_idx]
            has_all_modalities = True
            
            for modality in required_modalities:
                if sample.get(modality) is None:
                    has_all_modalities = False
                    break
            
            if has_all_modalities:
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def create_modality_subset(self, required_modalities: List[str]) -> 'UnifiedSarcasmDataset':
        """Create a subset with only samples that have required modalities."""
        filtered_indices = self.filter_by_modalities(required_modalities)
        
        new_dataset = UnifiedSarcasmDataset.__new__(UnifiedSarcasmDataset)
        new_dataset.data_dir = self.data_dir
        new_dataset.dataset_names = self.dataset_names
        new_dataset.split = self.split
        new_dataset.sampling_strategy = self.sampling_strategy
        new_dataset.datasets = self.datasets
        new_dataset.dataset_sizes = self.dataset_sizes
        new_dataset.logger = self.logger
        new_dataset.unified_indices = [self.unified_indices[i] for i in filtered_indices]
        
        return new_dataset
