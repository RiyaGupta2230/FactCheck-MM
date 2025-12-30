"""
Hardware-Aware Data Loaders for FactCheck-MM
Chunked loading for MacBook Air M2, full GPU mode for RTX 2050.
"""

import torch
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import gc

from .multimodal_dataset import MultimodalDataset, ChunkedDataset
from ..utils import get_logger


CLASSIFICATION_TASKS = ['sarcasm_detection', 'fact_verification', 'classification']
GENERATION_TASKS = ['paraphrasing', 'generation', 'similarity']


class MultimodalCollator:
    """
    Collate function for multimodal data with proper padding and tensor conversion.
    """
    
    def __init__(
        self,
        text_processor: Optional[Any] = None,
        pad_token_id: int = 0,
        max_length: int = 512,
        return_attention_mask: bool = True
    ):
        """
        Initialize multimodal collator.
        
        Args:
            text_processor: Text processor for tokenization
            pad_token_id: Padding token ID
            max_length: Maximum sequence length
            return_attention_mask: Whether to return attention masks
        """
        self.text_processor = text_processor
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
        self.logger = get_logger("MultimodalCollator")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of multimodal samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch tensors
        """
        collated = {}
        batch_size = len(batch)
        
        if 'text' in batch[0] or 'text_processed' in batch[0]:
            text_data = []
            for sample in batch:
                text = sample.get('text_processed', sample.get('text', ''))
                text_data.append(text)
            
            if self.text_processor and text_data[0]:
                tokenized = self.text_processor.tokenize(
                    text_data,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                collated.update(tokenized)
            else:
                collated['input_ids'] = torch.zeros(batch_size, self.max_length, dtype=torch.long)
                if self.return_attention_mask:
                    collated['attention_mask'] = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        
        if 'audio_processed' in batch[0]:
            audio_features = []
            for sample in batch:
                audio_data = sample['audio_processed']
                if isinstance(audio_data, dict) and 'audio' in audio_data:
                    audio_features.append(torch.tensor(audio_data['audio']))
                elif isinstance(audio_data, np.ndarray):
                    audio_features.append(torch.tensor(audio_data))
                else:
                    audio_features.append(torch.zeros(16000))
            
            collated['audio_features'] = pad_sequence(audio_features, batch_first=True)
        
        if 'image_processed' in batch[0]:
            image_tensors = []
            for sample in batch:
                image_data = sample['image_processed']
                if isinstance(image_data, dict) and 'images' in image_data:
                    image = image_data['images'][0]
                    if hasattr(image, 'convert'):
                        image_array = np.array(image.convert('RGB'))
                        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
                    else:
                        image_tensor = torch.tensor(image).float()
                    image_tensors.append(image_tensor)
                else:
                    image_tensors.append(torch.zeros(3, 224, 224))
            
            collated['image_features'] = torch.stack(image_tensors)
        
        if 'video_processed' in batch[0]:
            video_tensors = []
            for sample in batch:
                video_data = sample['video_processed']
                if isinstance(video_data, dict) and 'frames' in video_data:
                    frames = video_data['frames']
                    if isinstance(frames, np.ndarray):
                        video_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
                    else:
                        video_tensor = torch.tensor(frames).float()
                    video_tensors.append(video_tensor)
                else:
                    video_tensors.append(torch.zeros(16, 3, 224, 224))
            
            collated['video_features'] = pad_sequence(video_tensors, batch_first=True)
        
        if 'label' in batch[0]:
            labels = []
            for sample in batch:
                label = sample['label']
                if isinstance(label, str):
                    if label.upper() in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                        label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
                        labels.append(label_map[label.upper()])
                    else:
                        labels.append(0)
                else:
                    labels.append(int(label))
            
            collated['labels'] = torch.tensor(labels, dtype=torch.long)
        
        if 'target' in batch[0]:
            targets = [sample['target'] for sample in batch]
            if self.text_processor:
                target_tokenized = self.text_processor.tokenize(
                    targets,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                collated['target_input_ids'] = target_tokenized['input_ids']
                if 'attention_mask' in target_tokenized:
                    collated['target_attention_mask'] = target_tokenized['attention_mask']
        
        if 'evidence' in batch[0]:
            evidence_batch = []
            for sample in batch:
                evidence = sample.get('evidence', [])
                if isinstance(evidence, list):
                    evidence_text = ' '.join(evidence[:5])
                else:
                    evidence_text = str(evidence)
                evidence_batch.append(evidence_text)
            
            if self.text_processor and any(evidence_batch):
                evidence_tokenized = self.text_processor.tokenize(
                    evidence_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                collated['evidence_input_ids'] = evidence_tokenized['input_ids']
                if 'attention_mask' in evidence_tokenized:
                    collated['evidence_attention_mask'] = evidence_tokenized['attention_mask']
        
        collated['batch_size'] = torch.tensor(batch_size)
        return collated


class DatasetProportionalSampler(Sampler):
    """
    Sampler for proportional sampling across multiple concatenated datasets.
    """
    
    def __init__(
        self,
        dataset_lengths: List[int],
        sampling_strategy: str = "proportional",
        num_samples: Optional[int] = None,
        replacement: bool = False
    ):
        """
        Initialize dataset proportional sampler.
        
        Args:
            dataset_lengths: List of lengths for each dataset
            sampling_strategy: "proportional", "uniform_per_dataset", "inverse_size", "cap_max_samples"
            num_samples: Total number of samples to draw
            replacement: Whether to sample with replacement
        """
        self.dataset_lengths = dataset_lengths
        self.sampling_strategy = sampling_strategy
        self.num_samples = num_samples or sum(dataset_lengths)
        self.replacement = replacement
        self.logger = get_logger("DatasetProportionalSampler")
        
        self.dataset_boundaries = []
        cumsum = 0
        for length in dataset_lengths:
            self.dataset_boundaries.append((cumsum, cumsum + length))
            cumsum += length
        self.total_length = cumsum
        
        self.weights = self._calculate_weights()
        self.logger.info(
            f"Initialized sampler with strategy '{sampling_strategy}', "
            f"{len(dataset_lengths)} datasets, {self.total_length} total samples"
        )
        
        self._log_sampling_distribution()
    
    def _calculate_weights(self) -> torch.Tensor:
        """Calculate sampling weights based on strategy."""
        weights = torch.zeros(self.total_length)
        
        if self.sampling_strategy == "proportional":
            for i, (start, end) in enumerate(self.dataset_boundaries):
                weight = 1.0
                weights[start:end] = weight
        
        elif self.sampling_strategy == "uniform_per_dataset":
            for i, (start, end) in enumerate(self.dataset_boundaries):
                dataset_size = end - start
                weight = 1.0 / dataset_size
                weights[start:end] = weight
        
        elif self.sampling_strategy == "inverse_size":
            total_size = sum(self.dataset_lengths)
            for i, (start, end) in enumerate(self.dataset_boundaries):
                dataset_size = end - start
                weight = total_size / (dataset_size * len(self.dataset_lengths))
                weights[start:end] = weight
        
        elif self.sampling_strategy == "cap_max_samples":
            max_samples_per_dataset = 10000
            for i, (start, end) in enumerate(self.dataset_boundaries):
                dataset_size = end - start
                if dataset_size > max_samples_per_dataset:
                    weight = max_samples_per_dataset / dataset_size
                else:
                    weight = 1.0
                weights[start:end] = weight
        
        weights = weights / weights.sum()
        return weights
    
    def _log_sampling_distribution(self):
        """Log the expected sampling distribution."""
        self.logger.info("=== Expected Sampling Distribution ===")
        for i, (start, end) in enumerate(self.dataset_boundaries):
            dataset_weight = self.weights[start:end].sum().item()
            expected_samples = int(dataset_weight * self.num_samples)
            self.logger.info(
                f"  Dataset {i}: {expected_samples} samples ({dataset_weight*100:.2f}%)"
            )
        self.logger.info("=" * 40)
    
    def __iter__(self):
        """Generate sample indices."""
        if self.replacement:
            indices = torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True
            )
        else:
            all_indices = []
            for start, end in self.dataset_boundaries:
                dataset_indices = torch.randperm(end - start) + start
                all_indices.append(dataset_indices)
            
            if self.sampling_strategy == "proportional":
                indices = torch.cat(all_indices)
                if len(indices) > self.num_samples:
                    indices = indices[:self.num_samples]
            else:
                all_indices = torch.cat(all_indices)
                indices = torch.multinomial(
                    self.weights,
                    min(self.num_samples, len(all_indices)),
                    replacement=False
                )
        
        return iter(indices.tolist())
    
    def __len__(self):
        """Return number of samples."""
        return self.num_samples


class BalancedClassSampler(Sampler):
    """
    Sampler for balanced class sampling using WeightedRandomSampler.
    ONLY for classification tasks.
    """
    
    def __init__(
        self,
        labels: List[int],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        task_name: str = "classification"
    ):
        """
        Initialize balanced class sampler.
        
        Args:
            labels: List of labels for all samples
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            task_name: Task name for validation
        """
        self.labels = labels
        self.num_samples = num_samples or len(labels)
        self.replacement = replacement
        self.task_name = task_name
        self.logger = get_logger("BalancedClassSampler")
        
        if task_name not in CLASSIFICATION_TASKS:
            self.logger.error(
                f"BalancedClassSampler should ONLY be used for classification tasks, "
                f"but got task '{task_name}'. This may cause training issues."
            )
            raise ValueError(
                f"Cannot apply BalancedClassSampler to task '{task_name}'. "
                f"Only valid for: {CLASSIFICATION_TASKS}"
            )
        
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        
        self.weights = torch.DoubleTensor([class_weights[label] for label in labels])
        
        self.logger.info(
            f"Initialized balanced sampler for task '{task_name}': "
            f"{len(np.unique(labels))} classes, distribution: {class_counts.tolist()}"
        )
    
    def __iter__(self):
        """Generate sample indices."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement
        )
        return iter(indices.tolist())
    
    def __len__(self):
        """Return number of samples."""
        return self.num_samples


class ChunkedDataLoader:
    """
    Memory-efficient data loader for chunked training on resource-constrained devices.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        chunk_size: int = 1000,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_memory_gb: float = 7.0,
        sampler: Optional[Sampler] = None
    ):
        """
        Initialize chunked data loader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            chunk_size: Size of each chunk
            shuffle: Whether to shuffle
            collate_fn: Collate function
            drop_last: Whether to drop last incomplete batch
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            max_memory_gb: Maximum memory usage in GB
            sampler: Optional sampler for balanced sampling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle if sampler is None else False
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_workers = min(num_workers, 2)
        self.pin_memory = pin_memory
        self.max_memory_gb = max_memory_gb
        self.sampler = sampler
        self.logger = get_logger("ChunkedDataLoader")
        
        self.chunked_dataset = ChunkedDataset(
            dataset,
            chunk_size=chunk_size,
            shuffle_chunks=self.shuffle
        )
        
        self.current_chunk_idx = 0
        self.current_dataloader = None
        
        self.logger.info(
            f"Initialized chunked dataloader: {len(dataset)} samples, "
            f"{self.chunked_dataset.num_chunks} chunks, batch_size={batch_size}"
        )
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        memory_info = psutil.virtual_memory()
        memory_used_gb = (memory_info.total - memory_info.available) / (1024**3)
        return memory_used_gb < self.max_memory_gb
    
    def _create_chunk_dataloader(self, chunk_data: List[Any]) -> DataLoader:
        """Create DataLoader for a chunk."""
        class ChunkDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        chunk_dataset = ChunkDataset(chunk_data)
        
        return DataLoader(
            chunk_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def __iter__(self):
        """Iterate over chunks and batches."""
        for chunk_idx, chunk_data in self.chunked_dataset:
            if not self._check_memory():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            chunk_dataloader = self._create_chunk_dataloader(chunk_data)
            self.logger.debug(f"Processing chunk {chunk_idx}/{self.chunked_dataset.num_chunks}")
            
            for batch in chunk_dataloader:
                yield batch
            
            del chunk_dataloader, chunk_data
            gc.collect()
    
    def __len__(self) -> int:
        """Get total number of batches."""
        total_samples = len(self.dataset)
        batches_per_chunk = (self.chunk_size + self.batch_size - 1) // self.batch_size
        return self.chunked_dataset.num_chunks * batches_per_chunk


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    sampler: Optional[Sampler] = None,
    balance_classes: bool = False,
    task_name: str = "classification"
) -> DataLoader:
    """
    Create standard PyTorch DataLoader.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        collate_fn: Collate function
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last batch
        persistent_workers: Whether to keep workers persistent
        sampler: Optional custom sampler
        balance_classes: Whether to use balanced class sampling
        task_name: Task name for validation
        
    Returns:
        DataLoader instance
    """
    logger = get_logger("create_dataloader")
    
    if balance_classes and sampler is None:
        if task_name in CLASSIFICATION_TASKS:
            try:
                labels = [dataset[i]['label'] for i in range(len(dataset))]
                sampler = BalancedClassSampler(labels, task_name=task_name)
                shuffle = False
                logger.info(f"Created BalancedClassSampler for task '{task_name}'")
            except Exception as e:
                logger.warning(f"Could not create balanced sampler: {e}")
        else:
            logger.warning(
                f"Ignoring balance_classes=True for task '{task_name}'. "
                f"Class balancing only applies to: {CLASSIFICATION_TASKS}"
            )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        sampler=sampler
    )


def create_chunked_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    chunk_size: int = 1000,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    max_memory_gb: float = 7.0,
    sampler: Optional[Sampler] = None,
    **kwargs
) -> ChunkedDataLoader:
    """
    Create chunked DataLoader for memory-constrained environments.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        chunk_size: Chunk size
        shuffle: Whether to shuffle
        collate_fn: Collate function
        max_memory_gb: Memory limit
        sampler: Optional custom sampler
        **kwargs: Additional arguments
        
    Returns:
        ChunkedDataLoader instance
    """
    return ChunkedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        chunk_size=chunk_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        max_memory_gb=max_memory_gb,
        sampler=sampler,
        **kwargs
    )


def create_multi_dataset_loader(
    datasets: List[Dataset],
    batch_size: int = 16,
    sampling_strategy: str = "proportional",
    collate_fn: Optional[Callable] = None,
    task_name: str = "classification",
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for multiple concatenated datasets with proportional sampling.
    DatasetProportionalSampler is the DEFAULT for multi-dataset loaders.
    
    Args:
        datasets: List of datasets to concatenate
        batch_size: Batch size
        sampling_strategy: "proportional", "uniform_per_dataset", "inverse_size", "cap_max_samples"
        collate_fn: Collate function
        task_name: Task name for logging
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance with custom sampler
    """
    from torch.utils.data import ConcatDataset
    
    logger = get_logger("create_multi_dataset_loader")
    
    concat_dataset = ConcatDataset(datasets)
    dataset_lengths = [len(d) for d in datasets]
    
    logger.info(
        f"Creating multi-dataset loader for task '{task_name}': "
        f"{len(datasets)} datasets, lengths: {dataset_lengths}, "
        f"strategy: '{sampling_strategy}'"
    )
    
    sampler = DatasetProportionalSampler(
        dataset_lengths=dataset_lengths,
        sampling_strategy=sampling_strategy
    )
    
    return create_dataloader(
        dataset=concat_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=False,
        task_name=task_name,
        **kwargs
    )


def create_hardware_aware_dataloader(
    dataset: Dataset,
    device_type: str = "auto",
    batch_size: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    balance_classes: bool = False,
    sampling_strategy: Optional[str] = None,
    task_name: str = "classification",
    **kwargs
) -> Union[DataLoader, ChunkedDataLoader]:
    """
    Create hardware-aware DataLoader based on available resources.
    
    Args:
        dataset: Dataset
        device_type: Device type ('auto', 'macbook', 'gpu', 'cpu')
        batch_size: Batch size (auto-detected if None)
        collate_fn: Collate function
        balance_classes: Whether to use balanced class sampling
        sampling_strategy: Optional dataset sampling strategy
        task_name: Task name for validation
        **kwargs: Additional arguments
        
    Returns:
        Appropriate DataLoader instance
    """
    logger = get_logger("HardwareAwareDataLoader")
    
    if device_type == "auto":
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 6.0 and total_memory_gb >= 12.0:
                device_type = "gpu"
            else:
                device_type = "macbook"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "macbook"
        else:
            device_type = "cpu"
        
        logger.info(f"Detected device type: {device_type}")
    
    sampler = None
    if balance_classes and task_name in CLASSIFICATION_TASKS:
        try:
            labels = [dataset[i]['label'] for i in range(len(dataset))]
            sampler = BalancedClassSampler(labels, task_name=task_name)
        except Exception as e:
            logger.warning(f"Could not create balanced sampler: {e}")
    
    if device_type == "macbook":
        config = {
            'batch_size': batch_size or 8,
            'chunk_size': 1000,
            'max_memory_gb': 7.0,
            'num_workers': 2,
            'pin_memory': False,
            'use_chunked': True
        }
    elif device_type == "gpu":
        config = {
            'batch_size': batch_size or 16,
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'use_chunked': False
        }
    else:
        config = {
            'batch_size': batch_size or 4,
            'num_workers': min(4, psutil.cpu_count() or 1),
            'pin_memory': False,
            'use_chunked': len(dataset) > 10000
        }
    
    if config.get('use_chunked', False):
        return create_chunked_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            **{k: v for k, v in config.items() if k != 'use_chunked'}
        )
    else:
        return create_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            task_name=task_name,
            **{k: v for k, v in config.items() if k != 'use_chunked'}
        )
