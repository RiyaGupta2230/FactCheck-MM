"""
Hardware-Aware Data Loaders for FactCheck-MM
Chunked loading for MacBook Air M2, full GPU mode for RTX 2050.
"""

import torch
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import gc

from .multimodal_dataset import MultimodalDataset, ChunkedDataset
from ..utils import get_logger


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
        
        # Text processing
        if 'text' in batch[0] or 'text_processed' in batch[0]:
            text_data = []
            for sample in batch:
                text = sample.get('text_processed', sample.get('text', ''))
                text_data.append(text)
            
            if self.text_processor and text_data[0]:
                # Tokenize batch
                tokenized = self.text_processor.tokenize(
                    text_data,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                collated.update(tokenized)
            else:
                # Fallback: convert to basic tensors
                collated['input_ids'] = torch.zeros(batch_size, self.max_length, dtype=torch.long)
                if self.return_attention_mask:
                    collated['attention_mask'] = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        
        # Audio processing
        if 'audio_processed' in batch[0]:
            audio_features = []
            for sample in batch:
                audio_data = sample['audio_processed']
                if isinstance(audio_data, dict) and 'audio' in audio_data:
                    audio_features.append(torch.tensor(audio_data['audio']))
                elif isinstance(audio_data, np.ndarray):
                    audio_features.append(torch.tensor(audio_data))
                else:
                    # Placeholder for missing audio
                    audio_features.append(torch.zeros(16000))  # 1 second at 16kHz
            
            # Pad audio sequences
            collated['audio_features'] = pad_sequence(audio_features, batch_first=True)
        
        # Image processing
        if 'image_processed' in batch[0]:
            image_tensors = []
            for sample in batch:
                image_data = sample['image_processed']
                if isinstance(image_data, dict) and 'images' in image_data:
                    # Convert PIL Image to tensor
                    image = image_data['images'][0]  # Use first image
                    if hasattr(image, 'convert'):  # PIL Image
                        image_array = np.array(image.convert('RGB'))
                        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
                    else:
                        image_tensor = torch.tensor(image).float()
                    image_tensors.append(image_tensor)
                else:
                    # Placeholder for missing image
                    image_tensors.append(torch.zeros(3, 224, 224))
            
            collated['image_features'] = torch.stack(image_tensors)
        
        # Video processing
        if 'video_processed' in batch[0]:
            video_tensors = []
            for sample in batch:
                video_data = sample['video_processed']
                if isinstance(video_data, dict) and 'frames' in video_data:
                    frames = video_data['frames']
                    # Convert to tensor: [num_frames, channels, height, width]
                    if isinstance(frames, np.ndarray):
                        video_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
                    else:
                        video_tensor = torch.tensor(frames).float()
                    video_tensors.append(video_tensor)
                else:
                    # Placeholder for missing video
                    video_tensors.append(torch.zeros(16, 3, 224, 224))  # 16 frames
            
            # Pad video sequences
            collated['video_features'] = pad_sequence(video_tensors, batch_first=True)
        
        # Labels processing
        if 'label' in batch[0]:
            labels = []
            for sample in batch:
                label = sample['label']
                if isinstance(label, str):
                    # Convert string labels to indices (task-specific)
                    if label.upper() in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                        # FEVER labels
                        label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
                        labels.append(label_map[label.upper()])
                    else:
                        labels.append(0)  # Default
                else:
                    labels.append(int(label))
            
            collated['labels'] = torch.tensor(labels, dtype=torch.long)
        
        # Paraphrasing targets
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
        
        # Evidence for fact verification
        if 'evidence' in batch[0]:
            evidence_batch = []
            for sample in batch:
                evidence = sample.get('evidence', [])
                if isinstance(evidence, list):
                    evidence_text = ' '.join(evidence[:5])  # Use first 5 evidence pieces
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
        
        # Additional metadata
        collated['batch_size'] = torch.tensor(batch_size)
        
        return collated


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
        max_memory_gb: float = 7.0
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
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_workers = min(num_workers, 2)  # Limit workers for memory
        self.pin_memory = pin_memory
        self.max_memory_gb = max_memory_gb
        
        self.logger = get_logger("ChunkedDataLoader")
        
        # Create chunked dataset
        self.chunked_dataset = ChunkedDataset(
            dataset,
            chunk_size=chunk_size,
            shuffle_chunks=shuffle
        )
        
        # Current state
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
            # Check memory before processing chunk
            if not self._check_memory():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create dataloader for this chunk
            chunk_dataloader = self._create_chunk_dataloader(chunk_data)
            
            self.logger.debug(f"Processing chunk {chunk_idx}/{self.chunked_dataset.num_chunks}")
            
            # Yield batches from this chunk
            for batch in chunk_dataloader:
                yield batch
            
            # Clear chunk from memory
            del chunk_dataloader, chunk_data
            gc.collect()
    
    def __len__(self) -> int:
        """Get total number of batches."""
        total_samples = len(self.dataset)
        batches_per_chunk = (self.chunk_size + self.batch_size - 1) // self.batch_size
        return self.chunked_dataset.num_chunks * batches_per_chunk


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    max_memory_gb: float = 7.0,
    start_batch_size: int = 1,
    device: str = "cuda"
) -> int:
    """
    Find optimal batch size for given memory constraints.
    
    Args:
        model: Model to test
        sample_input: Sample input for testing
        max_memory_gb: Maximum memory in GB
        start_batch_size: Starting batch size
        device: Device to test on
        
    Returns:
        Optimal batch size
    """
    logger = get_logger("BatchSizeOptimizer")
    
    model = model.to(device)
    model.eval()
    
    batch_size = start_batch_size
    optimal_batch_size = 1
    
    while batch_size <= 128:  # Upper limit for safety
        try:
            # Create batch
            batch_input = {}
            for key, tensor in sample_input.items():
                if isinstance(tensor, torch.Tensor):
                    # Repeat tensor to create batch
                    batch_tensor = tensor.unsqueeze(0).repeat(batch_size, *([1] * (tensor.dim())))
                    batch_input[key] = batch_tensor.to(device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(**batch_input)
            
            # Check memory usage
            if device == "cuda" and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                if memory_used > max_memory_gb:
                    break
            else:
                memory_info = psutil.virtual_memory()
                memory_used = (memory_info.total - memory_info.available) / (1024**3)
                if memory_used > max_memory_gb:
                    break
            
            optimal_batch_size = batch_size
            batch_size *= 2
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.debug(f"OOM at batch size {batch_size}: {e}")
            break
        
        # Clear memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True
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
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )


def create_chunked_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    chunk_size: int = 1000,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    max_memory_gb: float = 7.0,
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
        **kwargs
    )


def create_hardware_aware_dataloader(
    dataset: Dataset,
    device_type: str = "auto",
    batch_size: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    **kwargs
) -> Union[DataLoader, ChunkedDataLoader]:
    """
    Create hardware-aware DataLoader based on available resources.
    
    Args:
        dataset: Dataset
        device_type: Device type ('auto', 'macbook', 'gpu', 'cpu')
        batch_size: Batch size (auto-detected if None)
        collate_fn: Collate function
        **kwargs: Additional arguments
        
    Returns:
        Appropriate DataLoader instance
    """
    logger = get_logger("HardwareAwareDataLoader")
    
    # Detect hardware if auto
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
    
    # Configure based on device type
    if device_type == "macbook":
        # MacBook Air M2 configuration
        config = {
            'batch_size': batch_size or 8,
            'chunk_size': 1000,
            'max_memory_gb': 7.0,
            'num_workers': 2,
            'pin_memory': False,
            'use_chunked': True
        }
    elif device_type == "gpu":
        # RTX 2050 configuration
        config = {
            'batch_size': batch_size or 16,
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'use_chunked': False
        }
    else:
        # CPU configuration
        config = {
            'batch_size': batch_size or 4,
            'num_workers': min(4, psutil.cpu_count() or 1),
            'pin_memory': False,
            'use_chunked': len(dataset) > 10000  # Use chunking for large datasets
        }
    
    # Create appropriate dataloader
    if config.get('use_chunked', False):
        return create_chunked_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            **{k: v for k, v in config.items() if k != 'use_chunked'}
        )
    else:
        return create_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            **{k: v for k, v in config.items() if k != 'use_chunked'}
        )
