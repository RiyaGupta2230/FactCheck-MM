#!/usr/bin/env python3
"""
ParaNMT-5M Dataset Loader

Loads the ParaNMT-5M dataset containing 5 million high-quality paraphrase pairs.
Implements memory-efficient chunked loading for resource-constrained environments.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class ParaNMTConfig:
    """Configuration for ParaNMT-5M dataset loading."""
    
    # Data paths
    data_dir: str = "data/paranmt"
    filename: str = "para-nmt-5m-processed.txt"
    
    # Processing parameters
    max_length: int = 128
    chunk_size: int = 10000
    quality_threshold: float = 0.0
    max_samples: Optional[int] = None
    
    # Memory optimization
    use_chunked_loading: bool = True
    cache_processed: bool = False
    
    # Text preprocessing
    tokenizer_name: str = "roberta-base"
    lowercase: bool = False
    remove_special_chars: bool = False
    
    # Filtering
    min_length: int = 5
    max_token_length: int = 30


class ParaNMTLoader(Dataset):
    """
    PyTorch Dataset for ParaNMT-5M paraphrase pairs.
    
    Supports memory-efficient loading for large-scale training with chunked processing
    and quality-based filtering.
    """
    
    def __init__(
        self,
        config: ParaNMTConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize ParaNMT-5M dataset loader.
        
        Args:
            config: Dataset configuration
            split: Data split (train/val/test) - ParaNMT uses full data
            transform: Optional data transformation function
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Setup logging
        self.logger = get_logger("ParaNMTLoader")
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
            lowercase=config.lowercase
        )
        
        # Data storage
        self.data_path = Path(config.data_dir) / config.filename
        self.samples = []
        self.current_chunk_idx = 0
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"ParaNMT data file not found: {self.data_path}")
        
        # Load data based on configuration
        if config.use_chunked_loading:
            self._init_chunked_loading()
        else:
            self._load_full_data()
        
        self.logger.info(f"Loaded ParaNMT-5M dataset: {len(self)} samples")
    
    def _init_chunked_loading(self):
        """Initialize chunked loading system."""
        # Count total lines for progress tracking
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.total_lines = sum(1 for _ in f)
        
        # Calculate number of chunks
        self.num_chunks = (self.total_lines + self.config.chunk_size - 1) // self.config.chunk_size
        
        # Load first chunk
        self._load_chunk(0)
        
        self.logger.info(f"Initialized chunked loading: {self.num_chunks} chunks, {self.total_lines} total lines")
    
    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk of data."""
        start_line = chunk_idx * self.config.chunk_size
        end_line = min(start_line + self.config.chunk_size, self.total_lines)
        
        chunk_data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if i >= end_line:
                    break
                
                # Parse line: reference \t paraphrase \t score
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    reference = parts[0].strip()
                    paraphrase = parts[1].strip()
                    score = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    # Apply filtering
                    if self._should_include_sample(reference, paraphrase, score):
                        chunk_data.append({
                            'reference': reference,
                            'paraphrase': paraphrase,
                            'score': score
                        })
        
        self.samples = chunk_data
        self.current_chunk_idx = chunk_idx
        
        self.logger.debug(f"Loaded chunk {chunk_idx}: {len(chunk_data)} samples")
    
    def _load_full_data(self):
        """Load entire dataset into memory."""
        self.logger.info("Loading full ParaNMT dataset...")
        
        samples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Loading ParaNMT")):
                # Apply max samples limit
                if self.config.max_samples and len(samples) >= self.config.max_samples:
                    break
                
                # Parse line
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    reference = parts[0].strip()
                    paraphrase = parts[1].strip()
                    score = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    # Apply filtering
                    if self._should_include_sample(reference, paraphrase, score):
                        samples.append({
                            'reference': reference,
                            'paraphrase': paraphrase,
                            'score': score
                        })
        
        self.samples = samples
        self.logger.info(f"Loaded {len(samples)} samples from ParaNMT dataset")
    
    def _should_include_sample(self, reference: str, paraphrase: str, score: float) -> bool:
        """Check if sample meets filtering criteria."""
        
        # Quality threshold filtering
        if score < self.config.quality_threshold:
            return False
        
        # Length filtering
        ref_tokens = len(reference.split())
        para_tokens = len(paraphrase.split())
        
        if (ref_tokens < self.config.min_length or para_tokens < self.config.min_length or
            ref_tokens > self.config.max_token_length or para_tokens > self.config.max_token_length):
            return False
        
        # Content filtering
        if not reference.strip() or not paraphrase.strip():
            return False
        
        return True
    
    def __len__(self) -> int:
        """Return number of samples in current chunk or full dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed sample data
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Extract text pairs
        reference = sample['reference']
        paraphrase = sample['paraphrase']
        score = sample['score']
        
        # Process texts
        reference_processed = self.text_processor.process_text(reference)
        paraphrase_processed = self.text_processor.process_text(paraphrase)
        
        # Create model inputs for both sentences
        ref_encoding = self.text_processor.tokenize_pair(reference, paraphrase)
        para_encoding = self.text_processor.tokenize_pair(paraphrase, reference)
        
        # Prepare output
        result = {
            # Raw text
            'reference_text': reference,
            'paraphrase_text': paraphrase,
            
            # Processed text
            'reference_processed': reference_processed,
            'paraphrase_processed': paraphrase_processed,
            
            # Model inputs (for classification/similarity)
            'input_ids': ref_encoding['input_ids'],
            'attention_mask': ref_encoding['attention_mask'],
            
            # Generation targets (for seq2seq)
            'target_ids': self.text_processor.tokenize_text(paraphrase)['input_ids'],
            
            # Metadata
            'quality_score': torch.tensor(score, dtype=torch.float),
            'sample_id': idx,
            
            # Task label (always 1 for paraphrase pairs)
            'labels': torch.tensor(1, dtype=torch.long)
        }
        
        # Apply optional transform
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_text_pairs(self) -> List[Tuple[str, str]]:
        """Return all text pairs in current chunk."""
        return [(sample['reference'], sample['paraphrase']) for sample in self.samples]
    
    def get_quality_scores(self) -> List[float]:
        """Return quality scores for current chunk."""
        return [sample['score'] for sample in self.samples]
    
    def next_chunk(self) -> bool:
        """
        Load next chunk if using chunked loading.
        
        Returns:
            True if next chunk loaded successfully, False if at end
        """
        if not self.config.use_chunked_loading:
            return False
        
        next_chunk_idx = self.current_chunk_idx + 1
        if next_chunk_idx >= self.num_chunks:
            return False
        
        self._load_chunk(next_chunk_idx)
        return True
    
    def reset_to_chunk(self, chunk_idx: int = 0):
        """Reset to specific chunk."""
        if not self.config.use_chunked_loading:
            return
        
        if 0 <= chunk_idx < self.num_chunks:
            self._load_chunk(chunk_idx)
        else:
            raise ValueError(f"Chunk index {chunk_idx} out of range [0, {self.num_chunks})")
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about current chunk."""
        return {
            'current_chunk': self.current_chunk_idx,
            'total_chunks': getattr(self, 'num_chunks', 1),
            'chunk_size': len(self.samples),
            'total_lines': getattr(self, 'total_lines', len(self.samples)),
            'using_chunked_loading': self.config.use_chunked_loading
        }


def create_paranmt_dataloader(
    config: ParaNMTConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader for ParaNMT dataset.
    
    Args:
        config: Dataset configuration
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        PyTorch DataLoader
    """
    dataset = ParaNMTLoader(config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_paranmt_batch,
        **kwargs
    )


def _collate_paranmt_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ParaNMT batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Initialize batch dictionary
    batched = {
        'input_ids': [],
        'attention_mask': [],
        'target_ids': [],
        'labels': [],
        'quality_score': [],
        'reference_text': [],
        'paraphrase_text': []
    }
    
    # Collect all samples
    for sample in batch:
        for key in batched.keys():
            if key in sample:
                batched[key].append(sample[key])
    
    # Convert to tensors where appropriate
    for key in ['input_ids', 'attention_mask', 'target_ids', 'labels']:
        if batched[key] and torch.is_tensor(batched[key][0]):
            batched[key] = torch.stack(batched[key])
    
    if batched['quality_score'] and torch.is_tensor(batched['quality_score'][0]):
        batched['quality_score'] = torch.stack(batched['quality_score'])
    
    return batched


def main():
    """Example usage of ParaNMT loader."""
    
    # Configuration
    config = ParaNMTConfig(
        data_dir="data/paranmt",
        max_samples=1000,  # Limit for testing
        chunk_size=500,
        use_chunked_loading=True
    )
    
    # Create dataset
    dataset = ParaNMTLoader(config)
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample access
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Reference: {sample['reference_text']}")
    print(f"Paraphrase: {sample['paraphrase_text']}")
    print(f"Quality score: {sample['quality_score']}")
    
    # Test chunked loading
    if config.use_chunked_loading:
        chunk_info = dataset.get_chunk_info()
        print(f"Chunk info: {chunk_info}")
        
        # Load next chunk if available
        if dataset.next_chunk():
            print("Loaded next chunk")
            print(f"New chunk size: {len(dataset)}")
    
    # Create dataloader
    dataloader = create_paranmt_dataloader(config, batch_size=4)
    
    # Test batch loading
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        break


if __name__ == "__main__":
    main()
