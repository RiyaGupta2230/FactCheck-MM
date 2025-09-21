#!/usr/bin/env python3
"""
MRPC Dataset Loader

Loads the Microsoft Research Paraphrase Corpus (MRPC) dataset from GLUE benchmark.
Handles tab-separated format with paraphrase classification labels.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class MRPCConfig:
    """Configuration for MRPC dataset loading."""
    
    # Data paths
    data_dir: str = "data/MRPC"
    
    # Processing parameters
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    
    # Text preprocessing
    lowercase: bool = False
    remove_special_chars: bool = False
    
    # Task configuration
    classification_task: bool = True  # True for classification, False for generation


class MRPCDataset(Dataset):
    """
    PyTorch Dataset for MRPC paraphrase classification.
    
    Loads Microsoft Research Paraphrase Corpus with proper handling of
    tab-separated format and binary paraphrase labels.
    """
    
    def __init__(
        self,
        config: MRPCConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize MRPC dataset loader.
        
        Args:
            config: Dataset configuration
            split: Data split (train/dev/test)
            transform: Optional data transformation function
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Setup logging
        self.logger = get_logger("MRPCDataset")
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
            lowercase=config.lowercase
        )
        
        # Load data
        self.data = self._load_data()
        
        self.logger.info(f"Loaded MRPC {split} dataset: {len(self.data)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """Load MRPC data from TSV files."""
        
        # Determine file name based on split
        if self.split == "train":
            filename = "train.tsv"
        elif self.split in ["val", "dev", "validation"]:
            filename = "dev.tsv"
        elif self.split == "test":
            filename = "test.tsv"
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        data_path = Path(self.config.data_dir) / filename
        
        if not data_path.exists():
            raise FileNotFoundError(f"MRPC data file not found: {data_path}")
        
        # Load TSV with proper column names
        try:
            # Read with tab separator and proper column names
            df = pd.read_csv(
                data_path,
                sep='\t',
                encoding='utf-8-sig',  # Handle BOM
                names=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'],
                skiprows=1  # Skip header row
            )
            
            # Clean column names
            df.columns = ['quality', 'id1', 'id2', 'sentence1', 'sentence2']
            
            # Validate data
            self._validate_data(df)
            
            self.logger.info(f"Loaded {len(df)} samples from {data_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading MRPC data: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate loaded MRPC data."""
        
        # Check required columns
        required_cols = ['quality', 'sentence1', 'sentence2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Found null values: {null_counts.to_dict()}")
        
        # Check label distribution
        if 'quality' in df.columns:
            label_dist = df['quality'].value_counts().sort_index()
            self.logger.info(f"Label distribution: {label_dist.to_dict()}")
        
        # Check text lengths
        text1_lengths = df['sentence1'].str.len()
        text2_lengths = df['sentence2'].str.len()
        
        self.logger.info(f"Sentence 1 length - Mean: {text1_lengths.mean():.1f}, Max: {text1_lengths.max()}")
        self.logger.info(f"Sentence 2 length - Mean: {text2_lengths.mean():.1f}, Max: {text2_lengths.max()}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed sample data
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        row = self.data.iloc[idx]
        
        # Extract data
        sentence1 = str(row['sentence1']).strip()
        sentence2 = str(row['sentence2']).strip()
        quality = int(row['quality']) if pd.notna(row['quality']) else 0
        
        # Process texts
        sentence1_processed = self.text_processor.process_text(sentence1)
        sentence2_processed = self.text_processor.process_text(sentence2)
        
        # Tokenize sentence pair for classification
        encoding = self.text_processor.tokenize_pair(sentence1, sentence2)
        
        # Prepare output based on task type
        result = {
            # Raw text
            'sentence1_text': sentence1,
            'sentence2_text': sentence2,
            
            # Processed text
            'sentence1_processed': sentence1_processed,
            'sentence2_processed': sentence2_processed,
            
            # Model inputs
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            
            # Labels
            'labels': torch.tensor(quality, dtype=torch.long),
            
            # Metadata
            'sample_id': idx,
            'id1': row.get('id1', -1),
            'id2': row.get('id2', -1)
        }
        
        # Add token type ids if available
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']
        
        # For generation tasks, add target
        if not self.config.classification_task:
            result['target_ids'] = self.text_processor.tokenize_text(sentence2)['input_ids']
        
        # Apply optional transform
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_text_pairs(self) -> List[Tuple[str, str]]:
        """Return all text pairs."""
        return list(zip(self.data['sentence1'], self.data['sentence2']))
    
    def get_labels(self) -> List[int]:
        """Return all labels."""
        return self.data['quality'].tolist()
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels."""
        return self.data['quality'].value_counts().sort_index().to_dict()
    
    def filter_by_length(self, min_length: int = 5, max_length: int = 200) -> 'MRPCDataset':
        """
        Create filtered dataset based on text length.
        
        Args:
            min_length: Minimum character length
            max_length: Maximum character length
            
        Returns:
            New filtered dataset instance
        """
        # Calculate text lengths
        text1_lengths = self.data['sentence1'].str.len()
        text2_lengths = self.data['sentence2'].str.len()
        
        # Apply length filters
        length_mask = (
            (text1_lengths >= min_length) & (text1_lengths <= max_length) &
            (text2_lengths >= min_length) & (text2_lengths <= max_length)
        )
        
        # Create new dataset with filtered data
        filtered_dataset = MRPCDataset(self.config, self.split, self.transform)
        filtered_dataset.data = self.data[length_mask].reset_index(drop=True)
        
        self.logger.info(f"Filtered dataset from {len(self.data)} to {len(filtered_dataset.data)} samples")
        
        return filtered_dataset


def create_mrpc_dataloader(
    config: MRPCConfig,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader for MRPC dataset.
    
    Args:
        config: Dataset configuration
        split: Data split
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        PyTorch DataLoader
    """
    dataset = MRPCDataset(config, split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_mrpc_batch,
        **kwargs
    )


def _collate_mrpc_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for MRPC batches.
    
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
        'sentence1_text': [],
        'sentence2_text': []
    }
    
    # Add token_type_ids if present
    if 'token_type_ids' in batch[0]:
        batched['token_type_ids'] = []
    
    # Add target_ids for generation
    if 'target_ids' in batch[0]:
        batched['target_ids'] = []
    
    # Collect all samples
    for sample in batch:
        for key in batched.keys():
            if key in sample:
                batched[key].append(sample[key])
    
    # Convert to tensors where appropriate
    tensor_keys = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in batched:
        tensor_keys.append('token_type_ids')
    if 'target_ids' in batched:
        tensor_keys.append('target_ids')
    
    for key in tensor_keys:
        if key in batched and batched[key] and torch.is_tensor(batched[key][0]):
            batched[key] = torch.stack(batched[key])
    
    return batched


def main():
    """Example usage of MRPC loader."""
    
    # Configuration
    config = MRPCConfig(
        data_dir="data/MRPC",
        max_length=128
    )
    
    # Test different splits
    for split in ["train", "dev", "test"]:
        try:
            dataset = MRPCDataset(config, split)
            print(f"\n{split.upper()} Split:")
            print(f"  Dataset size: {len(dataset)}")
            
            # Show label distribution
            label_dist = dataset.get_label_distribution()
            print(f"  Label distribution: {label_dist}")
            
            # Test sample access
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Sentence 1: {sample['sentence1_text'][:100]}...")
                print(f"  Sentence 2: {sample['sentence2_text'][:100]}...")
                print(f"  Label: {sample['labels']}")
            
        except FileNotFoundError as e:
            print(f"\n{split.upper()} Split: File not found - {e}")
    
    # Create dataloader for training
    try:
        dataloader = create_mrpc_dataloader(config, "train", batch_size=4)
        
        # Test batch loading
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            break
            
    except FileNotFoundError:
        print("Training data not available for DataLoader test")


if __name__ == "__main__":
    main()
