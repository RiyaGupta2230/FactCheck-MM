#!/usr/bin/env python3
"""
Quora Dataset Loader

Loads the Quora Question Pairs dataset for duplicate question detection.
Handles class imbalance through balanced sampling strategies.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class QuoraConfig:
    """Configuration for Quora dataset loading."""
    
    # Data paths
    data_dir: str = "data/quora"
    
    # Processing parameters
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    
    # Text preprocessing
    lowercase: bool = False
    remove_special_chars: bool = True
    handle_math_notation: bool = True
    
    # Sampling and balancing
    balanced_sampling: bool = True
    sample_ratio: float = 1.0  # Ratio of data to use (for memory efficiency)
    max_samples: Optional[int] = None
    
    # Train/validation split (for train.csv)
    val_split: float = 0.1
    random_state: int = 42
    
    # Task configuration
    classification_task: bool = True


class QuoraDataset(Dataset):
    """
    PyTorch Dataset for Quora Question Pairs.
    
    Handles duplicate question detection with proper class balancing
    and preprocessing of question pairs.
    """
    
    def __init__(
        self,
        config: QuoraConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize Quora dataset loader.
        
        Args:
            config: Dataset configuration
            split: Data split (train/val/test)
            transform: Optional data transformation function
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Setup logging
        self.logger = get_logger("QuoraDataset")
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
            lowercase=config.lowercase
        )
        
        # Load data
        self.data = self._load_data()
        
        # Setup balanced sampling if enabled
        self.sampler = None
        if config.balanced_sampling and split == "train":
            self.sampler = self._create_balanced_sampler()
        
        self.logger.info(f"Loaded Quora {split} dataset: {len(self.data)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """Load Quora data from CSV files."""
        
        if self.split == "test":
            # Load test data (no labels)
            data_path = Path(self.config.data_dir) / "test.csv"
            
            if not data_path.exists():
                raise FileNotFoundError(f"Quora test file not found: {data_path}")
            
            df = pd.read_csv(data_path, encoding='utf-8')
            
            # Rename columns for consistency
            df = df.rename(columns={
                'test_id': 'id',
                'question1': 'question1',
                'question2': 'question2'
            })
            
            # No labels in test set
            df['is_duplicate'] = -1  # Placeholder
            
        else:
            # Load training data
            data_path = Path(self.config.data_dir) / "train.csv"
            
            if not data_path.exists():
                raise FileNotFoundError(f"Quora train file not found: {data_path}")
            
            df = pd.read_csv(data_path, encoding='utf-8')
            
            # Split into train and validation if needed
            if self.split in ["train", "val", "validation"]:
                train_df, val_df = train_test_split(
                    df,
                    test_size=self.config.val_split,
                    random_state=self.config.random_state,
                    stratify=df['is_duplicate']
                )
                
                if self.split == "train":
                    df = train_df
                else:
                    df = val_df
        
        # Apply sampling if configured
        df = self._apply_sampling(df)
        
        # Clean and preprocess
        df = self._clean_data(df)
        
        # Validate data
        self._validate_data(df)
        
        return df
    
    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sampling strategies to manage dataset size."""
        
        original_size = len(df)
        
        # Apply sample ratio
        if self.config.sample_ratio < 1.0:
            sample_size = int(len(df) * self.config.sample_ratio)
            df = df.sample(n=sample_size, random_state=self.config.random_state)
        
        # Apply max samples limit
        if self.config.max_samples and len(df) > self.config.max_samples:
            df = df.sample(n=self.config.max_samples, random_state=self.config.random_state)
        
        if len(df) < original_size:
            self.logger.info(f"Sampled {len(df)} samples from {original_size}")
        
        return df.reset_index(drop=True)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess Quora data."""
        
        # Handle missing values
        df['question1'] = df['question1'].fillna('')
        df['question2'] = df['question2'].fillna('')
        
        # Remove empty questions
        mask = (df['question1'].str.strip() != '') & (df['question2'].str.strip() != '')
        df = df[mask].reset_index(drop=True)
        
        # Handle mathematical notation if configured
        if self.config.handle_math_notation:
            df['question1'] = df['question1'].apply(self._process_math_notation)
            df['question2'] = df['question2'].apply(self._process_math_notation)
        
        # Remove special characters if configured
        if self.config.remove_special_chars:
            df['question1'] = df['question1'].apply(self._clean_special_chars)
            df['question2'] = df['question2'].apply(self._clean_special_chars)
        
        return df
    
    def _process_math_notation(self, text: str) -> str:
        """Process mathematical notation in questions."""
        import re
        
        # Handle LaTeX-style math notation
        text = re.sub(r'\[math\](.*?)\[/math\]', r'(\1)', text)
        
        # Normalize mathematical symbols
        text = re.sub(r'\s*\^\s*', '^', text)
        text = re.sub(r'\s*_\s*', '_', text)
        
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """Clean special characters while preserving meaning."""
        import re
        
        # Remove or replace problematic characters
        text = re.sub(r'[^\w\s\?\.\!\,\;\:\'\"\(\)\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate loaded Quora data."""
        
        # Check required columns
        required_cols = ['question1', 'question2']
        if self.split != "test":
            required_cols.append('is_duplicate')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Found null values: {null_counts.to_dict()}")
        
        # Check label distribution (if labels available)
        if 'is_duplicate' in df.columns and self.split != "test":
            label_dist = df['is_duplicate'].value_counts().sort_index()
            total = len(df)
            self.logger.info(f"Label distribution:")
            for label, count in label_dist.items():
                self.logger.info(f"  {label}: {count} ({count/total*100:.1f}%)")
        
        # Check text statistics
        q1_lengths = df['question1'].str.len()
        q2_lengths = df['question2'].str.len()
        
        self.logger.info(f"Question 1 length - Mean: {q1_lengths.mean():.1f}, Max: {q1_lengths.max()}")
        self.logger.info(f"Question 2 length - Mean: {q2_lengths.mean():.1f}, Max: {q2_lengths.max()}")
    
    def _create_balanced_sampler(self) -> WeightedRandomSampler:
        """Create balanced sampler for training."""
        
        if 'is_duplicate' not in self.data.columns:
            return None
        
        labels = self.data['is_duplicate'].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        self.logger.info(f"Created balanced sampler with class weights: {class_weights}")
        
        return sampler
    
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
        question1 = str(row['question1']).strip()
        question2 = str(row['question2']).strip()
        is_duplicate = int(row['is_duplicate']) if row['is_duplicate'] != -1 else -1
        
        # Process texts
        question1_processed = self.text_processor.process_text(question1)
        question2_processed = self.text_processor.process_text(question2)
        
        # Tokenize question pair for classification
        encoding = self.text_processor.tokenize_pair(question1, question2)
        
        # Prepare output
        result = {
            # Raw text
            'question1_text': question1,
            'question2_text': question2,
            
            # Processed text
            'question1_processed': question1_processed,
            'question2_processed': question2_processed,
            
            # Model inputs
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            
            # Metadata
            'sample_id': idx,
        }
        
        # Add token type ids if available
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']
        
        # Add labels if available (not for test set)
        if is_duplicate != -1:
            result['labels'] = torch.tensor(is_duplicate, dtype=torch.long)
        
            # For generation tasks, add target
            if not self.config.classification_task:
                result['target_ids'] = self.text_processor.tokenize_text(question2)['input_ids']
        
        # Apply optional transform
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_text_pairs(self) -> List[Tuple[str, str]]:
        """Return all text pairs."""
        return list(zip(self.data['question1'], self.data['question2']))
    
    def get_labels(self) -> Optional[List[int]]:
        """Return all labels if available."""
        if 'is_duplicate' in self.data.columns and self.split != "test":
            return self.data['is_duplicate'].tolist()
        return None
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels."""
        if 'is_duplicate' in self.data.columns and self.split != "test":
            return self.data['is_duplicate'].value_counts().sort_index().to_dict()
        return {}
    
    def get_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get the balanced sampler if created."""
        return self.sampler


def create_quora_dataloader(
    config: QuoraConfig,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader for Quora dataset.
    
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
    dataset = QuoraDataset(config, split)
    
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
        collate_fn=_collate_quora_batch,
        **kwargs
    )


def _collate_quora_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Quora batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Initialize batch dictionary
    batched = {
        'input_ids': [],
        'attention_mask': [],
        'question1_text': [],
        'question2_text': []
    }
    
    # Add optional fields
    optional_fields = ['labels', 'token_type_ids', 'target_ids']
    for field in optional_fields:
        if field in batch[0]:
            batched[field] = []
    
    # Collect all samples
    for sample in batch:
        for key in batched.keys():
            if key in sample:
                batched[key].append(sample[key])
    
    # Convert to tensors where appropriate
    tensor_keys = ['input_ids', 'attention_mask']
    tensor_keys.extend([k for k in optional_fields if k in batched])
    
    for key in tensor_keys:
        if key in batched and batched[key] and torch.is_tensor(batched[key][0]):
            batched[key] = torch.stack(batched[key])
    
    return batched


def main():
    """Example usage of Quora loader."""
    
    # Configuration
    config = QuoraConfig(
        data_dir="data/quora",
        max_samples=1000,  # Limit for testing
        balanced_sampling=True
    )
    
    # Test different splits
    for split in ["train", "val", "test"]:
        try:
            dataset = QuoraDataset(config, split)
            print(f"\n{split.upper()} Split:")
            print(f"  Dataset size: {len(dataset)}")
            
            # Show label distribution
            if split != "test":
                label_dist = dataset.get_label_distribution()
                if label_dist:
                    print(f"  Label distribution: {label_dist}")
            
            # Test sample access
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Question 1: {sample['question1_text'][:100]}...")
                print(f"  Question 2: {sample['question2_text'][:100]}...")
                if 'labels' in sample:
                    print(f"  Label: {sample['labels']}")
            
        except FileNotFoundError as e:
            print(f"\n{split.upper()} Split: File not found - {e}")
    
    # Create dataloader with balanced sampling
    try:
        dataloader = create_quora_dataloader(config, "train", batch_size=4)
        
        # Test batch loading
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            if 'labels' in batch:
                print(f"  Labels shape: {batch['labels'].shape}")
                print(f"  Label distribution in batch: {torch.bincount(batch['labels'])}")
            break
            
    except FileNotFoundError:
        print("Training data not available for DataLoader test")


if __name__ == "__main__":
    main()
