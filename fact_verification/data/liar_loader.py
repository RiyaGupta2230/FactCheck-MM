#!/usr/bin/env python3
"""
LIAR Dataset Loader for Fact Verification

Implements comprehensive loading and preprocessing for the LIAR dataset, supporting
both binary and multi-class political fact-checking with flexible label mapping
and speaker metadata integration.

Example Usage:
    >>> from fact_verification.data import LiarDataset
    >>> 
    >>> # Load with binary labels (True/False)
    >>> liar_binary = LiarDataset('train', binary_labels=True, include_metadata=False)
    >>> print(f"Binary dataset size: {len(liar_binary)}")
    >>> 
    >>> # Load with full 6-class labels
    >>> liar_multiclass = LiarDataset('train', binary_labels=False, include_speaker_info=True)
    >>> 
    >>> # Access sample
    >>> sample = liar_binary[0]
    >>> print(f"Statement: {sample['statement']}")
    >>> print(f"Label: {sample['label']}")
"""

import sys
import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.preprocessing.text_processor import TextProcessor, TextProcessorConfig
from shared.datasets.data_loaders import ChunkedDataLoader
from shared.utils.logging_utils import get_logger


@dataclass
class LiarDatasetConfig:
    """Configuration for LIAR dataset loading."""
    
    # Data paths
    data_dir: str = "data/LIAR"
    train_file: str = "train_formatted.csv"
    test_file: str = "test.tsv"
    valid_file: str = "valid.tsv"
    
    # Text processing
    max_statement_length: int = 256
    max_context_length: int = 128
    combine_statement_context: bool = True
    
    # Label processing
    binary_labels: bool = True
    six_class_mapping: Dict[str, int] = field(default_factory=lambda: {
        "true": 0,
        "mostly-true": 1, 
        "half-true": 2,
        "mostly-false": 3,
        "false": 4,
        "pants-fire": 5
    })
    binary_mapping: Dict[str, int] = field(default_factory=lambda: {
        "true": 1,
        "mostly-true": 1,
        "half-true": 0,  # Uncertain -> False for binary
        "mostly-false": 0,
        "false": 0,
        "pants-fire": 0
    })
    
    # Metadata inclusion
    include_speaker_info: bool = False
    include_subject_info: bool = False
    include_context_info: bool = True
    include_historical_counts: bool = False
    
    # Data filtering
    min_statement_length: int = 5
    filter_missing_statements: bool = True
    filter_unknown_labels: bool = True
    
    # Performance
    chunk_size: int = 1000
    use_chunked_loading: bool = True
    cache_processed_data: bool = True


class LiarDataset(Dataset):
    """
    PyTorch Dataset for LIAR political fact-checking data.
    
    Supports both binary (True/False) and 6-class classification with
    optional speaker and context metadata.
    """
    
    def __init__(
        self,
        split: str = "train", 
        config: Optional[LiarDatasetConfig] = None,
        text_processor: Optional[TextProcessor] = None
    ):
        """
        Initialize LIAR dataset.
        
        Args:
            split: Data split ('train', 'test', or 'valid')
            config: Dataset configuration
            text_processor: Text preprocessing pipeline
        """
        self.split = split
        self.config = config or LiarDatasetConfig()
        self.logger = get_logger("LiarDataset")
        
        # Initialize text processor
        if text_processor is None:
            processor_config = TextProcessorConfig(
                model_name="roberta-base",
                max_length=self.config.max_statement_length,
                lowercase=False,
                remove_special_chars=False
            )
            self.text_processor = TextProcessor(processor_config)
        else:
            self.text_processor = text_processor
        
        # Data storage
        self.samples = []
        self.processed_cache = {}
        
        # Load dataset
        self._load_data()
        
        self.logger.info(
            f"Loaded {len(self.samples)} LIAR {split} samples "
            f"({'binary' if self.config.binary_labels else '6-class'} labels)"
        )
    
    def _get_data_file_path(self) -> Path:
        """Get path to data file for current split."""
        
        data_dir = Path(self.config.data_dir)
        
        if self.split == "train":
            file_path = data_dir / self.config.train_file
        elif self.split == "test":
            file_path = data_dir / self.config.test_file
        elif self.split == "valid":
            file_path = data_dir / self.config.valid_file
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"LIAR data file not found: {file_path}")
        
        return file_path
    
    def _load_data(self):
        """Load LIAR data from CSV/TSV files."""
        
        file_path = self._get_data_file_path()
        self.logger.info(f"Loading LIAR data from: {file_path}")
        
        # Determine file format and load accordingly
        if file_path.suffix == '.csv':
            self._load_csv_data(file_path)
        else:  # .tsv
            self._load_tsv_data(file_path)
        
        # Filter samples if configured
        if (self.config.filter_missing_statements or 
            self.config.filter_unknown_labels or 
            self.config.min_statement_length > 0):
            self._filter_samples()
        
        self.logger.info(f"Processed {len(self.samples)} samples after filtering")
    
    def _load_csv_data(self, file_path: Path):
        """Load data from CSV file (train_formatted.csv)."""
        
        try:
            # Read CSV with proper handling
            df = pd.read_csv(file_path, encoding='utf-8')
            
            for idx, row in df.iterrows():
                sample = self._parse_liar_sample_csv(row, idx)
                if sample:
                    self.samples.append(sample)
                    
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise
    
    def _load_tsv_data(self, file_path: Path):
        """Load data from TSV file (test.tsv, valid.tsv)."""
        
        try:
            # Read TSV with tab delimiter
            df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', header=None)
            
            # TSV files have standard LIAR format with 14 columns
            column_names = [
                'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                'state_info', 'party_affiliation', 'barely_true_count',
                'false_count', 'half_true_count', 'mostly_true_count', 
                'pants_on_fire_count', 'context'
            ]
            
            if len(df.columns) == len(column_names):
                df.columns = column_names
            
            for idx, row in df.iterrows():
                sample = self._parse_liar_sample_tsv(row, idx)
                if sample:
                    self.samples.append(sample)
                    
        except Exception as e:
            self.logger.error(f"Error loading TSV file: {e}")
            raise
    
    def _parse_liar_sample_csv(self, row: pd.Series, idx: int) -> Optional[Dict[str, Any]]:
        """Parse a sample from CSV format."""
        
        try:
            # Extract statement and label (assuming standard CSV format)
            statement = str(row.get('Text', row.get('statement', ''))).strip()
            label = str(row.get('Label', row.get('label', ''))).strip().lower()
            
            # Basic validation
            if not statement or not label:
                return None
            
            # Create sample
            sample = {
                'id': row.get('ID', idx),
                'statement': statement,
                'original_label': label,
                'subject': row.get('subject', ''),
                'speaker': row.get('speaker', ''),
                'context': row.get('context', '')
            }
            
            # Map label
            label_id = self._map_label(label)
            if label_id is None:
                return None
            
            sample['label_id'] = label_id
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Error parsing CSV sample at index {idx}: {e}")
            return None
    
    def _parse_liar_sample_tsv(self, row: pd.Series, idx: int) -> Optional[Dict[str, Any]]:
        """Parse a sample from TSV format."""
        
        try:
            # Extract core fields
            statement = str(row['statement'] if 'statement' in row else '').strip()
            label = str(row['label'] if 'label' in row else '').strip().lower()
            
            # Basic validation
            if not statement or not label:
                return None
            
            # Create sample with full metadata
            sample = {
                'id': row.get('id', idx),
                'statement': statement,
                'original_label': label,
                'subject': str(row.get('subject', '')).strip(),
                'speaker': str(row.get('speaker', '')).strip(),
                'job_title': str(row.get('job_title', '')).strip(),
                'state_info': str(row.get('state_info', '')).strip(),
                'party_affiliation': str(row.get('party_affiliation', '')).strip(),
                'context': str(row.get('context', '')).strip()
            }
            
            # Add historical counts if requested
            if self.config.include_historical_counts:
                sample.update({
                    'barely_true_count': int(row.get('barely_true_count', 0) or 0),
                    'false_count': int(row.get('false_count', 0) or 0),
                    'half_true_count': int(row.get('half_true_count', 0) or 0),
                    'mostly_true_count': int(row.get('mostly_true_count', 0) or 0),
                    'pants_on_fire_count': int(row.get('pants_on_fire_count', 0) or 0)
                })
            
            # Map label
            label_id = self._map_label(label)
            if label_id is None:
                return None
            
            sample['label_id'] = label_id
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Error parsing TSV sample at index {idx}: {e}")
            return None
    
    def _map_label(self, label: str) -> Optional[int]:
        """Map original label to integer based on configuration."""
        
        # Normalize label
        label = label.lower().strip()
        
        # Handle different label formats
        label_mappings = {
            'true': 'true',
            'mostly-true': 'mostly-true',
            'mostly true': 'mostly-true',
            'half-true': 'half-true', 
            'half true': 'half-true',
            'barely-true': 'half-true',  # Map barely-true to half-true
            'barely true': 'half-true',
            'mostly-false': 'mostly-false',
            'mostly false': 'mostly-false',
            'false': 'false',
            'pants-fire': 'pants-fire',
            'pants on fire': 'pants-fire',
            'pants-on-fire': 'pants-fire'
        }
        
        normalized_label = label_mappings.get(label, label)
        
        # Map to integer
        if self.config.binary_labels:
            return self.config.binary_mapping.get(normalized_label)
        else:
            return self.config.six_class_mapping.get(normalized_label)
    
    def _filter_samples(self):
        """Filter samples based on configuration criteria."""
        
        original_count = len(self.samples)
        filtered_samples = []
        
        for sample in self.samples:
            # Filter missing statements
            if self.config.filter_missing_statements and not sample['statement'].strip():
                continue
            
            # Filter by statement length
            if len(sample['statement'].split()) < self.config.min_statement_length:
                continue
            
            # Filter unknown labels
            if self.config.filter_unknown_labels and sample['label_id'] is None:
                continue
            
            filtered_samples.append(sample)
        
        self.samples = filtered_samples
        filtered_count = original_count - len(self.samples)
        
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} samples ({filtered_count/original_count*100:.1f}%)")
    
    def _prepare_text_inputs(self, statement: str, context: str = "") -> Dict[str, torch.Tensor]:
        """
        Prepare text inputs for the model.
        
        Args:
            statement: Main statement text
            context: Optional context information
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Combine statement and context if configured
        if self.config.combine_statement_context and context.strip():
            # Create statement-context pair
            text_input = (statement, context)
            cache_key = f"{hash(statement)}_{hash(context)}"
        else:
            text_input = statement
            cache_key = f"{hash(statement)}"
        
        # Check cache
        if self.config.cache_processed_data and cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        # Process text
        if isinstance(text_input, tuple):
            # Statement-context pair
            inputs = self.text_processor.process_text_pair(
                text_input[0], text_input[1],
                max_length_a=self.config.max_statement_length,
                max_length_b=self.config.max_context_length,
                truncation="longest_first"
            )
        else:
            # Statement only
            inputs = self.text_processor.process_text(
                text_input,
                max_length=self.config.max_statement_length,
                truncation=True
            )
        
        # Cache processed inputs
        if self.config.cache_processed_data:
            self.processed_cache[cache_key] = inputs
        
        return inputs
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with model inputs
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Prepare text inputs
        context = sample['context'] if self.config.include_context_info else ""
        text_inputs = self._prepare_text_inputs(sample['statement'], context)
        
        # Create return dictionary
        result = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'label': torch.tensor(sample['label_id'], dtype=torch.long)
        }
        
        # Add metadata features if requested
        if self.config.include_speaker_info:
            result['speaker'] = sample.get('speaker', '')
            result['job_title'] = sample.get('job_title', '')
            result['party_affiliation'] = sample.get('party_affiliation', '')
        
        if self.config.include_subject_info:
            result['subject'] = sample.get('subject', '')
        
        # Add text metadata
        result['sample_id'] = sample['id']
        result['statement_text'] = sample['statement']
        result['original_label'] = sample['original_label']
        
        return result
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        
        label_counts = {}
        for sample in self.samples:
            label = sample['original_label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """Get statistics about speakers in the dataset."""
        
        speaker_counts = {}
        party_counts = {}
        
        for sample in self.samples:
            speaker = sample.get('speaker', 'Unknown')
            party = sample.get('party_affiliation', 'Unknown')
            
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            party_counts[party] = party_counts.get(party, 0) + 1
        
        return {
            'unique_speakers': len(speaker_counts),
            'top_speakers': dict(sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'party_distribution': party_counts
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        
        stats = {
            'total_samples': len(self.samples),
            'label_type': 'binary' if self.config.binary_labels else '6-class',
            'label_distribution': self.get_label_distribution(),
            'avg_statement_length': 0,
            'samples_with_context': 0,
            'samples_with_speaker': 0
        }
        
        if self.samples:
            statement_lengths = [len(s['statement'].split()) for s in self.samples]
            stats['avg_statement_length'] = np.mean(statement_lengths)
            
            stats['samples_with_context'] = sum(
                1 for s in self.samples if s.get('context', '').strip()
            )
            stats['samples_with_speaker'] = sum(
                1 for s in self.samples if s.get('speaker', '').strip()
            )
            
            stats['context_coverage'] = stats['samples_with_context'] / len(self.samples) * 100
            stats['speaker_coverage'] = stats['samples_with_speaker'] / len(self.samples) * 100
        
        # Add speaker statistics
        stats.update(self.get_speaker_statistics())
        
        return stats
    
    def create_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        use_chunked: bool = False
    ) -> Union[torch.utils.data.DataLoader, ChunkedDataLoader]:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            use_chunked: Whether to use chunked loading
            
        Returns:
            DataLoader instance
        """
        if use_chunked:
            return ChunkedDataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                chunk_size=self.config.chunk_size
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_fn
            )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        
        # Extract and stack tensors
        collated = {}
        
        for key in ['input_ids', 'attention_mask', 'label']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Keep metadata as lists
        metadata_keys = [
            'sample_id', 'statement_text', 'original_label', 
            'speaker', 'job_title', 'party_affiliation', 'subject'
        ]
        
        for key in metadata_keys:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated


def main():
    """Example usage of LiarDataset."""
    
    # Test with both binary and multi-class configurations
    configs = [
        ("Binary", LiarDatasetConfig(binary_labels=True, include_speaker_info=True)),
        ("6-Class", LiarDatasetConfig(binary_labels=False, include_context_info=True))
    ]
    
    for config_name, config in configs:
        try:
            print(f"\n=== {config_name} Configuration ===")
            
            # Load training data
            print("Loading LIAR training dataset...")
            train_dataset = LiarDataset('train', config=config)
            
            print(f"Dataset size: {len(train_dataset)}")
            
            # Show statistics
            stats = train_dataset.get_dataset_statistics()
            print("\nDataset Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, (int, float)) and key != "top_speakers":
                            print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
            
            # Show sample
            print(f"\nSample data:")
            sample = train_dataset[0]
            
            print(f"Statement: {sample['statement_text'][:100]}...")
            print(f"Label: {sample['original_label']} ({sample['label'].item()})")
            print(f"Input shape: {sample['input_ids'].shape}")
            
            if 'speaker' in sample:
                print(f"Speaker: {sample['speaker']}")
                print(f"Party: {sample['party_affiliation']}")
            
        except FileNotFoundError as e:
            print(f"Data file not found for {config_name}: {e}")
        except Exception as e:
            print(f"Error with {config_name} configuration: {e}")
    
    print("\nNote: Ensure LIAR data files are in the correct location:")
    print("  data/LIAR/train_formatted.csv")
    print("  data/LIAR/test.tsv")
    print("  data/LIAR/valid.tsv")


if __name__ == "__main__":
    main()
