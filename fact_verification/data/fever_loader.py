#!/usr/bin/env python3
"""
FEVER Dataset Loader for Fact Verification

Implements comprehensive loading and preprocessing for the FEVER (Fact Extraction and 
Verification) dataset, supporting claim-evidence pairs with SUPPORTS, REFUTES, and 
NOT ENOUGH INFO labels for robust fact verification training.

Example Usage:
    >>> from fact_verification.data import FeverDataset
    >>> 
    >>> # Load training data
    >>> fever_train = FeverDataset('train', max_claim_length=128, max_evidence_length=256)
    >>> print(f"Training samples: {len(fever_train)}")
    >>> 
    >>> # Access sample with evidence
    >>> sample = fever_train[0]
    >>> print(f"Claim: {sample['claim']}")
    >>> print(f"Evidence: {sample['evidence']}")
    >>> print(f"Label: {sample['label']}")
    >>> 
    >>> # Use DataLoader for batching
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(fever_train, batch_size=16, shuffle=True)
"""

import sys
import os
import json
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
class FeverDatasetConfig:
    """Configuration for FEVER dataset loading."""
    
    # Data paths
    data_dir: str = "data/FEVER"
    train_file: str = "fever_train.jsonl"
    test_file: str = "fever_test.jsonl"
    
    # Text processing
    max_claim_length: int = 128
    max_evidence_length: int = 256
    truncation_strategy: str = "longest_first"  # "longest_first", "only_first", "only_second"
    
    # Evidence handling
    max_evidence_sentences: int = 5
    evidence_separator: str = " [SEP] "
    include_evidence_metadata: bool = False
    
    # Label processing
    label_mapping: Dict[str, int] = field(default_factory=lambda: {
        "SUPPORTS": 0,
        "REFUTES": 1, 
        "NOT ENOUGH INFO": 2
    })
    
    # Data filtering
    filter_no_evidence: bool = False
    min_claim_length: int = 5
    min_evidence_length: int = 10
    
    # Performance
    chunk_size: int = 1000
    use_chunked_loading: bool = True
    cache_processed_data: bool = True


class FeverDataset(Dataset):
    """
    PyTorch Dataset for FEVER fact verification data.
    
    Loads and preprocesses FEVER dataset with claim-evidence pairs and 
    three-class labels (SUPPORTS, REFUTES, NOT ENOUGH INFO).
    """
    
    def __init__(
        self,
        split: str = "train",
        config: Optional[FeverDatasetConfig] = None,
        text_processor: Optional[TextProcessor] = None
    ):
        """
        Initialize FEVER dataset.
        
        Args:
            split: Data split ('train' or 'test')
            config: Dataset configuration
            text_processor: Text preprocessing pipeline
        """
        self.split = split
        self.config = config or FeverDatasetConfig()
        self.logger = get_logger("FeverDataset")
        
        # Initialize text processor
        if text_processor is None:
            processor_config = TextProcessorConfig(
                model_name="roberta-base",
                max_length=max(self.config.max_claim_length, self.config.max_evidence_length),
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
        
        self.logger.info(f"Loaded {len(self.samples)} FEVER {split} samples")
    
    def _get_data_file_path(self) -> Path:
        """Get path to data file for current split."""
        
        data_dir = Path(self.config.data_dir)
        
        if self.split == "train":
            file_path = data_dir / self.config.train_file
        elif self.split == "test":
            file_path = data_dir / self.config.test_file
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"FEVER data file not found: {file_path}")
        
        return file_path
    
    def _load_data(self):
        """Load FEVER data from JSONL files."""
        
        file_path = self._get_data_file_path()
        self.logger.info(f"Loading FEVER data from: {file_path}")
        
        # Load data with chunked processing for memory efficiency
        if self.config.use_chunked_loading:
            self._load_data_chunked(file_path)
        else:
            self._load_data_direct(file_path)
        
        # Filter samples if configured
        if self.config.filter_no_evidence or self.config.min_claim_length > 0:
            self._filter_samples()
        
        self.logger.info(f"Processed {len(self.samples)} samples after filtering")
    
    def _load_data_chunked(self, file_path: Path):
        """Load data using chunked processing for memory efficiency."""
        
        chunk_samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Extract required fields
                    sample = self._parse_fever_sample(data, line_num)
                    if sample:
                        chunk_samples.append(sample)
                    
                    # Process chunk when full
                    if len(chunk_samples) >= self.config.chunk_size:
                        self.samples.extend(chunk_samples)
                        chunk_samples = []
                        
                        if line_num % 10000 == 0:
                            self.logger.debug(f"Processed {line_num} lines, {len(self.samples)} samples")
                
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
        
        # Add remaining samples
        if chunk_samples:
            self.samples.extend(chunk_samples)
    
    def _load_data_direct(self, file_path: Path):
        """Load data directly into memory."""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    sample = self._parse_fever_sample(data, line_num)
                    if sample:
                        self.samples.append(sample)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
    
    def _parse_fever_sample(self, data: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
        """
        Parse a single FEVER sample from JSON data.
        
        Args:
            data: Raw JSON data
            line_num: Line number for error reporting
            
        Returns:
            Parsed sample dictionary or None if invalid
        """
        try:
            # Extract core fields
            sample = {
                'id': data.get('id', line_num),
                'claim': data.get('claim', '').strip(),
                'label': data.get('label', '').strip(),
                'evidence': [],
                'evidence_metadata': []
            }
            
            # Validate required fields
            if not sample['claim'] or not sample['label']:
                return None
            
            # Map label to integer
            if sample['label'] not in self.config.label_mapping:
                self.logger.warning(f"Unknown label '{sample['label']}' at line {line_num}")
                return None
            
            sample['label_id'] = self.config.label_mapping[sample['label']]
            
            # Process evidence if available
            if 'evidence' in data and data['evidence']:
                evidence_texts = []
                evidence_meta = []
                
                # Extract evidence from annotation structure
                for evidence_set in data['evidence']:
                    for evidence_item in evidence_set:
                        if len(evidence_item) >= 3:
                            evidence_id = evidence_item[0]
                            evidence_sent_id = evidence_item[1] 
                            evidence_text = evidence_item[2] if len(evidence_item) > 2 else ""
                            
                            if evidence_text and evidence_text.strip():
                                evidence_texts.append(evidence_text.strip())
                                
                                if self.config.include_evidence_metadata:
                                    evidence_meta.append({
                                        'evidence_id': evidence_id,
                                        'sentence_id': evidence_sent_id
                                    })
                        
                        # Limit evidence sentences
                        if len(evidence_texts) >= self.config.max_evidence_sentences:
                            break
                    
                    if len(evidence_texts) >= self.config.max_evidence_sentences:
                        break
                
                sample['evidence'] = evidence_texts
                sample['evidence_metadata'] = evidence_meta
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Error parsing sample at line {line_num}: {e}")
            return None
    
    def _filter_samples(self):
        """Filter samples based on configuration criteria."""
        
        original_count = len(self.samples)
        filtered_samples = []
        
        for sample in self.samples:
            # Filter by claim length
            if len(sample['claim'].split()) < self.config.min_claim_length:
                continue
            
            # Filter samples with no evidence if configured
            if self.config.filter_no_evidence and not sample['evidence']:
                continue
            
            # Filter by evidence length
            if sample['evidence']:
                total_evidence_length = sum(len(ev.split()) for ev in sample['evidence'])
                if total_evidence_length < self.config.min_evidence_length:
                    continue
            
            filtered_samples.append(sample)
        
        self.samples = filtered_samples
        filtered_count = original_count - len(self.samples)
        
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} samples ({filtered_count/original_count*100:.1f}%)")
    
    def _prepare_text_inputs(self, claim: str, evidence: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare text inputs for the model.
        
        Args:
            claim: Claim text
            evidence: List of evidence sentences
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Combine evidence sentences
        if evidence:
            combined_evidence = self.config.evidence_separator.join(evidence)
            
            # Create input pair
            text_pair = (claim, combined_evidence)
        else:
            # Claim only
            text_pair = claim
            combined_evidence = ""
        
        # Cache key for processed inputs
        cache_key = f"{hash(claim)}_{hash(combined_evidence)}"
        
        if self.config.cache_processed_data and cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        # Tokenize with text processor
        if isinstance(text_pair, tuple):
            # Claim-evidence pair
            inputs = self.text_processor.process_text_pair(
                text_pair[0], text_pair[1],
                max_length_a=self.config.max_claim_length,
                max_length_b=self.config.max_evidence_length,
                truncation=self.config.truncation_strategy
            )
        else:
            # Claim only
            inputs = self.text_processor.process_text(
                text_pair,
                max_length=self.config.max_claim_length,
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
        text_inputs = self._prepare_text_inputs(sample['claim'], sample['evidence'])
        
        # Create return dictionary
        result = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'label': torch.tensor(sample['label_id'], dtype=torch.long)
        }
        
        # Add evidence-specific inputs if available
        if sample['evidence']:
            # Also provide separate evidence encoding
            evidence_text = self.config.evidence_separator.join(sample['evidence'])
            evidence_inputs = self.text_processor.process_text(
                evidence_text,
                max_length=self.config.max_evidence_length,
                truncation=True
            )
            
            result['evidence_input_ids'] = evidence_inputs['input_ids']
            result['evidence_attention_mask'] = evidence_inputs['attention_mask']
        else:
            # Dummy evidence inputs for consistency
            dummy_inputs = self.text_processor.process_text(
                "", max_length=self.config.max_evidence_length
            )
            result['evidence_input_ids'] = dummy_inputs['input_ids']
            result['evidence_attention_mask'] = dummy_inputs['attention_mask']
        
        # Add metadata
        result['sample_id'] = sample['id']
        result['claim_text'] = sample['claim']
        result['evidence_text'] = sample['evidence']
        result['label_text'] = sample['label']
        
        return result
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        
        stats = {
            'total_samples': len(self.samples),
            'label_distribution': self.get_label_distribution(),
            'avg_claim_length': 0,
            'avg_evidence_length': 0,
            'samples_with_evidence': 0,
            'avg_evidence_sentences': 0
        }
        
        if self.samples:
            claim_lengths = [len(s['claim'].split()) for s in self.samples]
            evidence_lengths = []
            evidence_counts = []
            
            for sample in self.samples:
                if sample['evidence']:
                    stats['samples_with_evidence'] += 1
                    evidence_counts.append(len(sample['evidence']))
                    total_ev_length = sum(len(ev.split()) for ev in sample['evidence'])
                    evidence_lengths.append(total_ev_length)
                else:
                    evidence_lengths.append(0)
                    evidence_counts.append(0)
            
            stats['avg_claim_length'] = np.mean(claim_lengths)
            stats['avg_evidence_length'] = np.mean(evidence_lengths)
            stats['avg_evidence_sentences'] = np.mean(evidence_counts) if evidence_counts else 0
            stats['evidence_coverage'] = stats['samples_with_evidence'] / len(self.samples) * 100
        
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
            use_chunked: Whether to use chunked loading for MacBook M2
            
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
        
        for key in ['input_ids', 'attention_mask', 'evidence_input_ids', 'evidence_attention_mask', 'label']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Keep metadata as lists
        for key in ['sample_id', 'claim_text', 'evidence_text', 'label_text']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated


def main():
    """Example usage of FeverDataset."""
    
    # Create dataset configuration
    config = FeverDatasetConfig(
        max_claim_length=128,
        max_evidence_length=256,
        max_evidence_sentences=3,
        filter_no_evidence=False
    )
    
    try:
        # Load training data
        print("Loading FEVER training dataset...")
        train_dataset = FeverDataset('train', config=config)
        
        print(f"Dataset size: {len(train_dataset)}")
        
        # Show statistics
        stats = train_dataset.get_dataset_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Show sample
        print("\nSample data:")
        sample = train_dataset[0]
        
        print(f"Claim: {sample['claim_text']}")
        print(f"Evidence: {sample['evidence_text']}")
        print(f"Label: {sample['label_text']} ({sample['label'].item()})")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Evidence shape: {sample['evidence_input_ids'].shape}")
        
        # Create dataloader
        print("\nCreating DataLoader...")
        dataloader = train_dataset.create_dataloader(batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure FEVER data files are in the correct location:")
        print("  data/FEVER/fever_train.jsonl")
        print("  data/FEVER/fever_test.jsonl")


if __name__ == "__main__":
    main()
