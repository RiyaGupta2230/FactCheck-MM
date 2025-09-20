"""
Base Dataset Classes for FactCheck-MM
Abstract dataset wrapper with common functionality.
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import logging

from ..utils import get_logger
from ..preprocessing import TextProcessor, AudioProcessor, ImageProcessor, VideoProcessor


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    name: str
    path: Path
    modalities: List[str]  # ['text', 'audio', 'image', 'video']
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    format: str = "json"  # json, csv, jsonl, tsv
    text_column: str = "text"
    label_column: str = "label"
    audio_column: Optional[str] = None
    image_column: Optional[str] = None
    video_column: Optional[str] = None
    max_samples: Optional[int] = None
    preprocessing_params: Optional[Dict[str, Any]] = None


class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset class for FactCheck-MM.
    Provides common functionality for loading and processing multimodal data.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = True,
        lazy_loading: bool = False
    ):
        """
        Initialize base dataset.
        
        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
            processors: Dictionary of modality processors
            cache_data: Whether to cache processed data
            lazy_loading: Whether to use lazy loading for large datasets
        """
        super().__init__()
        
        self.config = config
        self.split = split
        self.cache_data = cache_data
        self.lazy_loading = lazy_loading
        
        self.logger = get_logger(f"Dataset_{config.name}")
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors and 'text' in config.modalities:
            self.processors['text'] = TextProcessor()
        if 'audio' not in self.processors and 'audio' in config.modalities:
            self.processors['audio'] = AudioProcessor()
        if 'image' not in self.processors and 'image' in config.modalities:
            self.processors['image'] = ImageProcessor()
        if 'video' not in self.processors and 'video' in config.modalities:
            self.processors['video'] = VideoProcessor()
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(
            f"Initialized {config.name} dataset: {len(self.data)} samples, "
            f"split: {split}, modalities: {config.modalities}"
        )
    
    def _get_data_file(self) -> Optional[Path]:
        """Get the appropriate data file for the split."""
        if self.split == "train" and self.config.train_file:
            return self.config.path / self.config.train_file
        elif self.split == "val" and self.config.val_file:
            return self.config.path / self.config.val_file  
        elif self.split == "test" and self.config.test_file:
            return self.config.path / self.config.test_file
        elif self.config.train_file:  # Fallback to train file
            return self.config.path / self.config.train_file
        else:
            return None
    
    def _load_data(self):
        """Load data from file."""
        data_file = self._get_data_file()
        
        if data_file is None or not data_file.exists():
            self.logger.warning(f"Data file not found: {data_file}")
            return
        
        try:
            if self.config.format == "json":
                with open(data_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            elif self.config.format == "jsonl":
                raw_data = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        raw_data.append(json.loads(line.strip()))
            elif self.config.format == "csv":
                df = pd.read_csv(data_file)
                raw_data = df.to_dict('records')
            elif self.config.format == "tsv":
                df = pd.read_csv(data_file, sep='\t')
                raw_data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")
            
            # Process raw data
            self.data = self._process_raw_data(raw_data)
            
            # Apply max samples limit
            if self.config.max_samples and len(self.data) > self.config.max_samples:
                self.data = self.data[:self.config.max_samples]
            
            self.logger.info(f"Loaded {len(self.data)} samples from {data_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_file}: {e}")
            raise
    
    @abstractmethod
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw data into standardized format.
        
        Args:
            raw_data: Raw data from file
            
        Returns:
            Processed data samples
        """
        pass
    
    def _load_multimodal_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load multimodal data for a sample.
        
        Args:
            sample: Data sample
            
        Returns:
            Sample with loaded multimodal data
        """
        processed_sample = sample.copy()
        
        # Process text
        if 'text' in self.config.modalities and 'text' in sample:
            text_data = sample['text']
            if isinstance(text_data, str) and text_data:
                processed_sample['text_processed'] = self.processors['text'].preprocess_text(
                    text_data, 
                    task=getattr(self, 'task_name', 'classification')
                )
        
        # Process audio
        if 'audio' in self.config.modalities and self.config.audio_column in sample:
            audio_path = sample[self.config.audio_column]
            if audio_path and Path(audio_path).exists():
                try:
                    audio_data = self.processors['audio'].process_audio(audio_path)
                    processed_sample['audio_processed'] = audio_data
                except Exception as e:
                    self.logger.debug(f"Failed to process audio {audio_path}: {e}")
        
        # Process image
        if 'image' in self.config.modalities and self.config.image_column in sample:
            image_path = sample[self.config.image_column]
            if image_path and Path(image_path).exists():
                try:
                    image_data = self.processors['image'].process_image(image_path)
                    processed_sample['image_processed'] = image_data
                except Exception as e:
                    self.logger.debug(f"Failed to process image {image_path}: {e}")
        
        # Process video
        if 'video' in self.config.modalities and self.config.video_column in sample:
            video_path = sample[self.config.video_column]
            if video_path and Path(video_path).exists():
                try:
                    video_data = self.processors['video'].process_video(video_path)
                    processed_sample['video_processed'] = video_data
                except Exception as e:
                    self.logger.debug(f"Failed to process video {video_path}: {e}")
        
        return processed_sample
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Processed sample
        """
        # Check cache first
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        # Get raw sample
        sample = self.data[idx].copy()
        
        # Load multimodal data
        if not self.lazy_loading:
            sample = self._load_multimodal_data(sample)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Get sample by ID.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Sample data or None
        """
        for sample in self.data:
            if sample.get('id') == sample_id:
                return self._load_multimodal_data(sample)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(self.data),
            'modalities': self.config.modalities,
            'split': self.split
        }
        
        # Text statistics
        if 'text' in self.config.modalities:
            text_lengths = []
            for sample in self.data:
                if 'text' in sample and sample['text']:
                    text_lengths.append(len(sample['text'].split()))
            
            if text_lengths:
                stats['text_stats'] = {
                    'mean_length': np.mean(text_lengths),
                    'std_length': np.std(text_lengths),
                    'min_length': np.min(text_lengths),
                    'max_length': np.max(text_lengths)
                }
        
        # Label distribution
        if self.config.label_column in self.data[0]:
            labels = [sample[self.config.label_column] for sample in self.data]
            unique_labels, counts = np.unique(labels, return_counts=True)
            stats['label_distribution'] = dict(zip(unique_labels, counts.tolist()))
        
        return stats
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.logger.info("Dataset cache cleared")
    
    def save_processed_data(self, output_path: Path):
        """
        Save processed data to file.
        
        Args:
            output_path: Output file path
        """
        processed_data = []
        for i in range(len(self.data)):
            sample = self.__getitem__(i)
            processed_data.append(sample)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved processed data to {output_path}")
    
    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        random_seed: int = 42
    ) -> Tuple['BaseDataset', 'BaseDataset', 'BaseDataset']:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to stratify by labels
            random_seed: Random seed
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        np.random.seed(random_seed)
        
        # Get indices
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        
        if stratify and self.config.label_column in self.data[0]:
            # Stratified split
            from sklearn.model_selection import train_test_split
            
            labels = [sample[self.config.label_column] for sample in self.data]
            
            # First split: train vs (val + test)
            train_idx, temp_idx = train_test_split(
                indices, test_size=(val_ratio + test_ratio),
                stratify=labels, random_state=random_seed
            )
            
            # Second split: val vs test
            temp_labels = [labels[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_ratio/(val_ratio + test_ratio),
                stratify=temp_labels, random_state=random_seed
            )
        else:
            # Random split
            np.random.shuffle(indices)
            
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
        
        # Create split datasets
        train_data = [self.data[i] for i in train_idx]
        val_data = [self.data[i] for i in val_idx]
        test_data = [self.data[i] for i in test_idx]
        
        # Create new dataset instances
        train_dataset = self.__class__(self.config, split="train", processors=self.processors)
        val_dataset = self.__class__(self.config, split="val", processors=self.processors)
        test_dataset = self.__class__(self.config, split="test", processors=self.processors)
        
        train_dataset.data = train_data
        val_dataset.data = val_data
        test_dataset.data = test_data
        
        return train_dataset, val_dataset, test_dataset
