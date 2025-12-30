# shared/datasets/base_dataset.py

"""
Base Dataset Classes for FactCheck-MM
Abstract dataset wrapper with common functionality.
"""

import json
import csv
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    modalities: List[str]
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    format: str = "json"
    text_column: str = "text"
    label_column: str = "label"
    audio_column: Optional[str] = None
    image_column: Optional[str] = None
    video_column: Optional[str] = None
    max_samples: Optional[int] = None
    preprocessing_params: Optional[Dict[str, Any]] = None
    balance_strategy: Optional[str] = None
    balance_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    random_seed: int = 42
    audio_features_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate and resolve path to absolute."""
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        self.path = self.path.resolve().absolute()
        
        logger = get_logger(f"DatasetConfig_{self.name}")
        logger.info(f"Resolved dataset path: {self.path}")
        
        if not self.path.exists():
            logger.warning(f"Dataset path does not exist: {self.path}")


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
        
        np.random.seed(config.random_seed)
        
        self.processors = processors or {}
        
        if 'text' not in self.processors and 'text' in config.modalities:
            self.processors['text'] = TextProcessor()
        if 'audio' not in self.processors and 'audio' in config.modalities:
            self.processors['audio'] = AudioProcessor()
        if 'image' not in self.processors and 'image' in config.modalities:
            self.processors['image'] = ImageProcessor()
        if 'video' not in self.processors and 'video' in config.modalities:
            self.processors['video'] = VideoProcessor()
        
        self.data = []
        self.cache = {}
        self.was_balanced = False
        self.audio_features_dict = {}
        
        # Load precomputed audio features if specified
        if config.audio_features_file:
            self._load_audio_features()
        
        self._load_data()
        self.logger.info(
            f"Initialized {config.name} dataset: {len(self.data)} samples, "
            f"split: {split}, modalities: {config.modalities}"
        )
    
    def _load_audio_features(self):
        """Load precomputed audio features from pickle file."""
        audio_features_path = self.config.path / self.config.audio_features_file
        if audio_features_path.exists():
            try:
                with open(audio_features_path, 'rb') as f:
                    self.audio_features_dict = pickle.load(f)
                self.logger.info(f"Loaded audio features from {audio_features_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load audio features: {e}")
        else:
            self.logger.warning(f"Audio features file not found: {audio_features_path}")
    
    def _get_data_file(self) -> Optional[Path]:
        """Get the appropriate data file for the split."""
        if self.split == "train" and self.config.train_file:
            return self.config.path / self.config.train_file
        elif self.split == "val" and self.config.val_file:
            return self.config.path / self.config.val_file
        elif self.split == "test" and self.config.test_file:
            return self.config.path / self.config.test_file
        elif self.config.train_file:
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
                        if line.strip():
                            raw_data.append(json.loads(line.strip()))
            elif self.config.format == "csv":
                df = pd.read_csv(data_file)
                raw_data = df.to_dict('records')
            elif self.config.format == "tsv":
                df = pd.read_csv(data_file, sep='\t')
                raw_data = df.to_dict('records')
            elif self.config.format == "txt":
                # ParaNMT format: TSV per line
                raw_data = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            raw_data.append({'line': line.strip()})
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")
            
            original_size = len(raw_data)
            self.logger.info(f"Loaded {original_size} raw samples from {data_file}")
            
            # Process raw data
            self.data = self._process_raw_data(raw_data)
            
            # Validate schema
            if len(self.data) > 0:
                self._validate_schema(self.data[0])
            
            # Apply balancing strategy if specified (only for train split)
            if self.config.balance_strategy and self.split == "train":
                pre_balance_size = len(self.data)
                pre_balance_dist = self._get_label_distribution(self.data)
                self.logger.info(
                    f"Pre-balancing: {pre_balance_size} samples, "
                    f"distribution: {pre_balance_dist}"
                )
                
                self.data = self._apply_balancing(self.data)
                self.was_balanced = True
                
                post_balance_size = len(self.data)
                post_balance_dist = self._get_label_distribution(self.data)
                self.logger.info(
                    f"Post-balancing: {post_balance_size} samples, "
                    f"distribution: {post_balance_dist}"
                )
            
            # Inject metadata
            self._inject_metadata()
            
            # Apply max samples limit if specified
            if self.config.max_samples and len(self.data) > self.config.max_samples:
                self.data = self.data[:self.config.max_samples]
                self.logger.info(
                    f"Final dataset: {len(self.data)} samples "
                    f"(original: {original_size}, balanced: {self.was_balanced})"
                )
        
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_file}: {e}")
            raise
    
    def _validate_schema(self, sample: Dict[str, Any]):
        """Validate schema - flexible for different tasks."""
        pass
    
    def _get_label_distribution(self, data: List[Dict[str, Any]]) -> Dict[Any, int]:
        """Get label distribution from dataset."""
        if not data or self.config.label_column not in data[0]:
            return {}
        
        labels = [sample.get(self.config.label_column) for sample in data if self.config.label_column in sample]
        if not labels:
            return {}
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))
    
    def _inject_metadata(self):
        """Inject dataset contribution metadata into each sample."""
        for sample in self.data:
            sample['_dataset_name'] = self.config.name
            sample['_balanced'] = self.was_balanced
    
    def _apply_balancing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply balancing strategy to the dataset.
        
        NOTE: This is dataset-level balancing. Task-level balancing is handled
        in MultimodalDataset to ensure no single dataset dominates.
        """
        if not self.config.balance_strategy:
            return data
        
        np.random.seed(self.config.random_seed)
        
        # Skip balancing for already-balanced datasets (as per PDF spec)
        skip_datasets = ['mustard', 'sarc']  # MUStARD: 690 balanced, SARC: already balanced
        if self.config.name in skip_datasets:
            self.logger.info(f"Skipping balancing for {self.config.name} (already balanced per spec)")
            return data
        
        if not data or self.config.label_column not in data[0]:
            return data
        
        strategy = self.config.balance_strategy
        
        if strategy == "cap_max_samples":
            # Cap extremely large datasets (ParaNMT, SARC)
            max_cap = self.config.balance_config.get("max_samples", 10000)
            if len(data) > max_cap:
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                data = [data[i] for i in indices[:max_cap]]
                self.logger.info(f"Capped dataset to {max_cap} samples (research balancing)")
            return data
        
        elif strategy == "proportional":
            target_size = self.config.balance_config.get("target_size", 5000)
            if len(data) > target_size:
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                data = [data[i] for i in indices[:target_size]]
                self.logger.info(f"Proportionally sampled to {target_size} samples")
            return data
        
        elif strategy in ["undersample", "oversample"]:
            return self._balance_classes(data, strategy)
        
        return data
    
    def _balance_classes(self, data: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        """Balance classes using undersampling or oversampling."""
        np.random.seed(self.config.random_seed)
        
        label_groups = {}
        for sample in data:
            label = sample[self.config.label_column]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        class_counts = {label: len(samples) for label, samples in label_groups.items()}
        self.logger.info(f"Original class distribution: {class_counts}")
        
        balanced_data = []
        
        if strategy == "undersample":
            min_count = min(class_counts.values())
            for label, samples in label_groups.items():
                if len(samples) > min_count:
                    indices = np.arange(len(samples))
                    np.random.shuffle(indices)
                    balanced_data.extend([samples[i] for i in indices[:min_count]])
                else:
                    balanced_data.extend(samples)
            self.logger.info(f"Undersampled to {min_count} samples per class")
        
        elif strategy == "oversample":
            max_count = max(class_counts.values())
            for label, samples in label_groups.items():
                if len(samples) < max_count:
                    oversampled = list(samples)
                    while len(oversampled) < max_count:
                        idx = np.random.randint(len(samples))
                        oversampled.append(samples[idx])
                    balanced_data.extend(oversampled[:max_count])
                else:
                    balanced_data.extend(samples)
            self.logger.info(f"Oversampled to {max_count} samples per class")
        
        indices = np.arange(len(balanced_data))
        np.random.shuffle(indices)
        balanced_data = [balanced_data[i] for i in indices]
        
        return balanced_data
    
    @abstractmethod
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw data into standardized format."""
        pass
    
    def _load_multimodal_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load multimodal data for a sample."""
        processed_sample = sample.copy()
        
        # Text processing
        if 'text' in self.config.modalities and 'text' in sample:
            text_data = sample['text']
            if isinstance(text_data, str) and text_data:
                processed_sample['text_processed'] = self.processors['text'].preprocess_text(
                    text_data,
                    task=getattr(self, 'task_name', 'classification')
                )
        
        # Audio processing - handle precomputed features
        if 'audio' in self.config.modalities:
            if 'audio_features_key' in sample and self.audio_features_dict:
                audio_key = sample['audio_features_key']
                if audio_key in self.audio_features_dict:
                    processed_sample['audio_processed'] = {
                        'audio': self.audio_features_dict[audio_key]
                    }
                else:
                    self.logger.warning(f"Audio features not found for key: {audio_key}")
                    processed_sample['audio_missing'] = True
            elif 'audio_path' in sample:
                audio_path = Path(sample['audio_path'])
                if audio_path.exists():
                    try:
                        audio_data = self.processors['audio'].process_audio(str(audio_path))
                        processed_sample['audio_processed'] = audio_data
                    except Exception as e:
                        self.logger.warning(f"Failed to process audio {audio_path}: {e}")
                        processed_sample['audio_missing'] = True
        
        # Image processing
        if 'image' in self.config.modalities and 'image_path' in sample:
            image_path = Path(sample['image_path'])
            if image_path.exists():
                try:
                    image_data = self.processors['image'].process_image(str(image_path))
                    processed_sample['image_processed'] = image_data
                except Exception as e:
                    self.logger.warning(f"Failed to process image {image_path}: {e}")
                    processed_sample['image_missing'] = True
            else:
                processed_sample['image_missing'] = True
        
        # Video processing - handle both utterance and context videos
        if 'video' in self.config.modalities and 'video_path' in sample:
            video_path = Path(sample['video_path'])
            if video_path.exists():
                try:
                    video_data = self.processors['video'].process_video(str(video_path))
                    processed_sample['video_processed'] = video_data
                except Exception as e:
                    self.logger.warning(f"Failed to process video {video_path}: {e}")
                    processed_sample['video_missing'] = True
        
        # Context videos (MUStARD specific)
        if 'context_video_paths' in sample:
            context_videos_processed = []
            for ctx_video_path in sample['context_video_paths']:
                video_path = Path(ctx_video_path)
                if video_path.exists():
                    try:
                        video_data = self.processors['video'].process_video(str(video_path))
                        context_videos_processed.append(video_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to process context video {video_path}: {e}")
            
            if context_videos_processed:
                processed_sample['context_videos_processed'] = context_videos_processed
        
        return processed_sample
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample = self.data[idx].copy()
        
        if not self.lazy_loading:
            sample = self._load_multimodal_data(sample)
        
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get sample by ID."""
        for sample in self.data:
            if sample.get('id') == sample_id:
                return self._load_multimodal_data(sample)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.data),
            'modalities': self.config.modalities,
            'split': self.split,
            'was_balanced': self.was_balanced
        }
        
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
        
        if len(self.data) > 0 and self.config.label_column in self.data[0]:
            labels = [sample[self.config.label_column] for sample in self.data if self.config.label_column in sample]
            if labels:
                unique_labels, counts = np.unique(labels, return_counts=True)
                stats['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        return stats
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.logger.info("Dataset cache cleared")
