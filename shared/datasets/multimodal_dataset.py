"""
Multimodal Dataset Implementations for FactCheck-MM
Unified datasets for text, audio, image, and video processing.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, ConcatDataset

from .base_dataset import BaseDataset, DatasetConfig
from ..utils import get_logger
from ..preprocessing import TextProcessor, AudioProcessor, ImageProcessor, VideoProcessor


class MultimodalDataset(BaseDataset):
    """
    Unified multimodal dataset that handles text, audio, image, and video data.
    """
    
    def __init__(
        self,
        configs: List[DatasetConfig],
        split: str = "train",
        task_name: str = "classification",
        processors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            configs: List of dataset configurations
            split: Dataset split
            task_name: Task name for preprocessing
            processors: Modality processors
            **kwargs: Additional arguments
        """
        self.task_name = task_name
        self.configs = configs
        self.datasets = []
        
        self.logger = get_logger(f"MultimodalDataset_{task_name}")
        
        # Initialize individual datasets
        for config in configs:
            try:
                dataset = BaseDataset(config, split, processors, **kwargs)
                self.datasets.append(dataset)
                self.logger.info(f"Added dataset: {config.name} ({len(dataset)} samples)")
            except Exception as e:
                self.logger.error(f"Failed to load dataset {config.name}: {e}")
        
        # Combine all datasets
        if self.datasets:
            self.combined_dataset = ConcatDataset(self.datasets)
        else:
            raise ValueError("No datasets loaded successfully")
        
        self.logger.info(f"Initialized multimodal dataset: {len(self.combined_dataset)} total samples")
    
    def __len__(self) -> int:
        """Get total dataset length."""
        return len(self.combined_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multimodal sample."""
        return self.combined_dataset[idx]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about constituent datasets."""
        info = {
            'total_samples': len(self.combined_dataset),
            'num_datasets': len(self.datasets),
            'datasets': []
        }
        
        for dataset in self.datasets:
            dataset_info = {
                'name': dataset.config.name,
                'samples': len(dataset),
                'modalities': dataset.config.modalities,
                'split': dataset.split
            }
            info['datasets'].append(dataset_info)
        
        return info


class SarcasmDataset(MultimodalDataset):
    """Specialized dataset for sarcasm detection task."""
    
    def __init__(
        self,
        dataset_names: List[str],
        data_dir: Path,
        split: str = "train",
        **kwargs
    ):
        """
        Initialize sarcasm detection dataset.
        
        Args:
            dataset_names: Names of sarcasm datasets to use
            data_dir: Root data directory
            split: Dataset split
            **kwargs: Additional arguments
        """
        configs = []
        
        # Dataset configurations for sarcasm detection
        dataset_configs = {
            'mustard': DatasetConfig(
                name='mustard',
                path=data_dir / 'mustard_repo',
                modalities=['text', 'audio', 'video'],
                train_file='data/sarcasm_data.json',
                format='json',
                text_column='utterance',
                label_column='sarcasm',
                audio_column='audio_file',
                video_column='video_file'
            ),
            'mmsd2': DatasetConfig(
                name='mmsd2',
                path=data_dir / 'mmsd2',
                modalities=['text', 'image'],
                train_file='train_data.csv' if split == 'train' else f'{split}_data.csv',
                format='csv',
                text_column='text',
                label_column='label',
                image_column='image_path'
            ),
            'sarcnet': DatasetConfig(
                name='sarcnet',
                path=data_dir / 'sarcnet',
                modalities=['text', 'image'],
                train_file='train.json' if split == 'train' else f'{split}.json',
                format='json',
                text_column='text',
                label_column='sarcastic',
                image_column='image_path'
            ),
            'sarc': DatasetConfig(
                name='sarc',
                path=data_dir / 'sarc',
                modalities=['text'],
                train_file='train-balanced-sarcasm.csv',
                format='csv',
                text_column='comment',
                label_column='label'
            ),
            'sarcasm_headlines': DatasetConfig(
                name='sarcasm_headlines',
                path=data_dir / 'Sarcasm Headlines',
                modalities=['text'],
                train_file='Sarcasm_Headlines_Dataset.json',
                format='json',
                text_column='headline',
                label_column='is_sarcastic'
            ),
            'spanish_sarcasm': DatasetConfig(
                name='spanish_sarcasm',
                path=data_dir / 'spanish_sarcasm',
                modalities=['text'],
                train_file='spanish_sarcasm_ULTIMATE_OPTIMIZED.csv',
                format='csv',
                text_column='text',
                label_column='sarcastic'
            ),
            'ur_funny': DatasetConfig(
                name='ur_funny',
                path=data_dir / 'ur_funny',
                modalities=['text', 'audio', 'video'],
                train_file='ur_funny.csv',
                format='csv',
                text_column='text',
                label_column='humor',  # Use humor as proxy for sarcasm
                audio_column='audio_file',
                video_column='video_file'
            )
        }
        
        # Select requested datasets
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                self.logger.warning(f"Unknown sarcasm dataset: {dataset_name}")
        
        super().__init__(configs, split, "sarcasm_detection", **kwargs)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw sarcasm data."""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                'id': item.get('id', f"sample_{len(processed_data)}"),
                'text': item.get(self.config.text_column, ''),
                'label': int(item.get(self.config.label_column, 0)),
                'dataset': self.config.name,
                'split': self.split
            }
            
            # Add multimodal paths if available
            if self.config.audio_column and self.config.audio_column in item:
                processed_item['audio_path'] = item[self.config.audio_column]
            
            if self.config.image_column and self.config.image_column in item:
                processed_item['image_path'] = item[self.config.image_column]
            
            if self.config.video_column and self.config.video_column in item:
                processed_item['video_path'] = item[self.config.video_column]
            
            processed_data.append(processed_item)
        
        return processed_data


class ParaphraseDataset(MultimodalDataset):
    """Specialized dataset for paraphrase generation task."""
    
    def __init__(
        self,
        dataset_names: List[str],
        data_dir: Path,
        split: str = "train",
        **kwargs
    ):
        """
        Initialize paraphrase dataset.
        
        Args:
            dataset_names: Names of paraphrase datasets to use
            data_dir: Root data directory
            split: Dataset split
            **kwargs: Additional arguments
        """
        configs = []
        
        # Dataset configurations for paraphrasing
        dataset_configs = {
            'paranmt': DatasetConfig(
                name='paranmt',
                path=data_dir / 'paranmt',
                modalities=['text'],
                train_file='para-nmt-5m-processed.txt',
                format='txt',
                text_column='source',
                label_column='target'
            ),
            'mrpc': DatasetConfig(
                name='mrpc',
                path=data_dir / 'MRPC',
                modalities=['text'],
                train_file='train.tsv' if split == 'train' else f'{split}.tsv',
                format='tsv',
                text_column='sentence1',
                label_column='sentence2'
            ),
            'quora': DatasetConfig(
                name='quora',
                path=data_dir / 'quora',
                modalities=['text'],
                train_file='train.csv' if split == 'train' else f'{split}.csv',
                format='csv',
                text_column='question1',
                label_column='question2'
            )
        }
        
        # Select requested datasets
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                self.logger.warning(f"Unknown paraphrase dataset: {dataset_name}")
        
        super().__init__(configs, split, "paraphrasing", **kwargs)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw paraphrase data."""
        processed_data = []
        
        for item in raw_data:
            source_text = item.get(self.config.text_column, '')
            target_text = item.get(self.config.label_column, '')
            
            if source_text and target_text:  # Only include valid pairs
                processed_item = {
                    'id': item.get('id', f"sample_{len(processed_data)}"),
                    'source': source_text,
                    'target': target_text,
                    'dataset': self.config.name,
                    'split': self.split
                }
                
                # Add quality score if available
                if 'quality' in item:
                    processed_item['quality'] = float(item['quality'])
                
                processed_data.append(processed_item)
        
        return processed_data


class FactVerificationDataset(MultimodalDataset):
    """Specialized dataset for fact verification task."""
    
    def __init__(
        self,
        dataset_names: List[str],
        data_dir: Path,
        split: str = "train",
        **kwargs
    ):
        """
        Initialize fact verification dataset.
        
        Args:
            dataset_names: Names of fact verification datasets to use
            data_dir: Root data directory
            split: Dataset split
            **kwargs: Additional arguments
        """
        configs = []
        
        # Dataset configurations for fact verification
        dataset_configs = {
            'fever': DatasetConfig(
                name='fever',
                path=data_dir / 'FEVER',
                modalities=['text'],
                train_file='fever_train.jsonl' if split == 'train' else 'fever_test.jsonl',
                format='jsonl',
                text_column='claim',
                label_column='label'
            ),
            'liar': DatasetConfig(
                name='liar',
                path=data_dir / 'LIAR',
                modalities=['text'],
                train_file='train_formatted.csv' if split == 'train' else f'{split}.tsv',
                format='csv' if split == 'train' else 'tsv',
                text_column='statement',
                label_column='label'
            )
        }
        
        # Select requested datasets
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                self.logger.warning(f"Unknown fact verification dataset: {dataset_name}")
        
        super().__init__(configs, split, "fact_verification", **kwargs)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw fact verification data."""
        processed_data = []
        
        for item in raw_data:
            claim = item.get(self.config.text_column, '')
            label = item.get(self.config.label_column, 'NOT ENOUGH INFO')
            
            # Standardize FEVER labels
            if label.upper() in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                standardized_label = label.upper()
            elif label in ['true', 'false', 'mostly-true', 'mostly-false']:
                # LIAR labels
                if label in ['true', 'mostly-true']:
                    standardized_label = 'SUPPORTS'
                else:
                    standardized_label = 'REFUTES'
            else:
                standardized_label = 'NOT ENOUGH INFO'
            
            if claim:  # Only include valid claims
                processed_item = {
                    'id': item.get('id', f"sample_{len(processed_data)}"),
                    'claim': claim,
                    'label': standardized_label,
                    'evidence': item.get('evidence', []),
                    'dataset': self.config.name,
                    'split': self.split
                }
                
                # Add additional metadata
                if 'subject' in item:
                    processed_item['subject'] = item['subject']
                if 'context' in item:
                    processed_item['context'] = item['context']
                
                processed_data.append(processed_item)
        
        return processed_data


class ChunkedDataset:
    """
    Dataset wrapper for chunked loading on memory-constrained devices.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        chunk_size: int = 1000,
        shuffle_chunks: bool = True,
        cache_chunks: bool = True
    ):
        """
        Initialize chunked dataset.
        
        Args:
            base_dataset: Base dataset to chunk
            chunk_size: Size of each chunk
            shuffle_chunks: Whether to shuffle chunks
            cache_chunks: Whether to cache chunks in memory
        """
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.cache_chunks = cache_chunks
        
        self.logger = get_logger("ChunkedDataset")
        
        # Calculate chunks
        self.total_samples = len(base_dataset)
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        
        # Create chunk indices
        self.chunk_indices = self._create_chunk_indices()
        
        # Current chunk state
        self.current_chunk = 0
        self.current_chunk_data = None
        self.chunk_cache = {}
        
        self.logger.info(
            f"Initialized chunked dataset: {self.total_samples} samples, "
            f"{self.num_chunks} chunks of size {chunk_size}"
        )
    
    def _create_chunk_indices(self) -> List[List[int]]:
        """Create indices for each chunk."""
        indices = list(range(self.total_samples))
        
        if self.shuffle_chunks:
            np.random.shuffle(indices)
        
        chunks = []
        for i in range(0, self.total_samples, self.chunk_size):
            chunk_indices = indices[i:i + self.chunk_size]
            chunks.append(chunk_indices)
        
        return chunks
    
    def load_chunk(self, chunk_idx: int) -> List[Any]:
        """
        Load specific chunk into memory.
        
        Args:
            chunk_idx: Chunk index to load
            
        Returns:
            List of samples in chunk
        """
        if chunk_idx >= self.num_chunks:
            raise IndexError(f"Chunk index {chunk_idx} out of range (0-{self.num_chunks-1})")
        
        # Check cache first
        if self.cache_chunks and chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        # Load chunk data
        chunk_indices = self.chunk_indices[chunk_idx]
        chunk_data = []
        
        for idx in chunk_indices:
            try:
                sample = self.base_dataset[idx]
                chunk_data.append(sample)
            except Exception as e:
                self.logger.debug(f"Failed to load sample {idx}: {e}")
        
        # Cache if enabled
        if self.cache_chunks:
            self.chunk_cache[chunk_idx] = chunk_data
        
        self.logger.debug(f"Loaded chunk {chunk_idx}: {len(chunk_data)} samples")
        return chunk_data
    
    def __iter__(self):
        """Iterate over chunks."""
        for chunk_idx in range(self.num_chunks):
            chunk_data = self.load_chunk(chunk_idx)
            yield chunk_idx, chunk_data
    
    def clear_cache(self):
        """Clear chunk cache."""
        self.chunk_cache.clear()
        self.logger.info("Chunk cache cleared")
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about chunks."""
        return {
            'total_samples': self.total_samples,
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size,
            'cached_chunks': len(self.chunk_cache),
            'memory_usage_mb': sum(len(str(chunk)) for chunk in self.chunk_cache.values()) / (1024 * 1024)
        }
