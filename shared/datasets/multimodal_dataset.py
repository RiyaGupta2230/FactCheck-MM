# shared/datasets/multimodal_dataset.py

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
        max_dataset_contribution: float = 0.5,
        **kwargs
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            configs: List of dataset configurations
            split: Dataset split
            task_name: Task name for preprocessing
            processors: Modality processors
            max_dataset_contribution: Max proportion any single dataset can contribute
            **kwargs: Additional arguments
        """
        self.task_name = task_name
        self.configs = configs
        self.datasets = []
        self.dataset_lengths = []
        self.max_dataset_contribution = max_dataset_contribution
        self.logger = get_logger(f"MultimodalDataset_{task_name}")
        
        for config in configs:
            try:
                dataset = BaseDataset(config, split, processors, **kwargs)
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))
                self.logger.info(f"Added dataset: {config.name} ({len(dataset)} samples)")
            except Exception as e:
                self.logger.error(f"Failed to load dataset {config.name}: {e}")
                raise
        
        if not self.datasets:
            raise ValueError("No datasets loaded successfully")
        
        self._apply_dataset_balancing()
        self.combined_dataset = ConcatDataset(self.datasets)
        self._log_contribution_ratios()
        self.logger.info(f"Initialized multimodal dataset: {len(self.combined_dataset)} total samples")
    
    def _apply_dataset_balancing(self):
        """
        Enforce dataset-level balancing to prevent one dataset from dominating.
        This is critical for research-grade results.
        """
        if len(self.datasets) <= 1:
            return
        
        total_samples = sum(self.dataset_lengths)
        max_allowed = int(total_samples * self.max_dataset_contribution)
        
        balanced_datasets = []
        balanced_lengths = []
        
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            if length > max_allowed:
                self.logger.warning(
                    f"Dataset {dataset.config.name} has {length} samples, "
                    f"capping to {max_allowed} (max {self.max_dataset_contribution*100}%)"
                )
                indices = np.random.permutation(length)[:max_allowed]
                limited_data = [dataset.data[i] for i in indices]
                dataset.data = limited_data
                balanced_datasets.append(dataset)
                balanced_lengths.append(max_allowed)
            else:
                balanced_datasets.append(dataset)
                balanced_lengths.append(length)
        
        self.datasets = balanced_datasets
        self.dataset_lengths = balanced_lengths
    
    def _log_contribution_ratios(self):
        """Log final per-dataset contribution ratios."""
        total = sum(self.dataset_lengths)
        self.logger.info("=== Dataset Contribution Ratios ===")
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            ratio = length / total if total > 0 else 0
            self.logger.info(f"  {dataset.config.name}: {length} samples ({ratio*100:.2f}%)")
        self.logger.info("=" * 35)
    
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
        max_samples_per_dataset: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize sarcasm detection dataset.
        
        Args:
            dataset_names: Names of sarcasm datasets to use
            data_dir: Root data directory
            split: Dataset split
            max_samples_per_dataset: Maximum samples per dataset (for balancing)
            **kwargs: Additional arguments
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        data_dir = data_dir.resolve().absolute()
        
        configs = []
        logger = get_logger("SarcasmDataset")
        
        dataset_configs = {
            'mustard': DatasetConfig(
                name='mustard',
                path=data_dir / 'mustard_repo',
                modalities=['text', 'audio', 'video'],
                train_file='data/sarcasm_data.json',
                format='json',
                text_column='utterance',
                label_column='sarcasm',
                audio_features_file='data/audio_features.p',
                max_samples=max_samples_per_dataset,
                balance_strategy=None
            ),
            'mmsd2': DatasetConfig(
                name='mmsd2',
                path=data_dir / 'mmsd2',
                modalities=['text', 'image'],
                train_file='text_json_final/train.json',
                val_file='text_json_final/valid.json',
                test_file='text_json_final/test.json',
                format='json',
                text_column='text',
                label_column='label',
                image_column='imageid',
                max_samples=max_samples_per_dataset
            ),
            'sarcnet': DatasetConfig(
                name='sarcnet',
                path=data_dir / 'sarcnet' / 'SarcNet Image-Text',
                modalities=['text', 'image'],
                train_file='SarcNetTrain.csv',
                val_file='SarcNetVal.csv',
                test_file='SarcNetTest.csv',
                format='csv',
                text_column='Text',
                label_column='Multilabel',
                image_column='Imagepath',
                max_samples=max_samples_per_dataset
            ),
            'sarc': DatasetConfig(
                name='sarc',
                path=data_dir / 'sarc',
                modalities=['text'],
                train_file='train-balanced-sarcasm.csv',
                format='csv',
                text_column='comment',
                label_column='label',
                max_samples=max_samples_per_dataset,
                balance_strategy='cap_max_samples',
                balance_config={'max_samples': 50000}
            ),
            'sarcasm_headlines': DatasetConfig(
                name='sarcasm_headlines',
                path=data_dir / 'Sarcasm Headlines',
                modalities=['text'],
                train_file='Sarcasm_Headlines_Dataset.json',
                format='json',
                text_column='headline',
                label_column='is_sarcastic',
                max_samples=max_samples_per_dataset
            )
        }
        
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                logger.warning(f"Unknown sarcasm dataset: {dataset_name}")
        
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
            
            if 'context' in item:
                processed_item['context'] = item['context']
            
            if self.config.name == 'mustard':
                utterance_id = item.get('show', '') + "_" + str(item.get('id', ''))
                
                if self.audio_features_dict:
                    processed_item['audio_features_key'] = utterance_id
                
                video_path = self.config.path / 'data' / 'videos' / 'utterances_final' / f'{utterance_id}.mp4'
                if video_path.exists():
                    processed_item['video_path'] = str(video_path)
                
                context_videos = []
                context_texts = item.get('context', [])
                if context_texts:
                    context_video_dir = self.config.path / 'data' / 'videos' / 'context_final'
                    for ctx_idx in range(len(context_texts)):
                        context_video_id = f"{utterance_id}_C{ctx_idx}"
                        context_video_path = context_video_dir / f'{context_video_id}.mp4'
                        if context_video_path.exists():
                            context_videos.append(str(context_video_path))
                
                if context_videos:
                    processed_item['context_video_paths'] = context_videos
            
            elif self.config.name == 'mmsd2' and self.config.image_column in item:
                image_id = str(item[self.config.image_column])
                if not image_id.endswith('.jpg'):
                    image_id = f'{image_id}.jpg'
                image_path = self.config.path / 'dataset_image' / image_id
                if image_path.exists():
                    processed_item['image_path'] = str(image_path)
            
            elif self.config.name == 'sarcnet':
                if 'Textlabel' in item:
                    processed_item['text_label'] = int(item['Textlabel'])
                if 'Imagelabel' in item:
                    processed_item['image_label'] = int(item['Imagelabel'])
                
                if self.config.image_column and self.config.image_column in item:
                    image_path = self.config.path / item[self.config.image_column]
                    if image_path.exists():
                        processed_item['image_path'] = str(image_path)
            
            elif self.config.image_column and self.config.image_column in item:
                processed_item['image_path'] = item[self.config.image_column]
            
            processed_data.append(processed_item)
        
        return processed_data

class ParaphraseDataset(MultimodalDataset):
    """Specialized dataset for paraphrase generation task."""
    
    def __init__(
        self,
        dataset_names: List[str],
        data_dir: Path,
        split: str = "train",
        max_samples_per_dataset: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize paraphrase dataset.
        
        Args:
            dataset_names: Names of paraphrase datasets to use
            data_dir: Root data directory
            split: Dataset split
            max_samples_per_dataset: Maximum samples per dataset (for balancing)
            **kwargs: Additional arguments
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        data_dir = data_dir.resolve().absolute()
        
        configs = []
        logger = get_logger("ParaphraseDataset")
        
        dataset_configs = {
            'paranmt': DatasetConfig(
                name='paranmt',
                path=data_dir / 'paranmt',
                modalities=['text'],
                train_file='para-nmt-5m-processed.txt',
                format='txt',
                max_samples=max_samples_per_dataset,
                balance_strategy='cap_max_samples',
                balance_config={'max_samples': 100000}
            ),
            'mrpc': DatasetConfig(
                name='mrpc',
                path=data_dir / 'MRPC',
                modalities=['text'],
                train_file='train.tsv',
                val_file='dev.tsv',
                test_file='test.tsv',
                format='tsv',
                text_column='#1 String',
                label_column='Quality',
                max_samples=max_samples_per_dataset
            ),
            'quora': DatasetConfig(
                name='quora',
                path=data_dir / 'quora',
                modalities=['text'],
                train_file='train.csv',
                test_file='test.csv',
                format='csv',
                text_column='question1',
                label_column='is_duplicate',
                max_samples=max_samples_per_dataset
            )
        }
        
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                logger.warning(f"Unknown paraphrase dataset: {dataset_name}")
        
        super().__init__(configs, split, "paraphrasing", **kwargs)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw paraphrase data."""
        processed_data = []
        
        for item in raw_data:
            if self.config.name == 'paranmt':
                line = item.get('line', '')
                parts = line.split('\t')
                if len(parts) >= 2:
                    source_text = parts[0]
                    target_text = parts[1]
                    score = float(parts[2]) if len(parts) > 2 else 1.0
                else:
                    continue
                
                processed_item = {
                    'id': f"sample_{len(processed_data)}",
                    'source': source_text,
                    'target': target_text,
                    'score': score,
                    'dataset': self.config.name,
                    'split': self.split
                }
            
            elif self.config.name == 'mrpc':
                sentence1 = item.get('#1 String', '')
                sentence2 = item.get('#2 String', '')
                quality = int(item.get('Quality', 0))
                
                if sentence1 and sentence2:
                    processed_item = {
                        'id': item.get('#1 ID', f"sample_{len(processed_data)}"),
                        'source': sentence1,
                        'target': sentence2,
                        'label': quality,
                        'dataset': self.config.name,
                        'split': self.split
                    }
                else:
                    continue
            
            elif self.config.name == 'quora':
                question1 = item.get('question1', '')
                question2 = item.get('question2', '')
                is_duplicate = int(item.get('is_duplicate', 0))
                
                if question1 and question2:
                    processed_item = {
                        'id': item.get('id', f"sample_{len(processed_data)}"),
                        'source': question1,
                        'target': question2,
                        'label': is_duplicate,
                        'dataset': self.config.name,
                        'split': self.split
                    }
                else:
                    continue
            
            else:
                continue
            
            processed_data.append(processed_item)
        
        return processed_data

class FactVerificationDataset(MultimodalDataset):
    """Specialized dataset for fact verification task."""
    
    def __init__(
        self,
        dataset_names: List[str],
        data_dir: Path,
        split: str = "train",
        max_samples_per_dataset: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize fact verification dataset.
        
        Args:
            dataset_names: Names of fact verification datasets to use
            data_dir: Root data directory
            split: Dataset split
            max_samples_per_dataset: Maximum samples per dataset (for balancing)
            **kwargs: Additional arguments
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        data_dir = data_dir.resolve().absolute()
        
        configs = []
        logger = get_logger("FactVerificationDataset")
        
        dataset_configs = {
            'fever': DatasetConfig(
                name='fever',
                path=data_dir / 'FEVER',
                modalities=['text'],
                train_file='fever_train.jsonl',
                test_file='fever_test.jsonl',
                format='jsonl',
                text_column='claim',
                label_column='label',
                max_samples=max_samples_per_dataset
            ),
            'liar': DatasetConfig(
                name='liar',
                path=data_dir / 'LIAR',
                modalities=['text', 'metadata'],
                train_file='train_formatted.csv',
                val_file='valid.tsv',
                test_file='test.tsv',
                format='csv',
                text_column='Statement',
                label_column='Label',
                max_samples=max_samples_per_dataset
            )
        }
        
        for dataset_name in dataset_names:
            if dataset_name in dataset_configs:
                configs.append(dataset_configs[dataset_name])
            else:
                logger.warning(f"Unknown fact verification dataset: {dataset_name}")
        
        super().__init__(configs, split, "fact_verification", **kwargs)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw fact verification data."""
        processed_data = []
        
        for item in raw_data:
            if self.config.name == 'fever':
                claim = item.get('claim', '')
                label = item.get('label', '')
                
                if claim and label:
                    processed_item = {
                        'id': item.get('id', f"sample_{len(processed_data)}"),
                        'text': claim,
                        'label': label,
                        'evidence_id': item.get('evidence_id', -1),
                        'evidence_wiki_url': item.get('evidence_wiki_url', ''),
                        'dataset': self.config.name,
                        'split': self.split
                    }
                    processed_data.append(processed_item)
            
            elif self.config.name == 'liar':
                statement = item.get('Statement', '')
                label = item.get('Label', '')
                
                if statement and label:
                    processed_item = {
                        'id': item.get('ID', f"sample_{len(processed_data)}"),
                        'text': statement,
                        'label': label,
                        'subject': item.get('Subject', ''),
                        'speaker': item.get('Speaker', ''),
                        'party': item.get('Party Affiliation', ''),
                        'context': item.get('Context', ''),
                        'dataset': self.config.name,
                        'split': self.split
                    }
                    processed_data.append(processed_item)
        
        return processed_data
