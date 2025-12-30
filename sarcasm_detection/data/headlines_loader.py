# sarcasm_detection/data/headlines_loader.py

"""
Sarcasm Headlines Dataset Loader for Sarcasm Detection
News headlines dataset with 28.6K samples.
Strictly follows PDF specification.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor
from shared.utils import get_logger


class HeadlinesDataset(Dataset):
    """
    Sarcasm Headlines Dataset Loader.
    - 28,619 news headlines
    - Binary classification: sarcastic (The Onion) vs non-sarcastic (HuffPost)
    - Text-only modality
    
    Path structure (as per PDF):
    Sarcasm Headlines/
    └── Sarcasm_Headlines_Dataset.json
    
    JSON Schema (per object):
    - is_sarcastic (0/1)
    - headline (string)
    - article_link (string)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = True,
        random_seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.cache_data = cache_data
        self.random_seed = random_seed
        self.logger = get_logger("HeadlinesDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                max_length=128,  # Headlines are typically short
                add_sarcasm_markers=True,
                clean_html=True,
                normalize_whitespace=True
            )
        
        self.data = []
        self.cache = {}
        
        self._load_data()
        
        self.logger.info(f"Loaded Headlines dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load Headlines dataset from JSON file."""
        data_file = self.data_dir / "Sarcasm_Headlines_Dataset.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Headlines data file not found: {data_file}")
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            all_samples = []
            for idx, item in enumerate(raw_data):
                headline = str(item.get('headline', ''))
                
                if headline and headline.strip():
                    sample = {
                        'id': f"sarcasm_headlines_{idx}",
                        'text': headline,
                        'label': int(item.get('is_sarcastic', 0)),  # 0=non-sarcastic, 1=sarcastic
                        'article_link': item.get('article_link', ''),
                        'dataset': 'sarcasm_headlines',
                        'split': self.split
                    }
                    all_samples.append(sample)
            
            # Create split
            self.data = self._create_split(all_samples)
            
            # Apply max samples with stratification
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load Headlines dataset: {e}")
            raise
    
    def _create_split(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create train/val/test split."""
        np.random.shuffle(all_samples)
        
        n_samples = len(all_samples)
        if self.split == "train":
            return all_samples[:int(0.8 * n_samples)]
        elif self.split == "val":
            return all_samples[int(0.8 * n_samples):int(0.9 * n_samples)]
        elif self.split == "test":
            return all_samples[int(0.9 * n_samples):]
        else:
            return all_samples
    
    def _stratified_sample(self, data: List[Dict[str, Any]], n_samples: int) -> List[Dict[str, Any]]:
        """Stratified sampling to maintain class balance."""
        sarcastic = [s for s in data if s['label'] == 1]
        non_sarcastic = [s for s in data if s['label'] == 0]
        
        n_sarcastic = min(len(sarcastic), n_samples // 2)
        n_non_sarcastic = min(len(non_sarcastic), n_samples - n_sarcastic)
        
        np.random.shuffle(sarcastic)
        np.random.shuffle(non_sarcastic)
        
        sampled = sarcastic[:n_sarcastic] + non_sarcastic[:n_non_sarcastic]
        np.random.shuffle(sampled)
        
        return sampled
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample = self.data[idx].copy()
        
        # Process text
        text_tokens = None
        if sample['text']:
            processed_text = self.processors['text'].preprocess_text(
                sample['text'],
                task="sarcasm_detection"
            )
            tokenized = self.processors['text'].tokenize(
                processed_text,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            text_tokens = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': None,
            'video': None,
            'image': None,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'article_link': sample['article_link'],
                'original_text': sample['text']
            }
        }
        
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        labels = [sample['label'] for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'median_text_length': np.median([len(s['text'].split()) for s in self.data if s['text']]),
            'min_text_length': min([len(s['text'].split()) for s in self.data if s['text']], default=0),
            'max_text_length': max([len(s['text'].split()) for s in self.data if s['text']], default=0),
            'modalities': ['text']
        }
