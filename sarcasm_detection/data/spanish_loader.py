# sarcasm_detection/data/spanish_loader.py
"""
Spanish Sarcasm Dataset Loader for Sarcasm Detection
Multilingual sarcasm dataset with Spanish text.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor
from shared.utils import get_logger


class SpanishSarcasmDataset(Dataset):
    """
    Spanish Sarcasm Dataset Loader.
    
    Contains Spanish text samples for sarcasm detection,
    providing multilingual capability to the system.
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
        """
        Initialize Spanish Sarcasm dataset.
        
        Args:
            data_dir: Path to Spanish sarcasm dataset directory
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            processors: Dictionary of processors for each modality
            cache_data: Whether to cache processed data
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.cache_data = cache_data
        self.random_seed = random_seed
        
        self.logger = get_logger("SpanishSarcasmDataset")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                max_length=256,
                add_sarcasm_markers=True,
                lowercase=False,  # Keep original case for Spanish
                clean_html=True,
                normalize_whitespace=True
            )
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(f"Loaded Spanish Sarcasm dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load Spanish Sarcasm dataset from CSV file."""
        
        data_file = self.data_dir / "spanish_sarcasm_ULTIMATE_OPTIMIZED.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Spanish sarcasm data file not found: {data_file}")
        
        try:
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Process each row
            all_samples = []
            for idx, row in df.iterrows():
                text = str(row.get('text', ''))
                label = int(row.get('sarcastic', row.get('label', 0)))  # 1 for sarcastic, 0 for non-sarcastic
                
                # Skip empty or very short text
                if len(text.strip()) < 3:
                    continue
                
                sample = {
                    'id': f"spanish_{idx}",
                    'text': text,
                    'label': label,
                    'dataset': 'spanish_sarcasm'
                }
                
                # Add additional metadata if available
                if 'sentiment' in row and pd.notna(row['sentiment']):
                    sample['sentiment'] = str(row['sentiment'])
                if 'category' in row and pd.notna(row['category']):
                    sample['category'] = str(row['category'])
                if 'source' in row and pd.notna(row['source']):
                    sample['source'] = str(row['source'])
                if 'context' in row and pd.notna(row['context']):
                    sample['context'] = str(row['context'])
                
                all_samples.append(sample)
            
            # Create train/val/test split
            self.data = self._create_split(all_samples)
            
            # Apply max samples limit with stratified sampling
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Spanish Sarcasm dataset: {e}")
            raise
    
    def _create_split(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create train/val/test split."""
        
        # Shuffle all samples
        np.random.shuffle(all_samples)
        
        # Split: 80% train, 10% val, 10% test
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
        """Perform stratified sampling to maintain class balance."""
        
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
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        
        # Check cache
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
            
            # Tokenize
            tokenized = self.processors['text'].tokenize(
                processed_text,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            text_tokens = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        # Create final sample
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': None,  # Spanish dataset is text-only
            'video': None,  # Spanish dataset is text-only
            'image': None,  # Spanish dataset is text-only
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'language': 'spanish',
                'sentiment': sample.get('sentiment', ''),
                'category': sample.get('category', ''),
                'source': sample.get('source', ''),
                'context': sample.get('context', ''),
                'original_text': sample['text']
            }
        }
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        labels = [sample['label'] for sample in self.data]
        categories = [sample.get('category', '') for sample in self.data if sample.get('category')]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'unique_categories': len(set(categories)) if categories else 0,
            'categories': list(set(categories)) if categories else [],
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'language': 'Spanish',
            'modalities': ['text']
        }
