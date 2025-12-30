# sarcasm_detection/data/sarc_loader.py

"""
SARC Dataset Loader for Sarcasm Detection
Large-scale Reddit sarcasm dataset with 1.3M samples.
Research-grade balancing applied to prevent task domination.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor
from shared.utils import get_logger


class SARCDataset(Dataset):
    """
    SARC (Self-Annotated Reddit Corpus) Dataset Loader.
    - 1.3M Reddit comments
    - Already balanced by dataset creators
    - Text-only modality
    
    Path structure (as per PDF):
    sarc/
    └── train-balanced-sarcasm.csv
    
    CSV Schema:
    - label (0/1)
    - comment
    - author
    - subreddit
    - score
    - ups
    - downs
    - date
    - created_utc
    - parent_comment
    
    NOTE: Dataset is capped at 50k samples for research-grade balancing
    to prevent dominating sarcasm detection task.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = 50000,  # Research-grade cap
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = False,  # Disabled for large dataset
        random_seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.cache_data = cache_data
        self.random_seed = random_seed
        self.logger = get_logger("SARCDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                max_length=512,
                add_sarcasm_markers=True,
                clean_html=True,
                normalize_whitespace=True
            )
        
        self.data = []
        self.cache = {}
        
        self._load_data()
        
        self.logger.info(f"Loaded SARC dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load SARC dataset from CSV file with research-grade capping."""
        data_file = self.data_dir / "train-balanced-sarcasm.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"SARC data file not found: {data_file}")
        
        try:
            chunk_size = 10000
            chunks = []
            self.logger.info("Loading SARC dataset with research-grade cap...")
            
            # Load in chunks to handle large file
            for chunk in pd.read_csv(data_file, chunksize=chunk_size):
                processed_chunk = []
                for idx, row in chunk.iterrows():
                    comment = str(row.get('comment', ''))
                    
                    # Filter out very short or very long comments
                    if len(comment.strip()) < 3 or len(comment.split()) > 1000:
                        continue
                    
                    sample = {
                        'id': f"sarc_{len(self.data) + len(processed_chunk)}",
                        'text': comment,
                        'label': int(row.get('label', 0)),  # 0=non-sarcastic, 1=sarcastic
                        'dataset': 'sarc',
                        'split': self.split
                    }
                    
                    # Add optional metadata
                    for col in ['author', 'subreddit', 'score', 'date', 'parent_comment']:
                        if col in row and pd.notna(row[col]):
                            sample[col] = str(row[col]) if col in ['author', 'subreddit', 'date', 'parent_comment'] else int(row[col])
                    
                    processed_chunk.append(sample)
                
                chunks.append(processed_chunk)
                
                # Stop early if we have enough samples
                total_samples = sum(len(chunk) for chunk in chunks)
                if self.max_samples and total_samples >= self.max_samples:
                    break
            
            # Combine all chunks
            all_samples = []
            for chunk in chunks:
                all_samples.extend(chunk)
            
            # Create split
            self.data = self._create_split(all_samples)
            
            # Apply final cap with stratification
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(
                f"Loaded {len(self.data)} samples for split: {self.split} "
                f"(capped from 1.3M for research balance)"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to load SARC dataset: {e}")
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
                'author': sample.get('author', ''),
                'subreddit': sample.get('subreddit', ''),
                'score': sample.get('score', 0),
                'date': sample.get('date', ''),
                'parent_comment': sample.get('parent_comment', ''),
                'original_text': sample['text']
            }
        }
        
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        labels = [sample['label'] for sample in self.data]
        subreddits = [sample.get('subreddit', '') for sample in self.data if sample.get('subreddit')]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'unique_subreddits': len(set(subreddits)) if subreddits else 0,
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'median_text_length': np.median([len(s['text'].split()) for s in self.data if s['text']]),
            'max_text_length': max([len(s['text'].split()) for s in self.data if s['text']], default=0),
            'modalities': ['text'],
            'note': 'Dataset capped at 50k for research-grade balancing'
        }
