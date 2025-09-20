# sarcasm_detection/data/sarc_loader.py
"""
SARC Dataset Loader for Sarcasm Detection
Large-scale Reddit sarcasm dataset with 1.3M samples.
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
    
    Large-scale text-only sarcasm dataset from Reddit with 1.3M samples.
    Balanced dataset with sarcastic and non-sarcastic comments.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = False,  # Disabled by default for large dataset
        random_seed: int = 42
    ):
        """
        Initialize SARC dataset.
        
        Args:
            data_dir: Path to SARC dataset directory
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
        
        self.logger = get_logger("SARCDataset")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                max_length=512, 
                add_sarcasm_markers=True,
                clean_html=True,
                normalize_whitespace=True
            )
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(f"Loaded SARC dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load SARC dataset from CSV file."""
        
        data_file = self.data_dir / "train-balanced-sarcasm.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"SARC data file not found: {data_file}")
        
        try:
            # Load CSV with chunking for large file
            chunk_size = 10000
            chunks = []
            
            self.logger.info("Loading SARC dataset (this may take a while for large files)...")
            
            for chunk in pd.read_csv(data_file, chunksize=chunk_size):
                # Process chunk
                processed_chunk = []
                
                for idx, row in chunk.iterrows():
                    comment = str(row.get('comment', ''))
                    label = int(row.get('label', 0))  # 1 for sarcastic, 0 for non-sarcastic
                    
                    # Skip empty or very short comments
                    if len(comment.strip()) < 3:
                        continue
                    
                    # Skip very long comments (likely spam or irrelevant)
                    if len(comment.split()) > 1000:
                        continue
                    
                    sample = {
                        'id': f"sarc_{len(self.data) + len(processed_chunk)}",
                        'text': comment,
                        'label': label,
                        'dataset': 'sarc'
                    }
                    
                    # Add metadata if available
                    if 'author' in row and pd.notna(row['author']):
                        sample['author'] = str(row['author'])
                    if 'subreddit' in row and pd.notna(row['subreddit']):
                        sample['subreddit'] = str(row['subreddit'])
                    if 'score' in row and pd.notna(row['score']):
                        sample['score'] = int(row['score'])
                    if 'ups' in row and pd.notna(row['ups']):
                        sample['ups'] = int(row['ups'])
                    if 'downs' in row and pd.notna(row['downs']):
                        sample['downs'] = int(row['downs'])
                    if 'date' in row and pd.notna(row['date']):
                        sample['date'] = str(row['date'])
                    if 'parent_comment' in row and pd.notna(row['parent_comment']):
                        sample['parent_comment'] = str(row['parent_comment'])
                    
                    processed_chunk.append(sample)
                
                chunks.append(processed_chunk)
                
                # Stop if we have enough samples
                total_samples = sum(len(chunk) for chunk in chunks)
                if self.max_samples and total_samples >= self.max_samples:
                    break
            
            # Combine all chunks
            all_samples = []
            for chunk in chunks:
                all_samples.extend(chunk)
            
            # Create train/val/test split
            self.data = self._create_split(all_samples)
            
            # Apply final max samples limit with stratified sampling
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SARC dataset: {e}")
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
            'audio': None,  # SARC is text-only
            'video': None,  # SARC is text-only
            'image': None,  # SARC is text-only
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'author': sample.get('author', ''),
                'subreddit': sample.get('subreddit', ''),
                'score': sample.get('score', 0),
                'ups': sample.get('ups', 0),
                'downs': sample.get('downs', 0),
                'date': sample.get('date', ''),
                'parent_comment': sample.get('parent_comment', ''),
                'original_text': sample['text']
            }
        }
        
        # Cache if enabled (unlikely for large dataset)
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
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
            'modalities': ['text']
        }
