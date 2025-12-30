# paraphrasing_detection/data/quora_loader.py

"""
Quora Question Pairs Dataset Loader - CORRECTED & RESEARCH-GRADE
Strictly follows PDF specification:
- Path: data/quora/
- Files: train.csv, test.csv
- Columns: question1, question2, is_duplicate
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class QuoraConfig:
    """Configuration for Quora dataset loading."""
    data_dir: str = "data/quora"
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    max_samples: Optional[int] = None
    val_split: float = 0.1  # Create val split from train
    random_seed: int = 42


class QuoraDataset(Dataset):
    """
    Quora Question Pairs Dataset Loader.
    - ~400k question pairs
    - Binary duplicate detection
    - CSV format with EXACT column names from specification
    """
    
    def __init__(
        self,
        config: QuoraConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
        processors: Optional[Dict[str, Any]] = None,
        random_seed: int = 42
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.max_samples = max_samples or config.max_samples
        self.random_seed = random_seed
        self.logger = get_logger("QuoraDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                tokenizer_name=config.tokenizer_name,
                max_length=config.max_length
            )
        
        self.data = []
        self._load_data()
        
        self.logger.info(f"Loaded Quora {split} dataset: {len(self.data)} samples")
    
    def _load_data(self):
        """Load Quora data from CSV files with STRICT schema enforcement."""
        if self.split == "test":
            # Load test data (may not have labels)
            data_file = Path(self.config.data_dir) / "test.csv"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Quora test file not found: {data_file}")
            
            df = pd.read_csv(data_file, encoding='utf-8')
            
            # Test file may not have is_duplicate column
            if 'is_duplicate' not in df.columns:
                df['is_duplicate'] = -1  # Placeholder
        
        else:
            # Load training data
            data_file = Path(self.config.data_dir) / "train.csv"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Quora train file not found: {data_file}")
            
            df = pd.read_csv(data_file, encoding='utf-8')
            
            # ENFORCE EXACT COLUMN NAMES (as per specification)
            required_cols = ['question1', 'question2', 'is_duplicate']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create train/val split
            if self.split in ["train", "val"]:
                # Stratified split
                from sklearn.model_selection import train_test_split
                
                train_df, val_df = train_test_split(
                    df,
                    test_size=self.config.val_split,
                    random_state=self.random_seed,
                    stratify=df['is_duplicate']
                )
                
                df = train_df if self.split == "train" else val_df
        
        # Process data
        try:
            for idx, row in df.iterrows():
                question1 = str(row['question1']).strip()
                question2 = str(row['question2']).strip()
                
                # DROP rows with missing or empty sentence pairs
                if not question1 or not question2 or pd.isna(question1) or pd.isna(question2):
                    continue
                
                # EXPLICITLY CAST LABEL TO INT (as per requirement)
                is_duplicate = int(row['is_duplicate']) if pd.notna(row['is_duplicate']) and row['is_duplicate'] != -1 else None
                
                if is_duplicate is None and self.split != "test":
                    continue
                
                sample = {
                    'id': f"quora_{idx}",
                    'question1': question1,
                    'question2': question2,
                    'label': is_duplicate if is_duplicate is not None else -1,
                    'dataset': 'quora',
                    'split': self.split
                }
                
                self.data.append(sample)
            
            # Apply max samples with stratification
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load Quora dataset: {e}")
            raise
    
    def _stratified_sample(self, data: List[Dict], n_samples: int) -> List[Dict]:
        """Stratified sampling to maintain class balance."""
        duplicate = [s for s in data if s['label'] == 1]
        non_duplicate = [s for s in data if s['label'] == 0]
        
        n_duplicate = min(len(duplicate), n_samples // 2)
        n_non_duplicate = min(len(non_duplicate), n_samples - n_duplicate)
        
        np.random.shuffle(duplicate)
        np.random.shuffle(non_duplicate)
        
        sampled = duplicate[:n_duplicate] + non_duplicate[:n_non_duplicate]
        np.random.shuffle(sampled)
        
        return sampled
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        sample = self.data[idx].copy()
        
        # Process text pairs
        question1 = sample['question1']
        question2 = sample['question2']
        
        # Tokenize pair
        encoding = self.processors['text'].tokenize_pair(question1, question2)
        
        result = {
            'id': sample['id'],
            'text1': question1,
            'text2': question2,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'metadata': {
                'dataset': sample['dataset'],
                'split': self.split
            }
        }
        
        if sample['label'] != -1:
            result['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {'total_samples': 0}
        
        labels = [s['label'] for s in self.data if s['label'] != -1]
        
        return {
            'total_samples': len(self.data),
            'duplicate_samples': sum(labels) if labels else 0,
            'non_duplicate_samples': len(labels) - sum(labels) if labels else 0,
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'avg_question1_length': np.mean([len(s['question1'].split()) for s in self.data]),
            'avg_question2_length': np.mean([len(s['question2'].split()) for s in self.data])
        }
