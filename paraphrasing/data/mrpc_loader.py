# paraphrasing_detection/data/mrpc_loader.py

"""
MRPC Dataset Loader - CORRECTED & RESEARCH-GRADE
Strictly follows PDF specification:
- Path: data/MRPC/
- Files: train.tsv, dev.tsv, test.tsv
- Columns: Quality, #1 ID, #2 ID, #1 String, #2 String
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
class MRPCConfig:
    """Configuration for MRPC dataset loading."""
    data_dir: str = "data/MRPC"
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    max_samples: Optional[int] = None
    random_seed: int = 42


class MRPCDataset(Dataset):
    """
    MRPC (Microsoft Research Paraphrase Corpus) Dataset Loader.
    - ~5.8k samples total (3.7k train, 1.7k test, 0.4k dev)
    - TSV format with EXACT column names from specification
    - Binary paraphrase detection
    """
    
    def __init__(
        self,
        config: MRPCConfig,
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
        self.logger = get_logger("MRPCDataset")
        
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
        
        self.logger.info(f"Loaded MRPC {split} dataset: {len(self.data)} samples")
    
    def _load_data(self):
        """Load MRPC data from TSV files with STRICT schema enforcement."""
        # Map split names to file names (EXACT as per specification)
        split_files = {
            "train": "train.tsv",
            "dev": "dev.tsv",
            "val": "dev.tsv",  # Accept "val" as alias for "dev"
            "test": "test.tsv"
        }
        
        filename = split_files.get(self.split)
        if not filename:
            raise ValueError(f"Unknown split: {self.split}")
        
        data_file = Path(self.config.data_dir) / filename
        
        if not data_file.exists():
            raise FileNotFoundError(f"MRPC data file not found: {data_file}")
        
        try:
            # Read TSV with EXACT column names from specification
            df = pd.read_csv(
                data_file,
                sep='\t',
                encoding='utf-8',
                quoting=3  # QUOTE_NONE to handle special characters
            )
            
            # ENFORCE EXACT COLUMN NAMES (as per specification)
            expected_cols = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
            
            # Check if columns match specification
            if list(df.columns) != expected_cols:
                self.logger.warning(
                    f"Column names don't match specification. "
                    f"Expected: {expected_cols}, Got: {list(df.columns)}"
                )
            
            for idx, row in df.iterrows():
                # EXPLICITLY CAST LABEL TO INT (as per requirement)
                quality = int(row['Quality']) if pd.notna(row['Quality']) else None
                
                sentence1 = str(row['#1 String']).strip()
                sentence2 = str(row['#2 String']).strip()
                
                # DROP rows with missing or empty sentence pairs
                if not sentence1 or not sentence2 or pd.isna(sentence1) or pd.isna(sentence2):
                    continue
                
                if quality is None:
                    continue
                
                sample = {
                    'id': f"mrpc_{idx}",
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    'label': quality,  # 0 or 1
                    'id1': row['#1 ID'] if pd.notna(row['#1 ID']) else '',
                    'id2': row['#2 ID'] if pd.notna(row['#2 ID']) else '',
                    'dataset': 'mrpc',
                    'split': self.split
                }
                
                self.data.append(sample)
            
            # Apply max samples with stratification
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load MRPC dataset: {e}")
            raise
    
    def _stratified_sample(self, data: List[Dict], n_samples: int) -> List[Dict]:
        """Stratified sampling to maintain class balance."""
        paraphrase = [s for s in data if s['label'] == 1]
        non_paraphrase = [s for s in data if s['label'] == 0]
        
        n_paraphrase = min(len(paraphrase), n_samples // 2)
        n_non_paraphrase = min(len(non_paraphrase), n_samples - n_paraphrase)
        
        np.random.shuffle(paraphrase)
        np.random.shuffle(non_paraphrase)
        
        sampled = paraphrase[:n_paraphrase] + non_paraphrase[:n_non_paraphrase]
        np.random.shuffle(sampled)
        
        return sampled
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        sample = self.data[idx].copy()
        
        # Process text pairs
        sentence1 = sample['sentence1']
        sentence2 = sample['sentence2']
        
        # Tokenize pair
        encoding = self.processors['text'].tokenize_pair(sentence1, sentence2)
        
        result = {
            'id': sample['id'],
            'text1': sentence1,
            'text2': sentence2,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'id1': sample['id1'],
                'id2': sample['id2'],
                'split': self.split
            }
        }
        
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {'total_samples': 0}
        
        labels = [s['label'] for s in self.data]
        
        return {
            'total_samples': len(self.data),
            'paraphrase_samples': sum(labels),
            'non_paraphrase_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'avg_text1_length': np.mean([len(s['sentence1'].split()) for s in self.data]),
            'avg_text2_length': np.mean([len(s['sentence2'].split()) for s in self.data])
        }
