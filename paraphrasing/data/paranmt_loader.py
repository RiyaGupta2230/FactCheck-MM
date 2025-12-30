# paraphrasing_detection/data/paranmt_loader.py

"""
ParaNMT-5M Dataset Loader - CORRECTED & RESEARCH-GRADE
Strictly follows PDF specification:
- Path: data/paranmt/para-nmt-5m-processed.txt
- Format: TSV (reference_sentence \t paraphrase_sentence \t similarity_score)
- Train split only
- Research-grade capping to prevent task domination
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.preprocessing.text_processor import TextProcessor
from shared.utils.logging_utils import get_logger


@dataclass
class ParaNMTConfig:
    """Configuration for ParaNMT-5M dataset loading."""
    data_dir: str = "data/paranmt"
    filename: str = "para-nmt-5m-processed.txt"
    max_length: int = 128
    tokenizer_name: str = "roberta-base"
    
    # CRITICAL: Research-grade balancing (cap from 5M)
    max_samples: Optional[int] = 100000  # Default cap at 100k
    
    # Filtering
    min_length: int = 3
    max_token_length: int = 50
    quality_threshold: float = 0.0
    
    random_seed: int = 42


class ParaNMTDataset(Dataset):
    """
    ParaNMT-5M Dataset Loader.
    - 5M paraphrase pairs (CAPPED for research balance)
    - TSV format: reference \t paraphrase \t score
    - Train only (no official splits)
    """
    
    def __init__(
        self,
        config: ParaNMTConfig,
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
        self.logger = get_logger("ParaNMTDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # ParaNMT is TRAIN ONLY
        if split not in ["train"]:
            self.logger.warning(
                f"ParaNMT has no official {split} split. Using empty dataset."
            )
            self.data = []
            return
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(
                tokenizer_name=config.tokenizer_name,
                max_length=config.max_length
            )
        
        self.data = []
        self._load_data()
        
        self.logger.info(
            f"Loaded ParaNMT dataset: {len(self.data)} samples "
            f"(capped from 5M for research balance)"
        )
    
    def _load_data(self):
        """Load ParaNMT data from TSV file with strict schema enforcement."""
        data_file = Path(self.config.data_dir) / self.config.filename
        
        if not data_file.exists():
            raise FileNotFoundError(f"ParaNMT data file not found: {data_file}")
        
        try:
            samples = []
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_idx, line in enumerate(f):
                    # Apply max samples early to avoid loading entire 5M file
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
                    
                    # STRICT TSV PARSING (as per specification)
                    parts = line.strip().split('\t')
                    
                    # Must have at least 2 columns (reference, paraphrase)
                    if len(parts) < 2:
                        continue
                    
                    reference = parts[0].strip()
                    paraphrase = parts[1].strip()
                    similarity_score = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    # DROP rows with missing or empty sentence pairs
                    if not reference or not paraphrase:
                        continue
                    
                    # Apply filtering criteria
                    if not self._should_include(reference, paraphrase, similarity_score):
                        continue
                    
                    sample = {
                        'id': f"paranmt_{len(samples)}",
                        'reference': reference,
                        'paraphrase': paraphrase,
                        'similarity_score': similarity_score,
                        'dataset': 'paranmt',
                        'split': self.split
                    }
                    samples.append(sample)
            
            self.data = samples
            self.logger.info(f"Loaded {len(self.data)} valid samples from ParaNMT")
        
        except Exception as e:
            self.logger.error(f"Failed to load ParaNMT dataset: {e}")
            raise
    
    def _should_include(self, reference: str, paraphrase: str, score: float) -> bool:
        """Check if sample meets filtering criteria."""
        # Quality threshold
        if score < self.config.quality_threshold:
            return False
        
        # Length filtering
        ref_tokens = len(reference.split())
        para_tokens = len(paraphrase.split())
        
        if (ref_tokens < self.config.min_length or 
            para_tokens < self.config.min_length or
            ref_tokens > self.config.max_token_length or 
            para_tokens > self.config.max_token_length):
            return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        sample = self.data[idx].copy()
        
        # Process text pairs
        reference = sample['reference']
        paraphrase = sample['paraphrase']
        
        # Tokenize pair for classification
        encoding = self.processors['text'].tokenize_pair(reference, paraphrase)
        
        result = {
            'id': sample['id'],
            'text1': reference,
            'text2': paraphrase,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(1, dtype=torch.long),  # Always paraphrase pairs
            'metadata': {
                'dataset': sample['dataset'],
                'similarity_score': sample['similarity_score'],
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
        
        scores = [s['similarity_score'] for s in self.data]
        ref_lengths = [len(s['reference'].split()) for s in self.data]
        para_lengths = [len(s['paraphrase'].split()) for s in self.data]
        
        return {
            'total_samples': len(self.data),
            'avg_similarity_score': np.mean(scores),
            'min_similarity_score': np.min(scores),
            'max_similarity_score': np.max(scores),
            'avg_reference_length': np.mean(ref_lengths),
            'avg_paraphrase_length': np.mean(para_lengths),
            'note': f'Capped at {self.max_samples} from 5M for research balance'
        }
