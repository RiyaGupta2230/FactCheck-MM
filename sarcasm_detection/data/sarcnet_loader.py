# sarcasm_detection/data/sarcnet_loader.py

"""
SarcNet Dataset Loader for Sarcasm Detection
Multimodal dataset with separate modality annotations for text and image.
Strictly follows PDF specification for multi-label structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor, ImageProcessor
from shared.utils import get_logger


class SarcNetDataset(Dataset):
    """
    SarcNet Dataset Loader for multimodal sarcasm detection.
    - ~3,335 image-text pairs
    - Separate labels: Textlabel, Imagelabel, Multilabel
    - Primary target: Multilabel (as per PDF)
    
    Path structure (as per PDF):
    sarcnet/
    └── SarcNet Image-Text/
        ├── Image/
        │   ├── 1.jpg
        │   └── ...
        ├── SarcNetTrain.csv
        ├── SarcNetVal.csv
        └── SarcNetTest.csv
    
    CSV Schema:
    - Text
    - Imagepath
    - Textlabel (0/1)
    - Imagelabel (0/1)
    - Multilabel (0/1) <- Primary target
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
        self.logger = get_logger("SarcNetDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(max_length=256, add_sarcasm_markers=True)
        if 'image' not in self.processors:
            self.processors['image'] = ImageProcessor(image_size=224, augment_training=True)
        
        self.data = []
        self.cache = {}
        
        self._load_data()
        
        self.logger.info(f"Loaded SarcNet dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load SarcNet dataset from CSV files."""
        # Map split names to file names (as per PDF)
        split_files = {
            "train": "SarcNetTrain.csv",
            "val": "SarcNetVal.csv",
            "test": "SarcNetTest.csv"
        }
        
        data_file = self.data_dir / split_files.get(self.split, "SarcNetTrain.csv")
        
        if not data_file.exists():
            raise FileNotFoundError(f"SarcNet data file not found: {data_file}")
        
        try:
            df = pd.read_csv(data_file)
            
            for idx, row in df.iterrows():
                text = str(row.get('Text', ''))
                image_path = str(row.get('Imagepath', ''))
                
                sample = {
                    'id': f"sarcnet_{idx}",
                    'text': text,
                    'label': int(row.get('Multilabel', 0)),  # Primary target as per PDF
                    'text_label': int(row.get('Textlabel', 0)),
                    'image_label': int(row.get('Imagelabel', 0)),
                    'dataset': 'sarcnet',
                    'split': self.split
                }
                
                # Resolve image path relative to SarcNet Image-Text directory
                if image_path:
                    full_image_path = self.data_dir / image_path
                    if full_image_path.exists():
                        sample['image_path'] = str(full_image_path)
                    else:
                        self.logger.debug(f"Image not found: {full_image_path}")
                
                if text and text.strip():
                    self.data.append(sample)
            
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load SarcNet dataset: {e}")
            raise
    
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
        
        # Process image
        image_tensor = None
        if 'image_path' in sample:
            image_path = Path(sample['image_path'])
            if image_path.exists():
                try:
                    image_data = self.processors['image'].process_for_vit(
                        str(image_path),
                        training=(self.split == "train")
                    )
                    image_tensor = image_data['pixel_values'].squeeze(0)
                except Exception as e:
                    self.logger.debug(f"Failed to process image for {sample['id']}: {e}")
        
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': None,
            'video': None,
            'image': image_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'text_label': sample['text_label'],
                'image_label': sample['image_label'],
                'has_image': image_tensor is not None,
                'original_text': sample['text']
            }
        }
        
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        labels = [sample['label'] for sample in self.data]
        text_labels = [sample['text_label'] for sample in self.data]
        image_labels = [sample['image_label'] for sample in self.data]
        has_image = [bool('image_path' in sample) for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'text_sarcastic_samples': sum(text_labels),
            'image_sarcastic_samples': sum(image_labels),
            'samples_with_images': sum(has_image),
            'image_coverage': sum(has_image) / len(has_image) if has_image else 0,
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'modalities': ['text', 'image']
        }
