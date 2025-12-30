# sarcasm_detection/data/mmsd2_loader.py

"""
MMSD2 Dataset Loader for Sarcasm Detection
Multimodal dataset with text and image modalities.
Strictly follows PDF specification for image resolution.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor, ImageProcessor
from shared.utils import get_logger


class MMSD2Dataset(Dataset):
    """
    MMSD2 (Multimodal Sarcasm Detection Dataset 2) Loader.
    - 24,636 text-image pairs
    - Binary sarcasm detection
    
    Path structure (as per PDF):
    mmsd2/
    ├── dataset_image/
    │   └── <imageid>.jpg
    └── text_json_final/
        ├── train.json
        ├── test.json
        └── valid.json
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
        self.logger = get_logger("MMSD2Dataset")
        
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
        
        self.logger.info(f"Loaded MMSD2 dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load MMSD2 dataset from JSON files."""
        # Map split names to file names (as per PDF)
        split_files = {
            "train": "train.json",
            "val": "valid.json",
            "test": "test.json"
        }
        
        data_file = self.data_dir / "text_json_final" / split_files.get(self.split, "train.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"MMSD2 data file not found: {data_file}")
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for idx, item in enumerate(raw_data):
                sample = {
                    'id': item.get('id', f"mmsd2_{idx}"),
                    'text': str(item.get('text', '')),
                    'label': int(item.get('label', 0)),  # 0=non-sarcastic, 1=sarcastic
                    'dataset': 'mmsd2',
                    'split': self.split
                }
                
                # Resolve image path: dataset_image/<imageid>.jpg
                if 'imageid' in item:
                    image_id = str(item['imageid'])
                    if not image_id.endswith('.jpg'):
                        image_id = f"{image_id}.jpg"
                    
                    image_path = self.data_dir / "dataset_image" / image_id
                    if image_path.exists():
                        sample['image_path'] = str(image_path)
                    else:
                        self.logger.debug(f"Image not found: {image_path}")
                
                if sample['text'] and sample['text'].strip():
                    self.data.append(sample)
            
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load MMSD2 dataset: {e}")
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
                'has_image': image_tensor is not None,
                'original_text': sample['text']
            }
        }
        
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        labels = [sample['label'] for sample in self.data]
        has_image = [bool('image_path' in sample) for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'samples_with_images': sum(has_image),
            'image_coverage': sum(has_image) / len(has_image) if has_image else 0,
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'modalities': ['text', 'image']
        }
