# sarcasm_detection/data/mmsd2_loader.py
"""
MMSD2 Dataset Loader for Sarcasm Detection
Multimodal dataset with text and image modalities.
"""

import pandas as pd
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
    
    Contains text and image pairs for sarcasm detection.
    Approximately 24,000 samples with balanced sarcastic/non-sarcastic distribution.
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
        Initialize MMSD2 dataset.
        
        Args:
            data_dir: Path to MMSD2 dataset directory
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
        
        self.logger = get_logger("MMSD2Dataset")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(max_length=256, add_sarcasm_markers=True)
        if 'image' not in self.processors:
            self.processors['image'] = ImageProcessor(image_size=224, augment_training=True)
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(f"Loaded MMSD2 dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load MMSD2 dataset from CSV files."""
        
        # Determine file based on split
        if self.split == "train":
            data_file = self.data_dir / "train_data.csv"
        elif self.split == "val":
            data_file = self.data_dir / "val_data.csv"
        elif self.split == "test":
            data_file = self.data_dir / "test_data.csv"
        else:
            data_file = self.data_dir / "train_data.csv"  # Default fallback
        
        if not data_file.exists():
            raise FileNotFoundError(f"MMSD2 data file not found: {data_file}")
        
        try:
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Process each row
            for idx, row in df.iterrows():
                sample = {
                    'id': f"mmsd2_{idx}",
                    'text': str(row.get('text', '')),
                    'label': int(row.get('label', 0)),  # Assuming 1=sarcastic, 0=non-sarcastic
                    'image_path': self._get_image_path(row.get('image_path', row.get('image_id', ''))),
                    'dataset': 'mmsd2'
                }
                
                # Only add samples with valid text
                if sample['text'] and sample['text'].strip():
                    self.data.append(sample)
            
            # Apply max samples limit with stratified sampling
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
            
        except Exception as e:
            self.logger.error(f"Failed to load MMSD2 dataset: {e}")
            raise
    
    def _get_image_path(self, image_identifier: str) -> Optional[Path]:
        """Get image file path for sample."""
        if not image_identifier:
            return None
        
        image_dir = self.data_dir / "images"
        
        # Try different possible extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = image_dir / f"{image_identifier}{ext}"
            if image_path.exists():
                return image_path
        
        # Try direct path if provided
        direct_path = self.data_dir / image_identifier
        if direct_path.exists():
            return direct_path
        
        return None
    
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
        
        # Process image
        image_tensor = None
        if sample['image_path'] and sample['image_path'].exists():
            try:
                image_data = self.processors['image'].process_for_vit(
                    sample['image_path'],
                    training=(self.split == "train")
                )
                image_tensor = image_data['pixel_values'].squeeze(0)
            except Exception as e:
                self.logger.debug(f"Failed to process image for {sample['id']}: {e}")
        
        # Create final sample
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': None,  # MMSD2 doesn't have audio
            'video': None,  # MMSD2 doesn't have video
            'image': image_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'has_image': image_tensor is not None,
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
        has_image = [bool(sample['image_path'] and sample['image_path'].exists()) for sample in self.data]
        
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
