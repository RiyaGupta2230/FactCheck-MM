# sarcasm_detection/data/mustard_loader.py
"""
MUStARD Dataset Loader for Sarcasm Detection
Multimodal dataset with text, audio, and video modalities.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset
import logging

from shared.preprocessing import TextProcessor, AudioProcessor, VideoProcessor
from shared.utils import get_logger


class MustardDataset(Dataset):
    """
    MUStARD (Multimodal Sarcasm Detection) Dataset Loader.
    
    Contains 690 video clips from TV shows with text, audio, and video modalities.
    Balanced dataset with sarcastic and non-sarcastic samples.
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
        Initialize MUStARD dataset.
        
        Args:
            data_dir: Path to MUStARD dataset directory
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
        
        self.logger = get_logger("MustardDataset")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(max_length=128, add_sarcasm_markers=True)
        if 'audio' not in self.processors:
            self.processors['audio'] = AudioProcessor(sample_rate=16000, target_length=16000*10)
        if 'video' not in self.processors:
            self.processors['video'] = VideoProcessor(max_frames=16, frame_size=(224, 224))
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(f"Loaded MUStARD dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load MUStARD dataset from files."""
        
        # Dataset paths
        sarcasm_file = self.data_dir / "data" / "sarcasm_data.json"
        utterances_file = self.data_dir / "data" / "utterances_final.jsonl"
        
        if not sarcasm_file.exists():
            raise FileNotFoundError(f"Sarcasm data file not found: {sarcasm_file}")
        
        if not utterances_file.exists():
            raise FileNotFoundError(f"Utterances file not found: {utterances_file}")
        
        try:
            # Load sarcasm labels
            with open(sarcasm_file, 'r') as f:
                sarcasm_data = json.load(f)
            
            # Load utterances
            utterances_data = {}
            with open(utterances_file, 'r') as f:
                for line in f:
                    utterance = json.loads(line.strip())
                    key = f"{utterance['show']}_{utterance['episode']}_{utterance['scene']}_{utterance['utterance']}"
                    utterances_data[key] = utterance
            
            # Combine data
            all_samples = []
            for key, sarcasm_label in sarcasm_data.items():
                if key in utterances_data:
                    utterance_data = utterances_data[key]
                    
                    sample = {
                        'id': key,
                        'text': utterance_data.get('utterance', ''),
                        'speaker': utterance_data.get('speaker', ''),
                        'show': utterance_data.get('show', ''),
                        'episode': utterance_data.get('episode', ''),
                        'scene': utterance_data.get('scene', ''),
                        'label': int(sarcasm_label),  # 1 for sarcastic, 0 for non-sarcastic
                        'audio_path': self._get_audio_path(key),
                        'video_path': self._get_video_path(key),
                        'context': utterance_data.get('context', []),
                        'dataset': 'mustard'
                    }
                    all_samples.append(sample)
            
            # Split data
            self.data = self._create_split(all_samples)
            
            # Apply max samples limit
            if self.max_samples and len(self.data) > self.max_samples:
                # Stratified sampling to maintain class balance
                sarcastic = [s for s in self.data if s['label'] == 1]
                non_sarcastic = [s for s in self.data if s['label'] == 0]
                
                n_sarcastic = min(len(sarcastic), self.max_samples // 2)
                n_non_sarcastic = min(len(non_sarcastic), self.max_samples - n_sarcastic)
                
                np.random.shuffle(sarcastic)
                np.random.shuffle(non_sarcastic)
                
                self.data = sarcastic[:n_sarcastic] + non_sarcastic[:n_non_sarcastic]
                np.random.shuffle(self.data)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
            
        except Exception as e:
            self.logger.error(f"Failed to load MUStARD dataset: {e}")
            raise
    
    def _get_audio_path(self, key: str) -> Optional[Path]:
        """Get audio file path for sample."""
        audio_dir = self.data_dir / "audio"
        audio_file = audio_dir / f"{key}.wav"
        return audio_file if audio_file.exists() else None
    
    def _get_video_path(self, key: str) -> Optional[Path]:
        """Get video file path for sample."""
        video_dir = self.data_dir / "video"
        video_file = video_dir / f"{key}.mp4"
        return video_file if video_file.exists() else None
    
    def _create_split(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create train/val/test split."""
        
        # MUStARD doesn't have predefined splits, so we create them
        # Stratified split maintaining show distribution
        
        shows = list(set(sample['show'] for sample in all_samples))
        np.random.shuffle(shows)
        
        # Split shows: 70% train, 15% val, 15% test
        n_shows = len(shows)
        train_shows = shows[:int(0.7 * n_shows)]
        val_shows = shows[int(0.7 * n_shows):int(0.85 * n_shows)]
        test_shows = shows[int(0.85 * n_shows):]
        
        if self.split == "train":
            return [s for s in all_samples if s['show'] in train_shows]
        elif self.split == "val":
            return [s for s in all_samples if s['show'] in val_shows]
        elif self.split == "test":
            return [s for s in all_samples if s['show'] in test_shows]
        else:
            return all_samples
    
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
        if sample['text']:
            processed_text = self.processors['text'].preprocess_text(
                sample['text'], 
                task="sarcasm_detection"
            )
            sample['text_processed'] = processed_text
            
            # Tokenize
            tokenized = self.processors['text'].tokenize(
                processed_text,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            sample['text_tokens'] = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        # Process audio
        sample['audio'] = None
        if sample['audio_path'] and sample['audio_path'].exists():
            try:
                audio_data = self.processors['audio'].process_audio(
                    sample['audio_path'],
                    return_features=False
                )
                sample['audio'] = torch.tensor(audio_data['audio'], dtype=torch.float32)
            except Exception as e:
                self.logger.debug(f"Failed to process audio for {sample['id']}: {e}")
        
        # Process video
        sample['video'] = None
        if sample['video_path'] and sample['video_path'].exists():
            try:
                video_tensor = self.processors['video'].process_for_model(
                    sample['video_path'],
                    training=(self.split == "train")
                )
                sample['video'] = video_tensor
            except Exception as e:
                self.logger.debug(f"Failed to process video for {sample['id']}: {e}")
        
        # Create final sample
        final_sample = {
            'id': sample['id'],
            'text': sample['text_tokens'] if 'text_tokens' in sample else None,
            'audio': sample['audio'],
            'video': sample['video'],
            'image': None,  # MUStARD doesn't have static images
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'speaker': sample['speaker'],
                'show': sample['show'],
                'episode': sample['episode'],
                'scene': sample['scene'],
                'context': sample['context'],
                'dataset': sample['dataset']
            }
        }
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        labels = [sample['label'] for sample in self.data]
        shows = [sample['show'] for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'sarcastic_samples': sum(labels),
            'non_sarcastic_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'unique_shows': len(set(shows)),
            'shows': list(set(shows)),
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'modalities': ['text', 'audio', 'video']
        }
