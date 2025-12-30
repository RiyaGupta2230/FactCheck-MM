# sarcasm_detection/data/mustard_loader.py

"""
MUStARD Dataset Loader for Sarcasm Detection
Multimodal dataset with text, audio, and video modalities.
Strictly follows PDF specification for nested video structure.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor, AudioProcessor, VideoProcessor
from shared.utils import get_logger


class MustardDataset(Dataset):
    """
    MUStARD (Multimodal Sarcasm Detection) Dataset Loader.
    - 690 video clips from TV shows
    - Perfectly balanced: 50/50 sarcastic/non-sarcastic
    - Text + Audio + Video modalities
    
    Path structure (as per PDF):
    mustard_repo/
    └── data/
        ├── sarcasm_data.json
        ├── audio_features.p
        └── videos/
            ├── utterances_final/
            └── context_final/
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
        self.logger = get_logger("MustardDataset")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(max_length=128, add_sarcasm_markers=True)
        if 'audio' not in self.processors:
            self.processors['audio'] = AudioProcessor(sample_rate=16000, target_length=16000*10)
        if 'video' not in self.processors:
            self.processors['video'] = VideoProcessor(max_frames=16, frame_size=(224, 224))
        
        self.data = []
        self.cache = {}
        self.audio_features = {}
        
        # Load precomputed audio features
        self._load_audio_features()
        self._load_data()
        
        self.logger.info(f"Loaded MUStARD dataset: {len(self.data)} samples, split: {split}")
    
    def _load_audio_features(self):
        """Load precomputed audio features from pickle file."""
        audio_features_file = self.data_dir / "data" / "audio_features.p"
        if audio_features_file.exists():
            try:
                with open(audio_features_file, 'rb') as f:
                    self.audio_features = pickle.load(f)
                self.logger.info(f"Loaded audio features for {len(self.audio_features)} samples")
            except Exception as e:
                self.logger.warning(f"Failed to load audio features: {e}")
        else:
            self.logger.warning(f"Audio features file not found: {audio_features_file}")
    
    def _load_data(self):
        """Load MUStARD dataset from sarcasm_data.json."""
        sarcasm_file = self.data_dir / "data" / "sarcasm_data.json"
        
        if not sarcasm_file.exists():
            raise FileNotFoundError(f"Sarcasm data file not found: {sarcasm_file}")
        
        try:
            with open(sarcasm_file, 'r') as f:
                sarcasm_data = json.load(f)
            
            all_samples = []
            for key, item in sarcasm_data.items():
                # Construct utterance ID: <show>_<key>
                utterance_id = f"{item['show']}_{key}"
                
                sample = {
                    'id': utterance_id,
                    'text': item.get('utterance', ''),
                    'label': int(item.get('sarcasm', False)),  # Boolean to int
                    'speaker': item.get('speaker', ''),
                    'show': item.get('show', ''),
                    'context': item.get('context', []),
                    'context_speakers': item.get('context_speakers', []),
                    'dataset': 'mustard',
                    'split': self.split
                }
                
                # Audio features key
                if utterance_id in self.audio_features:
                    sample['audio_features_key'] = utterance_id
                
                # Video path: data/videos/utterances_final/<id>.mp4
                video_path = self.data_dir / "data" / "videos" / "utterances_final" / f"{utterance_id}.mp4"
                if video_path.exists():
                    sample['video_path'] = str(video_path)
                
                # Context videos: data/videos/context_final/<id>_C<idx>.mp4
                if item.get('context'):
                    context_videos = []
                    for ctx_idx in range(len(item['context'])):
                        ctx_video_id = f"{utterance_id}_C{ctx_idx}"
                        ctx_video_path = self.data_dir / "data" / "videos" / "context_final" / f"{ctx_video_id}.mp4"
                        if ctx_video_path.exists():
                            context_videos.append(str(ctx_video_path))
                    
                    if context_videos:
                        sample['context_video_paths'] = context_videos
                
                all_samples.append(sample)
            
            # Create split
            self.data = self._create_split(all_samples)
            
            # Apply max samples if specified
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
        
        except Exception as e:
            self.logger.error(f"Failed to load MUStARD dataset: {e}")
            raise
    
    def _create_split(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create train/val/test split by show."""
        shows = list(set(sample['show'] for sample in all_samples))
        np.random.shuffle(shows)
        
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
        
        # Process audio (from precomputed features)
        audio_data = None
        if 'audio_features_key' in sample and sample['audio_features_key'] in self.audio_features:
            audio_data = self.audio_features[sample['audio_features_key']]
        
        # Process video
        video_data = None
        if 'video_path' in sample:
            video_path = Path(sample['video_path'])
            if video_path.exists():
                try:
                    video_data = self.processors['video'].process_for_model(
                        str(video_path),
                        training=(self.split == "train")
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to process video {sample['id']}: {e}")
        
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': audio_data,
            'video': video_data,
            'image': None,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'speaker': sample['speaker'],
                'show': sample['show'],
                'context': sample.get('context', []),
                'context_speakers': sample.get('context_speakers', []),
                'dataset': sample['dataset'],
                'original_text': sample['text']
            }
        }
        
        if self.cache_data:
            self.cache[idx] = final_sample
        
        return final_sample
    
    def get_statistics(self) -> Dict[str, Any]:
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
