# sarcasm_detection/data/funny_loader.py
"""
UR-FUNNY Dataset Loader for Sarcasm Detection
Multimodal humor detection dataset used as sarcasm complement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset

from shared.preprocessing import TextProcessor, AudioProcessor, VideoProcessor
from shared.utils import get_logger


class URFunnyDataset(Dataset):
    """
    UR-FUNNY Dataset Loader for humor/sarcasm detection.
    
    Contains multimodal TED Talk data with text, audio, and video.
    Used as a complement to sarcasm detection by leveraging humor patterns.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
        processors: Optional[Dict[str, Any]] = None,
        cache_data: bool = True,
        use_humor_as_sarcasm: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize UR-FUNNY dataset.
        
        Args:
            data_dir: Path to UR-FUNNY dataset directory
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            processors: Dictionary of processors for each modality
            cache_data: Whether to cache processed data
            use_humor_as_sarcasm: Whether to treat humor labels as sarcasm
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.cache_data = cache_data
        self.use_humor_as_sarcasm = use_humor_as_sarcasm
        self.random_seed = random_seed
        
        self.logger = get_logger("URFunnyDataset")
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize processors
        self.processors = processors or {}
        if 'text' not in self.processors:
            self.processors['text'] = TextProcessor(max_length=256, add_sarcasm_markers=True)
        if 'audio' not in self.processors:
            self.processors['audio'] = AudioProcessor(sample_rate=16000, target_length=16000*5)
        if 'video' not in self.processors:
            self.processors['video'] = VideoProcessor(max_frames=16, frame_size=(224, 224))
        
        # Data storage
        self.data = []
        self.cache = {}
        
        # Load data
        self._load_data()
        
        self.logger.info(f"Loaded UR-FUNNY dataset: {len(self.data)} samples, split: {split}")
    
    def _load_data(self):
        """Load UR-FUNNY dataset from CSV file."""
        
        data_file = self.data_dir / "ur_funny.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"UR-FUNNY data file not found: {data_file}")
        
        try:
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Process each row
            all_samples = []
            for idx, row in df.iterrows():
                text = str(row.get('text', ''))
                humor_label = int(row.get('humor', row.get('label', 0)))
                
                # Skip empty text
                if len(text.strip()) < 3:
                    continue
                
                sample = {
                    'id': f"ur_funny_{idx}",
                    'text': text,
                    'humor_label': humor_label,
                    'label': humor_label if self.use_humor_as_sarcasm else 0,  # Use humor as sarcasm proxy
                    'dataset': 'ur_funny'
                }
                
                # Add file paths if available
                if 'audio_file' in row and pd.notna(row['audio_file']):
                    sample['audio_path'] = self._get_audio_path(str(row['audio_file']))
                if 'video_file' in row and pd.notna(row['video_file']):
                    sample['video_path'] = self._get_video_path(str(row['video_file']))
                
                # Add additional metadata
                if 'speaker' in row and pd.notna(row['speaker']):
                    sample['speaker'] = str(row['speaker'])
                if 'talk_id' in row and pd.notna(row['talk_id']):
                    sample['talk_id'] = str(row['talk_id'])
                if 'segment_id' in row and pd.notna(row['segment_id']):
                    sample['segment_id'] = str(row['segment_id'])
                if 'start_time' in row and pd.notna(row['start_time']):
                    sample['start_time'] = float(row['start_time'])
                if 'end_time' in row and pd.notna(row['end_time']):
                    sample['end_time'] = float(row['end_time'])
                
                all_samples.append(sample)
            
            # Create train/val/test split
            self.data = self._create_split(all_samples)
            
            # Apply max samples limit with stratified sampling
            if self.max_samples and len(self.data) > self.max_samples:
                self.data = self._stratified_sample(self.data, self.max_samples)
            
            self.logger.info(f"Loaded {len(self.data)} samples for split: {self.split}")
            
        except Exception as e:
            self.logger.error(f"Failed to load UR-FUNNY dataset: {e}")
            raise
    
    def _get_audio_path(self, audio_file: str) -> Optional[Path]:
        """Get audio file path for sample."""
        audio_dir = self.data_dir / "audio"
        
        # Try different possible paths and extensions
        for ext in ['.wav', '.mp3', '.flac']:
            audio_path = audio_dir / f"{audio_file}{ext}"
            if audio_path.exists():
                return audio_path
        
        # Try direct path
        direct_path = self.data_dir / audio_file
        if direct_path.exists():
            return direct_path
        
        return None
    
    def _get_video_path(self, video_file: str) -> Optional[Path]:
        """Get video file path for sample."""
        video_dir = self.data_dir / "video"
        
        # Try different possible paths and extensions
        for ext in ['.mp4', '.avi', '.mov']:
            video_path = video_dir / f"{video_file}{ext}"
            if video_path.exists():
                return video_path
        
        # Try direct path
        direct_path = self.data_dir / video_file
        if direct_path.exists():
            return direct_path
        
        return None
    
    def _create_split(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create train/val/test split."""
        
        # Group by talk_id to avoid data leakage
        talks = {}
        for sample in all_samples:
            talk_id = sample.get('talk_id', 'unknown')
            if talk_id not in talks:
                talks[talk_id] = []
            talks[talk_id].append(sample)
        
        # Split talks, not individual samples
        talk_ids = list(talks.keys())
        np.random.shuffle(talk_ids)
        
        n_talks = len(talk_ids)
        if self.split == "train":
            selected_talks = talk_ids[:int(0.7 * n_talks)]
        elif self.split == "val":
            selected_talks = talk_ids[int(0.7 * n_talks):int(0.85 * n_talks)]
        elif self.split == "test":
            selected_talks = talk_ids[int(0.85 * n_talks):]
        else:
            selected_talks = talk_ids
        
        # Collect samples from selected talks
        split_samples = []
        for talk_id in selected_talks:
            split_samples.extend(talks[talk_id])
        
        return split_samples
    
    def _stratified_sample(self, data: List[Dict[str, Any]], n_samples: int) -> List[Dict[str, Any]]:
        """Perform stratified sampling to maintain class balance."""
        
        humorous = [s for s in data if s['label'] == 1]
        non_humorous = [s for s in data if s['label'] == 0]
        
        n_humorous = min(len(humorous), n_samples // 2)
        n_non_humorous = min(len(non_humorous), n_samples - n_humorous)
        
        np.random.shuffle(humorous)
        np.random.shuffle(non_humorous)
        
        sampled = humorous[:n_humorous] + non_humorous[:n_non_humorous]
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
        
        # Process audio
        audio_tensor = None
        if sample.get('audio_path') and sample['audio_path'].exists():
            try:
                audio_data = self.processors['audio'].process_audio(
                    sample['audio_path'],
                    return_features=False
                )
                audio_tensor = torch.tensor(audio_data['audio'], dtype=torch.float32)
            except Exception as e:
                self.logger.debug(f"Failed to process audio for {sample['id']}: {e}")
        
        # Process video
        video_tensor = None
        if sample.get('video_path') and sample['video_path'].exists():
            try:
                video_tensor = self.processors['video'].process_for_model(
                    sample['video_path'],
                    training=(self.split == "train")
                )
            except Exception as e:
                self.logger.debug(f"Failed to process video for {sample['id']}: {e}")
        
        # Create final sample
        final_sample = {
            'id': sample['id'],
            'text': text_tokens,
            'audio': audio_tensor,
            'video': video_tensor,
            'image': None,  # UR-FUNNY doesn't have static images
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metadata': {
                'dataset': sample['dataset'],
                'humor_label': sample['humor_label'],
                'speaker': sample.get('speaker', ''),
                'talk_id': sample.get('talk_id', ''),
                'segment_id': sample.get('segment_id', ''),
                'start_time': sample.get('start_time', 0.0),
                'end_time': sample.get('end_time', 0.0),
                'has_audio': audio_tensor is not None,
                'has_video': video_tensor is not None,
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
        talk_ids = [sample.get('talk_id', '') for sample in self.data if sample.get('talk_id')]
        has_audio = [bool(sample.get('audio_path') and sample['audio_path'].exists()) for sample in self.data]
        has_video = [bool(sample.get('video_path') and sample['video_path'].exists()) for sample in self.data]
        
        return {
            'total_samples': len(self.data),
            'humorous_samples': sum(labels),
            'non_humorous_samples': len(labels) - sum(labels),
            'class_balance': sum(labels) / len(labels) if labels else 0,
            'unique_talks': len(set(talk_ids)),
            'samples_with_audio': sum(has_audio),
            'samples_with_video': sum(has_video),
            'audio_coverage': sum(has_audio) / len(has_audio) if has_audio else 0,
            'video_coverage': sum(has_video) / len(has_video) if has_video else 0,
            'avg_text_length': np.mean([len(s['text'].split()) for s in self.data if s['text']]),
            'modalities': ['text', 'audio', 'video']
        }
