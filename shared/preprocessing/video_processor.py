"""
Video Preprocessing for FactCheck-MM
Handles frame extraction, temporal sampling, and augmentations for multimodal analysis.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import tempfile
import imageio
from PIL import Image
import albumentations as A

from .image_processor import ImageProcessor
from ..utils import get_logger


class VideoProcessor:
    """
    Comprehensive video processor for multimodal analysis.
    Extracts frames, applies temporal sampling, and prepares for ViT processing.
    """
    
    def __init__(
        self,
        max_frames: int = 16,
        fps: Optional[int] = None,
        frame_size: Tuple[int, int] = (224, 224),
        sampling_strategy: str = "uniform",
        temporal_augmentation: bool = True,
        spatial_augmentation: bool = True,
        image_processor: Optional[ImageProcessor] = None,
        cache_frames: bool = False
    ):
        """
        Initialize video processor.
        
        Args:
            max_frames: Maximum number of frames to extract
            fps: Target FPS (None to use original)
            frame_size: Target frame size (width, height)
            sampling_strategy: Frame sampling strategy
            temporal_augmentation: Whether to apply temporal augmentations
            spatial_augmentation: Whether to apply spatial augmentations
            image_processor: Image processor for frame processing
            cache_frames: Whether to cache extracted frames
        """
        self.max_frames = max_frames
        self.fps = fps
        self.frame_size = frame_size
        self.sampling_strategy = sampling_strategy
        self.temporal_augmentation = temporal_augmentation
        self.spatial_augmentation = spatial_augmentation
        self.cache_frames = cache_frames
        self.logger = get_logger("VideoProcessor")
        
        if image_processor is None:
            self.image_processor = ImageProcessor(
                image_size=max(frame_size),
                augment_training=spatial_augmentation
            )
        else:
            self.image_processor = image_processor
        
        self._setup_temporal_augmentations()
        
        self.frame_cache = {}
        
        self.logger.info("Video processor initialized successfully")
    
    def _setup_temporal_augmentations(self):
        """Setup temporal augmentation strategies."""
        self.temporal_augs = {
            'temporal_shift': {
                'enabled': True,
                'max_shift_ratio': 0.1
            },
            'frame_dropout': {
                'enabled': True,
                'dropout_prob': 0.1
            },
            'temporal_reverse': {
                'enabled': True,
                'reverse_prob': 0.1
            },
            'playback_speed': {
                'enabled': True,
                'speed_range': (0.8, 1.2)
            }
        }
    
    def load_video(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load video from file.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds (None for full video)
            
        Returns:
            Tuple of (frames_array, metadata)
        """
        try:
            reader = imageio.get_reader(str(video_path))
            
            meta = reader.get_meta_data()
            original_fps = meta.get('fps', 30.0)
            total_frames = meta.get('nframes', len(reader))
            
            start_frame = int(start_time * original_fps)
            if duration is not None:
                end_frame = start_frame + int(duration * original_fps)
            else:
                end_frame = total_frames
            
            end_frame = min(end_frame, total_frames)
            
            frames = []
            for i in range(start_frame, end_frame):
                try:
                    frame = reader.get_data(i)
                    frames.append(frame)
                except IndexError:
                    break
            
            reader.close()
            
            if not frames:
                self.logger.warning(f"No frames extracted from video: {video_path}")
                return np.zeros((1, 224, 224, 3), dtype=np.uint8), {'error': 'no_frames'}
            
            frames_array = np.array(frames)
            
            metadata = {
                'original_fps': original_fps,
                'total_frames': total_frames,
                'extracted_frames': len(frames),
                'duration': len(frames) / original_fps,
                'frame_shape': frames[0].shape if frames else None,
                'start_frame': start_frame,
                'end_frame': end_frame
            }
            
            self.logger.debug(
                f"Loaded video: {video_path} "
                f"({len(frames)} frames, {original_fps:.1f} fps)"
            )
            
            return frames_array, metadata
        
        except Exception as e:
            self.logger.warning(f"Failed to load video {video_path}: {e}")
            return np.zeros((1, 224, 224, 3), dtype=np.uint8), {'error': str(e)}
    
    def extract_frames_opencv(
        self,
        video_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract frames using OpenCV (alternative method).
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Tuple of (frames_array, metadata)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.logger.warning(f"Cannot open video file: {video_path}")
            return np.zeros((1, 224, 224, 3), dtype=np.uint8), {'error': 'cannot_open'}
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * original_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        if duration is not None:
            end_frame = start_frame + int(duration * original_fps)
        else:
            end_frame = total_frames
        
        frames = []
        frame_count = start_frame
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            self.logger.warning(f"No frames extracted from video: {video_path}")
            return np.zeros((1, 224, 224, 3), dtype=np.uint8), {'error': 'no_frames'}
        
        frames_array = np.array(frames)
        
        metadata = {
            'original_fps': original_fps,
            'total_frames': total_frames,
            'extracted_frames': len(frames),
            'duration': len(frames) / original_fps,
            'frame_shape': frames[0].shape,
            'start_frame': start_frame,
            'end_frame': min(end_frame, total_frames)
        }
        
        return frames_array, metadata
    
    def sample_frames(
        self,
        frames: np.ndarray,
        strategy: Optional[str] = None,
        target_fps: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample frames according to specified strategy.
        
        Args:
            frames: Input frames array [num_frames, height, width, channels]
            strategy: Sampling strategy (None to use default)
            target_fps: Target FPS for sampling
            
        Returns:
            Sampled frames array
        """
        if strategy is None:
            strategy = self.sampling_strategy
        
        num_frames = len(frames)
        
        if num_frames <= self.max_frames:
            return frames
        
        if strategy == "uniform":
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
        elif strategy == "random":
            indices = np.sort(np.random.choice(num_frames, self.max_frames, replace=False))
        elif strategy == "keyframe":
            indices = self._detect_keyframes(frames)
        elif strategy == "center":
            start_idx = (num_frames - self.max_frames) // 2
            indices = np.arange(start_idx, start_idx + self.max_frames)
        elif strategy == "fps_based" and target_fps is not None:
            frame_interval = max(1, num_frames // (self.max_frames * target_fps))
            indices = np.arange(0, num_frames, frame_interval)[:self.max_frames]
        else:
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
        
        return frames[indices]
    
    def _detect_keyframes(self, frames: np.ndarray) -> np.ndarray:
        """
        Detect keyframes based on frame differences.
        
        Args:
            frames: Input frames
            
        Returns:
            Array of keyframe indices
        """
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            gray_frames.append(gray)
        
        gray_frames = np.array(gray_frames)
        
        diffs = []
        for i in range(1, len(gray_frames)):
            diff = np.mean(np.abs(gray_frames[i].astype(float) - gray_frames[i-1].astype(float)))
            diffs.append(diff)
        
        diffs = np.array(diffs)
        
        if len(diffs) < self.max_frames:
            return np.arange(len(frames))
        
        keyframe_indices = np.argsort(diffs)[-self.max_frames:]
        keyframe_indices = np.sort(keyframe_indices)
        
        if 0 not in keyframe_indices:
            keyframe_indices = np.concatenate([[0], keyframe_indices[:-1]])
        
        return keyframe_indices
    
    def apply_temporal_augmentations(
        self,
        frames: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply temporal augmentations to frame sequence.
        
        Args:
            frames: Input frames [num_frames, height, width, channels]
            training: Whether this is for training
            
        Returns:
            Augmented frames
        """
        if not (training and self.temporal_augmentation):
            return frames
        
        augmented_frames = frames.copy()
        
        if (self.temporal_augs['temporal_shift']['enabled'] and
            np.random.random() < 0.3):
            max_shift = int(len(frames) * self.temporal_augs['temporal_shift']['max_shift_ratio'])
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    if shift > 0:
                        augmented_frames = np.concatenate([
                            np.repeat(frames[:1], shift, axis=0),
                            frames[:-shift]
                        ])
                    else:
                        augmented_frames = np.concatenate([
                            frames[-shift:],
                            np.repeat(frames[-1:], -shift, axis=0)
                        ])
        
        if (self.temporal_augs['frame_dropout']['enabled'] and
            np.random.random() < 0.2):
            dropout_prob = self.temporal_augs['frame_dropout']['dropout_prob']
            keep_mask = np.random.random(len(augmented_frames)) > dropout_prob
            
            if np.sum(keep_mask) < len(augmented_frames) // 2:
                keep_mask = np.random.random(len(augmented_frames)) > dropout_prob / 2
            
            if np.sum(keep_mask) < len(augmented_frames):
                kept_indices = np.where(keep_mask)[0]
                
                for i, keep in enumerate(keep_mask):
                    if not keep:
                        distances = np.abs(kept_indices - i)
                        nearest_idx = kept_indices[np.argmin(distances)]
                        augmented_frames[i] = augmented_frames[nearest_idx]
        
        if (self.temporal_augs['temporal_reverse']['enabled'] and
            np.random.random() < self.temporal_augs['temporal_reverse']['reverse_prob']):
            augmented_frames = augmented_frames[::-1]
        
        return augmented_frames
    
    def resize_frames(
        self,
        frames: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize all frames to target size.
        
        Args:
            frames: Input frames
            size: Target size (width, height)
            
        Returns:
            Resized frames
        """
        if size is None:
            size = self.frame_size
        
        resized_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            resized_image = self.image_processor.resize_image(pil_image, size)
            resized_frames.append(np.array(resized_image))
        
        return np.array(resized_frames)
    
    def process_video(
        self,
        video_path: Union[str, Path],
        training: bool = False,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        return_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            training: Whether this is for training
            start_time: Start time in seconds
            duration: Duration in seconds
            return_metadata: Whether to return metadata
            
        Returns:
            Dictionary with processed video data
        """
        cache_key = f"{video_path}_{start_time}_{duration}_{training}"
        if self.cache_frames and cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        try:
            frames, metadata = self.load_video(video_path, start_time, duration)
        except Exception:
            try:
                frames, metadata = self.extract_frames_opencv(video_path, start_time, duration)
            except Exception as e:
                self.logger.warning(f"All video loading methods failed for {video_path}: {e}")
                frames = np.zeros((self.max_frames, 224, 224, 3), dtype=np.uint8)
                metadata = {'error': 'all_methods_failed'}
        
        sampled_frames = self.sample_frames(frames, target_fps=self.fps)
        
        augmented_frames = self.apply_temporal_augmentations(sampled_frames, training)
        
        processed_frames = self.resize_frames(augmented_frames)
        
        if training and self.spatial_augmentation:
            final_frames = []
            for frame in processed_frames:
                processed_frame = self.image_processor.apply_augmentations(frame, training=True)
                final_frames.append(processed_frame)
            processed_frames = np.array(final_frames)
        
        result = {
            'frames': processed_frames,
            'num_frames': len(processed_frames),
            'frame_shape': processed_frames.shape[1:],
        }
        
        if return_metadata:
            result['metadata'] = metadata
        
        if self.cache_frames:
            self.frame_cache[cache_key] = result
        
        return result
    
    def process_for_model(
        self,
        video_path: Union[str, Path],
        training: bool = False
    ) -> torch.Tensor:
        """
        Process video for model input (returns tensor).
        
        Args:
            video_path: Path to video file
            training: Whether this is for training
            
        Returns:
            Video tensor [num_frames, channels, height, width]
        """
        processed = self.process_video(video_path, training=training)
        frames = processed['frames']
        
        video_tensor = torch.from_numpy(frames).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        
        return video_tensor
    
    def process_batch(
        self,
        video_paths: List[Union[str, Path]],
        training: bool = False
    ) -> torch.Tensor:
        """
        Process a batch of videos.
        
        Args:
            video_paths: List of video paths
            training: Whether this is for training
            
        Returns:
            Batched video tensor [batch_size, num_frames, channels, height, width]
        """
        batch_videos = []
        
        for video_path in video_paths:
            video_tensor = self.process_for_model(video_path, training)
            batch_videos.append(video_tensor)
        
        batch_tensor = torch.stack(batch_videos, dim=0)
        
        return batch_tensor
    
    def extract_video_features(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Extract basic video features for analysis.
        
        Args:
            frames: Input frames
            
        Returns:
            Dictionary of video features
        """
        features = {}
        
        features['num_frames'] = float(len(frames))
        features['avg_frame_size'] = float(np.mean([frame.size for frame in frames]))
        
        if len(frames) > 1:
            motion_magnitudes = []
            for i in range(1, len(frames)):
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, None, None
                )[0]
                
                if flow is not None:
                    motion_mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                    motion_magnitudes.append(motion_mag)
            
            if motion_magnitudes:
                features['avg_motion'] = float(np.mean(motion_magnitudes))
                features['motion_variance'] = float(np.var(motion_magnitudes))
            else:
                features['avg_motion'] = 0.0
                features['motion_variance'] = 0.0
        else:
            features['avg_motion'] = 0.0
            features['motion_variance'] = 0.0
        
        if len(frames) > 1:
            scene_changes = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                scene_changes.append(diff)
            
            features['avg_scene_change'] = float(np.mean(scene_changes))
            features['scene_stability'] = float(1.0 / (1.0 + np.var(scene_changes)))
        else:
            features['avg_scene_change'] = 0.0
            features['scene_stability'] = 1.0
        
        return features
    
    def clear_cache(self):
        """Clear the frame cache."""
        self.frame_cache.clear()
        self.logger.info("Frame cache cleared")
    
    def __repr__(self) -> str:
        return (
            f"VideoProcessor(\n"
            f"  max_frames={self.max_frames},\n"
            f"  frame_size={self.frame_size},\n"
            f"  sampling_strategy='{self.sampling_strategy}',\n"
            f"  temporal_augmentation={self.temporal_augmentation},\n"
            f"  spatial_augmentation={self.spatial_augmentation}\n"
            f")"
        )
