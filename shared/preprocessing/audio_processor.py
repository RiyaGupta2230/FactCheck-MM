"""
Audio Preprocessing for FactCheck-MM
Handles resampling, normalization, silence removal, and VAD.
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path
import tempfile
import webrtcvad
from scipy import signal
from transformers import Wav2Vec2Processor

from ..utils import get_logger


class AudioProcessor:
    """
    Comprehensive audio processor for multimodal analysis.
    Handles preprocessing for speech recognition, emotion detection, and sarcasm analysis.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        target_length: Optional[int] = None,
        normalize: bool = True,
        remove_silence: bool = True,
        apply_vad: bool = True,
        vad_aggressiveness: int = 2,
        noise_reduction: bool = True,
        model_name: str = "facebook/wav2vec2-large-960h",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            target_length: Target length in samples (None for variable)
            normalize: Whether to normalize audio amplitude
            remove_silence: Whether to remove silence
            apply_vad: Whether to apply Voice Activity Detection
            vad_aggressiveness: VAD aggressiveness (0-3)
            noise_reduction: Whether to apply noise reduction
            model_name: Wav2Vec2 model name for feature extraction
            cache_dir: Cache directory
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.normalize = normalize
        self.remove_silence = remove_silence
        self.apply_vad = apply_vad
        self.vad_aggressiveness = vad_aggressiveness
        self.noise_reduction = noise_reduction
        
        self.logger = get_logger("AudioProcessor")
        
        # Initialize Wav2Vec2 processor
        try:
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.logger.info(f"Loaded Wav2Vec2 processor: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load Wav2Vec2 processor: {e}")
            raise
        
        # Initialize VAD if needed
        if apply_vad:
            try:
                self.vad = webrtcvad.Vad(vad_aggressiveness)
                self.logger.info(f"Initialized VAD (aggressiveness: {vad_aggressiveness})")
            except Exception as e:
                self.logger.warning(f"Failed to initialize VAD: {e}")
                self.apply_vad = False
        
        # Audio processing parameters
        self.frame_duration_ms = 30  # Frame duration for VAD
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        self.logger.info("Audio processor initialized successfully")
    
    def load_audio(
        self,
        audio_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            duration: Duration in seconds (None for full file)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=None,  # Keep original sample rate initially
                offset=offset,
                duration=duration
            )
            
            self.logger.debug(f"Loaded audio: {audio_path} (sr={sr}, length={len(audio)})")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    def resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Resampled audio
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            self.logger.debug(f"Resampled audio: {orig_sr} -> {target_sr} Hz")
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio: Input audio data
            
        Returns:
            Normalized audio
        """
        if len(audio) == 0:
            return audio
        
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1  # Normalize to reasonable level
        
        # Clip to [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def remove_silence_simple(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Remove silence using energy-based thresholding.
        
        Args:
            audio: Input audio
            threshold: Energy threshold for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            
        Returns:
            Audio with silence removed
        """
        # Compute energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Find non-silent frames
        non_silent = energy > threshold
        
        if not np.any(non_silent):
            self.logger.warning("All audio detected as silence")
            return audio
        
        # Convert frame indices to sample indices
        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Create mask for samples
        mask = np.zeros(len(audio), dtype=bool)
        for i, is_speech in enumerate(non_silent):
            if is_speech:
                start_sample = int(times[i] * self.sample_rate)
                end_sample = int((times[i] + hop_length / self.sample_rate) * self.sample_rate)
                end_sample = min(end_sample, len(audio))
                mask[start_sample:end_sample] = True
        
        return audio[mask]
    
    def apply_vad_processing(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Voice Activity Detection to remove silence.
        
        Args:
            audio: Input audio data
            
        Returns:
            Audio with VAD applied
        """
        if not self.apply_vad or self.vad is None:
            return audio
        
        # Convert to 16-bit PCM for VAD
        pcm_data = (audio * 32767).astype(np.int16)
        
        # Process in frames
        voiced_frames = []
        frame_size = self.frame_size
        
        for start in range(0, len(pcm_data), frame_size):
            end = min(start + frame_size, len(pcm_data))
            frame = pcm_data[start:end]
            
            # Pad frame if necessary
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            
            # Apply VAD
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                if is_speech:
                    voiced_frames.extend(range(start, end))
            except Exception as e:
                self.logger.debug(f"VAD failed for frame: {e}")
                # Include frame if VAD fails
                voiced_frames.extend(range(start, end))
        
        if not voiced_frames:
            self.logger.warning("VAD detected no speech")
            return audio
        
        # Extract voiced audio
        voiced_audio = audio[voiced_frames]
        return voiced_audio
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply basic noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio
            
        Returns:
            Denoised audio
        """
        if not self.noise_reduction:
            return audio
        
        try:
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * self.sample_rate / 512)  # hop_length=512
            noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Lower bound factor
            
            clean_magnitude = magnitude - alpha * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            self.logger.debug(f"Noise reduction failed: {e}")
            return audio
    
    def pad_or_truncate(
        self,
        audio: np.ndarray,
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Pad or truncate audio to target length.
        
        Args:
            audio: Input audio
            target_length: Target length in samples
            
        Returns:
            Processed audio
        """
        if target_length is None:
            target_length = self.target_length
        
        if target_length is None:
            return audio
        
        current_length = len(audio)
        
        if current_length > target_length:
            # Truncate from center
            start = (current_length - target_length) // 2
            audio = audio[start:start + target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            pad_before = padding // 2
            pad_after = padding - pad_before
            audio = np.pad(audio, (pad_before, pad_after), mode='constant')
        
        return audio
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features for analysis.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic features
        features['rms_energy'] = librosa.feature.rms(y=audio)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13
        )
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        
        # Temporal features (for sarcasm detection)
        features['tempo'] = librosa.beat.tempo(y=audio, sr=self.sample_rate)
        
        return features
    
    def process_audio(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Complete audio processing pipeline.
        
        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate if audio is numpy array
            return_features: Whether to extract additional features
            
        Returns:
            Dictionary with processed audio and metadata
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio_data, orig_sr = self.load_audio(audio)
        else:
            audio_data = audio
            orig_sr = sample_rate or self.sample_rate
        
        # Resample
        audio_data = self.resample_audio(audio_data, orig_sr, self.sample_rate)
        
        # Apply noise reduction
        if self.noise_reduction:
            audio_data = self.apply_noise_reduction(audio_data)
        
        # Remove silence
        if self.remove_silence:
            if self.apply_vad:
                audio_data = self.apply_vad_processing(audio_data)
            else:
                audio_data = self.remove_silence_simple(audio_data)
        
        # Normalize
        if self.normalize:
            audio_data = self.normalize_audio(audio_data)
        
        # Pad or truncate
        audio_data = self.pad_or_truncate(audio_data)
        
        # Prepare result
        result = {
            'audio': audio_data,
            'sample_rate': self.sample_rate,
            'duration': len(audio_data) / self.sample_rate,
            'length_samples': len(audio_data)
        }
        
        # Extract features if requested
        if return_features:
            result['features'] = self.extract_features(audio_data)
        
        return result
    
    def process_for_wav2vec(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio for Wav2Vec2 model input.
        
        Args:
            audio: Audio input
            sample_rate: Sample rate if numpy array
            
        Returns:
            Wav2Vec2 processor output
        """
        # Process audio
        processed = self.process_audio(audio, sample_rate)
        audio_data = processed['audio']
        
        # Use Wav2Vec2 processor
        inputs = self.wav2vec_processor(
            audio_data,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs
    
    def process_batch(
        self,
        audio_list: List[Union[str, Path, np.ndarray]],
        sample_rates: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of audio files.
        
        Args:
            audio_list: List of audio inputs
            sample_rates: List of sample rates (if numpy arrays)
            
        Returns:
            Batched processor outputs
        """
        processed_audio = []
        
        for i, audio in enumerate(audio_list):
            sr = sample_rates[i] if sample_rates else None
            processed = self.process_audio(audio, sr)
            processed_audio.append(processed['audio'])
        
        # Use Wav2Vec2 processor for batching
        inputs = self.wav2vec_processor(
            processed_audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Save processed audio to file.
        
        Args:
            audio: Audio data
            output_path: Output file path
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        sf.write(output_path, audio, sample_rate)
        self.logger.info(f"Saved audio to {output_path}")
    
    def __repr__(self) -> str:
        return (
            f"AudioProcessor(\n"
            f"  sample_rate={self.sample_rate},\n"
            f"  target_length={self.target_length},\n"
            f"  normalize={self.normalize},\n"
            f"  vad_enabled={self.apply_vad},\n"
            f"  noise_reduction={self.noise_reduction}\n"
            f")"
        )
