# sarcasm_detection/utils/data_augmentation.py
"""
Comprehensive Data Augmentation for Sarcasm Detection
Text, audio, image, and video augmentation techniques optimized for sarcasm.
"""

import numpy as np
import torch
import random
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import warnings

# NLP augmentation
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    import nlpaug.augmenter.char as nac
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False

# Audio augmentation
try:
    import librosa
    import soundfile as sf
    import audiomentations as AA
    AUDIO_AUG_AVAILABLE = True
except ImportError:
    AUDIO_AUG_AVAILABLE = False

# Image augmentation
try:
    import cv2
    import albumentations as A
    from PIL import Image
    IMAGE_AUG_AVAILABLE = True
except ImportError:
    IMAGE_AUG_AVAILABLE = False

# Video augmentation
try:
    import torchvision.transforms as T
    VIDEO_AUG_AVAILABLE = True
except ImportError:
    VIDEO_AUG_AVAILABLE = False

from shared.utils import get_logger


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # General settings
    augmentation_probability: float = 0.5
    preserve_label: bool = True
    max_augmentations_per_sample: int = 2
    
    # Text augmentation
    text_augmentation: bool = True
    text_methods: List[str] = field(default_factory=lambda: [
        'synonym_replacement', 'contextual_insertion', 'back_translation',
        'paraphrasing', 'punctuation_modification', 'capitalization_change'
    ])
    
    # Audio augmentation
    audio_augmentation: bool = True
    audio_methods: List[str] = field(default_factory=lambda: [
        'noise_addition', 'speed_change', 'pitch_shift', 'time_stretch',
        'volume_change', 'emphasis_modification'
    ])
    
    # Image augmentation
    image_augmentation: bool = True
    image_methods: List[str] = field(default_factory=lambda: [
        'brightness_contrast', 'color_jitter', 'gaussian_blur',
        'facial_expression_enhancement', 'context_modification'
    ])
    
    # Video augmentation
    video_augmentation: bool = True
    video_methods: List[str] = field(default_factory=lambda: [
        'temporal_crop', 'frame_rate_change', 'brightness_variation',
        'gesture_emphasis', 'expression_enhancement'
    ])
    
    # Sarcasm-specific settings
    preserve_sarcasm_markers: bool = True
    enhance_irony_signals: bool = True
    maintain_context: bool = True
    
    # Cross-modal consistency
    maintain_modal_consistency: bool = True
    cross_modal_alignment_weight: float = 0.7


class SarcasmDataAugmenter:
    """Comprehensive data augmenter for sarcasm detection."""
    
    def __init__(self, config: Union[AugmentationConfig, Dict[str, Any]] = None):
        """
        Initialize sarcasm data augmenter.
        
        Args:
            config: Augmentation configuration
        """
        if isinstance(config, dict):
            config = AugmentationConfig(**config)
        elif config is None:
            config = AugmentationConfig()
        
        self.config = config
        self.logger = get_logger("SarcasmDataAugmenter")
        
        # Initialize modality-specific augmenters
        self.text_augmenter = TextAugmenter(config)
        self.audio_augmenter = AudioAugmenter(config)
        self.image_augmenter = ImageAugmenter(config)
        self.video_augmenter = VideoAugmenter(config)
        
        # Initialize specialized augmenters
        self.contextual_augmenter = ContextualAugmenter(config)
        self.back_translation_augmenter = BackTranslationAugmenter(config)
        self.syntactic_augmenter = SyntacticAugmenter(config)
        
        self.logger.info("Initialized comprehensive sarcasm data augmenter")
    
    def augment_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Augment a single sample across all available modalities.
        
        Args:
            sample: Original sample
            
        Returns:
            List of augmented samples
        """
        augmented_samples = []
        
        # Determine number of augmentations to generate
        num_augmentations = random.randint(1, self.config.max_augmentations_per_sample)
        
        for _ in range(num_augmentations):
            if random.random() > self.config.augmentation_probability:
                continue
            
            augmented_sample = sample.copy()
            
            # Apply multimodal augmentation
            augmented_sample = self._apply_multimodal_augmentation(augmented_sample)
            
            # Ensure cross-modal consistency if required
            if self.config.maintain_modal_consistency:
                augmented_sample = self._ensure_cross_modal_consistency(
                    original_sample=sample,
                    augmented_sample=augmented_sample
                )
            
            augmented_samples.append(augmented_sample)
        
        return augmented_samples
    
    def _apply_multimodal_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation across all available modalities."""
        
        augmented_sample = sample.copy()
        
        # Text augmentation
        if 'text' in sample and self.config.text_augmentation:
            try:
                augmented_text = self.text_augmenter.augment_text(
                    sample['text'], 
                    preserve_sarcasm=self.config.preserve_sarcasm_markers
                )
                if augmented_text:
                    augmented_sample['text'] = augmented_text
            except Exception as e:
                self.logger.debug(f"Text augmentation failed: {e}")
        
        # Audio augmentation
        if 'audio' in sample and self.config.audio_augmentation:
            try:
                augmented_audio = self.audio_augmenter.augment_audio(
                    sample['audio'],
                    preserve_emphasis=self.config.preserve_sarcasm_markers
                )
                if augmented_audio is not None:
                    augmented_sample['audio'] = augmented_audio
            except Exception as e:
                self.logger.debug(f"Audio augmentation failed: {e}")
        
        # Image augmentation
        if 'image' in sample and self.config.image_augmentation:
            try:
                augmented_image = self.image_augmenter.augment_image(
                    sample['image'],
                    preserve_expressions=self.config.preserve_sarcasm_markers
                )
                if augmented_image is not None:
                    augmented_sample['image'] = augmented_image
            except Exception as e:
                self.logger.debug(f"Image augmentation failed: {e}")
        
        # Video augmentation  
        if 'video' in sample and self.config.video_augmentation:
            try:
                augmented_video = self.video_augmenter.augment_video(
                    sample['video'],
                    preserve_gestures=self.config.preserve_sarcasm_markers
                )
                if augmented_video is not None:
                    augmented_sample['video'] = augmented_video
            except Exception as e:
                self.logger.debug(f"Video augmentation failed: {e}")
        
        return augmented_sample
    
    def _ensure_cross_modal_consistency(
        self, 
        original_sample: Dict[str, Any], 
        augmented_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure consistency across modalities after augmentation."""
        
        # This is a placeholder for more sophisticated cross-modal consistency
        # In practice, this would involve checking that augmentations don't
        # contradict each other across modalities
        
        consistent_sample = augmented_sample.copy()
        
        # Example: If text sentiment changes drastically, adjust image/video accordingly
        if 'text' in original_sample and 'text' in augmented_sample:
            original_sentiment = self._estimate_sentiment(original_sample['text'])
            augmented_sentiment = self._estimate_sentiment(augmented_sample['text'])
            
            sentiment_change = abs(original_sentiment - augmented_sentiment)
            
            # If sentiment changed significantly, we might need to adjust other modalities
            if sentiment_change > 0.5:
                self.logger.debug("Large sentiment change detected, adjusting cross-modal consistency")
                # Could adjust image brightness, audio tone, etc.
        
        return consistent_sample
    
    def _estimate_sentiment(self, text: str) -> float:
        """Simple sentiment estimation for consistency checking."""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def augment_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        target_size_multiplier: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Augment entire dataset.
        
        Args:
            dataset: Original dataset
            target_size_multiplier: How much to multiply dataset size
            
        Returns:
            Augmented dataset
        """
        self.logger.info(f"Augmenting dataset of size {len(dataset)}")
        
        augmented_dataset = dataset.copy()  # Keep original samples
        target_new_samples = int(len(dataset) * (target_size_multiplier - 1))
        
        samples_added = 0
        attempts = 0
        max_attempts = target_new_samples * 2
        
        while samples_added < target_new_samples and attempts < max_attempts:
            # Randomly select sample to augment
            original_sample = random.choice(dataset)
            
            # Generate augmentations
            augmented_samples = self.augment_sample(original_sample)
            
            for aug_sample in augmented_samples:
                if samples_added >= target_new_samples:
                    break
                
                augmented_dataset.append(aug_sample)
                samples_added += 1
            
            attempts += 1
        
        self.logger.info(f"Dataset augmented from {len(dataset)} to {len(augmented_dataset)} samples")
        
        return augmented_dataset


class TextAugmenter:
    """Specialized text augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize text augmenter."""
        self.config = config
        self.logger = get_logger("TextAugmenter")
        
        # Sarcasm-specific patterns and markers
        self.sarcasm_markers = [
            r'\boh\s+(sure|great|wonderful|perfect)\b',
            r'\byeah\s+(right|sure)\b',
            r'\bas\s+if\b',
            r'\btotally\b',
            r'\babsolutely\b',
            r'\.{3,}',  # Ellipsis
            r'[!]{2,}', # Multiple exclamations
            r'\bwow\b.*\bimpressive\b'
        ]
        
        self.compiled_markers = [re.compile(pattern, re.IGNORECASE) for pattern in self.sarcasm_markers]
        
        # Initialize NLP augmenters if available
        self._setup_nlp_augmenters()
    
    def _setup_nlp_augmenters(self):
        """Setup NLP-based augmenters."""
        if not NLPAUG_AVAILABLE:
            self.logger.warning("nlpaug not available, using basic augmentation only")
            return
        
        try:
            # Synonym replacement
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
            
            # Contextual word insertion
            self.insert_aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', 
                action="insert"
            )
            
            # Character-level augmentation
            self.char_aug = nac.RandomCharAug(action="insert")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize NLP augmenters: {e}")
            self.synonym_aug = None
            self.insert_aug = None
            self.char_aug = None
    
    def augment_text(
        self, 
        text: str, 
        preserve_sarcasm: bool = True,
        method: Optional[str] = None
    ) -> Optional[str]:
        """
        Augment text while preserving sarcasm indicators.
        
        Args:
            text: Original text
            preserve_sarcasm: Whether to preserve sarcasm markers
            method: Specific augmentation method to use
            
        Returns:
            Augmented text
        """
        if not text or not text.strip():
            return text
        
        # Choose augmentation method
        if method is None:
            method = random.choice(self.config.text_methods)
        
        # Extract and preserve sarcasm markers
        preserved_markers = []
        if preserve_sarcasm:
            preserved_markers = self._extract_sarcasm_markers(text)
        
        # Apply augmentation
        try:
            if method == 'synonym_replacement':
                augmented = self._synonym_replacement(text)
            elif method == 'contextual_insertion':
                augmented = self._contextual_insertion(text)
            elif method == 'back_translation':
                augmented = self._back_translation(text)
            elif method == 'paraphrasing':
                augmented = self._paraphrasing(text)
            elif method == 'punctuation_modification':
                augmented = self._punctuation_modification(text)
            elif method == 'capitalization_change':
                augmented = self._capitalization_change(text)
            elif method == 'irony_enhancement':
                augmented = self._enhance_irony(text)
            else:
                augmented = text
            
            # Restore sarcasm markers if needed
            if preserve_sarcasm and preserved_markers:
                augmented = self._restore_sarcasm_markers(augmented, preserved_markers)
            
            return augmented
            
        except Exception as e:
            self.logger.debug(f"Text augmentation method {method} failed: {e}")
            return text
    
    def _extract_sarcasm_markers(self, text: str) -> List[Dict[str, Any]]:
        """Extract sarcasm markers from text."""
        markers = []
        
        for i, pattern in enumerate(self.compiled_markers):
            matches = list(pattern.finditer(text))
            for match in matches:
                markers.append({
                    'pattern_idx': i,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return markers
    
    def _restore_sarcasm_markers(self, text: str, markers: List[Dict[str, Any]]) -> str:
        """Restore sarcasm markers in augmented text."""
        # Simple approach: append important markers if missing
        restored_text = text
        
        for marker in markers:
            marker_text = marker['text']
            if marker_text.lower() not in restored_text.lower():
                # Try to insert marker naturally
                if '.' in restored_text:
                    parts = restored_text.rsplit('.', 1)
                    if len(parts) == 2:
                        restored_text = f"{parts[0]} {marker_text}.{parts[1]}"
                else:
                    restored_text = f"{restored_text} {marker_text}"
        
        return restored_text
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        if self.synonym_aug and NLPAUG_AVAILABLE:
            try:
                return self.synonym_aug.augment(text)
            except:
                pass
        
        # Fallback: simple word replacement
        return self._simple_synonym_replacement(text)
    
    def _simple_synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement fallback."""
        simple_synonyms = {
            'good': ['great', 'excellent', 'wonderful', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'petite'],
            'happy': ['joyful', 'cheerful', 'delighted', 'elated'],
            'sad': ['unhappy', 'miserable', 'depressed', 'gloomy']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in simple_synonyms and random.random() < 0.3:
                synonym = random.choice(simple_synonyms[word_lower])
                # Preserve original capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                words[i] = word.replace(word_lower, synonym)
        
        return ' '.join(words)
    
    def _contextual_insertion(self, text: str) -> str:
        """Insert contextually appropriate words."""
        if self.insert_aug and NLPAUG_AVAILABLE:
            try:
                return self.insert_aug.augment(text)
            except:
                pass
        
        # Fallback: insert sarcasm-enhancing words
        sarcasm_enhancers = ['really', 'totally', 'absolutely', 'definitely', 'certainly']
        
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            if sentence.strip() and random.random() < 0.4:
                words = sentence.strip().split()
                if len(words) > 2:
                    insert_pos = random.randint(1, len(words) - 1)
                    enhancer = random.choice(sarcasm_enhancers)
                    words.insert(insert_pos, enhancer)
                    sentences[i] = ' '.join(words)
        
        return '.'.join(sentences)
    
    def _back_translation(self, text: str) -> str:
        """Back-translation augmentation (placeholder)."""
        # This would involve translating to another language and back
        # For now, return original text
        return text
    
    def _paraphrasing(self, text: str) -> str:
        """Paraphrase the text while maintaining meaning."""
        # Simplified paraphrasing by restructuring
        paraphrase_patterns = [
            (r'I think (.+)', r'It seems to me that \1'),
            (r'This is (.+)', r'This appears to be \1'),
            (r'(.+) is really (.+)', r'\1 is quite \2'),
            (r'(.+) is very (.+)', r'\1 is extremely \2')
        ]
        
        augmented = text
        for pattern, replacement in paraphrase_patterns:
            if random.random() < 0.3:
                augmented = re.sub(pattern, replacement, augmented, flags=re.IGNORECASE)
        
        return augmented
    
    def _punctuation_modification(self, text: str) -> str:
        """Modify punctuation for sarcasm enhancement."""
        # Add ellipsis for dramatic effect
        if '.' in text and random.random() < 0.4:
            text = text.replace('.', '...', 1)
        
        # Add multiple exclamation marks
        if '!' in text and random.random() < 0.3:
            text = text.replace('!', '!!', 1)
        
        # Add question mark for rhetorical effect
        if not text.endswith('?') and random.random() < 0.2:
            text = text.rstrip('.!') + '?'
        
        return text
    
    def _capitalization_change(self, text: str) -> str:
        """Modify capitalization for emphasis."""
        words = text.split()
        
        for i, word in enumerate(words):
            if random.random() < 0.1:  # Low probability to maintain readability
                # Emphasize word with ALL CAPS
                if len(word) > 3 and word.isalpha():
                    words[i] = word.upper()
        
        return ' '.join(words)
    
    def _enhance_irony(self, text: str) -> str:
        """Enhance irony in the text."""
        irony_enhancers = [
            ('good', 'absolutely wonderful'),
            ('nice', 'just fantastic'), 
            ('great', 'truly spectacular'),
            ('perfect', 'absolutely perfect')
        ]
        
        enhanced = text
        for original, replacement in irony_enhancers:
            if original in enhanced.lower() and random.random() < 0.4:
                enhanced = re.sub(rf'\b{original}\b', replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced


class AudioAugmenter:
    """Specialized audio augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize audio augmenter."""
        self.config = config
        self.logger = get_logger("AudioAugmenter")
        
        if not AUDIO_AUG_AVAILABLE:
            self.logger.warning("Audio augmentation libraries not available")
            return
        
        # Setup audio augmentation pipeline
        self._setup_audio_pipeline()
    
    def _setup_audio_pipeline(self):
        """Setup audio augmentation pipeline."""
        if not AUDIO_AUG_AVAILABLE:
            return
        
        self.augment_pipeline = AA.Compose([
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            AA.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
        ])
    
    def augment_audio(
        self, 
        audio_data: Union[np.ndarray, str, Path], 
        preserve_emphasis: bool = True,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Augment audio while preserving sarcastic emphasis.
        
        Args:
            audio_data: Audio data or file path
            preserve_emphasis: Whether to preserve vocal emphasis
            sample_rate: Audio sample rate
            
        Returns:
            Augmented audio data
        """
        if not AUDIO_AUG_AVAILABLE:
            return None
        
        try:
            # Load audio if path provided
            if isinstance(audio_data, (str, Path)):
                audio, sr = librosa.load(audio_data, sr=sample_rate)
            else:
                audio = audio_data
                sr = sample_rate
            
            # Apply augmentation
            augmented = self.augment_pipeline(samples=audio, sample_rate=sr)
            
            # Preserve emphasis if requested
            if preserve_emphasis:
                augmented = self._preserve_vocal_emphasis(audio, augmented)
            
            return augmented
            
        except Exception as e:
            self.logger.debug(f"Audio augmentation failed: {e}")
            return None
    
    def _preserve_vocal_emphasis(
        self, 
        original: np.ndarray, 
        augmented: np.ndarray
    ) -> np.ndarray:
        """Preserve vocal emphasis patterns in augmented audio."""
        # Simple approach: blend high-energy regions from original
        
        # Calculate energy
        orig_energy = librosa.feature.rms(y=original)[0]
        aug_energy = librosa.feature.rms(y=augmented)[0]
        
        # Find high-energy regions (potential emphasis)
        energy_threshold = np.percentile(orig_energy, 75)
        emphasis_regions = orig_energy > energy_threshold
        
        # Blend regions to preserve emphasis
        preserved = augmented.copy()
        
        # This is a simplified approach - in practice would need more sophisticated 
        # analysis of prosodic features
        
        return preserved


class ImageAugmenter:
    """Specialized image augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize image augmenter."""
        self.config = config
        self.logger = get_logger("ImageAugmenter")
        
        if not IMAGE_AUG_AVAILABLE:
            self.logger.warning("Image augmentation libraries not available")
            return
        
        self._setup_image_pipeline()
    
    def _setup_image_pipeline(self):
        """Setup image augmentation pipeline."""
        if not IMAGE_AUG_AVAILABLE:
            return
        
        self.augment_pipeline = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ])
    
    def augment_image(
        self, 
        image: Union[np.ndarray, str, Path], 
        preserve_expressions: bool = True
    ) -> Optional[np.ndarray]:
        """
        Augment image while preserving facial expressions.
        
        Args:
            image: Image data or file path
            preserve_expressions: Whether to preserve facial expressions
            
        Returns:
            Augmented image
        """
        if not IMAGE_AUG_AVAILABLE:
            return None
        
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image
            
            # Apply augmentation
            augmented = self.augment_pipeline(image=img)['image']
            
            # Preserve facial expressions if needed
            if preserve_expressions:
                augmented = self._preserve_facial_expressions(img, augmented)
            
            return augmented
            
        except Exception as e:
            self.logger.debug(f"Image augmentation failed: {e}")
            return None
    
    def _preserve_facial_expressions(
        self, 
        original: np.ndarray, 
        augmented: np.ndarray
    ) -> np.ndarray:
        """Preserve facial expressions in augmented image."""
        # This would require facial landmark detection and expression analysis
        # For now, return augmented image
        return augmented


class VideoAugmenter:
    """Specialized video augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize video augmenter."""
        self.config = config
        self.logger = get_logger("VideoAugmenter")
        
        if not VIDEO_AUG_AVAILABLE:
            self.logger.warning("Video augmentation libraries not available")
    
    def augment_video(
        self, 
        video_frames: Union[torch.Tensor, np.ndarray, str, Path],
        preserve_gestures: bool = True
    ) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Augment video while preserving important gestures.
        
        Args:
            video_frames: Video frames tensor or file path
            preserve_gestures: Whether to preserve gesture information
            
        Returns:
            Augmented video frames
        """
        if not VIDEO_AUG_AVAILABLE:
            return None
        
        try:
            # Load video if path provided
            if isinstance(video_frames, (str, Path)):
                # Would need video loading logic here
                return None
            
            # Convert to tensor if numpy
            if isinstance(video_frames, np.ndarray):
                frames = torch.from_numpy(video_frames)
            else:
                frames = video_frames
            
            # Apply video augmentation
            augmented = self._apply_video_transforms(frames)
            
            return augmented
            
        except Exception as e:
            self.logger.debug(f"Video augmentation failed: {e}")
            return None
    
    def _apply_video_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply video-specific transformations."""
        # Basic video transforms
        transforms = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
        
        # Apply to each frame
        augmented_frames = []
        for frame in frames:
            aug_frame = transforms(frame)
            augmented_frames.append(aug_frame)
        
        return torch.stack(augmented_frames)


class MultimodalAugmenter:
    """Coordinated augmentation across multiple modalities."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize multimodal augmenter."""
        self.config = config
        self.logger = get_logger("MultimodalAugmenter")
    
    def coordinated_augmentation(
        self, 
        sample: Dict[str, Any], 
        consistency_weight: float = 0.7
    ) -> Dict[str, Any]:
        """Apply coordinated augmentation across modalities."""
        
        augmented_sample = sample.copy()
        
        # Choose a primary modality to guide augmentation
        available_modalities = [mod for mod in ['text', 'audio', 'image', 'video'] if mod in sample]
        
        if not available_modalities:
            return sample
        
        primary_modality = random.choice(available_modalities)
        
        # Determine augmentation theme based on primary modality
        if primary_modality == 'text':
            theme = self._analyze_text_theme(sample['text'])
        else:
            theme = 'neutral'  # Default theme
        
        # Apply theme-consistent augmentation
        if theme == 'high_sarcasm':
            augmented_sample = self._apply_high_sarcasm_augmentation(augmented_sample)
        elif theme == 'subtle_sarcasm':
            augmented_sample = self._apply_subtle_sarcasm_augmentation(augmented_sample)
        else:
            augmented_sample = self._apply_neutral_augmentation(augmented_sample)
        
        return augmented_sample
    
    def _analyze_text_theme(self, text: str) -> str:
        """Analyze text to determine sarcasm theme."""
        # Simple heuristic-based analysis
        sarcasm_indicators = ['oh sure', 'yeah right', 'totally', 'absolutely', '...', '!!']
        
        indicator_count = sum(1 for indicator in sarcasm_indicators if indicator in text.lower())
        
        if indicator_count >= 2:
            return 'high_sarcasm'
        elif indicator_count == 1:
            return 'subtle_sarcasm'
        else:
            return 'neutral'
    
    def _apply_high_sarcasm_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation for high sarcasm samples."""
        # Enhance sarcasm markers in text
        if 'text' in sample:
            sample['text'] = self._enhance_sarcasm_markers(sample['text'])
        
        # Increase emphasis in audio (if available)
        # Enhance facial expressions in image/video (if available)
        
        return sample
    
    def _apply_subtle_sarcasm_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation for subtle sarcasm samples."""
        # Add subtle sarcasm indicators
        if 'text' in sample:
            sample['text'] = self._add_subtle_indicators(sample['text'])
        
        return sample
    
    def _apply_neutral_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply neutral augmentation."""
        # Standard augmentation without sarcasm enhancement
        return sample
    
    def _enhance_sarcasm_markers(self, text: str) -> str:
        """Enhance existing sarcasm markers."""
        # Replace single punctuation with multiple
        text = re.sub(r'!(?!!)', '!!', text)
        text = re.sub(r'\.(?!\.)', '...', text)
        
        return text
    
    def _add_subtle_indicators(self, text: str) -> str:
        """Add subtle sarcasm indicators."""
        subtle_additions = ['I\'m sure', 'of course', 'naturally']
        
        if random.random() < 0.3:
            addition = random.choice(subtle_additions)
            # Insert at beginning with comma
            text = f"{addition}, {text.lower()}"
        
        return text


class ContextualAugmenter:
    """Context-aware augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize contextual augmenter."""
        self.config = config
        self.logger = get_logger("ContextualAugmenter")
    
    def context_aware_augmentation(
        self, 
        sample: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply context-aware augmentation."""
        
        if not context:
            return sample
        
        # Use context to guide augmentation
        # This could include conversation history, speaker information, etc.
        
        augmented_sample = sample.copy()
        
        # Example: Adjust augmentation based on conversation context
        if 'conversation_history' in context:
            history = context['conversation_history']
            if self._is_heated_conversation(history):
                augmented_sample = self._apply_heated_augmentation(augmented_sample)
            elif self._is_casual_conversation(history):
                augmented_sample = self._apply_casual_augmentation(augmented_sample)
        
        return augmented_sample
    
    def _is_heated_conversation(self, history: List[str]) -> bool:
        """Check if conversation is heated."""
        heated_indicators = ['angry', 'mad', 'frustrated', 'ridiculous', 'stupid']
        
        for message in history[-3:]:  # Check last 3 messages
            if any(indicator in message.lower() for indicator in heated_indicators):
                return True
        
        return False
    
    def _is_casual_conversation(self, history: List[str]) -> bool:
        """Check if conversation is casual."""
        casual_indicators = ['lol', 'haha', 'funny', 'cool', 'nice']
        
        indicator_count = 0
        for message in history[-5:]:  # Check last 5 messages
            if any(indicator in message.lower() for indicator in casual_indicators):
                indicator_count += 1
        
        return indicator_count >= 2
    
    def _apply_heated_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation for heated context."""
        if 'text' in sample:
            # Add intensity markers
            sample['text'] = sample['text'].replace('.', '!')
        
        return sample
    
    def _apply_casual_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation for casual context."""
        if 'text' in sample:
            # Add casual markers
            casual_additions = ['lol', 'haha', 'right']
            if random.random() < 0.3:
                addition = random.choice(casual_additions)
                sample['text'] = f"{sample['text']} {addition}"
        
        return sample


class BackTranslationAugmenter:
    """Back-translation augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize back-translation augmenter."""
        self.config = config
        self.logger = get_logger("BackTranslationAugmenter")
        
        # This would typically use translation APIs or models
        # For now, it's a placeholder implementation
    
    def back_translate(
        self, 
        text: str, 
        intermediate_language: str = 'es'
    ) -> str:
        """
        Back-translate text through intermediate language.
        
        Args:
            text: Original text
            intermediate_language: Language to translate through
            
        Returns:
            Back-translated text
        """
        # Placeholder implementation
        # In practice, would use Google Translate API, MarianMT, or similar
        
        self.logger.debug(f"Back-translation not implemented, returning original text")
        return text


class SyntacticAugmenter:
    """Syntactic structure augmentation for sarcasm detection."""
    
    def __init__(self, config: AugmentationConfig):
        """Initialize syntactic augmenter."""
        self.config = config
        self.logger = get_logger("SyntacticAugmenter")
    
    def syntactic_augmentation(self, text: str) -> str:
        """Apply syntactic transformations."""
        
        # Simple syntactic transformations
        transformations = [
            self._passive_to_active,
            self._active_to_passive,
            self._add_rhetorical_question,
            self._restructure_with_emphasis
        ]
        
        # Apply random transformation
        if random.random() < 0.4:
            transform = random.choice(transformations)
            return transform(text)
        
        return text
    
    def _passive_to_active(self, text: str) -> str:
        """Convert passive voice to active (simplified)."""
        # Very basic implementation
        passive_patterns = [
            (r'was (.+) by (.+)', r'\2 \1'),
            (r'is (.+) by (.+)', r'\2 \1s')
        ]
        
        for pattern, replacement in passive_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _active_to_passive(self, text: str) -> str:
        """Convert active voice to passive (simplified)."""
        # Very basic implementation - in practice would need proper parsing
        return text
    
    def _add_rhetorical_question(self, text: str) -> str:
        """Add rhetorical question for sarcastic effect."""
        if not text.endswith('?') and random.random() < 0.3:
            rhetorical_additions = [
                ", don't you think?",
                ", right?", 
                ", wouldn't you agree?",
                ", or am I wrong?"
            ]
            
            addition = random.choice(rhetorical_additions)
            text = text.rstrip('.!') + addition
        
        return text
    
    def _restructure_with_emphasis(self, text: str) -> str:
        """Restructure sentence with emphasis."""
        # Add emphasis through restructuring
        if ',' in text:
            parts = text.split(',', 1)
            if len(parts) == 2 and random.random() < 0.4:
                # Rearrange for emphasis
                text = f"{parts[1].strip()}, {parts[0].strip()}"
        
        return text
