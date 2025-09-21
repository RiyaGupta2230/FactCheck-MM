"""
Unit Tests for FactCheck-MM Preprocessing

Tests text, audio, image, and video processors from shared/preprocessing.
Validates tokenization, feature extraction, and data transformations.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.preprocessing.text_processor import TextProcessor
from shared.preprocessing.audio_processor import AudioProcessor
from shared.preprocessing.image_processor import ImageProcessor
from shared.preprocessing.video_processor import VideoProcessor
from tests import TEST_CONFIG


class TestTextProcessor:
    """Test text preprocessing functionality."""
    
    @pytest.fixture
    def text_processor_config(self):
        """Text processor configuration."""
        return {
            'model_name': 'distilbert-base-uncased',
            'max_length': TEST_CONFIG['test_sequence_length'],
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt'
        }
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is a great example of sarcasm.",
            "Wow, that's really helpful!",
            "Perfect timing as always.",
            "",  # Empty string
            "A very long sentence that might need to be truncated if it exceeds the maximum length limit."
        ]
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_text_processor_initialization(self, mock_tokenizer, text_processor_config):
        """Test text processor initialization."""
        mock_tok = Mock()
        mock_tokenizer.return_value = mock_tok
        
        processor = TextProcessor(text_processor_config)
        
        assert processor.max_length == text_processor_config['max_length']
        assert processor.padding == text_processor_config['padding']
        assert processor.truncation == text_processor_config['truncation']
        mock_tokenizer.assert_called_once_with(text_processor_config['model_name'])
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_text_tokenization(self, mock_tokenizer, text_processor_config, sample_texts):
        """Test text tokenization functionality."""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.return_value = {
            'input_ids': torch.randint(0, 1000, (len(sample_texts), text_processor_config['max_length'])),
            'attention_mask': torch.ones(len(sample_texts), text_processor_config['max_length'])
        }
        mock_tokenizer.return_value = mock_tok
        
        processor = TextProcessor(text_processor_config)
        result = processor.process(sample_texts)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert result['input_ids'].shape[0] == len(sample_texts)
        assert result['input_ids'].shape[1] == text_processor_config['max_length']
        assert result['attention_mask'].shape == result['input_ids'].shape
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_single_text_processing(self, mock_tokenizer, text_processor_config):
        """Test processing single text string."""
        mock_tok = Mock()
        mock_tok.return_value = {
            'input_ids': torch.randint(0, 1000, (1, text_processor_config['max_length'])),
            'attention_mask': torch.ones(1, text_processor_config['max_length'])
        }
        mock_tokenizer.return_value = mock_tok
        
        processor = TextProcessor(text_processor_config)
        single_text = "This is a single text."
        
        result = processor.process(single_text)
        
        assert result['input_ids'].shape[0] == 1
        assert result['attention_mask'].shape[0] == 1
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_text_cleaning(self, mock_tokenizer, text_processor_config):
        """Test text cleaning functionality."""
        mock_tok = Mock()
        mock_tok.return_value = {
            'input_ids': torch.randint(0, 1000, (1, text_processor_config['max_length'])),
            'attention_mask': torch.ones(1, text_processor_config['max_length'])
        }
        mock_tokenizer.return_value = mock_tok
        
        processor = TextProcessor(text_processor_config)
        
        # Test with messy text
        messy_text = "  This has   extra spaces\nand\nnewlines!!! 😀 "
        cleaned = processor.clean_text(messy_text)
        
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert '  ' not in cleaned  # No double spaces
        assert '\n' not in cleaned  # No newlines
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_batch_processing(self, mock_tokenizer, text_processor_config, sample_texts):
        """Test batch processing of texts."""
        mock_tok = Mock()
        mock_tok.return_value = {
            'input_ids': torch.randint(0, 1000, (len(sample_texts), text_processor_config['max_length'])),
            'attention_mask': torch.ones(len(sample_texts), text_processor_config['max_length'])
        }
        mock_tokenizer.return_value = mock_tok
        
        processor = TextProcessor(text_processor_config)
        
        # Test processing in batches
        batch_size = 2
        results = []
        
        for i in range(0, len(sample_texts), batch_size):
            batch = sample_texts[i:i + batch_size]
            batch_result = processor.process(batch)
            results.append(batch_result)
        
        # Verify batch results
        for result in results:
            assert 'input_ids' in result
            assert 'attention_mask' in result
            assert result['input_ids'].shape[1] == text_processor_config['max_length']


class TestAudioProcessor:
    """Test audio preprocessing functionality."""
    
    @pytest.fixture
    def audio_processor_config(self):
        """Audio processor configuration."""
        return {
            'sample_rate': 16000,
            'n_mels': 80,
            'n_fft': 1024,
            'hop_length': 256,
            'max_length': 500,  # Max frames
            'normalize': True
        }
    
    @pytest.fixture
    def sample_audio(self, audio_processor_config):
        """Generate sample audio data."""
        sample_rate = audio_processor_config['sample_rate']
        duration = 2.0  # 2 seconds
        samples = int(sample_rate * duration)
        
        # Generate sine wave
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return audio, sample_rate
    
    def test_audio_processor_initialization(self, audio_processor_config):
        """Test audio processor initialization."""
        processor = AudioProcessor(audio_processor_config)
        
        assert processor.sample_rate == audio_processor_config['sample_rate']
        assert processor.n_mels == audio_processor_config['n_mels']
        assert processor.n_fft == audio_processor_config['n_fft']
        assert processor.hop_length == audio_processor_config['hop_length']
    
    @patch('librosa.load')
    def test_audio_loading(self, mock_load, audio_processor_config, sample_audio):
        """Test audio file loading."""
        mock_load.return_value = sample_audio
        
        processor = AudioProcessor(audio_processor_config)
        
        # Mock audio file path
        audio_path = "mock_audio.wav"
        loaded_audio, sr = processor.load_audio(audio_path)
        
        mock_load.assert_called_once_with(audio_path, sr=audio_processor_config['sample_rate'])
        assert loaded_audio.shape == sample_audio[0].shape
        assert sr == sample_audio[1]
    
    def test_melspectrogram_extraction(self, audio_processor_config, sample_audio):
        """Test mel-spectrogram feature extraction."""
        processor = AudioProcessor(audio_processor_config)
        
        audio_data, _ = sample_audio
        features = processor.extract_melspectrogram(audio_data)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2  # [n_mels, time_frames]
        assert features.shape[0] == audio_processor_config['n_mels']
        assert features.shape[1] > 0  # Should have time frames
    
    def test_audio_normalization(self, audio_processor_config, sample_audio):
        """Test audio normalization."""
        processor = AudioProcessor(audio_processor_config)
        
        audio_data, _ = sample_audio
        
        # Add some outliers
        audio_with_outliers = audio_data.copy()
        audio_with_outliers[0] = 10.0  # Large value
        audio_with_outliers[-1] = -10.0  # Large negative value
        
        normalized = processor.normalize_audio(audio_with_outliers)
        
        # Check normalization bounds
        assert np.max(normalized) <= 1.0
        assert np.min(normalized) >= -1.0
        assert np.abs(np.max(normalized)) <= 1.1  # Allow small tolerance
    
    def test_audio_padding_truncation(self, audio_processor_config, sample_audio):
        """Test audio padding and truncation."""
        processor = AudioProcessor(audio_processor_config)
        
        audio_data, _ = sample_audio
        
        # Test truncation (long audio)
        long_audio = np.tile(audio_data, 10)  # Make it much longer
        processed_long = processor.process(long_audio)
        
        # Should be truncated to max_length frames
        max_frames = audio_processor_config['max_length']
        assert processed_long.shape[1] <= max_frames
        
        # Test padding (short audio)
        short_audio = audio_data[:1000]  # Very short
        processed_short = processor.process(short_audio)
        
        # Should have consistent output shape
        assert processed_short.shape[0] == audio_processor_config['n_mels']
    
    def test_batch_audio_processing(self, audio_processor_config):
        """Test batch processing of audio files."""
        processor = AudioProcessor(audio_processor_config)
        
        # Create multiple audio samples
        batch_size = 3
        audio_batch = []
        
        for i in range(batch_size):
            # Generate different frequency sine waves
            duration = 1.0 + i * 0.5  # Different lengths
            samples = int(audio_processor_config['sample_rate'] * duration)
            t = np.linspace(0, duration, samples)
            frequency = 440 + i * 100  # Different frequencies
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            audio_batch.append(audio)
        
        # Process batch
        results = []
        for audio in audio_batch:
            result = processor.process(audio)
            results.append(result)
        
        # Check consistency
        for result in results:
            assert result.shape[0] == audio_processor_config['n_mels']
            assert isinstance(result, np.ndarray)


class TestImageProcessor:
    """Test image preprocessing functionality."""
    
    @pytest.fixture
    def image_processor_config(self):
        """Image processor configuration."""
        return {
            'target_size': (224, 224),
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],  # ImageNet means
            'std': [0.229, 0.224, 0.225],   # ImageNet stds
            'augment': False  # Disable augmentation for tests
        }
    
    @pytest.fixture
    def sample_image(self, image_processor_config):
        """Generate sample image data."""
        height, width = image_processor_config['target_size']
        # Create RGB image
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return image
    
    def test_image_processor_initialization(self, image_processor_config):
        """Test image processor initialization."""
        processor = ImageProcessor(image_processor_config)
        
        assert processor.target_size == image_processor_config['target_size']
        assert processor.normalize == image_processor_config['normalize']
        assert processor.mean == image_processor_config['mean']
        assert processor.std == image_processor_config['std']
    
    @patch('cv2.imread')
    def test_image_loading(self, mock_imread, image_processor_config, sample_image):
        """Test image file loading."""
        # Mock OpenCV imread
        mock_imread.return_value = sample_image
        
        processor = ImageProcessor(image_processor_config)
        
        image_path = "mock_image.jpg"
        loaded_image = processor.load_image(image_path)
        
        mock_imread.assert_called_once_with(image_path)
        assert loaded_image.shape == sample_image.shape
    
    def test_image_resizing(self, image_processor_config):
        """Test image resizing functionality."""
        processor = ImageProcessor(image_processor_config)
        
        # Create image with different size
        original_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        resized = processor.resize_image(original_image)
        
        target_height, target_width = image_processor_config['target_size']
        assert resized.shape == (target_height, target_width, 3)
    
    def test_image_normalization(self, image_processor_config, sample_image):
        """Test image normalization."""
        processor = ImageProcessor(image_processor_config)
        
        # Convert to float and normalize to [0, 1]
        float_image = sample_image.astype(np.float32) / 255.0
        
        normalized = processor.normalize_image(float_image)
        
        # Check that normalization was applied
        assert normalized.dtype == np.float32
        # Values should be roughly in the range of normalized ImageNet data
        assert np.min(normalized) > -5  # Loose bounds
        assert np.max(normalized) < 5
    
    def test_image_to_tensor(self, image_processor_config, sample_image):
        """Test conversion of image to tensor."""
        processor = ImageProcessor(image_processor_config)
        
        processed = processor.process(sample_image)
        
        assert isinstance(processed, torch.Tensor)
        # Should be [C, H, W] format
        assert processed.shape[0] == 3  # RGB channels
        assert processed.shape[1] == image_processor_config['target_size'][0]  # Height
        assert processed.shape[2] == image_processor_config['target_size'][1]  # Width
    
    def test_batch_image_processing(self, image_processor_config):
        """Test batch processing of images."""
        processor = ImageProcessor(image_processor_config)
        
        # Create batch of images
        batch_size = 4
        height, width = image_processor_config['target_size']
        
        image_batch = []
        for i in range(batch_size):
            # Create images with different patterns
            image = np.random.randint(0, 256, (height + i * 10, width + i * 10, 3), dtype=np.uint8)
            image_batch.append(image)
        
        # Process batch
        processed_batch = []
        for image in image_batch:
            processed = processor.process(image)
            processed_batch.append(processed)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_batch)
        
        assert batch_tensor.shape[0] == batch_size
        assert batch_tensor.shape[1] == 3  # RGB channels
        assert batch_tensor.shape[2] == height  # Target height
        assert batch_tensor.shape[3] == width   # Target width
    
    def test_image_format_handling(self, image_processor_config):
        """Test handling of different image formats."""
        processor = ImageProcessor(image_processor_config)
        
        height, width = image_processor_config['target_size']
        
        # Test grayscale image (should be converted to RGB)
        grayscale_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        processed_gray = processor.process(grayscale_image)
        
        assert processed_gray.shape[0] == 3  # Should be converted to RGB
        
        # Test RGBA image (should remove alpha channel)
        rgba_image = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        processed_rgba = processor.process(rgba_image)
        
        assert processed_rgba.shape[0] == 3  # Should be RGB only


class TestVideoProcessor:
    """Test video preprocessing functionality."""
    
    @pytest.fixture
    def video_processor_config(self):
        """Video processor configuration."""
        return {
            'target_fps': 15,
            'max_frames': 32,
            'frame_size': (224, 224),
            'normalize': True,
            'temporal_sampling': 'uniform'
        }
    
    @pytest.fixture
    def sample_video_frames(self, video_processor_config):
        """Generate sample video frames."""
        num_frames = 60  # 4 seconds at 15 fps
        height, width = video_processor_config['frame_size']
        
        frames = []
        for i in range(num_frames):
            # Create frames with changing patterns
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            # Add some temporal consistency
            if i > 0:
                frame = (frame * 0.8 + frames[-1] * 0.2).astype(np.uint8)
            frames.append(frame)
        
        return np.stack(frames)
    
    def test_video_processor_initialization(self, video_processor_config):
        """Test video processor initialization."""
        processor = VideoProcessor(video_processor_config)
        
        assert processor.target_fps == video_processor_config['target_fps']
        assert processor.max_frames == video_processor_config['max_frames']
        assert processor.frame_size == video_processor_config['frame_size']
    
    @patch('cv2.VideoCapture')
    def test_video_loading(self, mock_video_capture, video_processor_config, sample_video_frames):
        """Test video file loading."""
        # Mock cv2.VideoCapture
        mock_cap = Mock()
        mock_cap.get.return_value = video_processor_config['target_fps']  # FPS
        mock_cap.read.side_effect = [(True, frame) for frame in sample_video_frames] + [(False, None)]
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_processor_config)
        
        video_path = "mock_video.mp4"
        frames = processor.load_video(video_path)
        
        mock_video_capture.assert_called_once_with(video_path)
        assert len(frames) == len(sample_video_frames)
    
    def test_temporal_sampling(self, video_processor_config, sample_video_frames):
        """Test temporal sampling of video frames."""
        processor = VideoProcessor(video_processor_config)
        
        max_frames = video_processor_config['max_frames']
        sampled_frames = processor.sample_frames(sample_video_frames, max_frames)
        
        assert len(sampled_frames) == max_frames
        assert sampled_frames[0].shape == sample_video_frames[0].shape
    
    def test_frame_preprocessing(self, video_processor_config, sample_video_frames):
        """Test individual frame preprocessing."""
        processor = VideoProcessor(video_processor_config)
        
        # Process a single frame
        single_frame = sample_video_frames[0]
        processed_frame = processor.preprocess_frame(single_frame)
        
        target_height, target_width = video_processor_config['frame_size']
        assert processed_frame.shape == (target_height, target_width, 3)
    
    def test_video_to_tensor(self, video_processor_config, sample_video_frames):
        """Test conversion of video to tensor."""
        processor = VideoProcessor(video_processor_config)
        
        processed = processor.process(sample_video_frames)
        
        assert isinstance(processed, torch.Tensor)
        # Should be [T, C, H, W] format
        max_frames = video_processor_config['max_frames']
        target_height, target_width = video_processor_config['frame_size']
        
        assert processed.shape[0] == max_frames  # Time
        assert processed.shape[1] == 3          # RGB channels
        assert processed.shape[2] == target_height  # Height
        assert processed.shape[3] == target_width   # Width
    
    def test_short_video_handling(self, video_processor_config):
        """Test handling of videos shorter than max_frames."""
        processor = VideoProcessor(video_processor_config)
        
        # Create short video (fewer frames than max_frames)
        short_frames = 10
        height, width = video_processor_config['frame_size']
        short_video = np.random.randint(0, 256, (short_frames, height, width, 3), dtype=np.uint8)
        
        processed = processor.process(short_video)
        
        max_frames = video_processor_config['max_frames']
        # Should be padded to max_frames
        assert processed.shape[0] == max_frames


class TestPreprocessingIntegration:
    """Test integration between different preprocessors."""
    
    def test_multimodal_preprocessing_pipeline(self):
        """Test processing pipeline with multiple modalities."""
        # Configure processors
        text_config = {
            'model_name': 'distilbert-base-uncased',
            'max_length': 32,
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt'
        }
        
        image_config = {
            'target_size': (224, 224),
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        
        audio_config = {
            'sample_rate': 16000,
            'n_mels': 80,
            'max_length': 100
        }
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Mock tokenizer
            mock_tok = Mock()
            mock_tok.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 32)),
                'attention_mask': torch.ones(1, 32)
            }
            mock_tokenizer.return_value = mock_tok
            
            # Initialize processors
            text_processor = TextProcessor(text_config)
            image_processor = ImageProcessor(image_config)
            audio_processor = AudioProcessor(audio_config)
            
            # Create sample data
            text_data = "This is a test sentence."
            image_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio
            
            # Process all modalities
            text_features = text_processor.process(text_data)
            image_features = image_processor.process(image_data)
            audio_features = audio_processor.process(audio_data)
            
            # Verify outputs
            assert isinstance(text_features, dict)
            assert 'input_ids' in text_features
            assert isinstance(image_features, torch.Tensor)
            assert isinstance(audio_features, np.ndarray)
            
            # Check shapes are reasonable
            assert text_features['input_ids'].shape[1] == 32
            assert image_features.shape == (3, 224, 224)
            assert audio_features.shape[0] == 80  # n_mels


if __name__ == "__main__":
    pytest.main([__file__])
