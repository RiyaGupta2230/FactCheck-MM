"""
Mock Models for FactCheck-MM Testing

Lightweight mock models and components that replace heavy model loading
during testing. These mocks simulate the interface and behavior of real
models without requiring GPU resources or large memory footprints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock
import random


class MockTextEncoder(nn.Module):
    """Mock text encoder that returns random embeddings."""
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768, max_length: int = 512):
        """
        Initialize mock text encoder.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            max_length: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Simple embedding layer for testing
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = Mock()
        self.config.hidden_size = hidden_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass returning mock embeddings."""
        batch_size, seq_len = input_ids.shape
        
        # Create mock output
        last_hidden_state = torch.randn(batch_size, seq_len, self.hidden_size)
        pooler_output = torch.randn(batch_size, self.hidden_size)
        
        # Mock transformer output
        output = Mock()
        output.last_hidden_state = last_hidden_state
        output.pooler_output = pooler_output
        output.hidden_states = [last_hidden_state]
        output.attentions = [torch.randn(batch_size, 12, seq_len, seq_len)]  # 12 heads
        
        return output


class MockAudioEncoder(nn.Module):
    """Mock audio encoder for multimodal testing."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512):
        """
        Initialize mock audio encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple linear layer for testing
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, audio_features: torch.Tensor, **kwargs):
        """Forward pass for audio features."""
        batch_size, time_steps, input_dim = audio_features.shape
        
        # Simple projection + pooling
        projected = self.projection(audio_features)
        
        # Global average pooling over time dimension
        pooled = projected.mean(dim=1)  # [batch_size, hidden_dim]
        
        return pooled


class MockImageEncoder(nn.Module):
    """Mock image encoder for multimodal testing."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 512):
        """
        Initialize mock image encoder.
        
        Args:
            input_channels: Input image channels (RGB = 3)
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Simple CNN for testing
        self.conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, hidden_dim)
        
    def forward(self, image: torch.Tensor, **kwargs):
        """Forward pass for image features."""
        batch_size = image.shape[0]
        
        # Simple convolution + pooling
        x = self.conv(image)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


class MockVideoEncoder(nn.Module):
    """Mock video encoder for multimodal testing."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 512):
        """
        Initialize mock video encoder.
        
        Args:
            input_channels: Input video channels (RGB = 3)
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # 3D convolution for temporal modeling
        self.conv3d = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, hidden_dim)
        
    def forward(self, video: torch.Tensor, **kwargs):
        """Forward pass for video features."""
        # Input: [batch_size, time, channels, height, width]
        # Rearrange to [batch_size, channels, time, height, width] for Conv3d
        video = video.permute(0, 2, 1, 3, 4)
        batch_size = video.shape[0]
        
        x = self.conv3d(video)
        x = self.pool3d(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


class MockSarcasmModel(nn.Module):
    """Mock sarcasm detection model for testing."""
    
    def __init__(self, num_classes: int = 2, hidden_dim: int = 768):
        """
        Initialize mock sarcasm model.
        
        Args:
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Mock components
        self.text_encoder = MockTextEncoder(hidden_size=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Add some mock attributes
        self.config = {
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
            'model_type': 'mock_sarcasm'
        }
        
    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor], **kwargs):
        """Forward pass for sarcasm detection."""
        if isinstance(inputs, dict):
            # Handle dictionary input
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                attention_mask = inputs.get('attention_mask')
                
                # Get text features
                text_output = self.text_encoder(input_ids, attention_mask)
                features = text_output.pooler_output
            else:
                # Handle multimodal input
                features = torch.randn(inputs[list(inputs.keys())[0]].shape[0], self.hidden_dim)
        else:
            # Handle tensor input
            batch_size = inputs.shape[0]
            features = torch.randn(batch_size, self.hidden_dim)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class MockMultimodalSarcasmModel(nn.Module):
    """Mock multimodal sarcasm detection model."""
    
    def __init__(self, modalities: List[str] = None, num_classes: int = 2, hidden_dim: int = 512):
        """
        Initialize mock multimodal sarcasm model.
        
        Args:
            modalities: List of modalities to support
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.modalities = modalities or ['text', 'audio', 'image', 'video']
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Mock encoders
        if 'text' in self.modalities:
            self.text_encoder = MockTextEncoder(hidden_size=hidden_dim)
        if 'audio' in self.modalities:
            self.audio_encoder = MockAudioEncoder(hidden_dim=hidden_dim)
        if 'image' in self.modalities:
            self.image_encoder = MockImageEncoder(hidden_dim=hidden_dim)
        if 'video' in self.modalities:
            self.video_encoder = MockVideoEncoder(hidden_dim=hidden_dim)
        
        # Fusion layer
        fusion_input_dim = len(self.modalities) * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Mock config
        self.config = {
            'modalities': self.modalities,
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
            'model_type': 'mock_multimodal_sarcasm'
        }
    
    def forward(self, inputs: Dict[str, Any], **kwargs):
        """Forward pass for multimodal sarcasm detection."""
        features = []
        
        # Process each modality
        if 'text' in inputs and 'text' in self.modalities:
            if isinstance(inputs['text'], dict):
                text_output = self.text_encoder(inputs['text']['input_ids'], 
                                              inputs['text'].get('attention_mask'))
                text_features = text_output.pooler_output
            else:
                # Handle direct text tensor input
                batch_size = inputs['text'].shape[0] if hasattr(inputs['text'], 'shape') else 1
                text_features = torch.randn(batch_size, self.hidden_dim)
            features.append(text_features)
        
        if 'audio' in inputs and 'audio' in self.modalities:
            if hasattr(inputs['audio'], 'shape'):
                audio_features = self.audio_encoder(inputs['audio'])
            else:
                batch_size = len(inputs.get('text', {}).get('input_ids', [1]))
                audio_features = torch.randn(batch_size, self.hidden_dim)
            features.append(audio_features)
        
        if 'image' in inputs and 'image' in self.modalities:
            if hasattr(inputs['image'], 'shape'):
                image_features = self.image_encoder(inputs['image'])
            else:
                batch_size = len(inputs.get('text', {}).get('input_ids', [1]))
                image_features = torch.randn(batch_size, self.hidden_dim)
            features.append(image_features)
        
        if 'video' in inputs and 'video' in self.modalities:
            if hasattr(inputs['video'], 'shape'):
                video_features = self.video_encoder(inputs['video'])
            else:
                batch_size = len(inputs.get('text', {}).get('input_ids', [1]))
                video_features = torch.randn(batch_size, self.hidden_dim)
            features.append(video_features)
        
        # Handle case where no features were extracted
        if not features:
            batch_size = 1
            features = [torch.randn(batch_size, self.hidden_dim)]
        
        # Ensure all features have the same batch size
        batch_size = features[0].shape[0]
        for i in range(len(features)):
            if features[i].shape[0] != batch_size:
                features[i] = features[i][:batch_size] if features[i].shape[0] > batch_size else \
                              torch.randn(batch_size, self.hidden_dim)
        
        # Fuse features
        if len(features) > 1:
            fused_features = torch.cat(features, dim=-1)
            fused_features = self.fusion(fused_features)
        else:
            fused_features = features[0]
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class MockFactVerificationModel(nn.Module):
    """Mock fact verification model."""
    
    def __init__(self, num_classes: int = 3, hidden_dim: int = 768):
        """
        Initialize mock fact verification model.
        
        Args:
            num_classes: Number of classes (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Mock components
        self.claim_encoder = MockTextEncoder(hidden_size=hidden_dim)
        self.evidence_encoder = MockTextEncoder(hidden_size=hidden_dim)
        
        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # claim + evidence + interaction
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.config = {
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
            'model_type': 'mock_fact_verification'
        }
    
    def forward(self, inputs: Dict[str, torch.Tensor], **kwargs):
        """Forward pass for fact verification."""
        claim_ids = inputs.get('claim_ids', inputs.get('input_ids'))
        evidence_ids = inputs.get('evidence_ids', inputs.get('input_ids'))
        
        batch_size = claim_ids.shape[0]
        
        # Encode claim and evidence
        claim_output = self.claim_encoder(claim_ids)
        evidence_output = self.evidence_encoder(evidence_ids)
        
        claim_features = claim_output.pooler_output
        evidence_features = evidence_output.pooler_output
        
        # Interaction features (element-wise product)
        interaction_features = claim_features * evidence_features
        
        # Concatenate all features
        combined_features = torch.cat([claim_features, evidence_features, interaction_features], dim=-1)
        
        # Final classification
        hidden = self.interaction(combined_features)
        logits = self.classifier(hidden)
        
        return logits


class MockParaphrasingModel(nn.Module):
    """Mock paraphrasing model."""
    
    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 768, max_length: int = 128):
        """
        Initialize mock paraphrasing model.
        
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension size
            max_length: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Mock encoder-decoder architecture
        self.encoder = MockTextEncoder(vocab_size, hidden_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'max_length': max_length,
            'model_type': 'mock_paraphrasing'
        }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                decoder_input_ids: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass for paraphrasing."""
        batch_size, seq_len = input_ids.shape
        
        # Encode input
        encoder_output = self.encoder(input_ids, attention_mask)
        encoder_hidden = encoder_output.pooler_output
        
        # If no decoder input, generate mock output
        if decoder_input_ids is None:
            # Generate mock paraphrase tokens
            output_length = min(seq_len + random.randint(-5, 10), self.max_length)
            output_ids = torch.randint(0, self.vocab_size, (batch_size, output_length))
            return output_ids
        
        # Decode with decoder input
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)
        decoder_output, _ = self.decoder(decoder_embeddings)
        logits = self.output_projection(decoder_output)
        
        return logits


class MockTextProcessor:
    """Mock text processor for testing."""
    
    def __init__(self, max_length: int = 128, vocab_size: int = 30522):
        """
        Initialize mock text processor.
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def process(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Mock text processing."""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        
        # Generate mock tokenized output
        input_ids = torch.randint(1, self.vocab_size - 1, (batch_size, self.max_length))
        attention_mask = torch.ones(batch_size, self.max_length)
        
        # Add padding tokens at random positions to make it more realistic
        for i in range(batch_size):
            # Random sequence length
            seq_len = random.randint(10, self.max_length - 10)
            attention_mask[i, seq_len:] = 0
            input_ids[i, seq_len:] = 0  # Padding token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def clean_text(self, text: str) -> str:
        """Mock text cleaning."""
        # Simple mock cleaning
        return text.strip().replace('\n', ' ').replace('  ', ' ')


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        """
        Initialize mock dataset.
        
        Args:
            samples: List of sample dictionaries
            transform: Optional transform function
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item by index."""
        sample = self.samples[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, dataset: MockDataset, batch_size: int = 4, shuffle: bool = False):
        """
        Initialize mock data loader.
        
        Args:
            dataset: Mock dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self._indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self._indices)
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self._indices[i:i + self.batch_size]
            batch_samples = [self.dataset[idx] for idx in batch_indices]
            
            # Collate batch
            batch = self._collate_fn(batch_samples)
            yield batch
    
    def _collate_fn(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch samples."""
        if not batch_samples:
            return {}
        
        # Get all keys from first sample
        keys = batch_samples[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch_samples]
            
            # Handle different data types
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            elif isinstance(values[0], str):
                collated[key] = values  # Keep as list of strings
            else:
                collated[key] = values  # Keep as list for other types
        
        return collated


class MockMetricsComputer:
    """Mock metrics computer for testing."""
    
    def __init__(self, task_name: str = "test"):
        """Initialize mock metrics computer."""
        self.task_name = task_name
    
    def compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Mock accuracy computation."""
        return float(np.mean(predictions == labels))
    
    def compute_f1_score(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Mock F1 score computation."""
        # Simple mock F1 calculation
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_classification_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                                     probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Mock classification metrics computation."""
        accuracy = self.compute_accuracy(predictions, labels)
        f1 = self.compute_f1_score(predictions, labels)
        
        # Mock additional metrics
        precision = f1 * 0.95  # Mock relationship
        recall = f1 * 1.05     # Mock relationship
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': min(precision, 1.0),
            'recall': min(recall, 1.0)
        }
        
        if probabilities is not None:
            # Mock confidence-based metrics
            confidence = np.max(probabilities, axis=1)
            metrics['mean_confidence'] = float(np.mean(confidence))
            metrics['calibration_error'] = abs(accuracy - np.mean(confidence))
        
        return metrics


# Factory functions for easy mock creation
def create_mock_sarcasm_model(num_classes: int = 2, multimodal: bool = False) -> nn.Module:
    """Create mock sarcasm detection model."""
    if multimodal:
        return MockMultimodalSarcasmModel(num_classes=num_classes)
    else:
        return MockSarcasmModel(num_classes=num_classes)


def create_mock_dataset(task: str = "sarcasm", num_samples: int = 10) -> MockDataset:
    """Create mock dataset for specified task."""
    if task == "sarcasm":
        samples = []
        for i in range(num_samples):
            sample = {
                'input_ids': torch.randint(0, 30522, (32,)),
                'attention_mask': torch.ones(32),
                'label': i % 2,
                'text': f"Mock sarcasm sample {i}"
            }
            samples.append(sample)
        return MockDataset(samples)
    
    elif task == "multimodal_sarcasm":
        samples = []
        for i in range(num_samples):
            sample = {
                'text': {
                    'input_ids': torch.randint(0, 30522, (32,)),
                    'attention_mask': torch.ones(32)
                },
                'audio': torch.randn(100, 128),
                'image': torch.randn(3, 224, 224),
                'label': i % 2
            }
            samples.append(sample)
        return MockDataset(samples)
    
    elif task == "fact_verification":
        samples = []
        for i in range(num_samples):
            sample = {
                'claim_ids': torch.randint(0, 30522, (32,)),
                'evidence_ids': torch.randint(0, 30522, (32,)),
                'attention_mask': torch.ones(32),
                'label': i % 3,  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
                'claim': f"Mock claim {i}",
                'evidence': f"Mock evidence {i}"
            }
            samples.append(sample)
        return MockDataset(samples)
    
    else:
        raise ValueError(f"Unknown task: {task}")


def create_mock_dataloader(dataset: MockDataset, batch_size: int = 4, shuffle: bool = False) -> MockDataLoader:
    """Create mock data loader."""
    return MockDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Pre-configured mock instances for common testing scenarios
MOCK_SARCASM_MODEL = create_mock_sarcasm_model()
MOCK_MULTIMODAL_SARCASM_MODEL = create_mock_sarcasm_model(multimodal=True)
MOCK_FACT_VERIFICATION_MODEL = MockFactVerificationModel()
MOCK_PARAPHRASING_MODEL = MockParaphrasingModel()

MOCK_TEXT_PROCESSOR = MockTextProcessor()
MOCK_METRICS_COMPUTER = MockMetricsComputer()

# Mock datasets
MOCK_SARCASM_DATASET = create_mock_dataset("sarcasm", 20)
MOCK_MULTIMODAL_DATASET = create_mock_dataset("multimodal_sarcasm", 15)
MOCK_FACT_DATASET = create_mock_dataset("fact_verification", 18)


if __name__ == "__main__":
    # Quick test of mock components
    print("Testing mock models...")
    
    # Test text model
    model = MOCK_SARCASM_MODEL
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 32)),
        'attention_mask': torch.ones(2, 32)
    }
    output = model(batch)
    print(f"Sarcasm model output shape: {output.shape}")
    
    # Test multimodal model
    mm_model = MOCK_MULTIMODAL_SARCASM_MODEL
    mm_batch = {
        'text': {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32)
        },
        'audio': torch.randn(2, 100, 128),
        'image': torch.randn(2, 3, 224, 224)
    }
    mm_output = mm_model(mm_batch)
    print(f"Multimodal sarcasm model output shape: {mm_output.shape}")
    
    # Test dataset
    dataset = MOCK_SARCASM_DATASET
    print(f"Mock dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    # Test data loader
    dataloader = create_mock_dataloader(dataset, batch_size=4)
    print(f"Mock dataloader batches: {len(dataloader)}")
    
    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch size: {batch['input_ids'].shape[0]}")
        break
    
    print("All mock components working correctly!")
