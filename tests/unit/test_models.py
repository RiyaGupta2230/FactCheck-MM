"""
Unit Tests for FactCheck-MM Models

Tests core model classes from shared/base_model.py and task-specific models.
Validates forward passes, output shapes, and loss computations with mock data.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.base_model import BaseMultimodalModel
from shared.multimodal_encoder import MultimodalEncoder
from shared.fusion_layers import CrossModalAttentionFusion
from sarcasm_detection.models.text_sarcasm_model import RobertaSarcasmModel
from sarcasm_detection.models.multimodal_sarcasm import MultimodalSarcasmModel
from tests import TEST_CONFIG


class TestBaseMultimodalModel:
    """Test the base multimodal model architecture."""
    
    @pytest.fixture
    def model_config(self):
        """Base model configuration for testing."""
        return {
            'modalities': ['text', 'audio', 'image'],
            'text_hidden_dim': 128,
            'audio_hidden_dim': 64,
            'image_hidden_dim': 64,
            'fusion_output_dim': 96,
            'num_classes': 2,
            'dropout_rate': 0.1
        }
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = TEST_CONFIG['test_batch_size']
        seq_len = TEST_CONFIG['test_sequence_length']
        
        return {
            'text': {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len)
            },
            'audio': torch.randn(batch_size, 100, 128),  # [batch, time, features]
            'image': torch.randn(batch_size, 3, 224, 224)  # [batch, channels, height, width]
        }
    
    def test_base_model_initialization(self, model_config):
        """Test base model initialization."""
        model = BaseMultimodalModel(model_config)
        
        assert model.config == model_config
        assert model.modalities == model_config['modalities']
        assert model.num_classes == model_config['num_classes']
        
        # Check that the model has required components
        assert hasattr(model, 'multimodal_encoder')
        assert hasattr(model, 'fusion_layer')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass_shape(self, model_config, sample_inputs):
        """Test forward pass output shapes."""
        model = BaseMultimodalModel(model_config)
        model.eval()
        
        with torch.no_grad():
            # Test with all modalities
            outputs = model(sample_inputs)
            
            batch_size = TEST_CONFIG['test_batch_size']
            num_classes = model_config['num_classes']
            
            assert outputs.shape == (batch_size, num_classes)
            assert not torch.isnan(outputs).any()
            assert torch.isfinite(outputs).all()
    
    def test_partial_modalities(self, model_config, sample_inputs):
        """Test model with missing modalities."""
        model = BaseMultimodalModel(model_config)
        model.eval()
        
        # Test with only text
        text_only_inputs = {'text': sample_inputs['text']}
        
        with torch.no_grad():
            outputs = model(text_only_inputs)
            
            batch_size = TEST_CONFIG['test_batch_size']
            num_classes = model_config['num_classes']
            
            assert outputs.shape == (batch_size, num_classes)
            assert not torch.isnan(outputs).any()
    
    def test_training_mode(self, model_config, sample_inputs):
        """Test model in training mode."""
        model = BaseMultimodalModel(model_config)
        model.train()
        
        outputs = model(sample_inputs)
        
        # Should be able to compute gradients
        loss = outputs.sum()
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultimodalEncoder:
    """Test the multimodal encoder component."""
    
    @pytest.fixture
    def encoder_config(self):
        """Encoder configuration for testing."""
        return {
            'text_encoder_name': 'distilbert-base-uncased',  # Smaller for tests
            'text_hidden_dim': 128,
            'audio_hidden_dim': 64,
            'image_hidden_dim': 64,
            'freeze_text_encoder': False
        }
    
    def test_encoder_initialization(self, encoder_config):
        """Test encoder initialization."""
        encoder = MultimodalEncoder(encoder_config)
        
        assert hasattr(encoder, 'text_encoder')
        assert hasattr(encoder, 'audio_encoder')
        assert hasattr(encoder, 'image_encoder')
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_text_encoding(self, mock_transformer, encoder_config):
        """Test text encoding functionality."""
        # Mock the transformer model
        mock_model = Mock()
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 32, 768)  # [batch, seq, hidden]
        mock_model.return_value = mock_output
        mock_transformer.return_value = mock_model
        
        encoder = MultimodalEncoder(encoder_config)
        
        text_input = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32)
        }
        
        text_features = encoder.encode_text(text_input)
        
        assert text_features.shape[0] == 2  # Batch size
        assert text_features.shape[-1] == encoder_config['text_hidden_dim']
    
    def test_audio_encoding(self, encoder_config):
        """Test audio encoding functionality."""
        encoder = MultimodalEncoder(encoder_config)
        
        audio_input = torch.randn(2, 100, 128)  # [batch, time, features]
        
        audio_features = encoder.encode_audio(audio_input)
        
        assert audio_features.shape[0] == 2  # Batch size
        assert audio_features.shape[-1] == encoder_config['audio_hidden_dim']
    
    def test_image_encoding(self, encoder_config):
        """Test image encoding functionality."""
        encoder = MultimodalEncoder(encoder_config)
        
        image_input = torch.randn(2, 3, 224, 224)  # [batch, channels, height, width]
        
        image_features = encoder.encode_image(image_input)
        
        assert image_features.shape[0] == 2  # Batch size
        assert image_features.shape[-1] == encoder_config['image_hidden_dim']


class TestCrossModalAttentionFusion:
    """Test the cross-modal attention fusion layer."""
    
    @pytest.fixture
    def fusion_config(self):
        """Fusion layer configuration."""
        return {
            'input_dims': {'text': 128, 'audio': 64, 'image': 64},
            'output_dim': 96,
            'num_heads': 4,
            'dropout_rate': 0.1
        }
    
    def test_fusion_initialization(self, fusion_config):
        """Test fusion layer initialization."""
        fusion = CrossModalAttentionFusion(fusion_config)
        
        assert fusion.output_dim == fusion_config['output_dim']
        assert fusion.num_heads == fusion_config['num_heads']
    
    def test_fusion_forward(self, fusion_config):
        """Test fusion forward pass."""
        fusion = CrossModalAttentionFusion(fusion_config)
        
        # Create sample modality features
        modality_features = {
            'text': torch.randn(2, 32, 128),    # [batch, seq, dim]
            'audio': torch.randn(2, 50, 64),    # [batch, time, dim]
            'image': torch.randn(2, 196, 64)    # [batch, patches, dim]
        }
        
        fused_features = fusion(modality_features)
        
        assert fused_features.shape == (2, fusion_config['output_dim'])
        assert not torch.isnan(fused_features).any()


class TestSarcasmModels:
    """Test sarcasm detection specific models."""
    
    @pytest.fixture
    def text_model_config(self):
        """Text sarcasm model configuration."""
        return {
            'model_name': 'distilbert-base-uncased',
            'num_classes': 2,
            'dropout_rate': 0.1,
            'freeze_encoder': False
        }
    
    @pytest.fixture
    def multimodal_sarcasm_config(self):
        """Multimodal sarcasm model configuration."""
        return {
            'modalities': ['text', 'audio', 'image'],
            'fusion_strategy': 'cross_modal_attention',
            'text_hidden_dim': 128,
            'audio_hidden_dim': 64,
            'image_hidden_dim': 64,
            'fusion_output_dim': 96,
            'num_classes': 2,
            'dropout_rate': 0.1
        }
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_text_sarcasm_model(self, mock_tokenizer, mock_model, text_model_config):
        """Test RoBERTa-based sarcasm model."""
        # Mock transformer components
        mock_transformer = Mock()
        mock_transformer.config.hidden_size = 768
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 32, 768)
        mock_transformer.return_value = mock_output
        mock_model.return_value = mock_transformer
        
        mock_tok = Mock()
        mock_tokenizer.return_value = mock_tok
        
        model = RobertaSarcasmModel(text_model_config)
        model.eval()
        
        # Test input
        batch_size = 2
        seq_len = 32
        
        inputs = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }
        
        with torch.no_grad():
            outputs = model(inputs)
            
            assert outputs.shape == (batch_size, text_model_config['num_classes'])
            assert not torch.isnan(outputs).any()
    
    def test_multimodal_sarcasm_model(self, multimodal_sarcasm_config):
        """Test multimodal sarcasm detection model."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock transformer
            mock_transformer = Mock()
            mock_transformer.config.hidden_size = 768
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(2, 32, 768)
            mock_transformer.return_value = mock_output
            mock_model.return_value = mock_transformer
            
            mock_tok = Mock()
            mock_tokenizer.return_value = mock_tok
            
            model = MultimodalSarcasmModel(multimodal_sarcasm_config)
            model.eval()
            
            # Test inputs
            batch_size = 2
            seq_len = 32
            
            inputs = {
                'text': {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                    'attention_mask': torch.ones(batch_size, seq_len)
                },
                'audio': torch.randn(batch_size, 100, 128),
                'image': torch.randn(batch_size, 3, 224, 224)
            }
            
            with torch.no_grad():
                outputs = model(inputs)
                
                assert outputs.shape == (batch_size, multimodal_sarcasm_config['num_classes'])
                assert not torch.isnan(outputs).any()
    
    def test_loss_computation(self, multimodal_sarcasm_config):
        """Test loss computation for sarcasm models."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock transformer
            mock_transformer = Mock()
            mock_transformer.config.hidden_size = 768
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(2, 32, 768)
            mock_transformer.return_value = mock_output
            mock_model.return_value = mock_transformer
            
            mock_tok = Mock()
            mock_tokenizer.return_value = mock_tok
            
            model = MultimodalSarcasmModel(multimodal_sarcasm_config)
            
            # Sample inputs and labels
            batch_size = 2
            seq_len = 32
            
            inputs = {
                'text': {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                    'attention_mask': torch.ones(batch_size, seq_len)
                }
            }
            
            labels = torch.randint(0, 2, (batch_size,))
            
            # Compute loss
            outputs = model(inputs)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            assert loss.item() > 0
            assert torch.isfinite(loss)
            
            # Test backward pass
            loss.backward()
            
            # Check gradients
            has_gradients = False
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break
            
            assert has_gradients


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_model_parameter_count(self):
        """Test parameter counting utility."""
        model_config = {
            'modalities': ['text'],
            'text_hidden_dim': 64,
            'fusion_output_dim': 32,
            'num_classes': 2
        }
        
        with patch('transformers.AutoModel.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock small transformer
            mock_transformer = Mock()
            mock_transformer.config.hidden_size = 64
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(1, 10, 64)
            mock_transformer.return_value = mock_output
            mock_model.return_value = mock_transformer
            
            mock_tok = Mock()
            mock_tokenizer.return_value = mock_tok
            
            model = BaseMultimodalModel(model_config)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert total_params > 0
            assert trainable_params <= total_params
    
    def test_model_device_placement(self):
        """Test model device placement."""
        model_config = {
            'modalities': ['text'],
            'text_hidden_dim': 64,
            'fusion_output_dim': 32,
            'num_classes': 2
        }
        
        with patch('transformers.AutoModel.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_transformer = Mock()
            mock_transformer.config.hidden_size = 64
            mock_model.return_value = mock_transformer
            
            mock_tok = Mock()
            mock_tokenizer.return_value = mock_tok
            
            model = BaseMultimodalModel(model_config)
            
            # Test CPU placement
            device = torch.device('cpu')
            model.to(device)
            
            for param in model.parameters():
                assert param.device == device


if __name__ == "__main__":
    pytest.main([__file__])
