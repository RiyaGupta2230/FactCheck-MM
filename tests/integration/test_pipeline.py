"""
Integration Tests for FactCheck-MM Pipeline

End-to-end tests of main.py pipeline with mock datasets.
Validates data loading, model initialization, and evaluation workflow.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests import TEST_CONFIG
from tests.fixtures.mock_models import MockSarcasmModel, MockTextProcessor
import main


class TestFullPipeline:
    """Test the complete FactCheck-MM pipeline."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Create minimal config files
            base_config = {
                "project_name": "FactCheck-MM-Test",
                "version": "1.0.0-test",
                "debug": True
            }
            
            with open(config_dir / "base_config.json", 'w') as f:
                json.dump(base_config, f)
            
            yield config_dir
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Create mock sarcasm dataset
            sarcasm_dir = data_dir / "test_sarcasm"
            sarcasm_dir.mkdir(exist_ok=True)
            
            mock_sarcasm_data = []
            for i in range(20):
                sample = {
                    "id": i,
                    "text": f"This is test sarcasm sample number {i}. {'So great!' if i % 2 else 'Perfect timing!'}",
                    "label": i % 2,  # Alternating labels
                    "source": "test_dataset"
                }
                mock_sarcasm_data.append(sample)
            
            with open(sarcasm_dir / "test_data.json", 'w') as f:
                json.dump(mock_sarcasm_data, f)
            
            yield data_dir
    
    @pytest.fixture
    def mock_model_components(self):
        """Mock all major model components."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock transformer model
            mock_transformer = Mock()
            mock_transformer.config.hidden_size = 768
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(2, 32, 768)
            mock_transformer.return_value = mock_output
            mock_model.return_value = mock_transformer
            
            # Mock tokenizer
            mock_tok = Mock()
            mock_tok.return_value = {
                'input_ids': torch.randint(0, 30522, (2, 32)),
                'attention_mask': torch.ones(2, 32)
            }
            mock_tokenizer.return_value = mock_tok
            
            yield {
                'model': mock_model,
                'tokenizer': mock_tokenizer,
                'transformer': mock_transformer,
                'tok': mock_tok
            }
    
    def test_pipeline_initialization(self, temp_config_dir, temp_data_dir, mock_model_components):
        """Test pipeline initialization with mock components."""
        
        # Mock sys.argv for main function
        test_args = [
            'main.py',
            '--config-dir', str(temp_config_dir),
            '--data-dir', str(temp_data_dir),
            '--task', 'sarcasm_detection',
            '--mode', 'test',
            '--device', 'cpu'
        ]
        
        with patch('sys.argv', test_args):
            with patch('main.initialize_pipeline') as mock_init:
                mock_pipeline = Mock()
                mock_init.return_value = mock_pipeline
                
                # This should not crash
                try:
                    # Import and initialize main components
                    from main import parse_arguments, setup_logging
                    
                    args = parse_arguments()
                    assert args.task == 'sarcasm_detection'
                    assert args.mode == 'test'
                    assert args.device == 'cpu'
                    
                except Exception as e:
                    pytest.fail(f"Pipeline initialization failed: {e}")
    
    @patch('main.SarcasmDetectionPipeline')
    def test_sarcasm_detection_pipeline(self, mock_pipeline_class, temp_data_dir, mock_model_components):
        """Test sarcasm detection pipeline execution."""
        
        # Mock the pipeline instance
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Mock pipeline methods
        mock_pipeline_instance.load_data.return_value = {
            'train': Mock(),
            'val': Mock(),
            'test': Mock()
        }
        
        mock_pipeline_instance.initialize_model.return_value = MockSarcasmModel()
        
        mock_pipeline_instance.train.return_value = {
            'loss': 0.5,
            'accuracy': 0.75,
            'f1': 0.7
        }
        
        mock_pipeline_instance.evaluate.return_value = {
            'test_accuracy': 0.8,
            'test_f1': 0.78,
            'test_precision': 0.79,
            'test_recall': 0.77
        }
        
        # Test pipeline execution
        from main import run_sarcasm_detection
        
        mock_args = Mock()
        mock_args.data_dir = str(temp_data_dir)
        mock_args.device = 'cpu'
        mock_args.batch_size = 4
        mock_args.epochs = 1
        mock_args.output_dir = tempfile.mkdtemp()
        
        results = run_sarcasm_detection(mock_args)
        
        # Verify pipeline was called correctly
        mock_pipeline_class.assert_called_once()
        mock_pipeline_instance.load_data.assert_called_once()
        mock_pipeline_instance.initialize_model.assert_called_once()
        
        # Check results structure
        assert isinstance(results, dict)
        if 'test_accuracy' in results:
            assert 0.0 <= results['test_accuracy'] <= 1.0
    
    def test_data_loading_integration(self, temp_data_dir, mock_model_components):
        """Test data loading integration."""
        
        with patch('sarcasm_detection.data.unified_loader.UnifiedSarcasmLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            # Create mock datasets
            train_samples = []
            val_samples = []
            test_samples = []
            
            for i in range(10):
                sample = {
                    'text': f"Test sample {i}",
                    'label': i % 2,
                    'input_ids': torch.randint(0, 1000, (32,)),
                    'attention_mask': torch.ones(32)
                }
                if i < 6:
                    train_samples.append(sample)
                elif i < 8:
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)
            
            mock_train_dataset = Mock()
            mock_train_dataset.__len__ = Mock(return_value=len(train_samples))
            mock_train_dataset.__getitem__ = Mock(side_effect=lambda i: train_samples[i])
            
            mock_val_dataset = Mock()
            mock_val_dataset.__len__ = Mock(return_value=len(val_samples))
            mock_val_dataset.__getitem__ = Mock(side_effect=lambda i: val_samples[i])
            
            mock_test_dataset = Mock()
            mock_test_dataset.__len__ = Mock(return_value=len(test_samples))
            mock_test_dataset.__getitem__ = Mock(side_effect=lambda i: test_samples[i])
            
            mock_loader.load_datasets.return_value = (mock_train_dataset, mock_val_dataset, mock_test_dataset)
            
            # Test data loading
            from main import load_sarcasm_data
            
            mock_args = Mock()
            mock_args.data_dir = str(temp_data_dir)
            mock_args.datasets = ['test_sarcasm']
            
            train_data, val_data, test_data = load_sarcasm_data(mock_args)
            
            assert len(train_data) == 6
            assert len(val_data) == 2
            assert len(test_data) == 2
    
    def test_model_training_integration(self, mock_model_components):
        """Test model training integration."""
        
        # Create mock training data
        batch_size = 4
        seq_length = 32
        
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'labels': torch.randint(0, 2, (batch_size,))
        }
        
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch] * 5))  # 5 batches
        mock_dataloader.__len__ = Mock(return_value=5)
        
        # Test training step
        from main import train_one_epoch
        
        model = MockSarcasmModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        epoch_loss = train_one_epoch(model, mock_dataloader, optimizer, device='cpu')
        
        assert isinstance(epoch_loss, float)
        assert epoch_loss > 0  # Should have some loss
    
    def test_model_evaluation_integration(self, mock_model_components):
        """Test model evaluation integration."""
        
        # Create mock evaluation data
        batch_size = 4
        seq_length = 32
        
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'labels': torch.randint(0, 2, (batch_size,))
        }
        
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch] * 3))  # 3 batches
        mock_dataloader.__len__ = Mock(return_value=3)
        
        # Test evaluation
        from main import evaluate_model
        
        model = MockSarcasmModel()
        model.eval()
        
        eval_results = evaluate_model(model, mock_dataloader, device='cpu')
        
        assert isinstance(eval_results, dict)
        assert 'accuracy' in eval_results
        assert 'loss' in eval_results
        assert 0.0 <= eval_results['accuracy'] <= 1.0
        assert eval_results['loss'] >= 0.0
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading."""
        
        model = MockSarcasmModel()
        
        # Create mock training state
        training_state = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': torch.optim.Adam(model.parameters()).state_dict(),
            'loss': 0.5,
            'accuracy': 0.8,
            'config': {'model_name': 'test_model'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Test saving
            from main import save_checkpoint
            save_checkpoint(training_state, str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Test loading
            from main import load_checkpoint
            loaded_state = load_checkpoint(str(checkpoint_path))
            
            assert loaded_state['epoch'] == 5
            assert loaded_state['loss'] == 0.5
            assert loaded_state['accuracy'] == 0.8
    
    def test_pipeline_error_handling(self, temp_data_dir):
        """Test pipeline error handling."""
        
        # Test with invalid task
        test_args = [
            'main.py',
            '--task', 'invalid_task',
            '--data-dir', str(temp_data_dir)
        ]
        
        with patch('sys.argv', test_args):
            with pytest.raises((ValueError, SystemExit)):
                from main import parse_arguments
                args = parse_arguments()
                main.main()
        
        # Test with missing data directory
        test_args = [
            'main.py',
            '--task', 'sarcasm_detection',
            '--data-dir', '/nonexistent/path'
        ]
        
        with patch('sys.argv', test_args):
            with pytest.raises((FileNotFoundError, SystemExit)):
                args = main.parse_arguments()
                # This should fail when trying to load data
    
    def test_memory_usage_monitoring(self, mock_model_components):
        """Test memory usage monitoring during pipeline execution."""
        
        from main import monitor_memory_usage
        
        # Mock process for memory monitoring
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            model = MockSarcasmModel()
            
            # Monitor memory during forward pass
            batch = {
                'input_ids': torch.randint(0, 1000, (4, 32)),
                'attention_mask': torch.ones(4, 32)
            }
            
            memory_before = monitor_memory_usage()
            _ = model(batch)
            memory_after = monitor_memory_usage()
            
            assert isinstance(memory_before, (int, float))
            assert isinstance(memory_after, (int, float))
            assert memory_before > 0
            assert memory_after > 0


class TestMultiModalPipeline:
    """Test multimodal pipeline functionality."""
    
    @pytest.fixture
    def multimodal_mock_data(self):
        """Create mock multimodal data."""
        batch_size = 2
        
        return {
            'text': {
                'input_ids': torch.randint(0, 1000, (batch_size, 32)),
                'attention_mask': torch.ones(batch_size, 32)
            },
            'audio': torch.randn(batch_size, 100, 128),
            'image': torch.randn(batch_size, 3, 224, 224),
            'labels': torch.randint(0, 2, (batch_size,))
        }
    
    @patch('sarcasm_detection.models.multimodal_sarcasm.MultimodalSarcasmModel')
    def test_multimodal_model_integration(self, mock_model_class, multimodal_mock_data, mock_model_components):
        """Test multimodal model integration."""
        
        # Mock multimodal model
        mock_model = Mock()
        mock_model.forward.return_value = torch.randn(2, 2)  # 2 samples, 2 classes
        mock_model_class.return_value = mock_model
        
        # Test multimodal forward pass
        from main import run_multimodal_inference
        
        results = run_multimodal_inference(mock_model, multimodal_mock_data)
        
        assert isinstance(results, torch.Tensor)
        assert results.shape == (2, 2)  # batch_size, num_classes
    
    def test_modality_missing_handling(self, multimodal_mock_data, mock_model_components):
        """Test handling of missing modalities."""
        
        # Test with missing audio
        incomplete_data = multimodal_mock_data.copy()
        del incomplete_data['audio']
        
        from main import handle_missing_modalities
        
        processed_data = handle_missing_modalities(incomplete_data, available_modalities=['text', 'image'])
        
        assert 'text' in processed_data
        assert 'image' in processed_data
        assert 'audio' not in processed_data
    
    def test_cross_modal_attention_integration(self, multimodal_mock_data, mock_model_components):
        """Test cross-modal attention in pipeline."""
        
        from main import test_cross_modal_attention
        
        # Mock attention weights
        attention_weights = {
            'text_to_audio': torch.rand(2, 32, 100),
            'text_to_image': torch.rand(2, 32, 196),
            'audio_to_text': torch.rand(2, 100, 32)
        }
        
        with patch('main.extract_attention_weights', return_value=attention_weights):
            results = test_cross_modal_attention(multimodal_mock_data)
            
            assert isinstance(results, dict)
            assert 'attention_weights' in results


class TestPipelineBenchmarking:
    """Test pipeline performance benchmarking."""
    
    def test_inference_speed_benchmark(self, mock_model_components):
        """Test inference speed measurement."""
        
        from main import benchmark_inference_speed
        
        model = MockSarcasmModel()
        model.eval()
        
        # Create test batch
        test_batch = {
            'input_ids': torch.randint(0, 1000, (8, 32)),
            'attention_mask': torch.ones(8, 32)
        }
        
        speed_results = benchmark_inference_speed(model, test_batch, num_runs=5)
        
        assert isinstance(speed_results, dict)
        assert 'avg_inference_time' in speed_results
        assert 'throughput_samples_per_sec' in speed_results
        assert speed_results['avg_inference_time'] > 0
        assert speed_results['throughput_samples_per_sec'] > 0
    
    def test_memory_benchmark(self, mock_model_components):
        """Test memory usage benchmarking."""
        
        from main import benchmark_memory_usage
        
        model = MockSarcasmModel()
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8]
        memory_results = {}
        
        for batch_size in batch_sizes:
            test_batch = {
                'input_ids': torch.randint(0, 1000, (batch_size, 32)),
                'attention_mask': torch.ones(batch_size, 32)
            }
            
            memory_usage = benchmark_memory_usage(model, test_batch)
            memory_results[batch_size] = memory_usage
        
        # Memory usage should generally increase with batch size
        assert memory_results[1] <= memory_results[8]
    
    def test_accuracy_benchmark(self, mock_model_components):
        """Test accuracy benchmarking across datasets."""
        
        from main import benchmark_accuracy
        
        model = MockSarcasmModel()
        
        # Mock different test datasets
        datasets = {
            'dataset_a': Mock(),
            'dataset_b': Mock(),
            'dataset_c': Mock()
        }
        
        for dataset_name, dataset in datasets.items():
            # Mock dataset to return consistent results
            mock_results = {
                'accuracy': np.random.uniform(0.7, 0.9),
                'f1': np.random.uniform(0.6, 0.8),
                'precision': np.random.uniform(0.7, 0.9),
                'recall': np.random.uniform(0.6, 0.8)
            }
            dataset.evaluate = Mock(return_value=mock_results)
        
        benchmark_results = benchmark_accuracy(model, datasets)
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) == len(datasets)
        
        for dataset_name in datasets.keys():
            assert dataset_name in benchmark_results
            assert 'accuracy' in benchmark_results[dataset_name]


if __name__ == "__main__":
    pytest.main([__file__])
