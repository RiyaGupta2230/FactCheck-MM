"""
Integration Tests for FactCheck-MM Training

Tests training loops using chunked_trainer.py with toy data.
Validates loss decrease, checkpoint saving/loading, and training convergence.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests import TEST_CONFIG
from tests.fixtures.mock_models import MockSarcasmModel, MockDataset, MockDataLoader
from sarcasm_detection.training.chunked_trainer import ChunkedTrainer, ChunkedTrainingConfig
from sarcasm_detection.training.train_multimodal import MultimodalSarcasmTrainer, MultimodalTrainingConfig


class TestChunkedTraining:
    """Test chunked training functionality."""
    
    @pytest.fixture
    def chunked_config(self):
        """Chunked training configuration for testing."""
        return ChunkedTrainingConfig(
            learning_rate=1e-3,
            batch_size=4,
            gradient_accumulation_steps=2,
            num_epochs=3,
            chunk_size=20,
            max_memory_gb=1.0,  # Small for testing
            save_every=2,
            eval_every=1,
            use_mixed_precision=False,  # Disable for CPU testing
            device="cpu"
        )
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for training."""
        samples = []
        for i in range(40):  # 2 chunks of 20 samples each
            sample = {
                'input_ids': torch.randint(0, 1000, (32,)),
                'attention_mask': torch.ones(32),
                'label': i % 2,  # Binary classification
                'text': f"Sample text number {i}"
            }
            samples.append(sample)
        
        return MockDataset(samples)
