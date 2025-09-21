"""
FactCheck-MM Test Suite

Comprehensive testing framework for the FactCheck-MM multimodal fact-checking pipeline.
Includes unit tests for individual components and integration tests for end-to-end workflows.

Test Structure:
- unit/: Unit tests for models, preprocessing, and evaluation components
- integration/: Integration tests for pipeline workflows and training loops
- fixtures/: Mock data and models for testing
"""

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"

# Test configuration
TEST_CONFIG = {
    'use_gpu': False,  # Force CPU for tests to ensure reproducibility
    'random_seed': 42,
    'test_batch_size': 2,
    'test_sequence_length': 32,
    'mock_data_size': 10
}
