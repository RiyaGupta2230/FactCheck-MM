"""
Unit Tests for FactCheck-MM Evaluation

Tests metrics computation from shared/utils/metrics.py and 
sarcasm_detection/utils/sarcasm_metrics.py with synthetic data.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils.metrics import MetricsComputer
from sarcasm_detection.utils.sarcasm_metrics import (
    SarcasmMetrics, SarcasmEvaluator, IronyDetectionMetrics, 
    ClassImbalanceMetrics, SarcasmMetricsConfig
)
from tests import TEST_CONFIG


class TestBasicMetrics:
    """Test basic classification metrics."""
    
    @pytest.fixture
    def sample_predictions_balanced(self):
        """Balanced sample predictions and labels."""
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        probabilities = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7],
            [0.4, 0.6], [0.8, 0.2], [0.9, 0.1], [0.2, 0.8], [0.6, 0.4]
        ])
        return predictions, labels, probabilities
    
    @pytest.fixture
    def sample_predictions_imbalanced(self):
        """Imbalanced sample predictions and labels."""
        # 80% class 0, 20% class 1
        predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
        probabilities = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.4, 0.6],
            [0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]
        ])
        return predictions, labels, probabilities
    
    def test_metrics_computer_initialization(self):
        """Test MetricsComputer initialization."""
        computer = MetricsComputer("test_task")
        
        assert computer.task_name == "test_task"
        assert hasattr(computer, 'logger')
    
    def test_accuracy_computation(self, sample_predictions_balanced):
        """Test accuracy computation."""
        predictions, labels, _ = sample_predictions_balanced
        
        computer = MetricsComputer("test")
        accuracy = computer.compute_accuracy(predictions, labels)
        
        # Manual calculation
        expected_accuracy = np.sum(predictions == labels) / len(labels)
        
        assert abs(accuracy - expected_accuracy) < 1e-6
        assert 0.0 <= accuracy <= 1.0
    
    def test_precision_recall_f1(self, sample_predictions_balanced):
        """Test precision, recall, and F1-score computation."""
        predictions, labels, _ = sample_predictions_balanced
        
        computer = MetricsComputer("test")
        metrics = computer.compute_classification_metrics(predictions, labels)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check ranges
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1'] <= 1.0
    
    def test_confusion_matrix(self, sample_predictions_balanced):
        """Test confusion matrix computation."""
        predictions, labels, _ = sample_predictions_balanced
        
        computer = MetricsComputer("test")
        cm = computer.compute_confusion_matrix(predictions, labels)
        
        # Should be 2x2 for binary classification
        assert cm.shape == (2, 2)
        
        # Sum should equal total samples
        assert np.sum(cm) == len(predictions)
        
        # All values should be non-negative
        assert np.all(cm >= 0)
    
    def test_auc_computation(self, sample_predictions_balanced):
        """Test AUC computation with probabilities."""
        predictions, labels, probabilities = sample_predictions_balanced
        
        computer = MetricsComputer("test")
        
        # Use probabilities for positive class
        prob_positive = probabilities[:, 1]
        auc = computer.compute_auc(labels, prob_positive)
        
        assert 0.0 <= auc <= 1.0


class TestSarcasmSpecificMetrics:
    """Test sarcasm-specific metrics."""
    
    @pytest.fixture
    def sarcasm_config(self):
        """Sarcasm metrics configuration."""
        return SarcasmMetricsConfig(
            compute_standard_metrics=True,
            compute_class_specific_metrics=True,
            compute_confidence_metrics=True,
            compute_irony_metrics=True,
            confidence_thresholds=[0.6, 0.8]
        )
    
    @pytest.fixture
    def sarcasm_sample_data(self):
        """Sample data for sarcasm testing."""
        # Create realistic sarcasm detection scenario
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0])
        probabilities = np.array([
            [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4],
            [0.7, 0.3], [0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
            [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]
        ])
        
        return predictions, labels, probabilities
    
    def test_sarcasm_metrics_initialization(self, sarcasm_config):
        """Test SarcasmMetrics initialization."""
        metrics = SarcasmMetrics("sarcasm_test", num_classes=2, config=sarcasm_config)
        
        assert metrics.task_name == "sarcasm_test"
        assert metrics.num_classes == 2
        assert metrics.config == sarcasm_config
    
    def test_sarcasm_detection_rate(self, sarcasm_config, sarcasm_sample_data):
        """Test sarcasm detection rate computation."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        metrics = SarcasmMetrics("test", config=sarcasm_config)
        results = metrics.compute_classification_metrics(predictions, labels, probabilities)
        
        assert 'test_sarcasm_detection_rate' in results
        
        # Manual calculation
        sarcasm_mask = (labels == 1)
        if np.sum(sarcasm_mask) > 0:
            expected_rate = np.sum(predictions[sarcasm_mask] == 1) / np.sum(sarcasm_mask)
            assert abs(results['test_sarcasm_detection_rate'] - expected_rate) < 1e-6
    
    def test_confidence_metrics(self, sarcasm_config, sarcasm_sample_data):
        """Test confidence-based metrics."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        metrics = SarcasmMetrics("test", config=sarcasm_config)
        results = metrics.compute_classification_metrics(predictions, labels, probabilities)
        
        # Check confidence statistics
        assert 'test_mean_confidence' in results
        assert 'test_std_confidence' in results
        assert 'test_min_confidence' in results
        assert 'test_max_confidence' in results
        
        # Check confidence thresholds
        for threshold in sarcasm_config.confidence_thresholds:
            assert f'test_accuracy_conf_{threshold}' in results or f'test_coverage_conf_{threshold}' in results
    
    def test_calibration_metrics(self, sarcasm_config, sarcasm_sample_data):
        """Test calibration metrics."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        metrics = SarcasmMetrics("test", config=sarcasm_config)
        results = metrics.compute_classification_metrics(predictions, labels, probabilities)
        
        # Check calibration metrics
        assert 'test_expected_calibration_error' in results
        assert 'test_brier_score' in results
        
        # Check ranges
        assert 0.0 <= results['test_expected_calibration_error'] <= 1.0
        assert 0.0 <= results['test_brier_score'] <= 2.0  # Brier score range
    
    def test_class_specific_metrics(self, sarcasm_config, sarcasm_sample_data):
        """Test class-specific metrics."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        metrics = SarcasmMetrics("test", config=sarcasm_config)
        results = metrics.compute_classification_metrics(predictions, labels, probabilities)
        
        # Check class-specific metrics exist
        assert 'test_precision_non_sarcastic' in results
        assert 'test_precision_sarcastic' in results
        assert 'test_recall_non_sarcastic' in results
        assert 'test_recall_sarcastic' in results
        assert 'test_f1_non_sarcastic' in results
        assert 'test_f1_sarcastic' in results
    
    def test_irony_specific_metrics(self, sarcasm_sample_data):
        """Test irony-specific metrics."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        irony_metrics = IronyDetectionMetrics()
        results = irony_metrics.compute_irony_specific_metrics(predictions, labels)
        
        # Check irony-specific metrics
        if 'irony_sensitivity' in results:
            assert 0.0 <= results['irony_sensitivity'] <= 1.0
        if 'non_irony_specificity' in results:
            assert 0.0 <= results['non_irony_specificity'] <= 1.0
    
    def test_text_feature_analysis(self, sarcasm_sample_data):
        """Test text feature-based analysis."""
        predictions, labels, probabilities = sarcasm_sample_data
        
        # Mock text features
        text_features = {
            'length': np.random.randint(10, 100, len(labels)),
            'sentiment': np.random.uniform(-1, 1, len(labels))
        }
        
        irony_metrics = IronyDetectionMetrics()
        results = irony_metrics.compute_irony_specific_metrics(
            predictions, labels, text_features=text_features
        )
        
        # Should include text-based analysis
        assert len(results) >= 2  # Should have some text-based metrics


class TestClassImbalanceMetrics:
    """Test class imbalance handling metrics."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Highly imbalanced dataset."""
        # 90% class 0, 10% class 1
        n_samples = 100
        n_minority = 10
        
        labels = np.concatenate([
            np.zeros(n_samples - n_minority),
            np.ones(n_minority)
        ])
        
        # Simulate poor performance on minority class
        predictions = labels.copy()
        # Flip some minority predictions to majority
        minority_indices = np.where(labels == 1)[0]
        flip_indices = np.random.choice(minority_indices, size=5, replace=False)
        predictions[flip_indices] = 0
        
        # Generate probabilities
        probabilities = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if predictions[i] == 1:
                probabilities[i] = [0.3, 0.7]
            else:
                probabilities[i] = [0.7, 0.3]
        
        return predictions, labels, probabilities
    
    def test_imbalance_analysis(self, imbalanced_data):
        """Test class imbalance analysis."""
        predictions, labels, probabilities = imbalanced_data
        
        imbalance_metrics = ClassImbalanceMetrics()
        analysis = imbalance_metrics.analyze_class_imbalance(labels, predictions, probabilities)
        
        # Check basic imbalance metrics
        assert 'imbalance_ratio' in analysis
        assert 'imbalance_severity' in analysis
        assert 'class_non_sarcastic_count' in analysis
        assert 'class_sarcastic_count' in analysis
        
        # Should detect severe imbalance
        assert analysis['imbalance_ratio'] > 5  # 90:10 ratio
        assert analysis['imbalance_severity'] in ['severe_imbalance', 'moderate_imbalance']
    
    def test_imbalance_aware_metrics(self, imbalanced_data):
        """Test imbalance-aware metrics computation."""
        predictions, labels, probabilities = imbalanced_data
        
        imbalance_metrics = ClassImbalanceMetrics()
        metrics = imbalance_metrics.compute_imbalance_aware_metrics(labels, predictions, probabilities)
        
        # Check imbalance-aware metrics
        assert 'balanced_accuracy' in metrics
        assert 'matthews_correlation' in metrics
        
        if 'area_under_roc' in metrics:
            assert 0.0 <= metrics['area_under_roc'] <= 1.0
        if 'average_precision' in metrics:
            assert 0.0 <= metrics['average_precision'] <= 1.0
    
    def test_handling_suggestions(self, imbalanced_data):
        """Test imbalance handling suggestions."""
        predictions, labels, probabilities = imbalanced_data
        
        imbalance_metrics = ClassImbalanceMetrics()
        analysis = imbalance_metrics.analyze_class_imbalance(labels, predictions, probabilities)
        suggestions = imbalance_metrics.suggest_imbalance_handling_strategies(analysis)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest appropriate strategies for severe imbalance
        suggestion_text = ' '.join(suggestions).lower()
        assert any(keyword in suggestion_text for keyword in [
            'weight', 'oversampling', 'focal', 'smote', 'ensemble'
        ])


class TestSarcasmEvaluator:
    """Test comprehensive sarcasm evaluator."""
    
    @pytest.fixture
    def evaluator_config(self):
        """Sarcasm evaluator configuration."""
        return SarcasmMetricsConfig(
            compute_standard_metrics=True,
            compute_class_specific_metrics=True,
            compute_confidence_metrics=True,
            compute_irony_metrics=True
        )
    
    @pytest.fixture
    def comprehensive_data(self):
        """Comprehensive evaluation data."""
        n_samples = 50
        
        predictions = np.random.randint(0, 2, n_samples)
        labels = np.random.randint(0, 2, n_samples)
        
        probabilities = np.random.rand(n_samples, 2)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Mock modality availability
        modality_availability = {
            'text': np.ones(n_samples, dtype=bool),
            'audio': np.random.rand(n_samples) > 0.3,  # 70% have audio
            'image': np.random.rand(n_samples) > 0.5   # 50% have image
        }
        
        # Mock modality predictions
        modality_predictions = {
            'text': np.random.randint(0, 2, n_samples),
            'audio': np.random.randint(0, 2, n_samples),
            'image': np.random.randint(0, 2, n_samples)
        }
        
        return predictions, labels, probabilities, modality_availability, modality_predictions
    
    def test_evaluator_initialization(self, evaluator_config):
        """Test sarcasm evaluator initialization."""
        evaluator = SarcasmEvaluator(evaluator_config)
        
        assert evaluator.config == evaluator_config
        assert hasattr(evaluator, 'basic_metrics')
        assert hasattr(evaluator, 'irony_metrics')
        assert hasattr(evaluator, 'multimodal_metrics')
        assert hasattr(evaluator, 'imbalance_metrics')
    
    def test_comprehensive_evaluation(self, evaluator_config, comprehensive_data):
        """Test comprehensive evaluation."""
        predictions, labels, probabilities, modality_availability, modality_predictions = comprehensive_data
        
        evaluator = SarcasmEvaluator(evaluator_config)
        
        results = evaluator.evaluate_comprehensive(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            modality_availability=modality_availability,
            modality_predictions=modality_predictions,
            mode="test"
        )
        
        # Check that all major components are present
        assert 'test_accuracy' in results
        assert 'test_precision' in results
        assert 'test_recall' in results
        assert 'test_f1' in results
        
        # Check component-specific results
        assert 'test_imbalance_analysis' in results
        assert 'test_recommendations' in results
        
        if 'test_multimodal_metrics' in results:
            assert isinstance(results['test_multimodal_metrics'], dict)
    
    def test_evaluation_report_generation(self, evaluator_config, comprehensive_data):
        """Test evaluation report generation."""
        predictions, labels, probabilities, modality_availability, modality_predictions = comprehensive_data
        
        evaluator = SarcasmEvaluator(evaluator_config)
        
        results = evaluator.evaluate_comprehensive(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            mode="test"
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert 'EVALUATION REPORT' in report
        assert 'ACCURACY' in report or 'accuracy' in report


class TestMetricsEdgeCases:
    """Test edge cases in metrics computation."""
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        predictions = np.array([])
        labels = np.array([])
        
        computer = MetricsComputer("test")
        
        # Should handle gracefully without crashing
        try:
            metrics = computer.compute_classification_metrics(predictions, labels)
            # If it doesn't crash, that's good
            assert isinstance(metrics, dict)
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise these errors for empty input
            pass
    
    def test_single_class_predictions(self):
        """Test handling when all predictions are same class."""
        predictions = np.ones(10)
        labels = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
        
        computer = MetricsComputer("test")
        metrics = computer.compute_classification_metrics(predictions, labels)
        
        # Should compute metrics without error
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
    
    def test_perfect_predictions(self):
        """Test handling of perfect predictions."""
        predictions = np.array([1, 0, 1, 0, 1, 0])
        labels = np.array([1, 0, 1, 0, 1, 0])
        
        computer = MetricsComputer("test")
        metrics = computer.compute_classification_metrics(predictions, labels)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_worst_predictions(self):
        """Test handling of worst possible predictions."""
        predictions = np.array([1, 0, 1, 0, 1, 0])
        labels = np.array([0, 1, 0, 1, 0, 1])  # Opposite of predictions
        
        computer = MetricsComputer("test")
        metrics = computer.compute_classification_metrics(predictions, labels)
        
        assert metrics['accuracy'] == 0.0
        # Precision, recall, F1 might be 0 or undefined
    
    def test_nan_handling(self):
        """Test handling of NaN values in predictions."""
        predictions = np.array([1, 0, 1, 0, 1])
        labels = np.array([1, 0, np.nan, 0, 1])  # Contains NaN
        
        computer = MetricsComputer("test")
        
        # Should either handle NaN gracefully or raise appropriate error
        try:
            metrics = computer.compute_classification_metrics(predictions, labels)
            # If successful, check that NaN didn't propagate
            for value in metrics.values():
                if isinstance(value, (int, float)):
                    assert not np.isnan(value)
        except (ValueError, TypeError):
            # Acceptable to raise error for NaN input
            pass


if __name__ == "__main__":
    pytest.main([__file__])
