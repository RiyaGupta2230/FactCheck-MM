# sarcasm_detection/evaluation/evaluator.py
"""
Comprehensive Sarcasm Detection Evaluator
Standard metrics computation (accuracy, F1, AUC) across all datasets.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from collections import defaultdict
import time

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

from shared.utils import get_logger, MetricsComputer
from shared.datasets import create_hardware_aware_dataloader, MultimodalCollator
from ..models import TextSarcasmModel, MultimodalSarcasmModel, EnsembleSarcasmModel
from ..utils import SarcasmMetrics


@dataclass
class EvaluationConfig:
    """Configuration for sarcasm detection evaluation."""
    
    # Evaluation settings
    batch_size: int = 16
    device: str = "auto"
    use_mixed_precision: bool = False
    
    # Metrics to compute
    compute_detailed_metrics: bool = True
    compute_per_class_metrics: bool = True
    compute_confidence_metrics: bool = True
    compute_calibration_metrics: bool = True
    
    # Dataset-specific evaluation
    evaluate_per_dataset: bool = True
    cross_dataset_evaluation: bool = True
    
    # Output settings
    save_predictions: bool = True
    save_detailed_results: bool = True
    save_visualizations: bool = True
    
    # Analysis settings
    analyze_failure_cases: bool = True
    confidence_threshold: float = 0.5
    top_k_errors: int = 100


class SarcasmEvaluator:
    """Base evaluator for sarcasm detection models."""
    
    def __init__(
        self,
        model: Union[TextSarcasmModel, MultimodalSarcasmModel, EnsembleSarcasmModel],
        config: Union[EvaluationConfig, Dict[str, Any]] = None
    ):
        """
        Initialize sarcasm evaluator.
        
        Args:
            model: Model to evaluate
            config: Evaluation configuration
        """
        if isinstance(config, dict):
            config = EvaluationConfig(**config)
        elif config is None:
            config = EvaluationConfig()
        
        self.model = model
        self.config = config
        
        self.logger = get_logger("SarcasmEvaluator")
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics computer
        self.metrics_computer = SarcasmMetrics("sarcasm_detection", num_classes=2)
        
        self.logger.info(f"Initialized sarcasm evaluator on {self.device}")
    
    def evaluate_dataset(
        self,
        dataset,
        dataset_name: str = "unknown",
        split_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single dataset.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset
            split_name: Split name (test, val, etc.)
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"Evaluating on {dataset_name} ({split_name}): {len(dataset)} samples")
        
        # Setup data loader
        from shared.preprocessing import TextProcessor
        text_processor = TextProcessor(max_length=512)
        
        collate_fn = MultimodalCollator(
            text_processor=text_processor.tokenizer,
            max_length=512
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Evaluation loop
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_sample_ids = []
        prediction_times = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                batch_start_time = time.time()
                
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if isinstance(self.model, MultimodalSarcasmModel):
                    logits = self._multimodal_forward(batch)
                elif isinstance(self.model, EnsembleSarcasmModel):
                    logits = self._ensemble_forward(batch)
                else:
                    # Text-only model
                    text_inputs = {k: v for k, v in batch.items() 
                                 if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                    logits = self.model(**text_inputs)
                
                # Compute loss
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(logits, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
                
                # Store sample IDs if available
                if 'id' in batch:
                    all_sample_ids.extend(batch['id'])
                else:
                    batch_size = len(predictions)
                    start_idx = len(all_sample_ids)
                    all_sample_ids.extend([f"{dataset_name}_{start_idx + i}" for i in range(batch_size)])
                
                # Track prediction time
                batch_time = time.time() - batch_start_time
                prediction_times.append(batch_time / len(predictions))  # Per sample
        
        # Compute comprehensive metrics
        results = self._compute_comprehensive_metrics(
            predictions=all_predictions,
            probabilities=all_probabilities,
            labels=all_labels,
            dataset_name=dataset_name,
            split_name=split_name
        )
        
        # Add loss and timing information
        results['metrics']['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        results['metrics']['avg_prediction_time_ms'] = np.mean(prediction_times) * 1000
        results['metrics']['total_samples'] = len(all_predictions)
        
        # Store predictions if requested
        if self.config.save_predictions:
            results['predictions'] = {
                'sample_ids': all_sample_ids,
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'labels': all_labels
            }
        
        return results
    
    def _multimodal_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for multimodal model."""
        text_input = batch.get('text')
        audio_input = batch.get('audio_features')
        image_input = batch.get('image_features')
        video_input = batch.get('video_features')
        
        return self.model(
            text=text_input,
            audio=audio_input,
            image=image_input,
            video=video_input
        )
    
    def _ensemble_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for ensemble model."""
        # Extract inputs for ensemble
        inputs = {}
        
        # Text inputs
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if key in batch:
                inputs[key] = batch[key]
        
        # Multimodal inputs
        if 'audio_features' in batch:
            inputs['audio'] = batch['audio_features']
        if 'image_features' in batch:
            inputs['image'] = batch['image_features']
        if 'video_features' in batch:
            inputs['video'] = batch['video_features']
        
        return self.model(**inputs)
    
    def _compute_comprehensive_metrics(
        self,
        predictions: List[int],
        probabilities: List[List[float]],
        labels: List[int],
        dataset_name: str,
        split_name: str
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        results = {
            'dataset': dataset_name,
            'split': split_name,
            'metrics': {},
            'detailed_metrics': {},
            'confusion_matrix': None,
            'classification_report': None
        }
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        labels = np.array(labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        # Store basic metrics
        results['metrics'].update({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro)
        })
        
        # Per-class metrics
        if self.config.compute_per_class_metrics:
            class_names = ['non_sarcastic', 'sarcastic']
            for i, class_name in enumerate(class_names):
                if i < len(precision_per_class):
                    results['detailed_metrics'][f'precision_{class_name}'] = float(precision_per_class[i])
                    results['detailed_metrics'][f'recall_{class_name}'] = float(recall_per_class[i])
                    results['detailed_metrics'][f'f1_{class_name}'] = float(f1_per_class[i])
                    results['detailed_metrics'][f'support_{class_name}'] = int(support[i])
        
        # AUC metrics
        try:
            # Binary AUC
            if len(np.unique(labels)) == 2:
                sarcastic_probs = probabilities[:, 1]  # Probability of sarcastic class
                auc = roc_auc_score(labels, sarcastic_probs)
                results['metrics']['auc'] = float(auc)
                
                # Average precision
                avg_precision = average_precision_score(labels, sarcastic_probs)
                results['metrics']['average_precision'] = float(avg_precision)
        except Exception as e:
            self.logger.debug(f"Could not compute AUC metrics: {e}")
        
        # Confidence metrics
        if self.config.compute_confidence_metrics:
            confidence_scores = np.max(probabilities, axis=1)
            results['detailed_metrics'].update({
                'mean_confidence': float(np.mean(confidence_scores)),
                'std_confidence': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores))
            })
            
            # Confidence-based accuracy
            high_conf_mask = confidence_scores > self.config.confidence_threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = accuracy_score(
                    labels[high_conf_mask], 
                    predictions[high_conf_mask]
                )
                results['detailed_metrics']['high_confidence_accuracy'] = float(high_conf_accuracy)
                results['detailed_metrics']['high_confidence_samples'] = int(np.sum(high_conf_mask))
        
        # Calibration metrics
        if self.config.compute_calibration_metrics:
            calibration_results = self._compute_calibration_metrics(probabilities, predictions, labels)
            results['detailed_metrics'].update(calibration_results)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_names = ['non_sarcastic', 'sarcastic']
        results['classification_report'] = classification_report(
            labels, predictions, 
            target_names=class_names,
            digits=4,
            output_dict=True
        )
        
        return results
    
    def _compute_calibration_metrics(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute model calibration metrics."""
        
        # Expected Calibration Error (ECE)
        confidence_scores = np.max(probabilities, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        # Brier Score
        # Convert labels to one-hot for binary case
        labels_one_hot = np.eye(2)[labels]
        brier_score = np.mean(np.sum((probabilities - labels_one_hot) ** 2, axis=1))
        
        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'brier_score': float(brier_score)
        }
    
    def evaluate_multiple_datasets(
        self,
        datasets: Dict[str, Any],
        split_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate model on multiple datasets.
        
        Args:
            datasets: Dictionary of {dataset_name: dataset} pairs
            split_name: Split name
            
        Returns:
            Combined evaluation results
        """
        self.logger.info(f"Evaluating on {len(datasets)} datasets")
        
        all_results = {}
        aggregate_metrics = defaultdict(list)
        
        for dataset_name, dataset in datasets.items():
            # Evaluate on single dataset
            dataset_results = self.evaluate_dataset(dataset, dataset_name, split_name)
            all_results[dataset_name] = dataset_results
            
            # Collect metrics for aggregation
            for metric_name, metric_value in dataset_results['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    aggregate_metrics[metric_name].append(metric_value)
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name, values in aggregate_metrics.items():
            aggregate_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'datasets': len(values)
            }
        
        return {
            'individual_results': all_results,
            'aggregate_metrics': aggregate_stats,
            'summary': {
                'total_datasets': len(datasets),
                'total_samples': sum(len(dataset) for dataset in datasets.values()),
                'evaluation_split': split_name
            }
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        experiment_name: str = "sarcasm_evaluation"
    ):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            experiment_name: Experiment name
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions if available
        if 'predictions' in results:
            predictions_file = output_dir / f"{experiment_name}_predictions.csv"
            predictions_df = pd.DataFrame(results['predictions'])
            predictions_df.to_csv(predictions_file, index=False)
        
        # Save summary CSV
        if 'individual_results' in results:
            summary_data = []
            for dataset_name, dataset_results in results['individual_results'].items():
                summary_row = {
                    'dataset': dataset_name,
                    **dataset_results['metrics']
                }
                summary_data.append(summary_row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / f"{experiment_name}_summary.csv" 
            summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Saved evaluation results to {output_dir}")


class ModelEvaluator(SarcasmEvaluator):
    """Evaluator focused on single model performance."""
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
    
    def benchmark_model(
        self,
        test_datasets: Dict[str, Any],
        validation_datasets: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model benchmarking.
        
        Args:
            test_datasets: Test datasets
            validation_datasets: Validation datasets
            
        Returns:
            Benchmark results
        """
        results = {}
        
        # Test evaluation
        test_results = self.evaluate_multiple_datasets(test_datasets, "test")
        results['test'] = test_results
        
        # Validation evaluation if provided
        if validation_datasets:
            val_results = self.evaluate_multiple_datasets(validation_datasets, "validation")
            results['validation'] = val_results
        
        # Model analysis
        results['model_analysis'] = self._analyze_model()
        
        return results
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model architecture and parameters."""
        
        analysis = {}
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        analysis['parameters'] = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        # Model type
        model_type = type(self.model).__name__
        analysis['model_type'] = model_type
        
        # Architecture details
        if hasattr(self.model, 'get_model_info'):
            analysis['architecture'] = self.model.get_model_info()
        
        return analysis


class DatasetEvaluator(SarcasmEvaluator):
    """Evaluator focused on dataset-specific analysis."""
    
    def __init__(self, model, datasets, config=None):
        super().__init__(model, config)
        self.datasets = datasets
    
    def analyze_dataset_performance(self) -> Dict[str, Any]:
        """Analyze performance across different datasets."""
        
        results = self.evaluate_multiple_datasets(self.datasets)
        
        # Dataset difficulty ranking
        dataset_f1_scores = {}
        for dataset_name, dataset_results in results['individual_results'].items():
            dataset_f1_scores[dataset_name] = dataset_results['metrics']['f1']
        
        # Sort by F1 score (ascending = more difficult)
        difficulty_ranking = sorted(dataset_f1_scores.items(), key=lambda x: x[1])
        
        # Dataset characteristics analysis
        dataset_analysis = {}
        for dataset_name, dataset in self.datasets.items():
            stats = dataset.get_statistics() if hasattr(dataset, 'get_statistics') else {}
            dataset_analysis[dataset_name] = {
                'size': len(dataset),
                'performance': dataset_f1_scores.get(dataset_name, 0.0),
                'statistics': stats
            }
        
        results['dataset_analysis'] = dataset_analysis
        results['difficulty_ranking'] = difficulty_ranking
        
        return results


class CrossDatasetEvaluator(SarcasmEvaluator):
    """Evaluator for cross-dataset generalization analysis."""
    
    def __init__(self, model, datasets, config=None):
        super().__init__(model, config)
        self.datasets = datasets
    
    def evaluate_cross_dataset_generalization(self) -> Dict[str, Any]:
        """
        Evaluate cross-dataset generalization by training on one dataset
        and testing on others.
        """
        
        self.logger.info("Evaluating cross-dataset generalization")
        
        results = {}
        dataset_names = list(self.datasets.keys())
        
        # Create cross-dataset evaluation matrix
        cross_matrix = {}
        
        for test_dataset_name in dataset_names:
            test_dataset = self.datasets[test_dataset_name]
            
            # Evaluate on this test dataset
            test_results = self.evaluate_dataset(
                test_dataset, 
                test_dataset_name, 
                "cross_dataset_test"
            )
            
            cross_matrix[test_dataset_name] = test_results['metrics']
        
        results['cross_dataset_matrix'] = cross_matrix
        
        # Analyze generalization patterns
        results['generalization_analysis'] = self._analyze_generalization_patterns(cross_matrix)
        
        return results
    
    def _analyze_generalization_patterns(
        self,
        cross_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze cross-dataset generalization patterns."""
        
        analysis = {}
        
        # Extract F1 scores for analysis
        f1_scores = {}
        for dataset_name, metrics in cross_matrix.items():
            f1_scores[dataset_name] = metrics.get('f1', 0.0)
        
        # Generalization statistics
        f1_values = list(f1_scores.values())
        analysis['f1_statistics'] = {
            'mean': float(np.mean(f1_values)),
            'std': float(np.std(f1_values)),
            'min': float(np.min(f1_values)),
            'max': float(np.max(f1_values)),
            'range': float(np.max(f1_values) - np.min(f1_values))
        }
        
        # Best and worst generalization
        best_dataset = max(f1_scores.items(), key=lambda x: x[1])
        worst_dataset = min(f1_scores.items(), key=lambda x: x[1])
        
        analysis['best_generalization'] = {
            'dataset': best_dataset[0],
            'f1': best_dataset[1]
        }
        
        analysis['worst_generalization'] = {
            'dataset': worst_dataset[0],
            'f1': worst_dataset[1]
        }
        
        return analysis
