# sarcasm_detection/utils/sarcasm_metrics.py
"""
Sarcasm-Specific Metrics and Evaluation Utilities
Comprehensive metrics tailored for sarcasm detection tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
import warnings

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, classification_report
)
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr, spearmanr

from shared.utils import get_logger, MetricsComputer


@dataclass
class SarcasmMetricsConfig:
    """Configuration for sarcasm metrics computation."""
    
    # Basic metrics
    compute_standard_metrics: bool = True
    compute_class_specific_metrics: bool = True
    compute_confidence_metrics: bool = True
    
    # Sarcasm-specific metrics
    compute_irony_metrics: bool = True
    compute_subtlety_metrics: bool = True
    compute_context_metrics: bool = True
    
    # Advanced metrics
    compute_calibration_metrics: bool = True
    compute_fairness_metrics: bool = True
    compute_robustness_metrics: bool = True
    
    # Thresholds
    confidence_thresholds: List[float] = None
    subtlety_threshold: float = 0.6
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.5, 0.7, 0.8, 0.9]


class SarcasmMetrics:
    """Comprehensive metrics computation for sarcasm detection."""
    
    def __init__(
        self,
        task_name: str = "sarcasm_detection",
        num_classes: int = 2,
        config: Union[SarcasmMetricsConfig, Dict[str, Any]] = None
    ):
        """
        Initialize sarcasm metrics.
        
        Args:
            task_name: Name of the task
            num_classes: Number of classes
            config: Metrics configuration
        """
        self.task_name = task_name
        self.num_classes = num_classes
        
        if isinstance(config, dict):
            config = SarcasmMetricsConfig(**config)
        elif config is None:
            config = SarcasmMetricsConfig()
        
        self.config = config
        self.logger = get_logger("SarcasmMetrics")
    
    def compute_classification_metrics(
        self,
        predictions: Union[List[int], np.ndarray],
        labels: Union[List[int], np.ndarray],
        probabilities: Optional[Union[List[List[float]], np.ndarray]] = None,
        mode: str = "test"
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics for sarcasm detection.
        
        Args:
            predictions: Model predictions
            labels: True labels
            probabilities: Prediction probabilities
            mode: Evaluation mode
            
        Returns:
            Dictionary of computed metrics
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        if probabilities is not None:
            probabilities = np.array(probabilities)
        
        metrics = {}
        
        # Standard classification metrics
        if self.config.compute_standard_metrics:
            metrics.update(self._compute_standard_metrics(predictions, labels, mode))
        
        # Class-specific metrics
        if self.config.compute_class_specific_metrics:
            metrics.update(self._compute_class_specific_metrics(predictions, labels, mode))
        
        # Confidence-based metrics
        if self.config.compute_confidence_metrics and probabilities is not None:
            metrics.update(self._compute_confidence_metrics(predictions, labels, probabilities, mode))
        
        # Sarcasm-specific metrics
        if self.config.compute_irony_metrics:
            metrics.update(self._compute_irony_metrics(predictions, labels, probabilities, mode))
        
        # Calibration metrics
        if self.config.compute_calibration_metrics and probabilities is not None:
            metrics.update(self._compute_calibration_metrics(predictions, labels, probabilities, mode))
        
        return metrics
    
    def _compute_standard_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        mode: str
    ) -> Dict[str, Any]:
        """Compute standard classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics[f'{mode}_accuracy'] = float(accuracy_score(labels, predictions))
        metrics[f'{mode}_balanced_accuracy'] = float(balanced_accuracy_score(labels, predictions))
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics[f'{mode}_precision'] = float(precision)
        metrics[f'{mode}_recall'] = float(recall)
        metrics[f'{mode}_f1'] = float(f1)
        
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        metrics[f'{mode}_precision_macro'] = float(precision_macro)
        metrics[f'{mode}_recall_macro'] = float(recall_macro)
        metrics[f'{mode}_f1_macro'] = float(f1_macro)
        
        # Matthews Correlation Coefficient
        try:
            mcc = matthews_corrcoef(labels, predictions)
            metrics[f'{mode}_mcc'] = float(mcc)
        except:
            metrics[f'{mode}_mcc'] = 0.0
        
        return metrics
    
    def _compute_class_specific_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        mode: str
    ) -> Dict[str, Any]:
        """Compute class-specific metrics."""
        
        metrics = {}
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        class_names = ['non_sarcastic', 'sarcastic']
        for i, class_name in enumerate(class_names[:len(precision)]):
            metrics[f'{mode}_precision_{class_name}'] = float(precision[i])
            metrics[f'{mode}_recall_{class_name}'] = float(recall[i])
            metrics[f'{mode}_f1_{class_name}'] = float(f1[i])
            metrics[f'{mode}_support_{class_name}'] = int(support[i])
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        for class_idx, count in zip(unique, counts):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
            metrics[f'{mode}_distribution_{class_name}'] = float(count / total_samples)
        
        # Class imbalance ratio
        if len(counts) >= 2:
            imbalance_ratio = max(counts) / min(counts)
            metrics[f'{mode}_imbalance_ratio'] = float(imbalance_ratio)
        
        return metrics
    
    def _compute_confidence_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        mode: str
    ) -> Dict[str, Any]:
        """Compute confidence-based metrics."""
        
        metrics = {}
        
        # Overall confidence statistics
        max_probs = np.max(probabilities, axis=1)
        metrics[f'{mode}_mean_confidence'] = float(np.mean(max_probs))
        metrics[f'{mode}_std_confidence'] = float(np.std(max_probs))
        metrics[f'{mode}_min_confidence'] = float(np.min(max_probs))
        metrics[f'{mode}_max_confidence'] = float(np.max(max_probs))
        
        # Confidence-based accuracy at different thresholds
        for threshold in self.config.confidence_thresholds:
            high_conf_mask = max_probs >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = accuracy_score(
                    labels[high_conf_mask],
                    predictions[high_conf_mask]
                )
                metrics[f'{mode}_accuracy_conf_{threshold}'] = float(high_conf_acc)
                metrics[f'{mode}_samples_conf_{threshold}'] = int(np.sum(high_conf_mask))
                metrics[f'{mode}_coverage_conf_{threshold}'] = float(np.sum(high_conf_mask) / len(labels))
        
        # Confidence vs correctness correlation
        correct_mask = (predictions == labels).astype(float)
        try:
            corr_pearson, _ = pearsonr(max_probs, correct_mask)
            corr_spearman, _ = spearmanr(max_probs, correct_mask)
            
            metrics[f'{mode}_confidence_correctness_pearson'] = float(corr_pearson)
            metrics[f'{mode}_confidence_correctness_spearman'] = float(corr_spearman)
        except:
            metrics[f'{mode}_confidence_correctness_pearson'] = 0.0
            metrics[f'{mode}_confidence_correctness_spearman'] = 0.0
        
        return metrics
    
    def _compute_irony_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray],
        mode: str
    ) -> Dict[str, Any]:
        """Compute sarcasm/irony-specific metrics."""
        
        metrics = {}
        
        # Sarcasm detection rate (sensitivity for sarcastic class)
        sarcasm_mask = (labels == 1)
        if np.sum(sarcasm_mask) > 0:
            sarcasm_detection_rate = np.sum(predictions[sarcasm_mask] == 1) / np.sum(sarcasm_mask)
            metrics[f'{mode}_sarcasm_detection_rate'] = float(sarcasm_detection_rate)
        
        # Non-sarcasm preservation rate (specificity for non-sarcastic class)
        non_sarcasm_mask = (labels == 0)
        if np.sum(non_sarcasm_mask) > 0:
            non_sarcasm_preservation_rate = np.sum(predictions[non_sarcasm_mask] == 0) / np.sum(non_sarcasm_mask)
            metrics[f'{mode}_non_sarcasm_preservation_rate'] = float(non_sarcasm_preservation_rate)
        
        # False positive rate for sarcasm (incorrectly predicting sarcasm)
        if np.sum(non_sarcasm_mask) > 0:
            false_sarcasm_rate = np.sum(predictions[non_sarcasm_mask] == 1) / np.sum(non_sarcasm_mask)
            metrics[f'{mode}_false_sarcasm_rate'] = float(false_sarcasm_rate)
        
        # Subtlety analysis (if probabilities available)
        if probabilities is not None:
            sarcasm_probs = probabilities[:, 1]  # Probability of sarcastic class
            
            # Subtle sarcasm detection (low confidence but correct sarcastic predictions)
            subtle_threshold = self.config.subtlety_threshold
            subtle_sarcasm_mask = (labels == 1) & (sarcasm_probs < subtle_threshold) & (predictions == 1)
            if np.sum(labels == 1) > 0:
                subtle_detection_rate = np.sum(subtle_sarcasm_mask) / np.sum(labels == 1)
                metrics[f'{mode}_subtle_sarcasm_detection_rate'] = float(subtle_detection_rate)
            
            # Obvious sarcasm detection (high confidence correct predictions)
            obvious_sarcasm_mask = (labels == 1) & (sarcasm_probs >= subtle_threshold) & (predictions == 1)
            if np.sum(labels == 1) > 0:
                obvious_detection_rate = np.sum(obvious_sarcasm_mask) / np.sum(labels == 1)
                metrics[f'{mode}_obvious_sarcasm_detection_rate'] = float(obvious_detection_rate)
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        mode: str,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute model calibration metrics."""
        
        metrics = {}
        
        try:
            # Expected Calibration Error (ECE)
            max_probs = np.max(probabilities, axis=1)
            accuracies = (predictions == labels).astype(float)
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            mce = 0.0  # Maximum Calibration Error
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = max_probs[in_bin].mean()
                    
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_error * prop_in_bin
                    mce = max(mce, bin_error)
            
            metrics[f'{mode}_expected_calibration_error'] = float(ece)
            metrics[f'{mode}_maximum_calibration_error'] = float(mce)
            
            # Brier Score
            labels_one_hot = np.eye(self.num_classes)[labels]
            brier_score = np.mean(np.sum((probabilities - labels_one_hot) ** 2, axis=1))
            metrics[f'{mode}_brier_score'] = float(brier_score)
            
            # Reliability diagram data
            if len(np.unique(labels)) == 2:  # Binary classification
                fraction_positives, mean_predicted_values = calibration_curve(
                    labels, probabilities[:, 1], n_bins=n_bins, strategy='uniform'
                )
                
                # Perfect calibration has slope=1, intercept=0
                if len(fraction_positives) > 1:
                    calibration_slope = np.polyfit(mean_predicted_values, fraction_positives, 1)[0]
                    metrics[f'{mode}_calibration_slope'] = float(calibration_slope)
        
        except Exception as e:
            self.logger.warning(f"Could not compute calibration metrics: {e}")
        
        return metrics


class IronyDetectionMetrics:
    """Specialized metrics for irony and sarcasm detection nuances."""
    
    def __init__(self):
        """Initialize irony detection metrics."""
        self.logger = get_logger("IronyDetectionMetrics")
    
    def compute_irony_specific_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        text_features: Optional[Dict[str, np.ndarray]] = None,
        linguistic_features: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Compute irony-specific metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            text_features: Text-based features
            linguistic_features: Linguistic analysis features
            
        Returns:
            Irony-specific metrics
        """
        metrics = {}
        
        # Basic irony detection metrics
        irony_mask = (labels == 1)
        non_irony_mask = (labels == 0)
        
        if np.sum(irony_mask) > 0 and np.sum(non_irony_mask) > 0:
            # Irony detection sensitivity
            irony_sensitivity = np.sum((predictions == 1) & irony_mask) / np.sum(irony_mask)
            metrics['irony_sensitivity'] = float(irony_sensitivity)
            
            # Non-irony specificity
            non_irony_specificity = np.sum((predictions == 0) & non_irony_mask) / np.sum(non_irony_mask)
            metrics['non_irony_specificity'] = float(non_irony_specificity)
        
        # Feature-based analysis
        if text_features is not None:
            metrics.update(self._analyze_text_feature_performance(predictions, labels, text_features))
        
        if linguistic_features is not None:
            metrics.update(self._analyze_linguistic_feature_performance(predictions, labels, linguistic_features))
        
        return metrics
    
    def _analyze_text_feature_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        text_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze performance based on text features."""
        
        metrics = {}
        
        # Length-based analysis
        if 'length' in text_features:
            lengths = text_features['length']
            
            # Short text performance
            short_mask = lengths <= np.percentile(lengths, 25)
            if np.sum(short_mask) > 0:
                short_acc = accuracy_score(labels[short_mask], predictions[short_mask])
                metrics['short_text_accuracy'] = float(short_acc)
            
            # Long text performance
            long_mask = lengths >= np.percentile(lengths, 75)
            if np.sum(long_mask) > 0:
                long_acc = accuracy_score(labels[long_mask], predictions[long_mask])
                metrics['long_text_accuracy'] = float(long_acc)
        
        # Sentiment-based analysis
        if 'sentiment' in text_features:
            sentiment = text_features['sentiment']
            
            # Negative sentiment (often associated with sarcasm)
            negative_mask = sentiment < -0.1
            if np.sum(negative_mask) > 0:
                neg_acc = accuracy_score(labels[negative_mask], predictions[negative_mask])
                metrics['negative_sentiment_accuracy'] = float(neg_acc)
            
            # Positive sentiment (potential sarcasm indicator)
            positive_mask = sentiment > 0.1
            if np.sum(positive_mask) > 0:
                pos_acc = accuracy_score(labels[positive_mask], predictions[positive_mask])
                metrics['positive_sentiment_accuracy'] = float(pos_acc)
        
        return metrics
    
    def _analyze_linguistic_feature_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        linguistic_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze performance based on linguistic features."""
        
        metrics = {}
        
        # Punctuation-based analysis
        if 'punctuation_score' in linguistic_features:
            punct_scores = linguistic_features['punctuation_score']
            
            # High punctuation (exclamations, questions)
            high_punct_mask = punct_scores >= np.percentile(punct_scores, 75)
            if np.sum(high_punct_mask) > 0:
                high_punct_acc = accuracy_score(labels[high_punct_mask], predictions[high_punct_mask])
                metrics['high_punctuation_accuracy'] = float(high_punct_acc)
        
        # Capitalization-based analysis
        if 'capitalization_ratio' in linguistic_features:
            cap_ratios = linguistic_features['capitalization_ratio']
            
            # High capitalization (emphasis)
            high_cap_mask = cap_ratios >= np.percentile(cap_ratios, 90)
            if np.sum(high_cap_mask) > 0:
                high_cap_acc = accuracy_score(labels[high_cap_mask], predictions[high_cap_mask])
                metrics['high_capitalization_accuracy'] = float(high_cap_acc)
        
        return metrics


class MultimodalSarcasmMetrics:
    """Metrics for multimodal sarcasm detection."""
    
    def __init__(self):
        """Initialize multimodal sarcasm metrics."""
        self.logger = get_logger("MultimodalSarcasmMetrics")
    
    def compute_multimodal_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        modality_availability: Dict[str, np.ndarray],
        modality_predictions: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Compute multimodal-specific metrics.
        
        Args:
            predictions: Final multimodal predictions
            labels: True labels
            modality_availability: Which modalities are available for each sample
            modality_predictions: Individual modality predictions
            
        Returns:
            Multimodal-specific metrics
        """
        metrics = {}
        
        # Performance by modality combination
        metrics.update(self._compute_modality_combination_metrics(
            predictions, labels, modality_availability
        ))
        
        # Modality contribution analysis
        if modality_predictions is not None:
            metrics.update(self._compute_modality_contribution_metrics(
                predictions, labels, modality_predictions
            ))
        
        # Cross-modal agreement
        if modality_predictions is not None:
            metrics.update(self._compute_cross_modal_agreement(modality_predictions))
        
        return metrics
    
    def _compute_modality_combination_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        modality_availability: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute performance metrics by modality combination."""
        
        metrics = {}
        
        # Single modality performance
        for modality, available_mask in modality_availability.items():
            if np.sum(available_mask) > 0:
                single_acc = accuracy_score(labels[available_mask], predictions[available_mask])
                metrics[f'{modality}_only_accuracy'] = float(single_acc)
                metrics[f'{modality}_only_samples'] = int(np.sum(available_mask))
        
        # Multi-modality combinations
        modalities = list(modality_availability.keys())
        
        # Dual modality combinations
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                dual_mask = modality_availability[mod1] & modality_availability[mod2]
                
                if np.sum(dual_mask) > 0:
                    dual_acc = accuracy_score(labels[dual_mask], predictions[dual_mask])
                    metrics[f'{mod1}_{mod2}_accuracy'] = float(dual_acc)
                    metrics[f'{mod1}_{mod2}_samples'] = int(np.sum(dual_mask))
        
        # All modalities available
        all_available_mask = np.ones(len(labels), dtype=bool)
        for available_mask in modality_availability.values():
            all_available_mask &= available_mask
        
        if np.sum(all_available_mask) > 0:
            all_acc = accuracy_score(labels[all_available_mask], predictions[all_available_mask])
            metrics['all_modalities_accuracy'] = float(all_acc)
            metrics['all_modalities_samples'] = int(np.sum(all_available_mask))
        
        return metrics
    
    def _compute_modality_contribution_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        modality_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute individual modality contribution metrics."""
        
        metrics = {}
        
        # Individual modality performance
        for modality, mod_predictions in modality_predictions.items():
            mod_acc = accuracy_score(labels, mod_predictions)
            metrics[f'{modality}_individual_accuracy'] = float(mod_acc)
            
            # Agreement with final prediction
            agreement = np.mean(mod_predictions == predictions)
            metrics[f'{modality}_final_agreement'] = float(agreement)
            
            # Complementary information (cases where modality is right but ensemble is wrong)
            mod_correct = (mod_predictions == labels)
            ensemble_wrong = (predictions != labels)
            complementary_cases = mod_correct & ensemble_wrong
            
            if np.sum(ensemble_wrong) > 0:
                complementary_rate = np.sum(complementary_cases) / np.sum(ensemble_wrong)
                metrics[f'{modality}_complementary_rate'] = float(complementary_rate)
        
        return metrics
    
    def _compute_cross_modal_agreement(
        self,
        modality_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute cross-modal agreement metrics."""
        
        metrics = {}
        
        modalities = list(modality_predictions.keys())
        
        # Pairwise agreement
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                agreement = np.mean(modality_predictions[mod1] == modality_predictions[mod2])
                metrics[f'{mod1}_{mod2}_agreement'] = float(agreement)
        
        # Overall consensus
        if len(modalities) > 2:
            # Majority vote consensus
            stacked_predictions = np.stack(list(modality_predictions.values()), axis=1)
            majority_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=stacked_predictions
            )
            
            # Consensus strength (how often all modalities agree)
            full_consensus = np.apply_along_axis(
                lambda x: len(np.unique(x)) == 1, axis=1, arr=stacked_predictions
            )
            
            metrics['full_consensus_rate'] = float(np.mean(full_consensus))
            
            # Majority consensus rate
            majority_consensus = np.apply_along_axis(
                lambda x: np.max(np.bincount(x)) / len(x), axis=1, arr=stacked_predictions
            )
            metrics['mean_majority_consensus'] = float(np.mean(majority_consensus))
        
        return metrics


class ClassImbalanceMetrics:
    """Metrics and utilities for handling class imbalance in sarcasm detection."""
    
    def __init__(self):
        """Initialize class imbalance metrics."""
        self.logger = get_logger("ClassImbalanceMetrics")
    
    def analyze_class_imbalance(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze class imbalance effects on performance.
        
        Args:
            labels: True labels
            predictions: Model predictions
            probabilities: Prediction probabilities
            
        Returns:
            Class imbalance analysis
        """
        metrics = {}
        
        # Class distribution analysis
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        for label, count in zip(unique_labels, counts):
            class_name = 'sarcastic' if label == 1 else 'non_sarcastic'
            metrics[f'class_{class_name}_count'] = int(count)
            metrics[f'class_{class_name}_ratio'] = float(count / total_samples)
        
        # Imbalance ratio
        if len(counts) >= 2:
            imbalance_ratio = max(counts) / min(counts)
            metrics['imbalance_ratio'] = float(imbalance_ratio)
            
            # Severity classification
            if imbalance_ratio < 2:
                severity = "balanced"
            elif imbalance_ratio < 5:
                severity = "mild_imbalance"
            elif imbalance_ratio < 10:
                severity = "moderate_imbalance"
            else:
                severity = "severe_imbalance"
            
            metrics['imbalance_severity'] = severity
        
        # Performance by class size
        minority_class = unique_labels[np.argmin(counts)]
        majority_class = unique_labels[np.argmax(counts)]
        
        minority_mask = (labels == minority_class)
        majority_mask = (labels == majority_class)
        
        if np.sum(minority_mask) > 0:
            minority_acc = accuracy_score(labels[minority_mask], predictions[minority_mask])
            metrics['minority_class_accuracy'] = float(minority_acc)
        
        if np.sum(majority_mask) > 0:
            majority_acc = accuracy_score(labels[majority_mask], predictions[majority_mask])
            metrics['majority_class_accuracy'] = float(majority_acc)
        
        # Bias towards majority class
        pred_distribution = np.bincount(predictions, minlength=len(unique_labels))
        pred_ratios = pred_distribution / np.sum(pred_distribution)
        true_ratios = counts / np.sum(counts)
        
        bias_score = np.sum(np.abs(pred_ratios - true_ratios))
        metrics['prediction_bias_score'] = float(bias_score)
        
        return metrics
    
    def compute_imbalance_aware_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute metrics that are aware of class imbalance."""
        
        metrics = {}
        
        # Balanced accuracy (accounts for class imbalance)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        metrics['balanced_accuracy'] = float(balanced_acc)
        
        # Matthews Correlation Coefficient (robust to imbalance)
        mcc = matthews_corrcoef(labels, predictions)
        metrics['matthews_correlation'] = float(mcc)
        
        # Geometric mean of sensitivities
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        if len(recall) >= 2 and all(r > 0 for r in recall):
            geometric_mean = np.sqrt(np.prod(recall))
            metrics['geometric_mean_sensitivity'] = float(geometric_mean)
        
        # Area Under ROC (if probabilities available)
        if probabilities is not None and len(np.unique(labels)) == 2:
            try:
                auc = roc_auc_score(labels, probabilities[:, 1])
                metrics['area_under_roc'] = float(auc)
                
                # Average Precision (better for imbalanced data)
                avg_precision = average_precision_score(labels, probabilities[:, 1])
                metrics['average_precision'] = float(avg_precision)
            except:
                pass
        
        return metrics
    
    def suggest_imbalance_handling_strategies(
        self,
        imbalance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Suggest strategies for handling class imbalance."""
        
        suggestions = []
        
        imbalance_ratio = imbalance_analysis.get('imbalance_ratio', 1.0)
        severity = imbalance_analysis.get('imbalance_severity', 'balanced')
        
        if severity == 'balanced':
            suggestions.append("Dataset is well-balanced. Standard training should work well.")
        
        elif severity == 'mild_imbalance':
            suggestions.extend([
                "Consider using class weights in loss function",
                "Monitor balanced accuracy in addition to regular accuracy",
                "Use stratified sampling for train/validation splits"
            ])
        
        elif severity == 'moderate_imbalance':
            suggestions.extend([
                "Apply class weighting with inverse frequency weighting",
                "Consider data augmentation for minority class",
                "Use focal loss or other imbalance-aware loss functions",
                "Evaluate using F1-score and AUC-PR instead of accuracy",
                "Consider ensemble methods with balanced sampling"
            ])
        
        else:  # severe_imbalance
            suggestions.extend([
                "Strong class weighting or cost-sensitive learning required",
                "Extensive data augmentation for minority class",
                "Consider SMOTE or other oversampling techniques",
                "Use ensemble methods with different sampling strategies",
                "Focal loss or similar imbalance-aware losses essential",
                "Threshold optimization may be necessary",
                "Consider collecting more minority class data"
            ])
        
        # Performance-based suggestions
        minority_acc = imbalance_analysis.get('minority_class_accuracy', 0.0)
        majority_acc = imbalance_analysis.get('majority_class_accuracy', 0.0)
        
        if minority_acc < majority_acc - 0.2:
            suggestions.append("Large performance gap between classes - focus on minority class improvements")
        
        bias_score = imbalance_analysis.get('prediction_bias_score', 0.0)
        if bias_score > 0.1:
            suggestions.append("Model shows bias towards majority class - consider threshold adjustment")
        
        return suggestions


class SarcasmEvaluator:
    """High-level evaluator combining all sarcasm-specific metrics."""
    
    def __init__(self, config: Union[SarcasmMetricsConfig, Dict[str, Any]] = None):
        """Initialize comprehensive sarcasm evaluator."""
        
        if isinstance(config, dict):
            config = SarcasmMetricsConfig(**config)
        elif config is None:
            config = SarcasmMetricsConfig()
        
        self.config = config
        self.logger = get_logger("SarcasmEvaluator")
        
        # Initialize specialized metrics
        self.basic_metrics = SarcasmMetrics("sarcasm_detection", 2, config)
        self.irony_metrics = IronyDetectionMetrics()
        self.multimodal_metrics = MultimodalSarcasmMetrics()
        self.imbalance_metrics = ClassImbalanceMetrics()
    
    def evaluate_comprehensive(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        text_features: Optional[Dict[str, np.ndarray]] = None,
        linguistic_features: Optional[Dict[str, np.ndarray]] = None,
        modality_availability: Optional[Dict[str, np.ndarray]] = None,
        modality_predictions: Optional[Dict[str, np.ndarray]] = None,
        mode: str = "test"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with all sarcasm-specific metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            probabilities: Prediction probabilities
            text_features: Text-based features
            linguistic_features: Linguistic features
            modality_availability: Multimodal availability
            modality_predictions: Individual modality predictions
            mode: Evaluation mode
            
        Returns:
            Comprehensive metrics dictionary
        """
        self.logger.info(f"Running comprehensive sarcasm evaluation ({mode})")
        
        results = {}
        
        # Basic classification metrics
        basic_results = self.basic_metrics.compute_classification_metrics(
            predictions, labels, probabilities, mode
        )
        results.update(basic_results)
        
        # Irony-specific metrics
        if self.config.compute_irony_metrics:
            irony_results = self.irony_metrics.compute_irony_specific_metrics(
                predictions, labels, text_features, linguistic_features
            )
            results[f'{mode}_irony_metrics'] = irony_results
        
        # Multimodal metrics
        if modality_availability is not None:
            multimodal_results = self.multimodal_metrics.compute_multimodal_metrics(
                predictions, labels, modality_availability, modality_predictions
            )
            results[f'{mode}_multimodal_metrics'] = multimodal_results
        
        # Class imbalance analysis
        imbalance_analysis = self.imbalance_metrics.analyze_class_imbalance(
            labels, predictions, probabilities
        )
        results[f'{mode}_imbalance_analysis'] = imbalance_analysis
        
        # Imbalance-aware metrics
        imbalance_metrics = self.imbalance_metrics.compute_imbalance_aware_metrics(
            labels, predictions, probabilities
        )
        results.update({f'{mode}_{k}': v for k, v in imbalance_metrics.items()})
        
        # Recommendations
        suggestions = self.imbalance_metrics.suggest_imbalance_handling_strategies(
            imbalance_analysis
        )
        results[f'{mode}_recommendations'] = suggestions
        
        return results
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate human-readable evaluation report."""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPREHENSIVE SARCASM DETECTION EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Extract mode from results
        mode = "test"
        for key in results.keys():
            if key.endswith("_accuracy"):
                mode = key.split("_")[0]
                break
        
        # Basic performance
        report_lines.append("BASIC PERFORMANCE METRICS")
        report_lines.append("-" * 30)
        
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
        for metric in basic_metrics:
            key = f'{mode}_{metric}'
            if key in results:
                report_lines.append(f"{metric.upper()}: {results[key]:.4f}")
        
        report_lines.append("")
        
        # Class-specific performance
        report_lines.append("CLASS-SPECIFIC PERFORMANCE")
        report_lines.append("-" * 30)
        
        for class_name in ['non_sarcastic', 'sarcastic']:
            f1_key = f'{mode}_f1_{class_name}'
            if f1_key in results:
                report_lines.append(f"{class_name.upper()} F1: {results[f1_key]:.4f}")
        
        report_lines.append("")
        
        # Sarcasm-specific insights
        if f'{mode}_sarcasm_detection_rate' in results:
            report_lines.append("SARCASM-SPECIFIC INSIGHTS")
            report_lines.append("-" * 30)
            report_lines.append(f"Sarcasm Detection Rate: {results[f'{mode}_sarcasm_detection_rate']:.4f}")
            
            if f'{mode}_false_sarcasm_rate' in results:
                report_lines.append(f"False Sarcasm Rate: {results[f'{mode}_false_sarcasm_rate']:.4f}")
            
            report_lines.append("")
        
        # Imbalance analysis
        imbalance_key = f'{mode}_imbalance_analysis'
        if imbalance_key in results:
            imbalance_data = results[imbalance_key]
            report_lines.append("CLASS IMBALANCE ANALYSIS")
            report_lines.append("-" * 30)
            report_lines.append(f"Imbalance Ratio: {imbalance_data.get('imbalance_ratio', 1.0):.2f}")
            report_lines.append(f"Severity: {imbalance_data.get('imbalance_severity', 'unknown')}")
            report_lines.append("")
        
        # Recommendations
        rec_key = f'{mode}_recommendations'
        if rec_key in results and results[rec_key]:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 30)
            for i, rec in enumerate(results[rec_key], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
