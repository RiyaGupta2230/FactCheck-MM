"""
Comprehensive metrics utilities for milestone1 sarcasm detection
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd

class SarcasmMetrics:
    """Comprehensive metrics calculator for sarcasm detection"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored predictions"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(self, predictions: List, labels: List, probabilities: List = None):
        """Update metrics with new predictions"""
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def calculate_basic_metrics(self, predictions=None, labels=None):
        """Calculate basic classification metrics"""
        
        preds = predictions if predictions is not None else self.all_predictions
        labs = labels if labels is not None else self.all_labels
        
        metrics = {
            'accuracy': accuracy_score(labs, preds),
            'precision': precision_score(labs, preds, average='binary'),
            'recall': recall_score(labs, preds, average='binary'),
            'f1': f1_score(labs, preds, average='binary'),
            'precision_macro': precision_score(labs, preds, average='macro'),
            'recall_macro': recall_score(labs, preds, average='macro'),
            'f1_macro': f1_score(labs, preds, average='macro')
        }
        
        return metrics
    
    def calculate_advanced_metrics(self, probabilities=None, labels=None):
        """Calculate advanced metrics requiring probabilities"""
        
        probs = probabilities if probabilities is not None else self.all_probabilities
        labs = labels if labels is not None else self.all_labels
        
        if not probs:
            return {}
        
        metrics = {
            'auc_roc': roc_auc_score(labs, probs),
            'auc_pr': average_precision_score(labs, probs)
        }
        
        return metrics
    
    def calculate_per_class_metrics(self, predictions=None, labels=None):
        """Calculate per-class metrics"""
        
        preds = predictions if predictions is not None else self.all_predictions
        labs = labels if labels is not None else self.all_labels
        
        # Per-class precision, recall, f1
        precision_per_class = precision_score(labs, preds, average=None)
        recall_per_class = recall_score(labs, preds, average=None)
        f1_per_class = f1_score(labs, preds, average=None)
        
        class_names = ['Non-Sarcastic', 'Sarcastic']
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[f'{class_name.lower()}_precision'] = precision_per_class[i]
            per_class_metrics[f'{class_name.lower()}_recall'] = recall_per_class[i]
            per_class_metrics[f'{class_name.lower()}_f1'] = f1_per_class[i]
        
        return per_class_metrics
    
    def get_confusion_matrix(self, predictions=None, labels=None):
        """Get confusion matrix"""
        
        preds = predictions if predictions is not None else self.all_predictions
        labs = labels if labels is not None else self.all_labels
        
        cm = confusion_matrix(labs, preds)
        
        return {
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    
    def calculate_comprehensive_metrics(self):
        """Calculate all metrics"""
        
        if not self.all_predictions or not self.all_labels:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Advanced metrics (if probabilities available)
        if self.all_probabilities:
            metrics.update(self.calculate_advanced_metrics())
        
        # Per-class metrics
        metrics.update(self.calculate_per_class_metrics())
        
        # Confusion matrix
        metrics.update(self.get_confusion_matrix())
        
        # Additional derived metrics
        metrics.update(self._calculate_derived_metrics())
        
        return metrics
    
    def _calculate_derived_metrics(self):
        """Calculate additional derived metrics"""
        
        cm_metrics = self.get_confusion_matrix()
        tp, tn, fp, fn = cm_metrics['tp'], cm_metrics['tn'], cm_metrics['fp'], cm_metrics['fn']
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Sensitivity (True Positive Rate / Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        }
    
    def generate_classification_report(self, predictions=None, labels=None):
        """Generate detailed classification report"""
        
        preds = predictions if predictions is not None else self.all_predictions
        labs = labels if labels is not None else self.all_labels
        
        target_names = ['Non-Sarcastic', 'Sarcastic']
        report = classification_report(labs, preds, target_names=target_names, output_dict=True)
        
        return report

class MetricsVisualizer:
    """Visualization utilities for metrics"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None, figsize=(8, 6)):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Sarcastic', 'Sarcastic'],
                   yticklabels=['Non-Sarcastic', 'Sarcastic'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob, save_path=None, figsize=(8, 6)):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_prob, save_path=None, figsize=(8, 6)):
        """Plot precision-recall curve"""
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(12, 8)):
        """Plot comparison of multiple metrics"""
        
        metrics_df = pd.DataFrame(metrics_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot different metric categories
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1']
        advanced_metrics = ['auc_roc', 'auc_pr', 'balanced_accuracy', 'mcc']
        
        if all(metric in metrics_df.columns for metric in basic_metrics):
            metrics_df[basic_metrics].plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Basic Classification Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
        
        if all(metric in metrics_df.columns for metric in advanced_metrics):
            metrics_df[advanced_metrics].plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Advanced Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def calculate_model_performance_summary(predictions, labels, probabilities=None):
    """Calculate comprehensive performance summary"""
    
    metrics_calc = SarcasmMetrics()
    metrics_calc.update(predictions, labels, probabilities)
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_comprehensive_metrics()
    
    # Generate classification report
    report = metrics_calc.generate_classification_report()
    
    # Create summary
    summary = {
        'overall_performance': {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'auc_roc': metrics.get('auc_roc', 'N/A')
        },
        'per_class_performance': {
            'non_sarcastic': {
                'precision': metrics['non-sarcastic_precision'],
                'recall': metrics['non-sarcastic_recall'],
                'f1': metrics['non-sarcastic_f1']
            },
            'sarcastic': {
                'precision': metrics['sarcastic_precision'],
                'recall': metrics['sarcastic_recall'],
                'f1': metrics['sarcastic_f1']
            }
        },
        'confusion_matrix': metrics,
        'advanced_metrics': {
            'balanced_accuracy': metrics['balanced_accuracy'],
            'mcc': metrics['mcc'],
            'specificity': metrics['specificity'],
            'sensitivity': metrics['sensitivity']
        },
        'classification_report': report
    }
    
    return summary
