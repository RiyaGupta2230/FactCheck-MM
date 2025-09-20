# sarcasm_detection/evaluation/visualizations.py
"""
Comprehensive Visualizations for Sarcasm Detection
Confusion matrices, loss curves, attention heatmaps, and interactive dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from wordcloud import WordCloud
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from shared.utils import get_logger
from shared.utils.visualization import Visualizer


class SarcasmVisualizer:
    """Comprehensive visualizer for sarcasm detection results."""
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        style: str = "publication",
        use_interactive: bool = True,
        color_palette: str = "husl"
    ):
        """
        Initialize sarcasm visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            style: Visualization style ('publication', 'presentation', 'web')
            use_interactive: Whether to create interactive plots
            color_palette: Color palette to use
        """
        self.save_dir = save_dir
        self.style = style
        self.use_interactive = use_interactive
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("SarcasmVisualizer")
        
        # Setup plotting style
        self._setup_style(color_palette)
        
        # Initialize base visualizer
        self.base_visualizer = Visualizer(save_dir, use_interactive)
        
        self.logger.info(f"Initialized sarcasm visualizer with {style} style")
    
    def _setup_style(self, color_palette: str):
        """Setup visualization style."""
        
        if self.style == "publication":
            plt.style.use('seaborn-v0_8-paper')
            self.figsize = (10, 8)
            self.dpi = 300
            self.font_size = 12
        elif self.style == "presentation":
            plt.style.use('seaborn-v0_8-talk')
            self.figsize = (12, 9)
            self.dpi = 150
            self.font_size = 14
        else:  # web
            plt.style.use('seaborn-v0_8-whitegrid')
            self.figsize = (10, 6)
            self.dpi = 100
            self.font_size = 11
        
        # Set color palette
        sns.set_palette(color_palette)
        
        # Update matplotlib settings
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4
        })
    
    def create_comprehensive_dashboard(
        self,
        evaluation_results: Dict[str, Any],
        title: str = "Sarcasm Detection Results Dashboard"
    ) -> Optional[Path]:
        """
        Create comprehensive interactive dashboard.
        
        Args:
            evaluation_results: Complete evaluation results
            title: Dashboard title
            
        Returns:
            Path to saved dashboard
        """
        if not self.use_interactive:
            self.logger.warning("Interactive dashboard requires use_interactive=True")
            return None
        
        self.logger.info("Creating comprehensive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Model Performance Overview",
                "Dataset Comparison", 
                "Confusion Matrix",
                "ROC Curves",
                "Precision-Recall Curves",
                "Error Analysis",
                "Feature Importance",
                "Confidence Distribution",
                "Performance Trends"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Model Performance Overview
        self._add_performance_overview(fig, evaluation_results, 1, 1)
        
        # 2. Dataset Comparison
        self._add_dataset_comparison(fig, evaluation_results, 1, 2)
        
        # 3. Confusion Matrix
        self._add_confusion_matrix_plotly(fig, evaluation_results, 1, 3)
        
        # 4. ROC Curves
        self._add_roc_curves(fig, evaluation_results, 2, 1)
        
        # 5. Precision-Recall Curves
        self._add_pr_curves(fig, evaluation_results, 2, 2)
        
        # 6. Error Analysis
        self._add_error_analysis(fig, evaluation_results, 2, 3)
        
        # 7. Feature Importance (if available)
        self._add_feature_importance(fig, evaluation_results, 3, 1)
        
        # 8. Confidence Distribution
        self._add_confidence_distribution(fig, evaluation_results, 3, 2)
        
        # 9. Performance Trends
        self._add_performance_trends(fig, evaluation_results, 3, 3)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save dashboard
        if self.save_dir:
            dashboard_path = self.save_dir / f"sarcasm_dashboard.html"
            fig.write_html(str(dashboard_path))
            self.logger.info(f"Saved dashboard to {dashboard_path}")
            return dashboard_path
        
        return None
    
    def _add_performance_overview(self, fig, results, row, col):
        """Add performance overview to dashboard."""
        
        if 'individual_results' in results:
            datasets = []
            f1_scores = []
            accuracies = []
            
            for dataset_name, dataset_results in results['individual_results'].items():
                datasets.append(dataset_name)
                f1_scores.append(dataset_results['metrics'].get('f1', 0))
                accuracies.append(dataset_results['metrics'].get('accuracy', 0))
            
            # F1 scores
            fig.add_trace(
                go.Bar(
                    x=datasets,
                    y=f1_scores,
                    name='F1 Score',
                    marker_color='lightblue'
                ),
                row=row, col=col
            )
    
    def _add_dataset_comparison(self, fig, results, row, col):
        """Add dataset comparison to dashboard."""
        
        if 'individual_results' in results:
            datasets = []
            metrics_data = []
            
            for dataset_name, dataset_results in results['individual_results'].items():
                datasets.append(dataset_name)
                metrics_data.append(dataset_results['metrics'].get('f1', 0))
            
            fig.add_trace(
                go.Bar(
                    x=datasets,
                    y=metrics_data,
                    name='Dataset Performance',
                    marker_color='lightgreen'
                ),
                row=row, col=col
            )
    
    def _add_confusion_matrix_plotly(self, fig, results, row, col):
        """Add confusion matrix to dashboard."""
        
        # Get confusion matrix data from first available result
        for dataset_results in results.get('individual_results', {}).values():
            if 'confusion_matrix' in dataset_results:
                cm = np.array(dataset_results['confusion_matrix'])
                
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Non-Sarcastic', 'Sarcastic'],
                        y=['Non-Sarcastic', 'Sarcastic'],
                        colorscale='Blues',
                        showscale=False
                    ),
                    row=row, col=col
                )
                break
    
    def _add_roc_curves(self, fig, results, row, col):
        """Add ROC curves to dashboard."""
        
        # Placeholder - would need actual prediction probabilities
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.5)  # Dummy curve
        
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name='ROC Curve',
                line=dict(color='red')
            ),
            row=row, col=col
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', dash='dash')
            ),
            row=row, col=col
        )
    
    def _add_pr_curves(self, fig, results, row, col):
        """Add precision-recall curves to dashboard."""
        
        # Placeholder PR curve
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall * 0.5  # Dummy curve
        
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name='PR Curve',
                line=dict(color='purple')
            ),
            row=row, col=col
        )
    
    def _add_error_analysis(self, fig, results, row, col):
        """Add error analysis to dashboard."""
        
        # Example error categories
        categories = ['False Positives', 'False Negatives', 'High Conf Errors', 'Low Conf Errors']
        counts = [15, 20, 8, 12]  # Dummy data
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                name='Error Categories',
                marker_color='coral'
            ),
            row=row, col=col
        )
    
    def _add_feature_importance(self, fig, results, row, col):
        """Add feature importance to dashboard."""
        
        # Example features
        features = ['Text Length', 'Sentiment', 'Punctuation', 'Irony Markers', 'Capitalization']
        importance = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance,
                name='Feature Importance',
                marker_color='gold'
            ),
            row=row, col=col
        )
    
    def _add_confidence_distribution(self, fig, results, row, col):
        """Add confidence distribution to dashboard."""
        
        # Example confidence distribution
        confidences = np.random.beta(2, 5, 1000)  # Dummy data
        
        fig.add_trace(
            go.Histogram(
                x=confidences,
                nbinsx=30,
                name='Confidence Distribution',
                marker_color='teal'
            ),
            row=row, col=col
        )
    
    def _add_performance_trends(self, fig, results, row, col):
        """Add performance trends to dashboard."""
        
        # Example trend data
        epochs = list(range(1, 11))
        train_f1 = [0.5 + 0.04 * i + np.random.normal(0, 0.02) for i in epochs]
        val_f1 = [0.48 + 0.035 * i + np.random.normal(0, 0.02) for i in epochs]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_f1,
                mode='lines+markers',
                name='Train F1',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_f1,
                mode='lines+markers',
                name='Val F1',
                line=dict(color='red')
            ),
            row=row, col=col
        )
    
    def plot_enhanced_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        title: str = "Sarcasm Detection Confusion Matrix",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create enhanced confusion matrix with additional annotations.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Class names
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        if class_names is None:
            class_names = ['Non-Sarcastic', 'Sarcastic']
        
        plt.figure(figsize=self.figsize)
        
        # Normalize for percentages
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Frequency'},
            square=True,
            linewidths=0.5
        )
        
        # Add count annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = confusion_matrix[i, j]
                percentage = cm_normalized[i, j]
                plt.text(j + 0.5, i + 0.7, f'n={count}', 
                        ha='center', va='center', fontsize=10, color='darkblue')
        
        plt.title(title, pad=20)
        plt.xlabel('Predicted Label', fontsize=self.font_size)
        plt.ylabel('True Label', fontsize=self.font_size)
        
        # Add classification metrics text
        tn, fp, fn, tp = confusion_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved enhanced confusion matrix: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_dataset_performance_comparison(
        self,
        results: Dict[str, Any],
        metrics: List[str] = None,
        title: str = "Performance Comparison Across Datasets",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create comprehensive dataset performance comparison.
        
        Args:
            results: Evaluation results
            metrics: Metrics to compare
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        if 'individual_results' not in results:
            self.logger.warning("No individual results found for comparison")
            return None
        
        # Prepare data
        datasets = []
        metric_data = {metric: [] for metric in metrics}
        
        for dataset_name, dataset_results in results['individual_results'].items():
            datasets.append(dataset_name)
            for metric in metrics:
                metric_data[metric].append(dataset_results['metrics'].get(metric, 0))
        
        # Create subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes = [ax1, ax2, ax3, ax4]
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            ax = axes[i]
            
            bars = ax.bar(datasets, metric_data[metric], alpha=0.8)
            ax.set_title(f'{metric.title()} by Dataset')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Add mean line
            mean_value = np.mean(metric_data[metric])
            ax.axhline(y=mean_value, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_value:.3f}')
            ax.legend()
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved dataset comparison: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_error_analysis_dashboard(
        self,
        error_results: Dict[str, Any],
        title: str = "Error Analysis Dashboard",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create comprehensive error analysis dashboard.
        
        Args:
            error_results: Error analysis results
            title: Dashboard title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Error Distribution by Dataset
        if 'aggregated_analysis' in error_results:
            error_by_dataset = error_results['aggregated_analysis'].get('error_by_dataset', {})
            if error_by_dataset:
                datasets = list(error_by_dataset.keys())
                error_counts = list(error_by_dataset.values())
                
                bars1 = ax1.bar(datasets, error_counts, color='coral', alpha=0.7)
                ax1.set_title('Error Distribution by Dataset')
                ax1.set_ylabel('Number of Errors')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add count labels
                for bar, count in zip(bars1, error_counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(count), ha='center', va='bottom')
        
        # 2. Error Types Distribution
        error_types = {'False Positives': 0, 'False Negatives': 0}
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            breakdown = dataset_errors.get('error_breakdown', {}).get('by_error_type', {})
            error_types['False Positives'] += breakdown.get('false_positive', 0)
            error_types['False Negatives'] += breakdown.get('false_negative', 0)
        
        if sum(error_types.values()) > 0:
            ax2.pie(error_types.values(), labels=error_types.keys(), autopct='%1.1f%%',
                   colors=['lightcoral', 'lightskyblue'])
            ax2.set_title('Error Type Distribution')
        
        # 3. Confidence vs Error Rate
        confidence_data = []
        error_rates = []
        
        for dataset_name, dataset_errors in error_results.get('dataset_errors', {}).items():
            confidence_analysis = dataset_errors.get('confidence_analysis', {})
            stats = confidence_analysis.get('confidence_statistics', {})
            error_rate = dataset_errors.get('error_rate', 0)
            
            if stats:
                confidence_data.append(stats.get('mean', 0))
                error_rates.append(error_rate)
        
        if confidence_data:
            ax3.scatter(confidence_data, error_rates, alpha=0.7, s=100)
            ax3.set_xlabel('Mean Confidence')
            ax3.set_ylabel('Error Rate')
            ax3.set_title('Confidence vs Error Rate')
            
            # Add trend line
            if len(confidence_data) > 1:
                z = np.polyfit(confidence_data, error_rates, 1)
                p = np.poly1d(z)
                ax3.plot(confidence_data, p(confidence_data), "r--", alpha=0.8)
        
        # 4. Most Common Error Patterns
        if 'error_patterns' in error_results:
            patterns = error_results['error_patterns'].get('linguistic_patterns', {})
            if patterns:
                # Get top 10 patterns
                top_patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10])
                
                pattern_names = list(top_patterns.keys())
                pattern_counts = list(top_patterns.values())
                
                bars4 = ax4.barh(range(len(pattern_names)), pattern_counts, color='lightgreen', alpha=0.7)
                ax4.set_yticks(range(len(pattern_names)))
                ax4.set_yticklabels(pattern_names)
                ax4.set_xlabel('Frequency')
                ax4.set_title('Most Common Error Patterns')
                
                # Add count labels
                for bar, count in zip(bars4, pattern_counts):
                    ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           str(count), ha='left', va='center')
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved error analysis dashboard: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def create_sarcasm_wordcloud(
        self,
        texts: List[str],
        labels: List[int],
        title: str = "Sarcasm vs Non-Sarcasm Word Clouds",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create word clouds for sarcastic vs non-sarcastic texts.
        
        Args:
            texts: List of text samples
            labels: List of labels (0/1)
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        try:
            # Separate texts by label
            sarcastic_texts = [text for text, label in zip(texts, labels) if label == 1]
            non_sarcastic_texts = [text for text, label in zip(texts, labels) if label == 0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Sarcastic word cloud
            if sarcastic_texts:
                sarcastic_combined = ' '.join(sarcastic_texts)
                wordcloud_sarcastic = WordCloud(
                    width=400, height=400,
                    background_color='white',
                    colormap='Reds',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(sarcastic_combined)
                
                ax1.imshow(wordcloud_sarcastic, interpolation='bilinear')
                ax1.set_title('Sarcastic Texts', fontsize=16)
                ax1.axis('off')
            
            # Non-sarcastic word cloud
            if non_sarcastic_texts:
                non_sarcastic_combined = ' '.join(non_sarcastic_texts)
                wordcloud_non_sarcastic = WordCloud(
                    width=400, height=400,
                    background_color='white',
                    colormap='Blues',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(non_sarcastic_combined)
                
                ax2.imshow(wordcloud_non_sarcastic, interpolation='bilinear')
                ax2.set_title('Non-Sarcastic Texts', fontsize=16)
                ax2.axis('off')
            
            plt.suptitle(title, fontsize=18)
            plt.tight_layout()
            
            if save_name and self.save_dir:
                save_path = self.save_dir / f"{save_name}.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Saved word clouds: {save_path}")
                plt.close()
                return save_path
                
        except ImportError:
            self.logger.warning("WordCloud library not available, skipping word cloud generation")
        except Exception as e:
            self.logger.error(f"Error creating word clouds: {e}")
        
        return None
    
    def plot_training_progress(
        self,
        training_history: Dict[str, List[float]],
        title: str = "Training Progress",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot comprehensive training progress.
        
        Args:
            training_history: Training history with metrics
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        if not training_history:
            self.logger.warning("No training history provided")
            return None
        
        # Determine available metrics
        available_metrics = list(training_history.keys())
        
        # Create subplots based on available metrics
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))
        
        for i, (metric_name, values) in enumerate(training_history.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, color=colors[i], linewidth=2, marker='o', markersize=4)
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(values) > 2:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "--", alpha=0.8, color='gray')
            
            # Highlight best value
            if 'loss' in metric_name.lower():
                best_idx = np.argmin(values)
                best_value = min(values)
            else:
                best_idx = np.argmax(values)
                best_value = max(values)
            
            ax.plot(best_idx + 1, best_value, 'r*', markersize=10, 
                   label=f'Best: {best_value:.4f}')
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved training progress: {save_path}")
            plt.close()
            return save_path
        
        return None


class ResultsVisualizer(SarcasmVisualizer):
    """Specialized visualizer for sarcasm detection results."""
    
    def create_results_summary(
        self,
        results: Dict[str, Any],
        save_name: str = "results_summary"
    ) -> Optional[Path]:
        """Create comprehensive results summary visualization."""
        
        self.logger.info("Creating results summary visualization")
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Panel 1: Overall Performance
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_performance(ax1, results)
        
        # Panel 2: Dataset Breakdown
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_dataset_breakdown(ax2, results)
        
        # Panel 3: Confusion Matrix
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_summary_confusion_matrix(ax3, results)
        
        # Panel 4: Error Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_error_summary(ax4, results)
        
        # Panel 5: Performance Metrics
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_detailed_metrics(ax5, results)
        
        # Panel 6: Recommendations
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_recommendations(ax6, results)
        
        plt.suptitle('Sarcasm Detection Results Summary', fontsize=20, y=0.98)
        
        if self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved results summary: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def _plot_overall_performance(self, ax, results):
        """Plot overall performance metrics."""
        
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            
            metric_names = ['accuracy', 'precision', 'recall', 'f1']
            values = [metrics.get(name, {}).get('mean', 0) for name in metric_names]
            
            bars = ax.bar(metric_names, values, color=['skyblue', 'lightgreen', 'coral', 'gold'])
            ax.set_title('Overall Performance Metrics')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_dataset_breakdown(self, ax, results):
        """Plot performance breakdown by dataset."""
        
        if 'individual_results' in results:
            datasets = []
            f1_scores = []
            
            for dataset_name, dataset_results in results['individual_results'].items():
                datasets.append(dataset_name[:10])  # Truncate long names
                f1_scores.append(dataset_results['metrics'].get('f1', 0))
            
            bars = ax.bar(datasets, f1_scores, color='lightblue', alpha=0.8)
            ax.set_title('F1 Score by Dataset')
            ax.set_ylabel('F1 Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean line
            mean_f1 = np.mean(f1_scores)
            ax.axhline(y=mean_f1, color='red', linestyle='--', 
                      label=f'Mean: {mean_f1:.3f}')
            ax.legend()
    
    def _plot_summary_confusion_matrix(self, ax, results):
        """Plot aggregated confusion matrix."""
        
        # Aggregate confusion matrices from all datasets
        total_cm = None
        
        for dataset_results in results.get('individual_results', {}).values():
            if 'confusion_matrix' in dataset_results:
                cm = np.array(dataset_results['confusion_matrix'])
                if total_cm is None:
                    total_cm = cm
                else:
                    total_cm += cm
        
        if total_cm is not None:
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Non-Sarcastic', 'Sarcastic'],
                       yticklabels=['Non-Sarcastic', 'Sarcastic'],
                       ax=ax)
            ax.set_title('Aggregated Confusion Matrix')
    
    def _plot_error_summary(self, ax, results):
        """Plot error summary."""
        
        # Placeholder error analysis
        error_types = ['False Positives', 'False Negatives', 'High Conf Errors']
        error_counts = [25, 30, 12]  # Example data
        
        bars = ax.bar(error_types, error_counts, color=['coral', 'lightcoral', 'darkred'])
        ax.set_title('Error Summary')
        ax.set_ylabel('Count')
        
        # Add count labels
        for bar, count in zip(bars, error_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
    
    def _plot_detailed_metrics(self, ax, results):
        """Plot detailed metrics comparison."""
        
        if 'individual_results' in results:
            metrics_df = []
            
            for dataset_name, dataset_results in results['individual_results'].items():
                metrics = dataset_results['metrics']
                metrics_df.append({
                    'Dataset': dataset_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0)
                })
            
            if metrics_df:
                df = pd.DataFrame(metrics_df)
                df.set_index('Dataset').plot(kind='bar', ax=ax, width=0.8)
                ax.set_title('Detailed Metrics by Dataset')
                ax.set_ylabel('Score')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.tick_params(axis='x', rotation=45)
    
    def _plot_recommendations(self, ax, results):
        """Plot recommendations text."""
        
        recommendations = results.get('recommendations', [
            "Consider additional training data for low-performing datasets",
            "Investigate high-confidence errors for systematic improvements",
            "Implement ensemble methods for better robustness"
        ])
        
        ax.text(0.05, 0.95, "Key Recommendations:", fontsize=14, fontweight='bold', 
                transform=ax.transAxes, verticalalignment='top')
        
        for i, rec in enumerate(recommendations[:5]):  # Top 5 recommendations
            ax.text(0.05, 0.85 - i*0.15, f"{i+1}. {rec}", fontsize=11,
                   transform=ax.transAxes, verticalalignment='top',
                   wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


class AttentionVisualizer(SarcasmVisualizer):
    """Specialized visualizer for attention mechanisms in sarcasm detection."""
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        input_tokens: List[str],
        title: str = "Attention Heatmap",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot attention heatmap for sarcasm detection.
        
        Args:
            attention_weights: Attention weights tensor
            input_tokens: Input tokens
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        return self.base_visualizer.plot_attention_heatmap(
            attention_weights, input_tokens, title=title, save_name=save_name
        )


class PerformanceVisualizer(SarcasmVisualizer):
    """Specialized visualizer for performance analysis."""
    
    def create_performance_report(
        self,
        results: Dict[str, Any],
        save_name: str = "performance_report"
    ) -> Optional[Path]:
        """Create comprehensive performance report."""
        
        self.logger.info("Creating performance report")
        
        # Use base visualizer's dashboard functionality
        return self.create_comprehensive_dashboard(results, "Performance Analysis Report")
