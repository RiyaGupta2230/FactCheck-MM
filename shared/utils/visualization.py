"""
Visualization Utilities for FactCheck-MM
Plots for training curves, confusion matrices, attention heatmaps, and dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from wordcloud import WordCloud

from .logging_utils import get_logger

# Set style
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization suite for FactCheck-MM experiments.
    """
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        use_plotly: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
            use_plotly: Whether to use Plotly for interactive plots
            figsize: Default figure size for matplotlib
            dpi: DPI for saved figures
        """
        self.save_dir = save_dir
        self.use_plotly = use_plotly
        self.figsize = figsize
        self.dpi = dpi
        
        self.logger = get_logger("Visualizer")
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Visualizer initialized, saving to: {save_dir}")
    
    def plot_training_curves(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot training and validation curves.
        
        Args:
            metrics_history: Dictionary of metric histories
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        if self.use_plotly:
            return self._plot_training_curves_plotly(metrics_history, title, save_name)
        else:
            return self._plot_training_curves_matplotlib(metrics_history, title, save_name)
    
    def _plot_training_curves_plotly(
        self,
        metrics_history: Dict[str, List[float]],
        title: str,
        save_name: Optional[str]
    ) -> Optional[Path]:
        """Plot training curves using Plotly."""
        
        # Group metrics by type
        train_metrics = {k: v for k, v in metrics_history.items() if k.startswith('train_')}
        val_metrics = {k: v for k, v in metrics_history.items() if k.startswith('val_') or k.startswith('eval_')}
        
        # Create subplots
        unique_metrics = set()
        for key in train_metrics.keys():
            metric_name = key.replace('train_', '')
            unique_metrics.add(metric_name)
        for key in val_metrics.keys():
            metric_name = key.replace('val_', '').replace('eval_', '')
            unique_metrics.add(metric_name)
        
        n_metrics = len(unique_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(unique_metrics),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Plot each metric
        for i, metric in enumerate(unique_metrics):
            row = i // cols + 1
            col = i % cols + 1
            
            # Train curve
            train_key = f'train_{metric}'
            if train_key in train_metrics:
                steps = list(range(len(train_metrics[train_key])))
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=train_metrics[train_key],
                        mode='lines',
                        name=f'Train {metric}',
                        line=dict(color='blue'),
                        showlegend=i == 0
                    ),
                    row=row, col=col
                )
            
            # Validation curve
            val_key = f'val_{metric}' if f'val_{metric}' in val_metrics else f'eval_{metric}'
            if val_key in val_metrics:
                steps = list(range(len(val_metrics[val_key])))
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=val_metrics[val_key],
                        mode='lines',
                        name=f'Val {metric}',
                        line=dict(color='red'),
                        showlegend=i == 0
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            self.logger.info(f"Saved training curves: {save_path}")
            return save_path
        
        return None
    
    def _plot_training_curves_matplotlib(
        self,
        metrics_history: Dict[str, List[float]],
        title: str,
        save_name: Optional[str]
    ) -> Optional[Path]:
        """Plot training curves using Matplotlib."""
        
        # Group metrics
        train_metrics = {k: v for k, v in metrics_history.items() if k.startswith('train_')}
        val_metrics = {k: v for k, v in metrics_history.items() if k.startswith('val_') or k.startswith('eval_')}
        
        # Determine subplot layout
        unique_metrics = set()
        for key in list(train_metrics.keys()) + list(val_metrics.keys()):
            metric_name = key.replace('train_', '').replace('val_', '').replace('eval_', '')
            unique_metrics.add(metric_name)
        
        n_metrics = len(unique_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(unique_metrics):
            ax = axes[i]
            
            # Plot training curve
            train_key = f'train_{metric}'
            if train_key in train_metrics:
                steps = list(range(len(train_metrics[train_key])))
                ax.plot(steps, train_metrics[train_key], 'b-', label=f'Train {metric}', alpha=0.8)
            
            # Plot validation curve
            val_key = f'val_{metric}' if f'val_{metric}' in val_metrics else f'eval_{metric}'
            if val_key in val_metrics:
                steps = list(range(len(val_metrics[val_key])))
                ax.plot(steps, val_metrics[val_key], 'r-', label=f'Val {metric}', alpha=0.8)
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch/Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved training curves: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Class names
            title: Plot title
            normalize: Whether to normalize
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=self.figsize)
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrix: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        input_tokens: List[str],
        output_tokens: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
        save_name: Optional[str] = None,
        head_idx: int = 0,
        layer_idx: int = -1
    ) -> Optional[Path]:
        """
        Plot attention heatmap.
        
        Args:
            attention_weights: Attention tensor [layers, heads, seq_len, seq_len]
            input_tokens: Input token strings
            output_tokens: Output token strings (for cross-attention)
            title: Plot title
            save_name: Save filename
            head_idx: Attention head index
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Path to saved plot
        """
        # Extract attention weights for specific layer and head
        if attention_weights.dim() == 4:
            attn = attention_weights[layer_idx, head_idx].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head_idx].detach().cpu().numpy()
        else:
            attn = attention_weights.detach().cpu().numpy()
        
        # Truncate tokens and attention if too long
        max_len = 50
        if len(input_tokens) > max_len:
            input_tokens = input_tokens[:max_len]
            attn = attn[:max_len, :max_len]
        
        if output_tokens is None:
            output_tokens = input_tokens
        elif len(output_tokens) > max_len:
            output_tokens = output_tokens[:max_len]
            attn = attn[:max_len, :max_len]
        
        plt.figure(figsize=(max(8, len(input_tokens) * 0.3), max(6, len(output_tokens) * 0.3)))
        
        sns.heatmap(
            attn,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
        plt.xlabel('Input Tokens')
        plt.ylabel('Output Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved attention heatmap: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot metric comparison across different models/experiments.
        
        Args:
            results: Dictionary of {experiment_name: {metric: value}}
            metric_name: Metric to compare
            title: Plot title
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        # Extract data
        experiments = list(results.keys())
        values = [results[exp].get(metric_name, 0.0) for exp in experiments]
        
        plt.figure(figsize=self.figsize)
        
        bars = plt.bar(experiments, values, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title or f"{metric_name.replace('_', ' ').title()} Comparison")
        plt.xlabel('Experiment')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved metric comparison: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def create_results_dashboard(
        self,
        experiment_results: Dict[str, Any],
        save_name: str = "dashboard"
    ) -> Optional[Path]:
        """
        Create comprehensive results dashboard.
        
        Args:
            experiment_results: Complete experiment results
            save_name: Save filename
            
        Returns:
            Path to saved dashboard
        """
        if not self.use_plotly:
            self.logger.warning("Dashboard requires Plotly")
            return None
        
        # Create dashboard with multiple tabs/sections
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Training Progress",
                "Model Performance",
                "Loss Curves",
                "Metrics Summary"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # Add training progress
        if 'metrics_history' in experiment_results:
            history = experiment_results['metrics_history']
            if 'train_loss' in history:
                steps = list(range(len(history['train_loss'])))
                fig.add_trace(
                    go.Scatter(x=steps, y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
                    row=1, col=1
                )
            if 'val_loss' in history or 'eval_loss' in history:
                loss_key = 'val_loss' if 'val_loss' in history else 'eval_loss'
                steps = list(range(len(history[loss_key])))
                fig.add_trace(
                    go.Scatter(x=steps, y=history[loss_key], name='Val Loss', line=dict(color='red')),
                    row=1, col=1
                )
        
        # Add performance metrics
        if 'final_metrics' in experiment_results:
            metrics = experiment_results['final_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Final Metrics'),
                row=1, col=2
            )
        
        # Add loss curves
        if 'loss_curves' in experiment_results:
            curves = experiment_results['loss_curves']
            for name, values in curves.items():
                steps = list(range(len(values)))
                fig.add_trace(
                    go.Scatter(x=steps, y=values, name=name),
                    row=2, col=1
                )
        
        # Add summary table
        if 'model_info' in experiment_results:
            info = experiment_results['model_info']
            fig.add_trace(
                go.Table(
                    header=dict(values=['Property', 'Value']),
                    cells=dict(values=[list(info.keys()), list(info.values())])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="FactCheck-MM Experiment Dashboard",
            height=800,
            showlegend=True
        )
        
        if self.save_dir:
            save_path = self.save_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            self.logger.info(f"Saved dashboard: {save_path}")
            return save_path
        
        return None
    
    def plot_data_distribution(
        self,
        data: Union[List, np.ndarray, pd.Series],
        title: str = "Data Distribution",
        bins: int = 30,
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot data distribution histogram.
        
        Args:
            data: Data to plot
            title: Plot title
            bins: Number of bins
            save_name: Save filename
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=self.figsize)
        
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1 STD: {mean_val + std_val:.3f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1 STD: {mean_val - std_val:.3f}')
        plt.legend()
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved distribution plot: {save_path}")
            plt.close()
            return save_path
        
        return None


# Convenience functions
def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves"
) -> None:
    """Convenience function to plot training curves."""
    visualizer = Visualizer(save_dir=save_path.parent if save_path else None)
    save_name = save_path.stem if save_path else None
    visualizer.plot_training_curves(metrics_history, title, save_name)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> None:
    """Convenience function to plot confusion matrix."""
    visualizer = Visualizer(save_dir=save_path.parent if save_path else None)
    save_name = save_path.stem if save_path else None
    visualizer.plot_confusion_matrix(cm, class_names, title, normalize, save_name)


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: Optional[Path] = None,
    title: str = "Attention Heatmap"
) -> None:
    """Convenience function to plot attention heatmap."""
    visualizer = Visualizer(save_dir=save_path.parent if save_path else None)
    save_name = save_path.stem if save_path else None
    visualizer.plot_attention_heatmap(attention_weights, tokens, title=title, save_name=save_name)


def create_results_dashboard(
    results: Dict[str, Any],
    save_path: Path
) -> None:
    """Convenience function to create results dashboard."""
    visualizer = Visualizer(save_dir=save_path.parent, use_plotly=True)
    visualizer.create_results_dashboard(results, save_path.stem)
