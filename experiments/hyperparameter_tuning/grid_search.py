#!/usr/bin/env python3
"""
Grid Search Hyperparameter Tuning for FactCheck-MM

Manual grid search implementation with comprehensive logging and visualization.
Supports systematic exploration of hyperparameter combinations.
"""

import sys
import os
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.metrics import MetricsComputer


@dataclass
class GridSearchConfig:
    """Configuration for grid search hyperparameter tuning."""
    
    # Search space definition
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 768])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    weight_decays: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    
    # Multimodal-specific parameters
    fusion_strategies: List[str] = field(default_factory=lambda: [
        "concatenation", "cross_modal_attention", "bilinear_pooling"
    ])
    
    # Training configuration
    max_epochs: int = 10
    early_stopping_patience: int = 3
    
    # Evaluation configuration
    metric_name: str = "val_f1"
    direction: str = "maximize"  # maximize or minimize
    
    # Resource management
    max_memory_gb: float = 15.0
    device: str = "auto"
    
    # Output configuration
    save_all_models: bool = False
    save_best_model: bool = True
    use_tensorboard: bool = True
    
    # Search optimization
    max_combinations: Optional[int] = None  # Limit total combinations
    random_sample: bool = False  # If True, randomly sample from grid
    random_seed: int = 42


class GridSearchTuner:
    """Grid search hyperparameter tuner for FactCheck-MM models."""
    
    def __init__(
        self,
        config: GridSearchConfig,
        model_factory: Callable,
        data_loaders: Dict[str, DataLoader],
        task_name: str = "sarcasm_detection",
        output_dir: str = "outputs/experiments/grid_search"
    ):
        """
        Initialize grid search tuner.
        
        Args:
            config: Grid search configuration
            model_factory: Function that creates model given config
            data_loaders: Dictionary of train/val/test data loaders
            task_name: Name of the task being optimized
            output_dir: Output directory for results
        """
        self.config = config
        self.model_factory = model_factory
        self.data_loaders = data_loaders
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("GridSearchTuner")
        
        # Device setup
        self.device = self._setup_device()
        
        # TensorBoard setup
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        # Results storage
        self.results = []
        self.best_result = None
        self.best_config = None
        
        # Generate parameter combinations
        self.parameter_combinations = self._generate_parameter_combinations()
        
        self.logger.info(f"Initialized grid search tuner for {task_name}")
        self.logger.info(f"Total parameter combinations: {len(self.parameter_combinations)}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        return device
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        
        # Base parameters that always exist
        base_params = {
            'learning_rate': self.config.learning_rates,
            'batch_size': self.config.batch_sizes,
            'hidden_dim': self.config.hidden_dims,
            'dropout_rate': self.config.dropout_rates,
            'weight_decay': self.config.weight_decays
        }
        
        # Add task-specific parameters
        if self.task_name in ['sarcasm_detection', 'multimodal_task']:
            base_params['fusion_strategy'] = self.config.fusion_strategies
        
        # Generate all combinations
        param_names = list(base_params.keys())
        param_values = list(base_params.values())
        
        all_combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # Add derived parameters
            param_dict.update({
                'text_hidden_dim': param_dict['hidden_dim'],
                'audio_hidden_dim': min(param_dict['hidden_dim'], 512),
                'image_hidden_dim': min(param_dict['hidden_dim'], 512),
                'video_hidden_dim': min(param_dict['hidden_dim'], 512),
                'fusion_output_dim': param_dict['hidden_dim'],
                'device': self.device,
                'task_name': self.task_name
            })
            
            all_combinations.append(param_dict)
        
        # Apply limits and sampling
        if self.config.max_combinations and len(all_combinations) > self.config.max_combinations:
            if self.config.random_sample:
                import random
                random.seed(self.config.random_seed)
                all_combinations = random.sample(all_combinations, self.config.max_combinations)
            else:
                all_combinations = all_combinations[:self.config.max_combinations]
        
        return all_combinations
    
    def search(self) -> Dict[str, Any]:
        """Run grid search optimization."""
        
        self.logger.info(f"Starting grid search with {len(self.parameter_combinations)} combinations")
        
        start_time = time.time()
        
        for idx, params in enumerate(self.parameter_combinations):
            self.logger.info(f"Training combination {idx + 1}/{len(self.parameter_combinations)}")
            self.logger.info(f"Parameters: {params}")
            
            try:
                # Train and evaluate with current parameters
                result = self._train_and_evaluate(params, combination_idx=idx)
                
                # Store result
                self.results.append(result)
                
                # Update best result
                current_metric = result.get(self.config.metric_name, 0.0)
                if self._is_better_result(current_metric):
                    self.best_result = result
                    self.best_config = params.copy()
                    self.logger.info(f"New best {self.config.metric_name}: {current_metric:.4f}")
                
                # Log to TensorBoard
                if self.writer:
                    self._log_to_tensorboard(result, idx)
                
            except Exception as e:
                self.logger.error(f"Combination {idx + 1} failed: {e}")
                
                # Create failed result entry
                failed_result = params.copy()
                failed_result.update({
                    'combination_idx': idx,
                    'status': 'failed',
                    'error': str(e),
                    self.config.metric_name: float('-inf') if self.config.direction == "maximize" else float('inf')
                })
                self.results.append(failed_result)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        search_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'task_name': self.task_name,
            'search_time': search_time,
            'total_combinations': len(self.parameter_combinations),
            'successful_combinations': len([r for r in self.results if r.get('status') != 'failed']),
            'best_result': self.best_result,
            'best_config': self.best_config,
            'all_results': self.results,
            'config': self.config.__dict__
        }
        
        # Save results
        self._save_results(final_results)
        
        if self.writer:
            self.writer.close()
        
        self.logger.info(f"Grid search completed in {search_time:.2f}s")
        if self.best_result:
            self.logger.info(f"Best {self.config.metric_name}: {self.best_result[self.config.metric_name]:.4f}")
        
        return final_results
    
    def _is_better_result(self, current_metric: float) -> bool:
        """Check if current result is better than best so far."""
        if self.best_result is None:
            return True
        
        best_metric = self.best_result.get(self.config.metric_name, 0.0)
        
        if self.config.direction == "maximize":
            return current_metric > best_metric
        else:
            return current_metric < best_metric
    
    def _train_and_evaluate(self, params: Dict[str, Any], combination_idx: int) -> Dict[str, Any]:
        """Train and evaluate model with given parameters."""
        
        combination_start_time = time.time()
        
        # Create model
        model = self.model_factory(params)
        model.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Setup scheduler
        num_training_steps = len(self.data_loaders['train']) * self.config.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
        
        # Training loop
        best_metric = float('-inf') if self.config.direction == "maximize" else float('inf')
        patience_counter = 0
        train_history = []
        
        for epoch in range(self.config.max_epochs):
            # Training
            model.train()
            train_metrics = self._train_epoch(model, optimizer, scheduler, epoch, params)
            
            # Validation
            model.eval()
            val_metrics = self._evaluate_epoch(model, 'val', epoch, params)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            train_history.append(epoch_metrics)
            
            current_metric = val_metrics.get(self.config.metric_name, 0.0)
            
            # Check for improvement
            improved = False
            if self.config.direction == "maximize":
                if current_metric > best_metric:
                    best_metric = current_metric
                    improved = True
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    improved = True
            
            if improved:
                patience_counter = 0
                # Save best model for this combination
                if self.config.save_all_models or (self.config.save_best_model and combination_idx == 0):
                    model_path = self.output_dir / f"combination_{combination_idx}_best_model.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'parameters': params,
                        'metrics': epoch_metrics,
                        'epoch': epoch
                    }, model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch} for combination {combination_idx}")
                break
        
        combination_time = time.time() - combination_start_time
        
        # Test evaluation (if test set available)
        test_metrics = {}
        if 'test' in self.data_loaders:
            test_metrics = self._evaluate_epoch(model, 'test', -1, params)
        
        # Compile result
        result = params.copy()
        result.update({
            'combination_idx': combination_idx,
            'training_time': combination_time,
            'best_epoch': train_history[patience_counter if patience_counter < len(train_history) else -1]['epoch'],
            'status': 'completed',
            'train_history': train_history,
            **test_metrics
        })
        
        # Add final metrics
        if train_history:
            final_metrics = train_history[-1]
            result.update(final_metrics)
        
        return result
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int,
                    params: Dict[str, Any]) -> Dict[str, float]:
        """Train for one epoch."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.data_loaders['train']):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            if 'labels' in batch:
                targets = batch['labels']
            elif 'label' in batch:
                targets = batch['label']
            else:
                targets = batch[list(batch.keys())[-1]]
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_lr': scheduler.get_last_lr()[0] if scheduler.get_last_lr() else params['learning_rate']
        }
    
    def _evaluate_epoch(self, model: nn.Module, split: str, epoch: int,
                       params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate for one epoch."""
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_loaders[split]:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Get predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                
                if 'labels' in batch:
                    targets = batch['labels']
                elif 'label' in batch:
                    targets = batch['label']
                else:
                    targets = batch[list(batch.keys())[-1]]
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, targets)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        metrics_computer = MetricsComputer(self.task_name)
        metrics = metrics_computer.compute_classification_metrics(
            predictions=all_predictions,
            labels=all_labels
        )
        
        # Add loss
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Add prefix for split
        return {f'{split}_{k}': v for k, v in metrics.items()}
    
    def _log_to_tensorboard(self, result: Dict[str, Any], combination_idx: int):
        """Log results to TensorBoard."""
        
        if not self.writer:
            return
        
        # Log main metrics
        main_metric = result.get(self.config.metric_name, 0.0)
        self.writer.add_scalar(f'GridSearch/{self.config.metric_name}', main_metric, combination_idx)
        
        # Log parameters as text
        param_text = "\n".join([f"{k}: {v}" for k, v in result.items() 
                               if k in ['learning_rate', 'batch_size', 'hidden_dim', 'dropout_rate']])
        self.writer.add_text(f'GridSearch/Parameters_{combination_idx}', param_text, combination_idx)
        
        # Log training history if available
        if 'train_history' in result:
            for epoch_idx, epoch_metrics in enumerate(result['train_history']):
                for metric_name, value in epoch_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(
                            f'GridSearch/Combination_{combination_idx}/{metric_name}',
                            value, epoch_idx
                        )
    
    def _save_results(self, results: Dict[str, Any]):
        """Save grid search results."""
        
        # Save main results
        results_file = self.output_dir / "grid_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best configuration
        if self.best_config:
            best_config_file = self.output_dir / "best_configuration.json"
            with open(best_config_file, 'w') as f:
                json.dump(self.best_config, f, indent=2)
        
        # Create results summary
        summary = self._create_results_summary(results)
        summary_file = self.output_dir / "grid_search_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Create results visualization
        self._create_results_visualization(results)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _create_results_summary(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary of results."""
        
        lines = []
        lines.append("=" * 60)
        lines.append("FACTCHECK-MM GRID SEARCH RESULTS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Task: {results['task_name']}")
        lines.append(f"Search time: {results['search_time']:.2f} seconds")
        lines.append(f"Total combinations: {results['total_combinations']}")
        lines.append(f"Successful combinations: {results['successful_combinations']}")
        lines.append("")
        
        # Best result
        if results['best_result']:
            lines.append("BEST CONFIGURATION:")
            lines.append("-" * 30)
            best = results['best_result']
            
            for key in ['learning_rate', 'batch_size', 'hidden_dim', 'dropout_rate', 'fusion_strategy']:
                if key in best:
                    lines.append(f"{key}: {best[key]}")
            
            lines.append("")
            lines.append(f"Best {self.config.metric_name}: {best.get(self.config.metric_name, 'N/A'):.4f}")
            lines.append(f"Training time: {best.get('training_time', 'N/A'):.2f}s")
            lines.append("")
        
        # Top 5 results
        lines.append("TOP 5 RESULTS:")
        lines.append("-" * 30)
        
        # Sort results by metric
        valid_results = [r for r in results['all_results'] if r.get('status') == 'completed']
        sorted_results = sorted(
            valid_results,
            key=lambda x: x.get(self.config.metric_name, float('-inf')),
            reverse=(self.config.direction == "maximize")
        )
        
        for i, result in enumerate(sorted_results[:5]):
            lines.append(f"{i+1}. {self.config.metric_name}: {result.get(self.config.metric_name, 'N/A'):.4f}")
            lines.append(f"   LR: {result.get('learning_rate', 'N/A')}, "
                        f"BS: {result.get('batch_size', 'N/A')}, "
                        f"HD: {result.get('hidden_dim', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_results_visualization(self, results: Dict[str, Any]):
        """Create visualization of grid search results."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            valid_results = [r for r in results['all_results'] if r.get('status') == 'completed']
            
            if not valid_results:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Metric vs Learning Rate
            lrs = [r.get('learning_rate', 0) for r in valid_results]
            metrics = [r.get(self.config.metric_name, 0) for r in valid_results]
            
            axes[0, 0].scatter(lrs, metrics, alpha=0.6)
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_xlabel('Learning Rate')
            axes[0, 0].set_ylabel(self.config.metric_name)
            axes[0, 0].set_title('Metric vs Learning Rate')
            axes[0, 0].grid(True)
            
            # Plot 2: Metric vs Batch Size
            batch_sizes = [r.get('batch_size', 0) for r in valid_results]
            
            axes[0, 1].scatter(batch_sizes, metrics, alpha=0.6)
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel(self.config.metric_name)
            axes[0, 1].set_title('Metric vs Batch Size')
            axes[0, 1].grid(True)
            
            # Plot 3: Metric vs Hidden Dimension
            hidden_dims = [r.get('hidden_dim', 0) for r in valid_results]
            
            axes[1, 0].scatter(hidden_dims, metrics, alpha=0.6)
            axes[1, 0].set_xlabel('Hidden Dimension')
            axes[1, 0].set_ylabel(self.config.metric_name)
            axes[1, 0].set_title('Metric vs Hidden Dimension')
            axes[1, 0].grid(True)
            
            # Plot 4: Training Time vs Metric
            training_times = [r.get('training_time', 0) for r in valid_results]
            
            axes[1, 1].scatter(training_times, metrics, alpha=0.6)
            axes[1, 1].set_xlabel('Training Time (s)')
            axes[1, 1].set_ylabel(self.config.metric_name)
            axes[1, 1].set_title('Training Time vs Metric')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / "grid_search_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Results visualization saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualization")
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration from grid search."""
        return self.best_config.copy() if self.best_config else None
    
    def get_top_k_configs(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get top k configurations from grid search."""
        valid_results = [r for r in self.results if r.get('status') == 'completed']
        sorted_results = sorted(
            valid_results,
            key=lambda x: x.get(self.config.metric_name, float('-inf')),
            reverse=(self.config.direction == "maximize")
        )
        
        return sorted_results[:k]


def main():
    """Example usage of grid search hyperparameter tuning."""
    
    # Example model factory
    def create_sarcasm_model(params):
        from sarcasm_detection.models import MultimodalSarcasmModel
        
        model_config = {
            'modalities': ['text', 'audio', 'image'],
            'fusion_strategy': params.get('fusion_strategy', 'cross_modal_attention'),
            'text_hidden_dim': params['text_hidden_dim'],
            'audio_hidden_dim': params['audio_hidden_dim'],
            'image_hidden_dim': params['image_hidden_dim'],
            'fusion_output_dim': params['fusion_output_dim'],
            'num_classes': 2,
            'dropout_rate': params['dropout_rate']
        }
        
        return MultimodalSarcasmModel(model_config)
    
    # Example data loaders
    from tests.fixtures.mock_models import create_mock_dataloader, create_mock_dataset
    
    mock_train_dataset = create_mock_dataset("multimodal_sarcasm", 100)
    mock_val_dataset = create_mock_dataset("multimodal_sarcasm", 30)
    mock_test_dataset = create_mock_dataset("multimodal_sarcasm", 20)
    
    data_loaders = {
        'train': create_mock_dataloader(mock_train_dataset, batch_size=8),
        'val': create_mock_dataloader(mock_val_dataset, batch_size=8),
        'test': create_mock_dataloader(mock_test_dataset, batch_size=8)
    }
    
    # Setup configuration (smaller for example)
    config = GridSearchConfig(
        learning_rates=[1e-4, 5e-4],
        batch_sizes=[8, 16],
        hidden_dims=[256, 512],
        dropout_rates=[0.1, 0.2],
        max_epochs=3,
        max_combinations=8  # Limit for example
    )
    
    # Run grid search
    tuner = GridSearchTuner(
        config=config,
        model_factory=create_sarcasm_model,
        data_loaders=data_loaders,
        task_name="sarcasm_detection"
    )
    
    results = tuner.search()
    
    print("Grid search completed!")
    print(f"Best configuration: {tuner.get_best_config()}")


if __name__ == "__main__":
    main()
