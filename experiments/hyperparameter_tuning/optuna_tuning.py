#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Tuning for FactCheck-MM

Automated hyperparameter optimization using Optuna's Bayesian optimization.
Optimizes model architecture, training parameters, and fusion strategies.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.metrics import MetricsComputer
from config.training_configs import get_base_training_config


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter tuning."""
    
    # Study configuration
    study_name: str = "factcheck_mm_optimization"
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    
    # Optimization objective
    direction: str = "maximize"  # maximize or minimize
    metric_name: str = "val_f1"
    
    # Pruning configuration
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5
    
    # Search space bounds
    learning_rate_bounds: tuple = (1e-5, 1e-2)
    batch_size_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    hidden_dim_choices: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    dropout_bounds: tuple = (0.1, 0.5)
    fusion_strategy_choices: List[str] = field(default_factory=lambda: [
        "concatenation", "cross_modal_attention", "bilinear_pooling"
    ])
    
    # Training configuration
    max_epochs: int = 10
    early_stopping_patience: int = 3
    
    # Resource limits
    max_memory_gb: float = 15.0
    device: str = "auto"
    
    # Output configuration
    save_best_model: bool = True
    save_all_configs: bool = True
    log_intermediate_values: bool = True


class OptunaHyperparameterTuner:
    """Optuna-based hyperparameter tuner for FactCheck-MM models."""
    
    def __init__(
        self,
        config: OptunaConfig,
        model_factory: Callable,
        data_loaders: Dict[str, DataLoader],
        task_name: str = "sarcasm_detection",
        output_dir: str = "outputs/experiments/hyperparameter_tuning"
    ):
        """
        Initialize Optuna hyperparameter tuner.
        
        Args:
            config: Optuna configuration
            model_factory: Function that creates model given config
            data_loaders: Dictionary of train/val/test data loaders
            task_name: Name of the task being optimized
            output_dir: Output directory for results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required but not installed. Install with: pip install optuna")
        
        self.config = config
        self.model_factory = model_factory
        self.data_loaders = data_loaders
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("OptunaHyperparameterTuner")
        
        # Initialize study
        self.study = None
        self.best_trial = None
        self.best_config = None
        
        # Device setup
        self.device = self._setup_device()
        
        self.logger.info(f"Initialized Optuna tuner for {task_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup compute device for training."""
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
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        
        # Learning rate (log-uniform distribution)
        learning_rate = trial.suggest_float(
            'learning_rate', 
            self.config.learning_rate_bounds[0], 
            self.config.learning_rate_bounds[1],
            log=True
        )
        
        # Batch size (categorical)
        batch_size = trial.suggest_categorical('batch_size', self.config.batch_size_choices)
        
        # Hidden dimensions
        text_hidden_dim = trial.suggest_categorical('text_hidden_dim', self.config.hidden_dim_choices)
        
        # Dropout rate
        dropout_rate = trial.suggest_float('dropout_rate', *self.config.dropout_bounds)
        
        # Fusion strategy (for multimodal models)
        fusion_strategy = trial.suggest_categorical('fusion_strategy', self.config.fusion_strategy_choices)
        
        # Advanced hyperparameters
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
        
        # Model architecture parameters
        num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8, 12, 16])
        num_fusion_layers = trial.suggest_int('num_fusion_layers', 1, 3)
        
        # Multimodal-specific parameters
        if self.task_name in ['sarcasm_detection', 'multimodal_task']:
            audio_hidden_dim = trial.suggest_categorical('audio_hidden_dim', [256, 512, 768])
            image_hidden_dim = trial.suggest_categorical('image_hidden_dim', [256, 512, 768])
            video_hidden_dim = trial.suggest_categorical('video_hidden_dim', [256, 512, 768])
            modality_dropout = trial.suggest_float('modality_dropout', 0.0, 0.3)
        else:
            audio_hidden_dim = 512
            image_hidden_dim = 512
            video_hidden_dim = 512
            modality_dropout = 0.1
        
        # Training parameters
        gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.5, 5.0)
        
        return {
            # Training parameters
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'gradient_clip_norm': gradient_clip_norm,
            'dropout_rate': dropout_rate,
            
            # Model architecture
            'text_hidden_dim': text_hidden_dim,
            'audio_hidden_dim': audio_hidden_dim,
            'image_hidden_dim': image_hidden_dim,
            'video_hidden_dim': video_hidden_dim,
            'fusion_strategy': fusion_strategy,
            'num_attention_heads': num_attention_heads,
            'num_fusion_layers': num_fusion_layers,
            'modality_dropout': modality_dropout,
            
            # Task-specific parameters
            'task_name': self.task_name,
            'device': self.device,
            'max_epochs': self.config.max_epochs,
            'early_stopping_patience': self.config.early_stopping_patience
        }
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        
        # Get hyperparameters for this trial
        hyperparams = self._suggest_hyperparameters(trial)
        
        trial_start_time = time.time()
        
        try:
            # Create model with suggested hyperparameters
            model = self.model_factory(hyperparams)
            model.to(self.device)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hyperparams['learning_rate'],
                weight_decay=hyperparams['weight_decay']
            )
            
            # Setup scheduler
            num_training_steps = len(self.data_loaders['train']) * self.config.max_epochs
            num_warmup_steps = int(hyperparams['warmup_ratio'] * num_training_steps)
            
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=num_warmup_steps
            )
            
            # Training loop
            best_metric = float('-inf') if self.config.direction == "maximize" else float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                model.train()
                train_loss = self._train_epoch(model, optimizer, scheduler, epoch, trial)
                
                # Validation
                model.eval()
                val_metrics = self._evaluate_epoch(model, 'val', epoch, trial)
                
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
                    # Save best model for this trial
                    if self.config.save_best_model:
                        trial_model_path = self.output_dir / f"trial_{trial.number}_best_model.pt"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'hyperparameters': hyperparams,
                            'metrics': val_metrics,
                            'epoch': epoch
                        }, trial_model_path)
                else:
                    patience_counter += 1
                
                # Report intermediate value for pruning
                if self.config.enable_pruning:
                    trial.report(current_metric, epoch)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        self.logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.TrialPruned()
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch} for trial {trial.number}")
                    break
                
                # Log progress
                if self.config.log_intermediate_values:
                    self.logger.info(
                        f"Trial {trial.number}, Epoch {epoch}: "
                        f"train_loss={train_loss:.4f}, {self.config.metric_name}={current_metric:.4f}"
                    )
            
            trial_time = time.time() - trial_start_time
            self.logger.info(f"Trial {trial.number} completed in {trial_time:.2f}s, best {self.config.metric_name}={best_metric:.4f}")
            
            return best_metric
        
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible value for failed trials
            return float('-inf') if self.config.direction == "maximize" else float('inf')
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, 
                    trial: optuna.Trial) -> float:
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
                # Assume last element is target
                targets = batch[list(batch.keys())[-1]]
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip_norm' in trial.params:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trial.params['gradient_clip_norm'])
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _evaluate_epoch(self, model: nn.Module, split: str, epoch: int, trial: optuna.Trial) -> Dict[str, float]:
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
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        self.logger.info(f"Starting Optuna optimization with {self.config.n_trials} trials")
        
        # Setup study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=self.config.pruning_warmup_steps) if self.config.enable_pruning else None
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization
        start_time = time.time()
        
        try:
            self.study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # Get best trial
        self.best_trial = self.study.best_trial
        self.best_config = self.best_trial.params
        
        self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Best {self.config.metric_name}: {self.best_trial.value:.4f}")
        self.logger.info(f"Best hyperparameters: {self.best_config}")
        
        # Save results
        results = self._save_results()
        
        return results
    
    def _save_results(self) -> Dict[str, Any]:
        """Save optimization results."""
        
        # Compile results
        results = {
            'study_name': self.config.study_name,
            'task_name': self.task_name,
            'n_trials': len(self.study.trials),
            'optimization_time': time.time(),
            'best_trial': {
                'number': self.best_trial.number,
                'value': self.best_trial.value,
                'params': self.best_trial.params,
                'state': self.best_trial.state.name
            },
            'config': self.config.__dict__
        }
        
        # Save main results
        results_file = self.output_dir / f"{self.config.study_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best configuration
        best_config_file = self.output_dir / f"{self.config.study_name}_best_config.json"
        with open(best_config_file, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        # Save all trial results if requested
        if self.config.save_all_configs:
            all_trials_file = self.output_dir / f"{self.config.study_name}_all_trials.json"
            all_trials_data = []
            
            for trial in self.study.trials:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                all_trials_data.append(trial_data)
            
            with open(all_trials_file, 'w') as f:
                json.dump(all_trials_data, f, indent=2)
        
        # Generate optimization plots
        self._create_optimization_plots()
        
        self.logger.info(f"Results saved to: {self.output_dir}")
        
        return results
    
    def _create_optimization_plots(self):
        """Create optimization visualization plots."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Optimization history plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot optimization history
            trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            values = [t.value for t in trials]
            
            ax1.plot(values, 'b-', alpha=0.7, linewidth=1)
            ax1.axhline(y=self.best_trial.value, color='r', linestyle='--', 
                       label=f'Best: {self.best_trial.value:.4f}')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel(self.config.metric_name)
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True)
            
            # Plot parameter importance (if available)
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())[:10]  # Top 10 parameters
                importances = [importance[p] for p in params]
                
                ax2.barh(params, importances)
                ax2.set_xlabel('Importance')
                ax2.set_title('Parameter Importance')
                ax2.grid(True)
                
            except Exception as e:
                ax2.text(0.5, 0.5, f'Parameter importance\nnot available:\n{str(e)}', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{self.config.study_name}_optimization_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Optimization plots saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping optimization plots")
        except Exception as e:
            self.logger.error(f"Failed to create optimization plots: {e}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration from optimization."""
        if self.best_config is None:
            raise ValueError("Optimization has not been run yet")
        
        return self.best_config.copy()
    
    def load_best_model(self) -> Optional[torch.nn.Module]:
        """Load the best model from optimization."""
        if self.best_trial is None:
            raise ValueError("Optimization has not been run yet")
        
        trial_model_path = self.output_dir / f"trial_{self.best_trial.number}_best_model.pt"
        
        if trial_model_path.exists():
            checkpoint = torch.load(trial_model_path, map_location=self.device)
            
            # Recreate model with best hyperparameters
            model = self.model_factory(checkpoint['hyperparameters'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"Loaded best model from trial {self.best_trial.number}")
            return model
        else:
            self.logger.warning(f"Best model file not found: {trial_model_path}")
            return None


def main():
    """Example usage of Optuna hyperparameter tuning."""
    
    # Example model factory
    def create_sarcasm_model(hyperparams):
        from sarcasm_detection.models import MultimodalSarcasmModel
        
        model_config = {
            'modalities': ['text', 'audio', 'image'],
            'fusion_strategy': hyperparams['fusion_strategy'],
            'text_hidden_dim': hyperparams['text_hidden_dim'],
            'audio_hidden_dim': hyperparams['audio_hidden_dim'],
            'image_hidden_dim': hyperparams['image_hidden_dim'],
            'fusion_output_dim': hyperparams['text_hidden_dim'],
            'num_classes': 2,
            'dropout_rate': hyperparams['dropout_rate']
        }
        
        return MultimodalSarcasmModel(model_config)
    
    # Example data loaders (would be created from actual datasets)
    from tests.fixtures.mock_models import create_mock_dataloader, create_mock_dataset
    
    mock_train_dataset = create_mock_dataset("multimodal_sarcasm", 100)
    mock_val_dataset = create_mock_dataset("multimodal_sarcasm", 30)
    
    data_loaders = {
        'train': create_mock_dataloader(mock_train_dataset, batch_size=8),
        'val': create_mock_dataloader(mock_val_dataset, batch_size=8)
    }
    
    # Setup configuration
    config = OptunaConfig(
        study_name="sarcasm_detection_optimization",
        n_trials=20,
        max_epochs=5,
        direction="maximize",
        metric_name="val_f1"
    )
    
    # Run optimization
    tuner = OptunaHyperparameterTuner(
        config=config,
        model_factory=create_sarcasm_model,
        data_loaders=data_loaders,
        task_name="sarcasm_detection"
    )
    
    results = tuner.optimize()
    
    print("Optimization completed!")
    print(f"Best configuration: {tuner.get_best_config()}")


if __name__ == "__main__":
    main()
