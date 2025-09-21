#!/usr/bin/env python3
"""
Multitask Joint Trainer for FactCheck-MM

Joint training across sarcasm detection, paraphrasing, and fact verification
with shared encoder and task-specific heads. Implements sophisticated 
loss balancing and gradient management strategies.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.metrics import MetricsComputer
from shared.base_model import BaseMultimodalModel
from .task_scheduling import TaskScheduler, SchedulerConfig


@dataclass
class MultitaskConfig:
    """Configuration for multitask learning."""
    
    # Tasks to train jointly
    tasks: List[str] = field(default_factory=lambda: ['sarcasm_detection', 'paraphrasing', 'fact_verification'])
    
    # Shared architecture configuration
    shared_encoder_config: Dict[str, Any] = field(default_factory=dict)
    
    # Task-specific head configurations
    task_head_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Training configuration
    max_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Loss balancing
    loss_balancing_strategy: str = "weighted"  # weighted, adaptive, uncertainty
    initial_task_weights: Dict[str, float] = field(default_factory=dict)
    adaptive_loss_alpha: float = 0.16  # For uncertainty-based loss balancing
    
    # Task scheduling
    use_task_scheduling: bool = True
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Evaluation configuration
    eval_every_n_epochs: int = 1
    save_best_model_per_task: bool = True
    
    # Resource management
    device: str = "auto"
    mixed_precision: bool = True
    
    # Output configuration
    save_intermediate_results: bool = True


class MultitaskModel(nn.Module):
    """Multitask model with shared encoder and task-specific heads."""
    
    def __init__(self, config: MultitaskConfig):
        """
        Initialize multitask model.
        
        Args:
            config: Multitask configuration
        """
        super().__init__()
        self.config = config
        self.tasks = config.tasks
        
        # Shared encoder
        self.shared_encoder = self._create_shared_encoder()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in self.tasks:
            self.task_heads[task] = self._create_task_head(task)
        
        # Task-specific loss functions
        self.loss_functions = {}
        for task in self.tasks:
            self.loss_functions[task] = self._create_loss_function(task)
        
        # Loss balancing parameters
        if config.loss_balancing_strategy == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))
    
    def _create_shared_encoder(self) -> nn.Module:
        """Create shared encoder architecture."""
        
        encoder_config = self.config.shared_encoder_config
        
        # Use base multimodal model as shared encoder
        return BaseMultimodalModel(encoder_config)
    
    def _create_task_head(self, task: str) -> nn.Module:
        """Create task-specific head."""
        
        head_config = self.config.task_head_configs.get(task, {})
        shared_dim = self.config.shared_encoder_config.get('fusion_output_dim', 768)
        
        if task == 'sarcasm_detection':
            # Binary classification head
            return nn.Sequential(
                nn.Linear(shared_dim, head_config.get('hidden_dim', 256)),
                nn.ReLU(),
                nn.Dropout(head_config.get('dropout', 0.1)),
                nn.Linear(head_config.get('hidden_dim', 256), 2)
            )
        
        elif task == 'paraphrasing':
            # Sequence generation head (simplified)
            vocab_size = head_config.get('vocab_size', 30522)
            return nn.Sequential(
                nn.Linear(shared_dim, head_config.get('hidden_dim', 512)),
                nn.ReLU(),
                nn.Dropout(head_config.get('dropout', 0.1)),
                nn.Linear(head_config.get('hidden_dim', 512), vocab_size)
            )
        
        elif task == 'fact_verification':
            # 3-class classification (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
            return nn.Sequential(
                nn.Linear(shared_dim, head_config.get('hidden_dim', 256)),
                nn.ReLU(),
                nn.Dropout(head_config.get('dropout', 0.1)),
                nn.Linear(head_config.get('hidden_dim', 256), 3)
            )
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _create_loss_function(self, task: str) -> nn.Module:
        """Create task-specific loss function."""
        
        if task in ['sarcasm_detection', 'fact_verification']:
            return nn.CrossEntropyLoss()
        elif task == 'paraphrasing':
            return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        else:
            return nn.CrossEntropyLoss()
    
    def forward(self, inputs: Dict[str, Any], task: str) -> torch.Tensor:
        """
        Forward pass for specific task.
        
        Args:
            inputs: Input data
            task: Task name
            
        Returns:
            Task-specific output
        """
        # Get shared representation
        shared_repr = self.shared_encoder(inputs)
        
        # Apply task-specific head
        if task in self.task_heads:
            output = self.task_heads[task](shared_repr)
            return output
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def compute_loss(self, inputs: Dict[str, Any], targets: torch.Tensor, task: str) -> torch.Tensor:
        """
        Compute loss for specific task.
        
        Args:
            inputs: Input data
            targets: Target labels
            task: Task name
            
        Returns:
            Task loss
        """
        outputs = self.forward(inputs, task)
        loss_fn = self.loss_functions[task]
        
        if task == 'paraphrasing':
            # For sequence generation, reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        
        return loss_fn(outputs, targets)
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights for loss balancing."""
        
        if self.config.loss_balancing_strategy == "uncertainty":
            # Uncertainty-based weighting
            weights = {}
            for i, task in enumerate(self.tasks):
                precision = torch.exp(-self.log_vars[i])
                weights[task] = precision.item()
            return weights
        
        elif self.config.loss_balancing_strategy == "weighted":
            # Fixed weights
            return self.config.initial_task_weights
        
        else:  # equal
            return {task: 1.0 for task in self.tasks}


class MultitaskTrainer:
    """Joint trainer for multiple FactCheck-MM tasks."""
    
    def __init__(
        self,
        config: MultitaskConfig,
        data_loaders: Dict[str, Dict[str, DataLoader]],  # {task: {split: dataloader}}
        output_dir: str = "outputs/experiments/multitask_learning"
    ):
        """
        Initialize multitask trainer.
        
        Args:
            config: Multitask configuration
            data_loaders: Data loaders for each task and split
            output_dir: Output directory for results
        """
        self.config = config
        self.data_loaders = data_loaders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("MultitaskTrainer")
        
        # Device setup
        self.device = self._setup_device()
        
        # Initialize model
        self.model = MultitaskModel(config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        total_steps = self._estimate_total_steps()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # Initialize task scheduler
        if config.use_task_scheduling:
            self.task_scheduler = TaskScheduler(config.scheduler_config, list(data_loaders.keys()))
        else:
            self.task_scheduler = None
        
        # Mixed precision scaler
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {task: 0.0 for task in config.tasks}
        self.training_history = []
        
        # Metrics computers
        self.metrics_computers = {}
        for task in config.tasks:
            self.metrics_computers[task] = MetricsComputer(task)
        
        self.logger.info(f"Initialized multitask trainer for tasks: {config.tasks}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
    
    def _estimate_total_steps(self) -> int:
        """Estimate total training steps."""
        total_steps = 0
        for task, splits in self.data_loaders.items():
            if 'train' in splits:
                total_steps += len(splits['train']) * self.config.max_epochs
        return total_steps
    
    def train(self) -> Dict[str, Any]:
        """Run multitask training."""
        
        self.logger.info("Starting multitask training")
        training_start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_start_time = time.time()
            train_metrics = self._train_epoch()
            
            # Validation epoch
            val_metrics = self._validate_epoch()
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                'training_time': time.time() - epoch_start_time,
                **train_metrics,
                **val_metrics
            }
            
            self.training_history.append(epoch_metrics)
            
            # Log epoch results
            self._log_epoch_results(epoch_metrics)
            
            # Save best models
            self._save_best_models(val_metrics)
            
            # Update task scheduler
            if self.task_scheduler:
                self.task_scheduler.update_epoch(epoch, val_metrics)
        
        training_time = time.time() - training_start_time
        
        # Final evaluation
        test_metrics = self._test_all_tasks()
        
        # Compile results
        results = {
            'config': self.config.__dict__,
            'training_time': training_time,
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'final_test_metrics': test_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        # Save results
        self._save_results(results)
        
        self.logger.info(f"Multitask training completed in {training_time:.2f}s")
        
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        epoch_losses = defaultdict(list)
        task_steps = defaultdict(int)
        
        # Create task iterators
        task_iterators = {}
        for task in self.config.tasks:
            if task in self.data_loaders and 'train' in self.data_loaders[task]:
                task_iterators[task] = iter(self.data_loaders[task]['train'])
        
        # Determine number of steps per epoch
        max_steps = max(len(self.data_loaders[task]['train']) for task in task_iterators.keys())
        
        for step in range(max_steps):
            # Determine which task to train on this step
            if self.task_scheduler:
                current_task = self.task_scheduler.get_next_task(self.global_step)
            else:
                # Round-robin scheduling
                current_task = self.config.tasks[step % len(self.config.tasks)]
            
            # Skip if no data loader for this task
            if current_task not in task_iterators:
                continue
            
            # Get batch for current task
            try:
                batch = next(task_iterators[current_task])
            except StopIteration:
                # Restart iterator if exhausted
                task_iterators[current_task] = iter(self.data_loaders[current_task]['train'])
                batch = next(task_iterators[current_task])
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            loss = self._training_step(batch, current_task)
            
            # Record loss
            epoch_losses[current_task].append(loss.item())
            task_steps[current_task] += 1
            self.global_step += 1
        
        # Calculate average losses
        avg_losses = {}
        for task in self.config.tasks:
            if task in epoch_losses and epoch_losses[task]:
                avg_losses[f'train_{task}_loss'] = np.mean(epoch_losses[task])
                avg_losses[f'train_{task}_steps'] = task_steps[task]
            else:
                avg_losses[f'train_{task}_loss'] = 0.0
                avg_losses[f'train_{task}_steps'] = 0
        
        # Add overall loss
        all_losses = []
        for losses in epoch_losses.values():
            all_losses.extend(losses)
        avg_losses['train_overall_loss'] = np.mean(all_losses) if all_losses else 0.0
        
        return avg_losses
    
    def _training_step(self, batch: Dict[str, Any], task: str) -> torch.Tensor:
        """Perform single training step."""
        
        # Extract inputs and targets
        targets = self._extract_targets(batch, task)
        inputs = {k: v for k, v in batch.items() if k not in ['labels', 'label', 'targets']}
        
        # Forward pass with mixed precision if available
        if self.scaler and self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                loss = self.model.compute_loss(inputs, targets, task)
                
                # Apply uncertainty-based loss balancing if configured
                if self.config.loss_balancing_strategy == "uncertainty":
                    task_idx = self.config.tasks.index(task)
                    precision = torch.exp(-self.model.log_vars[task_idx])
                    loss = precision * loss + self.config.adaptive_loss_alpha * self.model.log_vars[task_idx]
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            # Standard training step
            loss = self.model.compute_loss(inputs, targets, task)
            
            # Apply loss balancing
            if self.config.loss_balancing_strategy == "weighted":
                task_weight = self.config.initial_task_weights.get(task, 1.0)
                loss = loss * task_weight
            elif self.config.loss_balancing_strategy == "uncertainty":
                task_idx = self.config.tasks.index(task)
                precision = torch.exp(-self.model.log_vars[task_idx])
                loss = precision * loss + self.config.adaptive_loss_alpha * self.model.log_vars[task_idx]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return loss.detach()
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate all tasks."""
        
        self.model.eval()
        val_metrics = {}
        
        for task in self.config.tasks:
            if task in self.data_loaders and 'val' in self.data_loaders[task]:
                task_metrics = self._evaluate_task(task, 'val')
                val_metrics.update(task_metrics)
        
        return val_metrics
    
    def _evaluate_task(self, task: str, split: str) -> Dict[str, float]:
        """Evaluate single task on given split."""
        
        dataloader = self.data_loaders[task][split]
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Extract inputs and targets
                targets = self._extract_targets(batch, task)
                inputs = {k: v for k, v in batch.items() if k not in ['labels', 'label', 'targets']}
                
                # Forward pass
                outputs = self.model.forward(inputs, task)
                loss = self.model.compute_loss(inputs, targets, task)
                
                # Get predictions
                if task == 'paraphrasing':
                    # For generation tasks, use greedy decoding
                    predictions = torch.argmax(outputs, dim=-1)
                    predictions = predictions.view(-1)
                    targets_flat = targets.view(-1)
                    
                    # Filter out padding tokens
                    mask = targets_flat != 0
                    predictions = predictions[mask]
                    targets_flat = targets_flat[mask]
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets_flat.cpu().numpy())
                else:
                    # Classification tasks
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        metrics_computer = self.metrics_computers[task]
        metrics = metrics_computer.compute_classification_metrics(
            predictions=all_predictions,
            labels=all_targets
        )
        
        # Add loss and prefix
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Add prefix for split and task
        prefixed_metrics = {}
        for metric_name, value in metrics.items():
            prefixed_metrics[f'{split}_{task}_{metric_name}'] = value
        
        return prefixed_metrics
    
    def _test_all_tasks(self) -> Dict[str, float]:
        """Test all tasks on test set."""
        
        test_metrics = {}
        
        for task in self.config.tasks:
            if task in self.data_loaders and 'test' in self.data_loaders[task]:
                task_metrics = self._evaluate_task(task, 'test')
                test_metrics.update(task_metrics)
        
        return test_metrics
    
    def _extract_targets(self, batch: Dict[str, Any], task: str) -> torch.Tensor:
        """Extract target labels from batch."""
        
        if 'targets' in batch:
            return batch['targets']
        elif 'labels' in batch:
            return batch['labels']
        elif 'label' in batch:
            return batch['label']
        else:
            # Try task-specific target keys
            possible_keys = [f'{task}_labels', f'{task}_targets', 'target']
            for key in possible_keys:
                if key in batch:
                    return batch[key]
            
            raise ValueError(f"No target labels found in batch for task {task}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def _log_epoch_results(self, metrics: Dict[str, float]):
        """Log epoch results."""
        
        epoch = metrics['epoch']
        
        # Log overall progress
        self.logger.info(f"Epoch {epoch}/{self.config.max_epochs}")
        
        # Log task-specific metrics
        for task in self.config.tasks:
            val_f1_key = f'val_{task}_f1'
            val_loss_key = f'val_{task}_loss'
            train_loss_key = f'train_{task}_loss'
            
            if val_f1_key in metrics and val_loss_key in metrics:
                val_f1 = metrics[val_f1_key]
                val_loss = metrics[val_loss_key]
                train_loss = metrics.get(train_loss_key, 0.0)
                
                self.logger.info(f"  {task}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")
        
        # Log task weights if using adaptive balancing
        if self.config.loss_balancing_strategy == "uncertainty":
            task_weights = self.model.get_task_weights()
            weight_str = ", ".join([f"{task}={weight:.3f}" for task, weight in task_weights.items()])
            self.logger.info(f"  Task weights: {weight_str}")
    
    def _save_best_models(self, val_metrics: Dict[str, float]):
        """Save best models for each task."""
        
        if not self.config.save_best_model_per_task:
            return
        
        for task in self.config.tasks:
            val_f1_key = f'val_{task}_f1'
            
            if val_f1_key in val_metrics:
                current_f1 = val_metrics[val_f1_key]
                
                if current_f1 > self.best_metrics[task]:
                    self.best_metrics[task] = current_f1
                    
                    # Save model
                    model_path = self.output_dir / f"best_model_{task}.pt"
                    torch.save({
                        'epoch': self.current_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_f1': current_f1,
                        'task': task,
                        'config': self.config.__dict__
                    }, model_path)
                    
                    self.logger.info(f"Saved new best model for {task} (F1: {current_f1:.4f})")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results."""
        
        # Save main results
        results_file = self.output_dir / "multitask_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save final model
        final_model_path = self.output_dir / "final_multitask_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'final_metrics': results['final_test_metrics'],
            'training_history': results['training_history']
        }, final_model_path)
        
        # Create summary report
        summary = self._create_summary_report(results)
        summary_file = self.output_dir / "multitask_training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create human-readable summary report."""
        
        lines = []
        lines.append("=" * 60)
        lines.append("FACTCHECK-MM MULTITASK TRAINING REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Tasks: {', '.join(self.config.tasks)}")
        lines.append(f"Training time: {results['training_time']:.2f} seconds")
        lines.append(f"Total epochs: {self.config.max_epochs}")
        lines.append(f"Model parameters: {results['model_info']['total_parameters']:,}")
        lines.append("")
        
        # Best validation metrics
        lines.append("BEST VALIDATION METRICS:")
        lines.append("-" * 30)
        for task, best_f1 in results['best_metrics'].items():
            lines.append(f"{task}: F1 = {best_f1:.4f}")
        lines.append("")
        
        # Final test metrics
        lines.append("FINAL TEST METRICS:")
        lines.append("-" * 30)
        test_metrics = results['final_test_metrics']
        for task in self.config.tasks:
            f1_key = f'test_{task}_f1'
            acc_key = f'test_{task}_accuracy'
            
            if f1_key in test_metrics:
                f1 = test_metrics[f1_key]
                acc = test_metrics.get(acc_key, 0.0)
                lines.append(f"{task}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")
        lines.append("")
        
        # Training configuration
        lines.append("TRAINING CONFIGURATION:")
        lines.append("-" * 30)
        lines.append(f"Learning rate: {self.config.learning_rate}")
        lines.append(f"Loss balancing: {self.config.loss_balancing_strategy}")
        lines.append(f"Task scheduling: {self.config.use_task_scheduling}")
        lines.append(f"Mixed precision: {self.config.mixed_precision}")
        lines.append("")
        
        return "\n".join(lines)


def main():
    """Example usage of multitask trainer."""
    
    # Example configuration
    config = MultitaskConfig(
        tasks=['sarcasm_detection', 'fact_verification'],
        shared_encoder_config={
            'modalities': ['text'],
            'text_hidden_dim': 768,
            'fusion_output_dim': 768,
            'num_classes': 2  # Will be overridden by task heads
        },
        task_head_configs={
            'sarcasm_detection': {'hidden_dim': 256, 'dropout': 0.1},
            'fact_verification': {'hidden_dim': 256, 'dropout': 0.1}
        },
        max_epochs=5,
        loss_balancing_strategy="weighted",
        initial_task_weights={'sarcasm_detection': 1.0, 'fact_verification': 1.0}
    )
    
    # Example data loaders
    from tests.fixtures.mock_models import create_mock_dataloader, create_mock_dataset
    
    data_loaders = {
        'sarcasm_detection': {
            'train': create_mock_dataloader(create_mock_dataset("sarcasm", 100), batch_size=8),
            'val': create_mock_dataloader(create_mock_dataset("sarcasm", 30), batch_size=8),
            'test': create_mock_dataloader(create_mock_dataset("sarcasm", 20), batch_size=8)
        },
        'fact_verification': {
            'train': create_mock_dataloader(create_mock_dataset("fact_verification", 100), batch_size=8),
            'val': create_mock_dataloader(create_mock_dataset("fact_verification", 30), batch_size=8),
            'test': create_mock_dataloader(create_mock_dataset("fact_verification", 20), batch_size=8)
        }
    }
    
    # Run multitask training
    trainer = MultitaskTrainer(config, data_loaders)
    results = trainer.train()
    
    print("Multitask training completed!")
    print(f"Best metrics: {results['best_metrics']}")


if __name__ == "__main__":
    main()
