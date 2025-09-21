#!/usr/bin/env python3
"""
Architecture Ablation Study for FactCheck-MM

Systematic removal of architectural components to understand their contributions.
Tests effects of disabling fusion layers, attention heads, encoder freezing, etc.
"""

import sys
import os
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.metrics import MetricsComputer


@dataclass
class ArchitectureAblationConfig:
    """Configuration for architecture ablation study."""
    
    # Architectural components to ablate
    test_fusion_strategies: bool = True
    test_attention_heads: bool = True
    test_encoder_freezing: bool = True
    test_dropout_variations: bool = True
    test_layer_depths: bool = True
    
    # Component-specific settings
    fusion_strategies_to_test: List[str] = field(default_factory=lambda: [
        "concatenation", "cross_modal_attention", "bilinear_pooling", "none"
    ])
    attention_head_counts: List[int] = field(default_factory=lambda: [1, 4, 8, 12, 16])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    layer_depth_configs: List[Dict[str, int]] = field(default_factory=lambda: [
        {"num_transformer_layers": 6, "num_fusion_layers": 1},
        {"num_transformer_layers": 12, "num_fusion_layers": 2},
        {"num_transformer_layers": 24, "num_fusion_layers": 3}
    ])
    
    # Training configuration
    max_epochs: int = 10
    early_stopping_patience: int = 3
    
    # Evaluation configuration
    metrics_to_track: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])
    
    # Resource management
    device: str = "auto"
    
    # Output configuration
    save_all_models: bool = False
    create_visualizations: bool = True


class ArchitectureAblationStudy:
    """Conducts systematic architecture ablation studies."""
    
    def __init__(
        self,
        config: ArchitectureAblationConfig,
        model_factory: Callable,
        data_loaders: Dict[str, DataLoader],
        base_model_config: Dict[str, Any],
        task_name: str = "multimodal_task",
        output_dir: str = "outputs/experiments/architecture_ablation"
    ):
        """
        Initialize architecture ablation study.
        
        Args:
            config: Ablation configuration
            model_factory: Function to create models with different architectures
            data_loaders: Dictionary of train/val/test data loaders
            base_model_config: Base model configuration
            task_name: Name of the task
            output_dir: Output directory for results
        """
        self.config = config
        self.model_factory = model_factory
        self.data_loaders = data_loaders
        self.base_model_config = base_model_config.copy()
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("ArchitectureAblationStudy")
        
        # Device setup
        self.device = self._setup_device()
        
        # Results storage
        self.results = {}
        self.baseline_performance = None
        
        self.logger.info(f"Initialized architecture ablation study for {task_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
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
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run the complete architecture ablation study."""
        
        self.logger.info("Starting architecture ablation study")
        study_start_time = time.time()
        
        # First, establish baseline performance with default configuration
        self.logger.info("Establishing baseline performance...")
        self.baseline_performance = self._test_architecture_config(
            self.base_model_config, "baseline"
        )
        self.results["baseline"] = self.baseline_performance
        
        baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
        self.logger.info(f"Baseline F1: {baseline_f1:.4f}")
        
        # Test fusion strategies
        if self.config.test_fusion_strategies:
            self.logger.info("Testing fusion strategies...")
            fusion_results = self._test_fusion_strategies()
            self.results.update(fusion_results)
        
        # Test attention head configurations
        if self.config.test_attention_heads:
            self.logger.info("Testing attention head configurations...")
            attention_results = self._test_attention_heads()
            self.results.update(attention_results)
        
        # Test encoder freezing
        if self.config.test_encoder_freezing:
            self.logger.info("Testing encoder freezing strategies...")
            freezing_results = self._test_encoder_freezing()
            self.results.update(freezing_results)
        
        # Test dropout variations
        if self.config.test_dropout_variations:
            self.logger.info("Testing dropout variations...")
            dropout_results = self._test_dropout_variations()
            self.results.update(dropout_results)
        
        # Test layer depths
        if self.config.test_layer_depths:
            self.logger.info("Testing layer depth configurations...")
            depth_results = self._test_layer_depths()
            self.results.update(depth_results)
        
        study_time = time.time() - study_start_time
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Compile final results
        final_results = {
            'study_time': study_time,
            'task_name': self.task_name,
            'baseline_performance': self.baseline_performance,
            'total_experiments': len(self.results),
            'successful_experiments': len([r for r in self.results.values() if r.get('status') != 'failed']),
            'individual_results': self.results,
            'analysis': analysis,
            'config': self.config.__dict__
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info(f"Architecture ablation study completed in {study_time:.2f}s")
        
        return final_results
    
    def _test_fusion_strategies(self) -> Dict[str, Any]:
        """Test different fusion strategies."""
        
        results = {}
        
        for strategy in self.config.fusion_strategies_to_test:
            self.logger.info(f"Testing fusion strategy: {strategy}")
            
            # Create modified config
            config = self.base_model_config.copy()
            config['fusion_strategy'] = strategy
            
            # Handle special case for no fusion
            if strategy == "none":
                config['modalities'] = ['text']  # Fall back to text-only
            
            experiment_name = f"fusion_{strategy}"
            
            try:
                result = self._test_architecture_config(config, experiment_name)
                results[experiment_name] = result
                
                # Calculate performance drop from baseline
                baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
                current_f1 = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - current_f1
                
                self.logger.info(f"  {strategy}: F1 = {current_f1:.4f} (drop: {performance_drop:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to test fusion strategy {strategy}: {e}")
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config
                }
        
        return results
    
    def _test_attention_heads(self) -> Dict[str, Any]:
        """Test different attention head configurations."""
        
        results = {}
        
        for num_heads in self.config.attention_head_counts:
            self.logger.info(f"Testing {num_heads} attention heads")
            
            # Create modified config
            config = self.base_model_config.copy()
            config['num_attention_heads'] = num_heads
            
            # Ensure hidden_dim is divisible by num_heads
            hidden_dim = config.get('text_hidden_dim', 768)
            if hidden_dim % num_heads != 0:
                # Adjust hidden dimension to be divisible
                adjusted_dim = ((hidden_dim // num_heads) + 1) * num_heads
                config['text_hidden_dim'] = adjusted_dim
                config['fusion_output_dim'] = adjusted_dim
            
            experiment_name = f"attention_heads_{num_heads}"
            
            try:
                result = self._test_architecture_config(config, experiment_name)
                results[experiment_name] = result
                
                # Calculate performance drop from baseline
                baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
                current_f1 = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - current_f1
                
                self.logger.info(f"  {num_heads} heads: F1 = {current_f1:.4f} (drop: {performance_drop:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to test {num_heads} attention heads: {e}")
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config
                }
        
        return results
    
    def _test_encoder_freezing(self) -> Dict[str, Any]:
        """Test encoder freezing strategies."""
        
        results = {}
        
        freezing_strategies = [
            {'freeze_text_encoder': True, 'freeze_other_encoders': False},
            {'freeze_text_encoder': False, 'freeze_other_encoders': True},
            {'freeze_text_encoder': True, 'freeze_other_encoders': True},
            {'freeze_text_encoder': False, 'freeze_other_encoders': False}  # Baseline
        ]
        
        for i, strategy in enumerate(freezing_strategies):
            strategy_name = f"freeze_text_{strategy['freeze_text_encoder']}_others_{strategy['freeze_other_encoders']}"
            self.logger.info(f"Testing encoder freezing: {strategy_name}")
            
            # Create modified config
            config = self.base_model_config.copy()
            config.update(strategy)
            
            experiment_name = f"encoder_freezing_{i}"
            
            try:
                result = self._test_architecture_config(config, experiment_name)
                result['freezing_strategy'] = strategy
                results[experiment_name] = result
                
                # Calculate performance drop from baseline
                baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
                current_f1 = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - current_f1
                
                self.logger.info(f"  {strategy_name}: F1 = {current_f1:.4f} (drop: {performance_drop:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to test encoder freezing {strategy_name}: {e}")
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config,
                    'freezing_strategy': strategy
                }
        
        return results
    
    def _test_dropout_variations(self) -> Dict[str, Any]:
        """Test different dropout rates."""
        
        results = {}
        
        for dropout_rate in self.config.dropout_rates:
            self.logger.info(f"Testing dropout rate: {dropout_rate}")
            
            # Create modified config
            config = self.base_model_config.copy()
            config['dropout_rate'] = dropout_rate
            
            experiment_name = f"dropout_{dropout_rate}"
            
            try:
                result = self._test_architecture_config(config, experiment_name)
                results[experiment_name] = result
                
                # Calculate performance drop from baseline
                baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
                current_f1 = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - current_f1
                
                self.logger.info(f"  Dropout {dropout_rate}: F1 = {current_f1:.4f} (drop: {performance_drop:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to test dropout {dropout_rate}: {e}")
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config
                }
        
        return results
    
    def _test_layer_depths(self) -> Dict[str, Any]:
        """Test different layer depth configurations."""
        
        results = {}
        
        for i, depth_config in enumerate(self.config.layer_depth_configs):
            depth_name = f"layers_{depth_config['num_transformer_layers']}_fusion_{depth_config['num_fusion_layers']}"
            self.logger.info(f"Testing layer depth: {depth_name}")
            
            # Create modified config
            config = self.base_model_config.copy()
            config.update(depth_config)
            
            experiment_name = f"layer_depth_{i}"
            
            try:
                result = self._test_architecture_config(config, experiment_name)
                result['depth_config'] = depth_config
                results[experiment_name] = result
                
                # Calculate performance drop from baseline
                baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
                current_f1 = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - current_f1
                
                self.logger.info(f"  {depth_name}: F1 = {current_f1:.4f} (drop: {performance_drop:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to test layer depth {depth_name}: {e}")
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config,
                    'depth_config': depth_config
                }
        
        return results
    
    def _test_architecture_config(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Test a specific architecture configuration."""
        
        experiment_start_time = time.time()
        
        # Create and train model
        model = self.model_factory(config)
        model.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config.max_epochs):
            # Training
            model.train()
            train_metrics = self._train_epoch(model, optimizer)
            
            # Validation
            model.eval()
            val_metrics = self._evaluate_epoch(model, 'val')
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            training_history.append(epoch_metrics)
            
            current_f1 = val_metrics.get('val_f1', 0.0)
            
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                
                # Save best model if requested
                if self.config.save_all_models:
                    model_path = self.output_dir / f"model_{experiment_name}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'metrics': epoch_metrics,
                        'epoch': epoch
                    }, model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                break
        
        # Test evaluation
        test_metrics = {}
        if 'test' in self.data_loaders:
            test_metrics = self._evaluate_epoch(model, 'test')
        
        experiment_time = time.time() - experiment_start_time
        
        # Compile result
        result = {
            'experiment_name': experiment_name,
            'config': config,
            'training_time': experiment_time,
            'best_val_f1': best_val_f1,
            'status': 'completed',
            'training_history': training_history,
            **test_metrics
        }
        
        # Add final metrics from best epoch
        if training_history:
            best_epoch_idx = np.argmax([h.get('val_f1', 0) for h in training_history])
            best_metrics = training_history[best_epoch_idx]
            for metric in self.config.metrics_to_track:
                val_key = f'val_{metric}'
                if val_key in best_metrics:
                    result[val_key] = best_metrics[val_key]
        
        return result
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.data_loaders['train']:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'train_loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _evaluate_epoch(self, model: nn.Module, split: str) -> Dict[str, float]:
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
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze ablation study results."""
        
        analysis = {}
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v.get('status') == 'completed'}
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        baseline_f1 = self.baseline_performance.get('val_f1', 0.0)
        
        # Component importance analysis
        analysis['component_importance'] = self._analyze_component_importance(successful_results, baseline_f1)
        
        # Best and worst configurations
        analysis['best_configuration'] = self._find_best_configuration(successful_results)
        analysis['worst_configuration'] = self._find_worst_configuration(successful_results)
        
        # Performance drops by component type
        analysis['performance_drops'] = self._analyze_performance_drops(successful_results, baseline_f1)
        
        # Recommendations
        analysis['recommendations'] = self._generate_recommendations(successful_results, baseline_f1)
        
        return analysis
    
    def _analyze_component_importance(self, results: Dict[str, Any], baseline_f1: float) -> Dict[str, Any]:
        """Analyze importance of different architectural components."""
        
        component_analysis = {}
        
        # Group results by component type
        component_groups = {
            'fusion_strategies': [k for k in results.keys() if k.startswith('fusion_')],
            'attention_heads': [k for k in results.keys() if k.startswith('attention_heads_')],
            'encoder_freezing': [k for k in results.keys() if k.startswith('encoder_freezing_')],
            'dropout_rates': [k for k in results.keys() if k.startswith('dropout_')],
            'layer_depths': [k for k in results.keys() if k.startswith('layer_depth_')]
        }
        
        for component_type, experiment_keys in component_groups.items():
            if not experiment_keys:
                continue
            
            component_results = []
            for key in experiment_keys:
                result = results[key]
                f1_score = result.get('val_f1', 0.0)
                performance_drop = baseline_f1 - f1_score
                
                component_results.append({
                    'experiment': key,
                    'f1_score': f1_score,
                    'performance_drop': performance_drop,
                    'config': result.get('config', {})
                })
            
            # Sort by performance drop (ascending = less important to remove)
            component_results.sort(key=lambda x: x['performance_drop'])
            
            component_analysis[component_type] = {
                'results': component_results,
                'most_important': component_results[-1] if component_results else None,  # Highest drop when removed
                'least_important': component_results[0] if component_results else None,   # Lowest drop when removed
                'average_drop': np.mean([r['performance_drop'] for r in component_results]) if component_results else 0.0
            }
        
        return component_analysis
    
    def _find_best_configuration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing architecture configuration."""
        
        best_config = None
        best_score = -1.0
        
        for experiment_name, result in results.items():
            score = result.get('val_f1', 0.0)
            if score > best_score:
                best_score = score
                best_config = {
                    'experiment': experiment_name,
                    'score': score,
                    'config': result.get('config', {}),
                    'result': result
                }
        
        return best_config
    
    def _find_worst_configuration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the worst performing architecture configuration."""
        
        worst_config = None
        worst_score = float('inf')
        
        for experiment_name, result in results.items():
            score = result.get('val_f1', 0.0)
            if score < worst_score:
                worst_score = score
                worst_config = {
                    'experiment': experiment_name,
                    'score': score,
                    'config': result.get('config', {}),
                    'result': result
                }
        
        return worst_config
    
    def _analyze_performance_drops(self, results: Dict[str, Any], baseline_f1: float) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze performance drops for different component types."""
        
        drops_by_type = {}
        
        # Group by component type and calculate drops
        for experiment_name, result in results.items():
            if experiment_name == 'baseline':
                continue
            
            f1_score = result.get('val_f1', 0.0)
            drop = baseline_f1 - f1_score
            
            # Determine component type
            if experiment_name.startswith('fusion_'):
                component_type = 'fusion_strategies'
            elif experiment_name.startswith('attention_heads_'):
                component_type = 'attention_heads'
            elif experiment_name.startswith('encoder_freezing_'):
                component_type = 'encoder_freezing'
            elif experiment_name.startswith('dropout_'):
                component_type = 'dropout_rates'
            elif experiment_name.startswith('layer_depth_'):
                component_type = 'layer_depths'
            else:
                component_type = 'other'
            
            if component_type not in drops_by_type:
                drops_by_type[component_type] = []
            
            drops_by_type[component_type].append((experiment_name, drop))
        
        # Sort each type by drop magnitude
        for component_type in drops_by_type:
            drops_by_type[component_type].sort(key=lambda x: x[1], reverse=True)
        
        return drops_by_type
    
    def _generate_recommendations(self, results: Dict[str, Any], baseline_f1: float) -> List[str]:
        """Generate architectural recommendations based on results."""
        
        recommendations = []
        
        # Find components with minimal performance impact
        minimal_impact_threshold = 0.01  # 1% F1 drop or less
        
        for experiment_name, result in results.items():
            if experiment_name == 'baseline':
                continue
            
            f1_score = result.get('val_f1', 0.0)
            drop = baseline_f1 - f1_score
            
            if drop <= minimal_impact_threshold:
                if 'fusion_' in experiment_name:
                    fusion_strategy = experiment_name.replace('fusion_', '')
                    recommendations.append(f"Fusion strategy '{fusion_strategy}' has minimal impact (<1% F1 drop)")
                elif 'dropout_' in experiment_name:
                    dropout_rate = experiment_name.replace('dropout_', '')
                    recommendations.append(f"Dropout rate {dropout_rate} maintains performance well")
        
        # Find critical components (large performance drops)
        critical_threshold = 0.05  # 5% F1 drop or more
        
        for experiment_name, result in results.items():
            if experiment_name == 'baseline':
                continue
            
            f1_score = result.get('val_f1', 0.0)
            drop = baseline_f1 - f1_score
            
            if drop >= critical_threshold:
                if 'fusion_' in experiment_name:
                    fusion_strategy = experiment_name.replace('fusion_', '')
                    recommendations.append(f"Fusion strategy '{fusion_strategy}' is critical (>{critical_threshold*100}% F1 drop)")
                elif 'attention_heads_' in experiment_name:
                    num_heads = experiment_name.replace('attention_heads_', '')
                    recommendations.append(f"Using {num_heads} attention heads significantly impacts performance")
        
        if not recommendations:
            recommendations.append("All architectural components tested have moderate impact on performance")
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ablation study results."""
        
        # Save main results
        results_file = self.output_dir / "architecture_ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary = self._create_summary_report(results)
        summary_file = self.output_dir / "architecture_ablation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(results)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary report."""
        
        lines = []
        lines.append("=" * 60)
        lines.append("FACTCHECK-MM ARCHITECTURE ABLATION STUDY REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Task: {results['task_name']}")
        lines.append(f"Study time: {results['study_time']:.2f} seconds")
        lines.append(f"Total experiments: {results['total_experiments']}")
        lines.append(f"Successful experiments: {results['successful_experiments']}")
        lines.append("")
        
        # Baseline performance
        baseline = results.get('baseline_performance', {})
        baseline_f1 = baseline.get('val_f1', 0.0)
        lines.append(f"Baseline F1 Score: {baseline_f1:.4f}")
        lines.append("")
        
        # Best and worst configurations
        analysis = results.get('analysis', {})
        
        if 'best_configuration' in analysis and analysis['best_configuration']:
            best = analysis['best_configuration']
            lines.append("BEST PERFORMING CONFIGURATION:")
            lines.append("-" * 30)
            lines.append(f"Experiment: {best['experiment']}")
            lines.append(f"F1 Score: {best['score']:.4f}")
            lines.append(f"Improvement over baseline: {best['score'] - baseline_f1:.4f}")
            lines.append("")
        
        if 'worst_configuration' in analysis and analysis['worst_configuration']:
            worst = analysis['worst_configuration']
            lines.append("WORST PERFORMING CONFIGURATION:")
            lines.append("-" * 30)
            lines.append(f"Experiment: {worst['experiment']}")
            lines.append(f"F1 Score: {worst['score']:.4f}")
            lines.append(f"Drop from baseline: {baseline_f1 - worst['score']:.4f}")
            lines.append("")
        
        # Component importance
        if 'component_importance' in analysis:
            lines.append("COMPONENT IMPORTANCE ANALYSIS:")
            lines.append("-" * 30)
            
            for component_type, component_data in analysis['component_importance'].items():
                if component_data and 'average_drop' in component_data:
                    avg_drop = component_data['average_drop']
                    lines.append(f"{component_type.replace('_', ' ').title()}: avg drop = {avg_drop:.4f}")
            lines.append("")
        
        # Recommendations
        if 'recommendations' in analysis:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 30)
            for i, rec in enumerate(analysis['recommendations']):
                lines.append(f"{i+1}. {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Create visualization plots."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            successful_results = {k: v for k, v in results['individual_results'].items() 
                                if v.get('status') == 'completed'}
            baseline_f1 = results['baseline_performance'].get('val_f1', 0.0)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Performance by experiment
            experiment_names = list(successful_results.keys())
            f1_scores = [successful_results[name].get('val_f1', 0) for name in experiment_names]
            
            axes[0, 0].bar(range(len(experiment_names)), f1_scores, color='skyblue')
            axes[0, 0].axhline(y=baseline_f1, color='red', linestyle='--', label=f'Baseline: {baseline_f1:.3f}')
            axes[0, 0].set_title('Architecture Ablation Results')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].set_xlabel('Experiment')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Performance drops
            performance_drops = [baseline_f1 - score for score in f1_scores]
            
            axes[0, 1].bar(range(len(experiment_names)), performance_drops, 
                          color=['red' if drop > 0 else 'green' for drop in performance_drops])
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].set_title('Performance Drops from Baseline')
            axes[0, 1].set_ylabel('F1 Score Drop')
            axes[0, 1].set_xlabel('Experiment')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Component type analysis
            analysis = results.get('analysis', {})
            if 'component_importance' in analysis:
                component_types = []
                avg_drops = []
                
                for comp_type, comp_data in analysis['component_importance'].items():
                    if comp_data and 'average_drop' in comp_data:
                        component_types.append(comp_type.replace('_', ' ').title())
                        avg_drops.append(comp_data['average_drop'])
                
                if component_types:
                    axes[1, 0].bar(component_types, avg_drops, color='lightcoral')
                    axes[1, 0].set_title('Average Performance Drop by Component Type')
                    axes[1, 0].set_ylabel('Average F1 Drop')
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Training time vs performance
            training_times = [successful_results[name].get('training_time', 0) for name in experiment_names]
            
            axes[1, 1].scatter(training_times, f1_scores, alpha=0.7, color='gold')
            axes[1, 1].axhline(y=baseline_f1, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Training Time vs Performance')
            axes[1, 1].set_xlabel('Training Time (s)')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / "architecture_ablation_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualizations saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")


def main():
    """Example usage of architecture ablation study."""
    
    # Example model factory
    def create_model(config):
        from sarcasm_detection.models import MultimodalSarcasmModel
        return MultimodalSarcasmModel(config)
    
    # Example data loaders
    from tests.fixtures.mock_models import create_mock_dataloader, create_mock_dataset
    
    data_loaders = {
        'train': create_mock_dataloader(create_mock_dataset("multimodal_sarcasm", 100), batch_size=8),
        'val': create_mock_dataloader(create_mock_dataset("multimodal_sarcasm", 30), batch_size=8),
        'test': create_mock_dataloader(create_mock_dataset("multimodal_sarcasm", 20), batch_size=8)
    }
    
    # Base model configuration
    base_config = {
        'modalities': ['text', 'audio', 'image'],
        'fusion_strategy': 'cross_modal_attention',
        'text_hidden_dim': 512,
        'audio_hidden_dim': 256,
        'image_hidden_dim': 256,
        'fusion_output_dim': 512,
        'num_classes': 2,
        'dropout_rate': 0.1,
        'num_attention_heads': 8
    }
    
    # Configuration (smaller for example)
    config = ArchitectureAblationConfig(
        fusion_strategies_to_test=['concatenation', 'cross_modal_attention'],
        attention_head_counts=[4, 8],
        dropout_rates=[0.1, 0.2],
        max_epochs=3
    )
    
    # Run ablation study
    study = ArchitectureAblationStudy(
        config=config,
        model_factory=create_model,
        data_loaders=data_loaders,
        base_model_config=base_config,
        task_name="sarcasm_detection"
    )
    
    results = study.run_ablation_study()
    
    print("Architecture ablation study completed!")
    print(f"Best configuration: {results['analysis']['best_configuration']['experiment']}")


if __name__ == "__main__":
    main()
