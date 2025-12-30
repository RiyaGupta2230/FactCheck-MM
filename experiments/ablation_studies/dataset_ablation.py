#!/usr/bin/env python3
"""
Dataset Ablation Study for FactCheck-MM

Systematic evaluation of dataset contributions to model performance.
Tests training on individual datasets and combinations to understand 
which datasets contribute most to final performance.
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
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.metrics import MetricsComputer


@dataclass
class DatasetAblationConfig:
    """Configuration for dataset ablation study."""
    
    # Available datasets for ablation
    available_datasets: List[str] = field(default_factory=lambda: [
        'sarc', 'mmsd2', 'mustard', 'sarcnet', 'sarcasm_headlines'
    ])
    
    # Ablation strategy
    test_individual_datasets: bool = True
    test_dataset_pairs: bool = True
    test_dataset_removal: bool = True  # Remove one dataset at a time from full set
    
    # Training configuration
    max_epochs: int = 10
    early_stopping_patience: int = 3
    
    # Data sampling configuration
    max_samples_per_dataset: Optional[int] = 10000  # Limit for computational efficiency
    balance_datasets: bool = True  # Balance dataset sizes
    
    # Evaluation configuration
    metrics_to_track: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 3
    
    # Resource management
    device: str = "auto"
    
    # Output configuration
    save_all_models: bool = False
    create_visualizations: bool = True


class DatasetAblationStudy:
    """Conducts systematic dataset ablation studies."""
    
    def __init__(
        self,
        config: DatasetAblationConfig,
        model_factory,
        dataset_loaders: Dict[str, Dict[str, DataLoader]],  # {dataset_name: {split: dataloader}}
        base_model_config: Dict[str, Any],
        task_name: str = "sarcasm_detection",
        output_dir: str = "outputs/experiments/dataset_ablation"
    ):
        """
        Initialize dataset ablation study.
        
        Args:
            config: Ablation configuration
            model_factory: Function to create models
            dataset_loaders: Dictionary mapping dataset names to their data loaders
            base_model_config: Base model configuration
            task_name: Name of the task
            output_dir: Output directory for results
        """
        self.config = config
        self.model_factory = model_factory
        self.dataset_loaders = dataset_loaders
        self.base_model_config = base_model_config.copy()
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("DatasetAblationStudy")
        
        # Device setup
        self.device = self._setup_device()
        
        # Results storage
        self.results = {}
        self.dataset_statistics = {}
        
        # Analyze available datasets
        self._analyze_datasets()
        
        self.logger.info(f"Initialized dataset ablation study for {task_name}")
        self.logger.info(f"Available datasets: {list(self.dataset_loaders.keys())}")
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
    
    def _analyze_datasets(self):
        """Analyze dataset characteristics."""
        
        for dataset_name, splits in self.dataset_loaders.items():
            stats = {}
            
            for split_name, dataloader in splits.items():
                # Count samples
                num_samples = len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else 0
                
                # Sample a few batches to analyze data characteristics
                sample_batches = []
                for i, batch in enumerate(dataloader):
                    sample_batches.append(batch)
                    if i >= 2:  # Only sample a few batches
                        break
                
                # Analyze label distribution if available
                label_counts = defaultdict(int)
                total_samples_seen = 0
                
                for batch in sample_batches:
                    if 'labels' in batch:
                        labels = batch['labels']
                    elif 'label' in batch:
                        labels = batch['label']
                    else:
                        labels = None
                    
                    if labels is not None:
                        for label in labels:
                            label_counts[label.item()] += 1
                            total_samples_seen += 1
                
                stats[split_name] = {
                    'num_samples': num_samples,
                    'label_distribution': dict(label_counts),
                    'samples_analyzed': total_samples_seen
                }
            
            self.dataset_statistics[dataset_name] = stats
            
            # Log dataset info
            train_samples = stats.get('train', {}).get('num_samples', 0)
            val_samples = stats.get('val', {}).get('num_samples', 0)
            test_samples = stats.get('test', {}).get('num_samples', 0)
            
            self.logger.info(f"Dataset {dataset_name}: train={train_samples}, val={val_samples}, test={test_samples}")
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run the complete dataset ablation study."""
        
        self.logger.info("Starting dataset ablation study")
        study_start_time = time.time()
        
        # Test individual datasets
        if self.config.test_individual_datasets:
            self.logger.info("Testing individual datasets...")
            individual_results = self._test_individual_datasets()
            self.results.update(individual_results)
        
        # Test dataset pairs
        if self.config.test_dataset_pairs:
            self.logger.info("Testing dataset pairs...")
            pair_results = self._test_dataset_pairs()
            self.results.update(pair_results)
        
        # Test dataset removal (leave-one-out)
        if self.config.test_dataset_removal:
            self.logger.info("Testing dataset removal...")
            removal_results = self._test_dataset_removal()
            self.results.update(removal_results)
        
        # Test full dataset combination
        self.logger.info("Testing full dataset combination...")
        full_result = self._test_dataset_combination(list(self.dataset_loaders.keys()), "full_dataset")
        self.results["full_dataset"] = full_result
        
        study_time = time.time() - study_start_time
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Compile final results
        final_results = {
            'study_time': study_time,
            'task_name': self.task_name,
            'total_experiments': len(self.results),
            'successful_experiments': len([r for r in self.results.values() if r.get('status') != 'failed']),
            'dataset_statistics': self.dataset_statistics,
            'individual_results': self.results,
            'analysis': analysis,
            'config': self.config.__dict__
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info(f"Dataset ablation study completed in {study_time:.2f}s")
        
        return final_results
    
    def _test_individual_datasets(self) -> Dict[str, Any]:
        """Test individual datasets."""
        
        results = {}
        
        for dataset_name in self.config.available_datasets:
            if dataset_name not in self.dataset_loaders:
                self.logger.warning(f"Dataset {dataset_name} not available, skipping")
                continue
            
            self.logger.info(f"Testing individual dataset: {dataset_name}")
            
            try:
                result = self._test_dataset_combination([dataset_name], f"individual_{dataset_name}")
                results[f"individual_{dataset_name}"] = result
                
                f1_score = result.get('val_f1', 0.0)
                self.logger.info(f"  {dataset_name}: F1 = {f1_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to test individual dataset {dataset_name}: {e}")
                results[f"individual_{dataset_name}"] = {
                    'datasets': [dataset_name],
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _test_dataset_pairs(self) -> Dict[str, Any]:
        """Test dataset pairs."""
        
        results = {}
        available_datasets = [d for d in self.config.available_datasets if d in self.dataset_loaders]
        
        # Test all pairs
        for i, dataset1 in enumerate(available_datasets):
            for j, dataset2 in enumerate(available_datasets):
                if i >= j:  # Avoid duplicates and self-pairs
                    continue
                
                pair_name = f"{dataset1}+{dataset2}"
                self.logger.info(f"Testing dataset pair: {pair_name}")
                
                try:
                    result = self._test_dataset_combination([dataset1, dataset2], f"pair_{pair_name}")
                    results[f"pair_{pair_name}"] = result
                    
                    f1_score = result.get('val_f1', 0.0)
                    self.logger.info(f"  {pair_name}: F1 = {f1_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to test dataset pair {pair_name}: {e}")
                    results[f"pair_{pair_name}"] = {
                        'datasets': [dataset1, dataset2],
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return results
    
    def _test_dataset_removal(self) -> Dict[str, Any]:
        """Test removing one dataset at a time (leave-one-out)."""
        
        results = {}
        available_datasets = [d for d in self.config.available_datasets if d in self.dataset_loaders]
        
        if len(available_datasets) < 2:
            self.logger.warning("Need at least 2 datasets for removal testing")
            return results
        
        for dataset_to_remove in available_datasets:
            remaining_datasets = [d for d in available_datasets if d != dataset_to_remove]
            experiment_name = f"remove_{dataset_to_remove}"
            
            self.logger.info(f"Testing removal of {dataset_to_remove}")
            
            try:
                result = self._test_dataset_combination(remaining_datasets, experiment_name)
                result['removed_dataset'] = dataset_to_remove
                results[experiment_name] = result
                
                f1_score = result.get('val_f1', 0.0)
                self.logger.info(f"  Without {dataset_to_remove}: F1 = {f1_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to test removal of {dataset_to_remove}: {e}")
                results[experiment_name] = {
                    'datasets': remaining_datasets,
                    'removed_dataset': dataset_to_remove,
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _test_dataset_combination(self, dataset_names: List[str], experiment_name: str) -> Dict[str, Any]:
        """Test a specific combination of datasets."""
        
        experiment_start_time = time.time()
        
        # Create combined data loaders
        combined_loaders = self._create_combined_loaders(dataset_names)
        
        # Create and train model
        model = self.model_factory(self.base_model_config)
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
            train_metrics = self._train_epoch(model, optimizer, combined_loaders['train'])
            
            # Validation
            model.eval()
            val_metrics = self._evaluate_epoch(model, combined_loaders['val'])
            
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
                        'datasets': dataset_names,
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
        if 'test' in combined_loaders:
            test_metrics = self._evaluate_epoch(model, combined_loaders['test'])
        
        experiment_time = time.time() - experiment_start_time
        
        # Calculate dataset statistics
        dataset_stats = {}
        total_train_samples = 0
        for dataset_name in dataset_names:
            if dataset_name in self.dataset_statistics:
                train_samples = self.dataset_statistics[dataset_name].get('train', {}).get('num_samples', 0)
                dataset_stats[dataset_name] = train_samples
                total_train_samples += train_samples
        
        # Compile result
        result = {
            'experiment_name': experiment_name,
            'datasets': dataset_names,
            'num_datasets': len(dataset_names),
            'total_train_samples': total_train_samples,
            'dataset_sample_counts': dataset_stats,
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
    
    def _create_combined_loaders(self, dataset_names: List[str]) -> Dict[str, DataLoader]:
        """Create combined data loaders from multiple datasets."""
        
        combined_loaders = {}
        
        for split in ['train', 'val', 'test']:
            datasets_for_split = []
            
            for dataset_name in dataset_names:
                if dataset_name in self.dataset_loaders and split in self.dataset_loaders[dataset_name]:
                    datasets_for_split.append(self.dataset_loaders[dataset_name][split].dataset)
            
            if datasets_for_split:
                # Combine datasets
                if len(datasets_for_split) == 1:
                    combined_dataset = datasets_for_split[0]
                else:
                    combined_dataset = ConcatDataset(datasets_for_split)
                
                # Apply sampling if needed
                if self.config.max_samples_per_dataset and len(combined_dataset) > self.config.max_samples_per_dataset:
                    indices = np.random.choice(
                        len(combined_dataset),
                        self.config.max_samples_per_dataset,
                        replace=False
                    )
                    combined_dataset = Subset(combined_dataset, indices)
                
                # Create data loader
                batch_size = 8 if split == 'train' else 16
                combined_loaders[split] = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    collate_fn=self._collate_fn
                )
        
        return combined_loaders
    
    def _collate_fn(self, batch):
        """Collate function for combining different dataset formats."""
        
        # Simple collation - assumes all datasets have similar format
        if not batch:
            return {}
        
        # Get all keys from first item
        keys = batch[0].keys() if isinstance(batch[0], dict) else []
        
        collated = {}
        for key in keys:
            values = [item[key] for item in batch if key in item]
            
            if values:
                if isinstance(values[0], torch.Tensor):
                    try:
                        collated[key] = torch.stack(values)
                    except:
                        # If stacking fails, pad sequences
                        collated[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
                elif isinstance(values[0], (int, float)):
                    collated[key] = torch.tensor(values)
                else:
                    collated[key] = values
        
        return collated
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
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
    
    def _evaluate_epoch(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch."""
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
        
        # Add prefix for validation/test
        return {f'val_{k}': v for k, v in metrics.items()}
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze dataset ablation results."""
        
        analysis = {}
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v.get('status') == 'completed'}
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        # Individual dataset contributions
        analysis['individual_dataset_performance'] = self._analyze_individual_performance(successful_results)
        
        # Dataset importance ranking
        analysis['dataset_importance_ranking'] = self._rank_dataset_importance(successful_results)
        
        # Synergy analysis
        analysis['dataset_synergy'] = self._analyze_dataset_synergy(successful_results)
        
        # Data efficiency analysis
        analysis['data_efficiency'] = self._analyze_data_efficiency(successful_results)
        
        # Best and worst combinations
        analysis['best_combination'] = self._find_best_combination(successful_results)
        analysis['worst_combination'] = self._find_worst_combination(successful_results)
        
        return analysis
    
    def _analyze_individual_performance(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze individual dataset performance."""
        
        individual_performance = {}
        
        for experiment_name, result in results.items():
            if experiment_name.startswith('individual_'):
                dataset_name = experiment_name.replace('individual_', '')
                
                performance = {}
                for metric in self.config.metrics_to_track:
                    val_key = f'val_{metric}'
                    if val_key in result:
                        performance[metric] = result[val_key]
                
                # Add training info
                performance['training_samples'] = result.get('total_train_samples', 0)
                performance['training_time'] = result.get('training_time', 0)
                
                individual_performance[dataset_name] = performance
        
        return individual_performance
    
    def _rank_dataset_importance(self, results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank datasets by their individual performance."""
        
        individual_perf = self._analyze_individual_performance(results)
        
        # Rank by F1 score
        ranking_metric = 'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]
        
        dataset_scores = []
        for dataset_name, performance in individual_perf.items():
            score = performance.get(ranking_metric, 0.0)
            dataset_scores.append((dataset_name, score))
        
        # Sort by score (descending)
        dataset_scores.sort(key=lambda x: x[1], reverse=True)
        
        return dataset_scores
    
    def _analyze_dataset_synergy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synergy between datasets."""
        
        synergy_analysis = {}
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        # Get individual dataset performances
        individual_perf = {}
        for experiment_name, result in results.items():
            if experiment_name.startswith('individual_'):
                dataset_name = experiment_name.replace('individual_', '')
                individual_perf[dataset_name] = result.get(ranking_metric, 0.0)
        
        # Analyze pair combinations for synergy
        for experiment_name, result in results.items():
            if experiment_name.startswith('pair_'):
                datasets = result.get('datasets', [])
                
                if len(datasets) == 2:
                    dataset1, dataset2 = datasets
                    
                    # Calculate expected additive performance
                    ind1 = individual_perf.get(dataset1, 0.0)
                    ind2 = individual_perf.get(dataset2, 0.0)
                    expected_additive = (ind1 + ind2) / 2
                    
                    # Actual combined performance
                    actual_combined = result.get(ranking_metric, 0.0)
                    
                    # Synergy score
                    synergy_score = actual_combined - expected_additive
                    
                    pair_name = f"{dataset1}+{dataset2}"
                    synergy_analysis[pair_name] = {
                        'datasets': datasets,
                        'individual_performances': [ind1, ind2],
                        'expected_additive': expected_additive,
                        'actual_combined': actual_combined,
                        'synergy_score': synergy_score
                    }
        
        return synergy_analysis
    
    def _analyze_data_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data efficiency (performance per sample)."""
        
        efficiency_analysis = {}
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        for experiment_name, result in results.items():
            if result.get('status') == 'completed':
                performance = result.get(ranking_metric, 0.0)
                num_samples = result.get('total_train_samples', 1)
                training_time = result.get('training_time', 0.0)
                
                # Efficiency metrics
                efficiency_analysis[experiment_name] = {
                    'performance_per_sample': performance / num_samples if num_samples > 0 else 0,
                    'performance_per_minute': performance / (training_time / 60) if training_time > 0 else 0,
                    'samples_per_f1_point': num_samples / performance if performance > 0 else float('inf')
                }
        
        return efficiency_analysis
    
    def _find_best_combination(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing dataset combination."""
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        best_combo = None
        best_score = -1.0
        
        for experiment_name, result in results.items():
            score = result.get(ranking_metric, 0.0)
            if score > best_score:
                best_score = score
                best_combo = {
                    'experiment': experiment_name,
                    'datasets': result.get('datasets', []),
                    'score': score,
                    'result': result
                }
        
        return best_combo
    
    def _find_worst_combination(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the worst performing dataset combination."""
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        worst_combo = None
        worst_score = float('inf')
        
        for experiment_name, result in results.items():
            score = result.get(ranking_metric, 0.0)
            if score < worst_score:
                worst_score = score
                worst_combo = {
                    'experiment': experiment_name,
                    'datasets': result.get('datasets', []),
                    'score': score,
                    'result': result
                }
        
        return worst_combo
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ablation study results."""
        
        # Save main results
        results_file = self.output_dir / "dataset_ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary = self._create_summary_report(results)
        summary_file = self.output_dir / "dataset_ablation_summary.txt"
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
        lines.append("FACTCHECK-MM DATASET ABLATION STUDY REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Task: {results['task_name']}")
        lines.append(f"Study time: {results['study_time']:.2f} seconds")
        lines.append(f"Total experiments: {results['total_experiments']}")
        lines.append(f"Successful experiments: {results['successful_experiments']}")
        lines.append("")
        
        # Dataset statistics
        lines.append("DATASET STATISTICS:")
        lines.append("-" * 30)
        for dataset_name, stats in results['dataset_statistics'].items():
            train_samples = stats.get('train', {}).get('num_samples', 0)
            lines.append(f"{dataset_name}: {train_samples:,} training samples")
        lines.append("")
        
        # Best and worst combinations
        analysis = results.get('analysis', {})
        
        if 'best_combination' in analysis and analysis['best_combination']:
            best = analysis['best_combination']
            lines.append("BEST PERFORMING COMBINATION:")
            lines.append("-" * 30)
            lines.append(f"Datasets: {', '.join(best['datasets'])}")
            lines.append(f"F1 Score: {best['score']:.4f}")
            lines.append("")
        
        # Individual dataset ranking
        if 'dataset_importance_ranking' in analysis:
            lines.append("DATASET IMPORTANCE RANKING:")
            lines.append("-" * 30)
            for i, (dataset, score) in enumerate(analysis['dataset_importance_ranking']):
                lines.append(f"{i+1}. {dataset}: {score:.4f}")
            lines.append("")
        
        # Synergy analysis
        if 'dataset_synergy' in analysis:
            lines.append("TOP SYNERGISTIC DATASET PAIRS:")
            lines.append("-" * 30)
            synergy_data = analysis['dataset_synergy']
            if synergy_data:
                synergy_pairs = sorted(synergy_data.items(), key=lambda x: x[1]['synergy_score'], reverse=True)
                
                for pair_name, synergy_info in synergy_pairs[:3]:
                    lines.append(f"{pair_name}: synergy score = {synergy_info['synergy_score']:.4f}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Create visualization plots."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            analysis = results.get('analysis', {})
            successful_results = {k: v for k, v in results['individual_results'].items() 
                                if v.get('status') == 'completed'}
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Individual dataset performance
            if 'individual_dataset_performance' in analysis:
                individual_perf = analysis['individual_dataset_performance']
                datasets = list(individual_perf.keys())
                f1_scores = [individual_perf[d].get('f1', 0) for d in datasets]
                
                axes[0, 0].bar(datasets, f1_scores, color='skyblue')
                axes[0, 0].set_title('Individual Dataset Performance')
                axes[0, 0].set_ylabel('F1 Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Data efficiency
            if 'data_efficiency' in analysis:
                efficiency_data = analysis['data_efficiency']
                experiment_names = []
                efficiency_scores = []
                
                for exp_name, efficiency in efficiency_data.items():
                    if exp_name.startswith('individual_'):
                        experiment_names.append(exp_name.replace('individual_', ''))
                        efficiency_scores.append(efficiency.get('performance_per_sample', 0) * 1000)  # Scale for visibility
                
                if experiment_names:
                    axes[0, 1].bar(experiment_names, efficiency_scores, color='lightcoral')
                    axes[0, 1].set_title('Data Efficiency (F1 per 1000 samples)')
                    axes[0, 1].set_ylabel('Efficiency Score')
                    axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Performance vs dataset size
            dataset_sizes = []
            performances = []
            
            for exp_name, result in successful_results.items():
                if exp_name.startswith('individual_'):
                    size = result.get('total_train_samples', 0)
                    perf = result.get('val_f1', 0)
                    if size > 0:
                        dataset_sizes.append(size)
                        performances.append(perf)
            
            if dataset_sizes:
                axes[1, 0].scatter(dataset_sizes, performances, alpha=0.7, color='gold')
                axes[1, 0].set_title('Performance vs Dataset Size')
                axes[1, 0].set_xlabel('Training Samples')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Synergy scores
            if 'dataset_synergy' in analysis:
                synergy_data = analysis['dataset_synergy']
                if synergy_data:
                    pairs = list(synergy_data.keys())[:5]  # Top 5 pairs
                    synergy_scores = [synergy_data[pair]['synergy_score'] for pair in pairs]
                    
                    colors = ['green' if score > 0 else 'red' for score in synergy_scores]
                    axes[1, 1].bar(range(len(pairs)), synergy_scores, color=colors)
                    axes[1, 1].set_title('Dataset Synergy Scores')
                    axes[1, 1].set_ylabel('Synergy Score')
                    axes[1, 1].set_xticks(range(len(pairs)))
                    axes[1, 1].set_xticklabels([p.replace('+', '+\n') for p in pairs], rotation=45)
                    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / "dataset_ablation_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualizations saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")


def main():
    """Example usage of dataset ablation study."""
    
    # Example model factory
    def create_model(config):
        from sarcasm_detection.models import MultimodalSarcasmModel
        return MultimodalSarcasmModel(config)
    
    # Example dataset loaders
    from tests.fixtures.mock_models import create_mock_dataloader, create_mock_dataset
    
    # Create mock datasets
    dataset_loaders = {}
    for dataset_name in ['sarc', 'mmsd2', 'mustard']:
        dataset_loaders[dataset_name] = {
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
        'dropout_rate': 0.1
    }
    
    # Configuration (smaller for example)
    config = DatasetAblationConfig(
        available_datasets=['sarc', 'mmsd2', 'mustard'],
        test_individual_datasets=True,
        test_dataset_pairs=True,
        test_dataset_removal=True,
        max_epochs=3
    )
    
    # Run ablation study
    study = DatasetAblationStudy(
        config=config,
        model_factory=create_model,
        dataset_loaders=dataset_loaders,
        base_model_config=base_config,
        task_name="sarcasm_detection"
    )
    
    results = study.run_ablation_study()
    
    print("Dataset ablation study completed!")
    if results['analysis']['best_combination']:
        best_datasets = results['analysis']['best_combination']['datasets']
        print(f"Best dataset combination: {', '.join(best_datasets)}")


if __name__ == "__main__":
    main()