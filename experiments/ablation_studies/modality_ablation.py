#!/usr/bin/env python3
"""
Modality Ablation Study for FactCheck-MM

Systematic removal of modalities to understand their individual contributions
to multimodal model performance. Tests all possible modality combinations.
"""

import sys
import os
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
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
class ModalityAblationConfig:
    """Configuration for modality ablation study."""
    
    # Available modalities
    available_modalities: List[str] = field(default_factory=lambda: ['text', 'audio', 'image', 'video'])
    
    # Ablation strategy
    test_single_modalities: bool = True
    test_pair_combinations: bool = True  
    test_all_combinations: bool = False  # Can be expensive
    
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


class ModalityAblationStudy:
    """Conducts systematic modality ablation studies."""
    
    def __init__(
        self,
        config: ModalityAblationConfig,
        model_factory,
        data_loaders: Dict[str, DataLoader],
        base_model_config: Dict[str, Any],
        task_name: str = "multimodal_task",
        output_dir: str = "outputs/experiments/modality_ablation"
    ):
        """
        Initialize modality ablation study.
        
        Args:
            config: Ablation configuration
            model_factory: Function to create models with different modality configs
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
        self.logger = get_logger("ModalityAblationStudy")
        
        # Device setup
        self.device = self._setup_device()
        
        # Results storage
        self.results = {}
        self.modality_combinations = []
        
        # Generate modality combinations to test
        self._generate_modality_combinations()
        
        self.logger.info(f"Initialized modality ablation study for {task_name}")
        self.logger.info(f"Testing {len(self.modality_combinations)} modality combinations")
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
    
    def _generate_modality_combinations(self):
        """Generate all modality combinations to test."""
        
        modalities = self.config.available_modalities
        combinations = []
        
        # Single modalities
        if self.config.test_single_modalities:
            for modality in modalities:
                combinations.append([modality])
        
        # Pair combinations
        if self.config.test_pair_combinations:
            for pair in itertools.combinations(modalities, 2):
                combinations.append(list(pair))
        
        # All combinations (can be expensive)
        if self.config.test_all_combinations:
            for r in range(3, len(modalities) + 1):
                for combo in itertools.combinations(modalities, r):
                    combinations.append(list(combo))
        else:
            # Add full modality combination
            combinations.append(modalities.copy())
        
        # Remove duplicates and sort
        unique_combinations = []
        for combo in combinations:
            sorted_combo = sorted(combo)
            if sorted_combo not in unique_combinations:
                unique_combinations.append(sorted_combo)
        
        self.modality_combinations = unique_combinations
        
        self.logger.info(f"Generated {len(self.modality_combinations)} unique modality combinations")
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run the complete modality ablation study."""
        
        self.logger.info("Starting modality ablation study")
        
        study_start_time = time.time()
        
        for combo_idx, modalities in enumerate(self.modality_combinations):
            combo_name = "+".join(sorted(modalities))
            
            self.logger.info(f"Testing combination {combo_idx + 1}/{len(self.modality_combinations)}: {combo_name}")
            
            try:
                # Train and evaluate model with this modality combination
                result = self._test_modality_combination(modalities, combo_idx)
                self.results[combo_name] = result
                
                # Log key metrics
                for metric in self.config.metrics_to_track:
                    val_metric = result.get(f'val_{metric}', 0.0)
                    self.logger.info(f"  {metric}: {val_metric:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to test combination {combo_name}: {e}")
                self.results[combo_name] = {
                    'modalities': modalities,
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        study_time = time.time() - study_start_time
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Compile final results
        final_results = {
            'study_time': study_time,
            'task_name': self.task_name,
            'total_combinations': len(self.modality_combinations),
            'successful_combinations': len([r for r in self.results.values() if r.get('status') != 'failed']),
            'individual_results': self.results,
            'analysis': analysis,
            'config': self.config.__dict__
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info(f"Modality ablation study completed in {study_time:.2f}s")
        
        return final_results
    
    def _test_modality_combination(self, modalities: List[str], combo_idx: int) -> Dict[str, Any]:
        """Test a specific modality combination."""
        
        combination_start_time = time.time()
        
        # Create model configuration for this modality combination
        model_config = self.base_model_config.copy()
        model_config['modalities'] = modalities
        
        # Adjust fusion strategy if needed
        if len(modalities) == 1:
            model_config['fusion_strategy'] = 'single_modality'
        
        # Create and train model
        model = self.model_factory(model_config)
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
            train_metrics = self._train_epoch(model, optimizer, modalities)
            
            # Validation
            model.eval()
            val_metrics = self._evaluate_epoch(model, 'val', modalities)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            training_history.append(epoch_metrics)
            
            current_f1 = val_metrics.get('val_f1', 0.0)
            
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                
                # Save best model if requested
                if self.config.save_all_models:
                    model_path = self.output_dir / f"model_{'_'.join(sorted(modalities))}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'modalities': modalities,
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
            test_metrics = self._evaluate_epoch(model, 'test', modalities)
        
        combination_time = time.time() - combination_start_time
        
        # Compile result
        result = {
            'modalities': modalities,
            'num_modalities': len(modalities),
            'training_time': combination_time,
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
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, modalities: List[str]) -> Dict[str, float]:
        """Train for one epoch with specific modalities."""
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.data_loaders['train']:
            # Filter batch to only include specified modalities
            filtered_batch = self._filter_batch_modalities(batch, modalities)
            
            # Forward pass
            outputs = model(filtered_batch)
            
            # Compute loss
            if 'labels' in filtered_batch:
                targets = filtered_batch['labels']
            elif 'label' in filtered_batch:
                targets = filtered_batch['label']
            else:
                targets = filtered_batch[list(filtered_batch.keys())[-1]]
            
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
    
    def _evaluate_epoch(self, model: nn.Module, split: str, modalities: List[str]) -> Dict[str, float]:
        """Evaluate for one epoch with specific modalities."""
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_loaders[split]:
                # Filter batch to only include specified modalities
                filtered_batch = self._filter_batch_modalities(batch, modalities)
                
                # Forward pass
                outputs = model(filtered_batch)
                
                # Get predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                
                if 'labels' in filtered_batch:
                    targets = filtered_batch['labels']
                elif 'label' in filtered_batch:
                    targets = filtered_batch['label']
                else:
                    targets = filtered_batch[list(filtered_batch.keys())[-1]]
                
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
    
    def _filter_batch_modalities(self, batch: Dict[str, Any], modalities: List[str]) -> Dict[str, Any]:
        """Filter batch to only include specified modalities."""
        
        filtered_batch = {}
        
        for key, value in batch.items():
            # Always include labels/targets
            if key in ['labels', 'label']:
                filtered_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
                continue
            
            # Include modality-specific data
            if key in modalities:
                if isinstance(value, dict):
                    # Handle nested modality data (e.g., text with input_ids and attention_mask)
                    filtered_value = {}
                    for sub_key, sub_value in value.items():
                        filtered_value[sub_key] = sub_value.to(self.device) if isinstance(sub_value, torch.Tensor) else sub_value
                    filtered_batch[key] = filtered_value
                else:
                    filtered_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
            
            # Handle text-specific keys for text modality
            elif key in ['input_ids', 'attention_mask'] and 'text' in modalities:
                filtered_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        
        return filtered_batch
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze ablation study results."""
        
        analysis = {}
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v.get('status') == 'completed'}
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        # Performance by number of modalities
        analysis['performance_by_modality_count'] = self._analyze_by_modality_count(successful_results)
        
        # Individual modality contribution
        analysis['individual_modality_contribution'] = self._analyze_individual_modalities(successful_results)
        
        # Modality importance ranking
        analysis['modality_importance_ranking'] = self._rank_modality_importance(successful_results)
        
        # Best and worst combinations
        analysis['best_combination'] = self._find_best_combination(successful_results)
        analysis['worst_combination'] = self._find_worst_combination(successful_results)
        
        # Synergy analysis
        analysis['modality_synergy'] = self._analyze_modality_synergy(successful_results)
        
        return analysis
    
    def _analyze_by_modality_count(self, results: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        """Analyze performance by number of modalities."""
        
        by_count = {}
        
        for combo_name, result in results.items():
            num_modalities = result['num_modalities']
            
            if num_modalities not in by_count:
                by_count[num_modalities] = []
            
            by_count[num_modalities].append(result)
        
        # Compute statistics for each count
        count_stats = {}
        for count, results_list in by_count.items():
            metrics = {}
            for metric in self.config.metrics_to_track:
                val_key = f'val_{metric}'
                values = [r.get(val_key, 0.0) for r in results_list if val_key in r]
                if values:
                    metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            count_stats[count] = metrics
        
        return count_stats
    
    def _analyze_individual_modalities(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze individual modality performance."""
        
        individual_performance = {}
        
        for combo_name, result in results.items():
            modalities = result['modalities']
            
            # Only consider single modality results
            if len(modalities) == 1:
                modality = modalities[0]
                
                performance = {}
                for metric in self.config.metrics_to_track:
                    val_key = f'val_{metric}'
                    if val_key in result:
                        performance[metric] = result[val_key]
                
                individual_performance[modality] = performance
        
        return individual_performance
    
    def _rank_modality_importance(self, results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank modalities by their importance (single modality performance)."""
        
        individual_perf = self._analyze_individual_modalities(results)
        
        # Rank by F1 score (or first available metric)
        ranking_metric = 'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]
        
        modality_scores = []
        for modality, performance in individual_perf.items():
            score = performance.get(ranking_metric, 0.0)
            modality_scores.append((modality, score))
        
        # Sort by score (descending)
        modality_scores.sort(key=lambda x: x[1], reverse=True)
        
        return modality_scores
    
    def _find_best_combination(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing modality combination."""
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        best_combo = None
        best_score = -1.0
        
        for combo_name, result in results.items():
            score = result.get(ranking_metric, 0.0)
            if score > best_score:
                best_score = score
                best_combo = {
                    'combination': combo_name,
                    'modalities': result['modalities'],
                    'score': score,
                    'result': result
                }
        
        return best_combo
    
    def _find_worst_combination(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the worst performing modality combination."""
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        worst_combo = None
        worst_score = float('inf')
        
        for combo_name, result in results.items():
            score = result.get(ranking_metric, 0.0)
            if score < worst_score:
                worst_score = score
                worst_combo = {
                    'combination': combo_name,
                    'modalities': result['modalities'],
                    'score': score,
                    'result': result
                }
        
        return worst_combo
    
    def _analyze_modality_synergy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synergy between modalities."""
        
        synergy_analysis = {}
        
        ranking_metric = f"val_{'f1' if 'f1' in self.config.metrics_to_track else self.config.metrics_to_track[0]}"
        
        # Get individual modality performances
        individual_perf = {}
        for combo_name, result in results.items():
            if len(result['modalities']) == 1:
                modality = result['modalities'][0]
                individual_perf[modality] = result.get(ranking_metric, 0.0)
        
        # Analyze pair combinations for synergy
        for combo_name, result in results.items():
            modalities = result['modalities']
            
            if len(modalities) == 2:
                mod1, mod2 = modalities
                
                # Calculate expected additive performance
                ind1 = individual_perf.get(mod1, 0.0)
                ind2 = individual_perf.get(mod2, 0.0)
                expected_additive = (ind1 + ind2) / 2  # Simple average
                
                # Actual combined performance
                actual_combined = result.get(ranking_metric, 0.0)
                
                # Synergy score (positive = synergistic, negative = interference)
                synergy_score = actual_combined - expected_additive
                
                synergy_analysis[combo_name] = {
                    'modalities': modalities,
                    'individual_performances': [ind1, ind2],
                    'expected_additive': expected_additive,
                    'actual_combined': actual_combined,
                    'synergy_score': synergy_score
                }
        
        return synergy_analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ablation study results."""
        
        # Save main results
        results_file = self.output_dir / "modality_ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary = self._create_summary_report(results)
        summary_file = self.output_dir / "modality_ablation_summary.txt"
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
        lines.append("FACTCHECK-MM MODALITY ABLATION STUDY REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Task: {results['task_name']}")
        lines.append(f"Study time: {results['study_time']:.2f} seconds")
        lines.append(f"Total combinations tested: {results['total_combinations']}")
        lines.append(f"Successful experiments: {results['successful_combinations']}")
        lines.append("")
        
        # Best and worst combinations
        analysis = results.get('analysis', {})
        
        if 'best_combination' in analysis and analysis['best_combination']:
            best = analysis['best_combination']
            lines.append("BEST PERFORMING COMBINATION:")
            lines.append("-" * 30)
            lines.append(f"Modalities: {', '.join(best['modalities'])}")
            lines.append(f"Score: {best['score']:.4f}")
            lines.append("")
        
        if 'worst_combination' in analysis and analysis['worst_combination']:
            worst = analysis['worst_combination']
            lines.append("WORST PERFORMING COMBINATION:")
            lines.append("-" * 30)
            lines.append(f"Modalities: {', '.join(worst['modalities'])}")
            lines.append(f"Score: {worst['score']:.4f}")
            lines.append("")
        
        # Individual modality ranking
        if 'modality_importance_ranking' in analysis:
            lines.append("MODALITY IMPORTANCE RANKING:")
            lines.append("-" * 30)
            for i, (modality, score) in enumerate(analysis['modality_importance_ranking']):
                lines.append(f"{i+1}. {modality}: {score:.4f}")
            lines.append("")
        
        # Performance by modality count
        if 'performance_by_modality_count' in analysis:
            lines.append("PERFORMANCE BY NUMBER OF MODALITIES:")
            lines.append("-" * 30)
            count_stats = analysis['performance_by_modality_count']
            for count in sorted(count_stats.keys()):
                stats = count_stats[count]
                f1_stats = stats.get('f1', stats.get(self.config.metrics_to_track[0], {}))
                if f1_stats:
                    lines.append(f"{count} modalities: {f1_stats.get('mean', 0):.4f} ± {f1_stats.get('std', 0):.4f}")
            lines.append("")
        
        # Synergy analysis
        if 'modality_synergy' in analysis:
            lines.append("TOP SYNERGISTIC PAIRS:")
            lines.append("-" * 30)
            synergy_data = analysis['modality_synergy']
            synergy_pairs = sorted(synergy_data.items(), key=lambda x: x[1]['synergy_score'], reverse=True)
            
            for combo_name, synergy_info in synergy_pairs[:3]:
                lines.append(f"{combo_name}: synergy score = {synergy_info['synergy_score']:.4f}")
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
            
            # Plot 1: Individual modality performance
            if 'individual_modality_contribution' in analysis:
                individual_perf = analysis['individual_modality_contribution']
                modalities = list(individual_perf.keys())
                f1_scores = [individual_perf[mod].get('f1', 0) for mod in modalities]
                
                axes[0, 0].bar(modalities, f1_scores, color='skyblue')
                axes[0, 0].set_title('Individual Modality Performance')
                axes[0, 0].set_ylabel('F1 Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Performance by modality count
            if 'performance_by_modality_count' in analysis:
                count_stats = analysis['performance_by_modality_count']
                counts = sorted(count_stats.keys())
                means = [count_stats[c].get('f1', {}).get('mean', 0) for c in counts]
                stds = [count_stats[c].get('f1', {}).get('std', 0) for c in counts]
                
                axes[0, 1].bar(counts, means, yerr=stds, capsize=5, color='lightcoral')
                axes[0, 1].set_title('Performance by Number of Modalities')
                axes[0, 1].set_xlabel('Number of Modalities')
                axes[0, 1].set_ylabel('F1 Score')
            
            # Plot 3: All combination results
            combo_names = list(successful_results.keys())
            f1_scores = [successful_results[name].get('val_f1', 0) for name in combo_names]
            
            axes[1, 0].bar(range(len(combo_names)), f1_scores, color='lightgreen')
            axes[1, 0].set_title('All Modality Combinations')
            axes[1, 0].set_xlabel('Combination Index')
            axes[1, 0].set_ylabel('F1 Score')
            
            # Plot 4: Synergy heatmap (if possible)
            if 'modality_synergy' in analysis:
                synergy_data = analysis['modality_synergy']
                if synergy_data:
                    pairs = list(synergy_data.keys())
                    synergy_scores = [synergy_data[pair]['synergy_score'] for pair in pairs]
                    
                    axes[1, 1].bar(range(len(pairs)), synergy_scores, color='gold')
                    axes[1, 1].set_title('Modality Synergy Scores')
                    axes[1, 1].set_xlabel('Modality Pair')
                    axes[1, 1].set_ylabel('Synergy Score')
                    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    
                    # Rotate x-axis labels
                    axes[1, 1].set_xticks(range(len(pairs)))
                    axes[1, 1].set_xticklabels([p.replace('+', '+\n') for p in pairs], rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / "modality_ablation_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualizations saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")


def main():
    """Example usage of modality ablation study."""
    
    # Example model factory
    def create_multimodal_model(config):
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
        'modalities': ['text', 'audio', 'image', 'video'],
        'fusion_strategy': 'cross_modal_attention',
        'text_hidden_dim': 512,
        'audio_hidden_dim': 256,
        'image_hidden_dim': 256,
        'video_hidden_dim': 256,
        'fusion_output_dim': 512,
        'num_classes': 2,
        'dropout_rate': 0.1
    }
    
    # Configuration
    config = ModalityAblationConfig(
        available_modalities=['text', 'audio', 'image', 'video'],
        test_single_modalities=True,
        test_pair_combinations=True,
        max_epochs=3
    )
    
    # Run ablation study
    study = ModalityAblationStudy(
        config=config,
        model_factory=create_multimodal_model,
        data_loaders=data_loaders,
        base_model_config=base_config,
        task_name="sarcasm_detection"
    )
    
    results = study.run_ablation_study()
    
    print("Modality ablation study completed!")
    print(f"Best combination: {results['analysis']['best_combination']['combination']}")


if __name__ == "__main__":
    main()
