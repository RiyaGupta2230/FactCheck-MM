# sarcasm_detection/evaluation/ablation_study.py
"""
Ablation Studies for Sarcasm Detection
Comprehensive ablation analysis for modalities, architectures, and datasets.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import copy
import json
from itertools import combinations

from shared.utils import get_logger
from ..models import MultimodalSarcasmModel, EnsembleSarcasmModel
from .evaluator import SarcasmEvaluator, EvaluationConfig


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    
    # Modality ablation
    modalities_to_test: List[str] = field(default_factory=lambda: ['text', 'audio', 'image', 'video'])
    test_individual_modalities: bool = True
    test_modality_combinations: bool = True
    max_combination_size: int = 3
    
    # Architecture ablation
    architecture_components: List[str] = field(default_factory=lambda: [
        'attention_pooling', 'fusion_layer', 'modality_attention', 'cross_modal_layers'
    ])
    
    # Dataset ablation
    test_dataset_combinations: bool = True
    min_dataset_size: int = 100
    
    # Evaluation settings
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    save_intermediate_results: bool = True
    
    # Output settings
    generate_visualizations: bool = True
    save_ablation_report: bool = True


class ModalityAblationStudy:
    """Ablation study for multimodal components."""
    
    def __init__(
        self,
        model: MultimodalSarcasmModel,
        datasets: Dict[str, Any],
        config: Union[AblationConfig, Dict[str, Any]] = None
    ):
        """
        Initialize modality ablation study.
        
        Args:
            model: Multimodal model to ablate
            datasets: Datasets for evaluation
            config: Ablation configuration
        """
        if isinstance(config, dict):
            config = AblationConfig(**config)
        elif config is None:
            config = AblationConfig()
        
        self.model = model
        self.datasets = datasets
        self.config = config
        
        self.logger = get_logger("ModalityAblationStudy")
        
        # Initialize evaluator
        eval_config = EvaluationConfig(**config.evaluation_config)
        self.evaluator = SarcasmEvaluator(model, eval_config)
        
        self.logger.info(f"Initialized modality ablation study with {len(datasets)} datasets")
    
    def run_modality_ablation(self) -> Dict[str, Any]:
        """
        Run comprehensive modality ablation study.
        
        Returns:
            Ablation results
        """
        self.logger.info("Starting modality ablation study")
        
        results = {
            'baseline': {},
            'individual_modalities': {},
            'modality_combinations': {},
            'analysis': {}
        }
        
        # Baseline: All modalities
        self.logger.info("Evaluating baseline (all modalities)")
        baseline_results = self.evaluator.evaluate_multiple_datasets(self.datasets)
        results['baseline'] = baseline_results
        
        # Individual modality evaluation
        if self.config.test_individual_modalities:
            results['individual_modalities'] = self._test_individual_modalities()
        
        # Modality combination evaluation
        if self.config.test_modality_combinations:
            results['modality_combinations'] = self._test_modality_combinations()
        
        # Analysis
        results['analysis'] = self._analyze_modality_importance(results)
        
        return results
    
    def _test_individual_modalities(self) -> Dict[str, Any]:
        """Test each modality individually."""
        
        self.logger.info("Testing individual modalities")
        individual_results = {}
        
        for modality in self.config.modalities_to_test:
            self.logger.info(f"Testing modality: {modality}")
            
            # Create single-modality datasets
            modality_datasets = self._create_single_modality_datasets(modality)
            
            if modality_datasets:
                # Evaluate with only this modality
                modality_results = self.evaluator.evaluate_multiple_datasets(modality_datasets)
                individual_results[modality] = modality_results
            else:
                self.logger.warning(f"No samples found for modality: {modality}")
        
        return individual_results
    
    def _test_modality_combinations(self) -> Dict[str, Any]:
        """Test combinations of modalities."""
        
        self.logger.info("Testing modality combinations")
        combination_results = {}
        
        modalities = self.config.modalities_to_test
        
        # Test all combinations up to max_combination_size
        for r in range(2, min(len(modalities), self.config.max_combination_size) + 1):
            for modality_combo in combinations(modalities, r):
                combo_name = "+".join(sorted(modality_combo))
                self.logger.info(f"Testing combination: {combo_name}")
                
                # Create multi-modality datasets
                combo_datasets = self._create_multi_modality_datasets(modality_combo)
                
                if combo_datasets:
                    combo_results = self.evaluator.evaluate_multiple_datasets(combo_datasets)
                    combination_results[combo_name] = combo_results
        
        return combination_results
    
    def _create_single_modality_datasets(self, modality: str) -> Dict[str, Any]:
        """Create datasets with only specified modality."""
        
        from torch.utils.data import Dataset
        
        class SingleModalityDataset(Dataset):
            def __init__(self, base_dataset, target_modality):
                self.base_dataset = base_dataset
                self.target_modality = target_modality
                
                # Filter samples that have the target modality
                self.valid_indices = []
                for i in range(len(base_dataset)):
                    sample = base_dataset[i]
                    if target_modality in sample and sample[target_modality] is not None:
                        self.valid_indices.append(i)
            
            def __len__(self):
                return len(self.valid_indices)
            
            def __getitem__(self, idx):
                real_idx = self.valid_indices[idx]
                sample = self.base_dataset[real_idx].copy()
                
                # Zero out all modalities except target
                for mod in ['text', 'audio', 'image', 'video']:
                    if mod != self.target_modality:
                        sample[mod] = None
                
                return sample
        
        modality_datasets = {}
        for dataset_name, dataset in self.datasets.items():
            single_mod_dataset = SingleModalityDataset(dataset, modality)
            if len(single_mod_dataset) >= self.config.min_dataset_size:
                modality_datasets[f"{dataset_name}_{modality}"] = single_mod_dataset
        
        return modality_datasets
    
    def _create_multi_modality_datasets(self, modalities: Tuple[str, ...]) -> Dict[str, Any]:
        """Create datasets with specified modality combinations."""
        
        from torch.utils.data import Dataset
        
        class MultiModalityDataset(Dataset):
            def __init__(self, base_dataset, target_modalities):
                self.base_dataset = base_dataset
                self.target_modalities = set(target_modalities)
                
                # Filter samples that have all target modalities
                self.valid_indices = []
                for i in range(len(base_dataset)):
                    sample = base_dataset[i]
                    has_all_modalities = all(
                        mod in sample and sample[mod] is not None 
                        for mod in target_modalities
                    )
                    if has_all_modalities:
                        self.valid_indices.append(i)
            
            def __len__(self):
                return len(self.valid_indices)
            
            def __getitem__(self, idx):
                real_idx = self.valid_indices[idx]
                sample = self.base_dataset[real_idx].copy()
                
                # Zero out modalities not in target set
                for mod in ['text', 'audio', 'image', 'video']:
                    if mod not in self.target_modalities:
                        sample[mod] = None
                
                return sample
        
        combo_datasets = {}
        combo_name = "+".join(sorted(modalities))
        
        for dataset_name, dataset in self.datasets.items():
            combo_dataset = MultiModalityDataset(dataset, modalities)
            if len(combo_dataset) >= self.config.min_dataset_size:
                combo_datasets[f"{dataset_name}_{combo_name}"] = combo_dataset
        
        return combo_datasets
    
    def _analyze_modality_importance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the importance of each modality."""
        
        analysis = {}
        
        # Get baseline performance
        baseline_metrics = results['baseline']['aggregate_metrics']
        baseline_f1 = baseline_metrics.get('f1', {}).get('mean', 0.0)
        
        # Individual modality contributions
        if 'individual_modalities' in results:
            modality_contributions = {}
            for modality, modality_results in results['individual_modalities'].items():
                modality_f1 = modality_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
                contribution = modality_f1 / baseline_f1 if baseline_f1 > 0 else 0.0
                modality_contributions[modality] = {
                    'f1_score': modality_f1,
                    'relative_contribution': contribution,
                    'absolute_drop': baseline_f1 - modality_f1
                }
            
            # Rank modalities by importance
            sorted_modalities = sorted(
                modality_contributions.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            analysis['individual_modality_ranking'] = [
                {'modality': mod, **metrics} for mod, metrics in sorted_modalities
            ]
        
        # Combination analysis
        if 'modality_combinations' in results:
            combination_analysis = {}
            for combo_name, combo_results in results['modality_combinations'].items():
                combo_f1 = combo_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
                combination_analysis[combo_name] = {
                    'f1_score': combo_f1,
                    'relative_to_baseline': combo_f1 / baseline_f1 if baseline_f1 > 0 else 0.0,
                    'modality_count': len(combo_name.split('+'))
                }
            
            analysis['combination_analysis'] = combination_analysis
        
        # Modality complementarity analysis
        if 'individual_modalities' in results and 'modality_combinations' in results:
            analysis['complementarity'] = self._analyze_modality_complementarity(results)
        
        return analysis
    
    def _analyze_modality_complementarity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well modalities complement each other."""
        
        complementarity = {}
        
        # For each pair of modalities, check if combination > sum of individuals
        individual_scores = {}
        for modality, modality_results in results['individual_modalities'].items():
            individual_scores[modality] = modality_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
        
        combination_scores = {}
        for combo_name, combo_results in results['modality_combinations'].items():
            combination_scores[combo_name] = combo_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
        
        # Analyze pairs
        for combo_name, combo_score in combination_scores.items():
            modalities = combo_name.split('+')
            if len(modalities) == 2:
                mod1_score = individual_scores.get(modalities[0], 0.0)
                mod2_score = individual_scores.get(modalities[1], 0.0)
                
                expected_combined = (mod1_score + mod2_score) / 2  # Average
                actual_combined = combo_score
                
                synergy = actual_combined - expected_combined
                complementarity[combo_name] = {
                    'individual_scores': [mod1_score, mod2_score],
                    'expected_combined': expected_combined,
                    'actual_combined': actual_combined,
                    'synergy': synergy,
                    'is_complementary': synergy > 0.01  # Threshold for significance
                }
        
        return complementarity


class ArchitectureAblationStudy:
    """Ablation study for architectural components."""
    
    def __init__(
        self,
        model: Union[MultimodalSarcasmModel, EnsembleSarcasmModel],
        datasets: Dict[str, Any],
        config: Union[AblationConfig, Dict[str, Any]] = None
    ):
        """
        Initialize architecture ablation study.
        
        Args:
            model: Model to ablate
            datasets: Datasets for evaluation
            config: Ablation configuration
        """
        if isinstance(config, dict):
            config = AblationConfig(**config)
        elif config is None:
            config = AblationConfig()
        
        self.model = model
        self.datasets = datasets
        self.config = config
        
        self.logger = get_logger("ArchitectureAblationStudy")
        
        # Initialize evaluator
        eval_config = EvaluationConfig(**config.evaluation_config)
        self.evaluator = SarcasmEvaluator(model, eval_config)
    
    def run_architecture_ablation(self) -> Dict[str, Any]:
        """Run architecture ablation study."""
        
        self.logger.info("Starting architecture ablation study")
        
        results = {
            'baseline': {},
            'component_ablations': {},
            'analysis': {}
        }
        
        # Baseline evaluation
        baseline_results = self.evaluator.evaluate_multiple_datasets(self.datasets)
        results['baseline'] = baseline_results
        
        # Test each architectural component
        for component in self.config.architecture_components:
            self.logger.info(f"Ablating component: {component}")
            
            # Create modified model without this component
            modified_model = self._create_ablated_model(component)
            
            if modified_model is not None:
                # Evaluate modified model
                modified_evaluator = SarcasmEvaluator(
                    modified_model, 
                    EvaluationConfig(**self.config.evaluation_config)
                )
                component_results = modified_evaluator.evaluate_multiple_datasets(self.datasets)
                results['component_ablations'][component] = component_results
        
        # Analysis
        results['analysis'] = self._analyze_component_importance(results)
        
        return results
    
    def _create_ablated_model(self, component: str):
        """Create model with specified component removed/disabled."""
        
        # This is a simplified approach - in practice, you'd need to modify
        # the model architecture or create variants
        
        try:
            # Create a copy of the model
            ablated_model = copy.deepcopy(self.model)
            
            # Disable/modify specific components
            if component == 'attention_pooling' and hasattr(ablated_model, 'use_attention_pooling'):
                ablated_model.use_attention_pooling = False
            elif component == 'modality_attention' and hasattr(ablated_model, 'modality_attention'):
                # Replace modality attention with identity
                for modality in ablated_model.modality_attention:
                    ablated_model.modality_attention[modality] = torch.nn.Identity()
            elif component == 'cross_modal_layers' and hasattr(ablated_model, 'cross_modal_layers'):
                # Remove cross-modal interactions
                ablated_model.cross_modal_layers = torch.nn.ModuleDict()
            elif component == 'fusion_layer' and hasattr(ablated_model, 'fusion_layer'):
                # Replace with simple concatenation
                from ..models.fusion_strategies import ConcatenationFusion
                input_dims = {mod: 768 for mod in ['text', 'audio', 'image', 'video']}  # Placeholder
                ablated_model.fusion_layer = ConcatenationFusion(
                    input_dims=input_dims,
                    output_dim=512,
                    use_modality_weights=False
                )
            
            return ablated_model
            
        except Exception as e:
            self.logger.error(f"Failed to create ablated model for {component}: {e}")
            return None
    
    def _analyze_component_importance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the importance of each architectural component."""
        
        analysis = {}
        
        # Get baseline performance
        baseline_f1 = results['baseline']['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
        
        # Component importance
        component_importance = {}
        for component, component_results in results['component_ablations'].items():
            component_f1 = component_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
            
            performance_drop = baseline_f1 - component_f1
            relative_importance = performance_drop / baseline_f1 if baseline_f1 > 0 else 0.0
            
            component_importance[component] = {
                'baseline_f1': baseline_f1,
                'ablated_f1': component_f1,
                'performance_drop': performance_drop,
                'relative_importance': relative_importance
            }
        
        # Rank components by importance
        sorted_components = sorted(
            component_importance.items(),
            key=lambda x: x[1]['performance_drop'],
            reverse=True
        )
        
        analysis['component_ranking'] = [
            {'component': comp, **metrics} for comp, metrics in sorted_components
        ]
        
        return analysis


class DatasetAblationStudy:
    """Ablation study for dataset contributions."""
    
    def __init__(
        self,
        model,
        datasets: Dict[str, Any],
        config: Union[AblationConfig, Dict[str, Any]] = None
    ):
        """
        Initialize dataset ablation study.
        
        Args:
            model: Model to evaluate
            datasets: Datasets for ablation
            config: Ablation configuration
        """
        if isinstance(config, dict):
            config = AblationConfig(**config)
        elif config is None:
            config = AblationConfig()
        
        self.model = model
        self.datasets = datasets
        self.config = config
        
        self.logger = get_logger("DatasetAblationStudy")
        
        # Initialize evaluator
        eval_config = EvaluationConfig(**config.evaluation_config)
        self.evaluator = SarcasmEvaluator(model, eval_config)
    
    def run_dataset_ablation(self) -> Dict[str, Any]:
        """Run dataset ablation study."""
        
        self.logger.info("Starting dataset ablation study")
        
        results = {
            'baseline': {},
            'individual_datasets': {},
            'dataset_combinations': {},
            'analysis': {}
        }
        
        # Baseline: All datasets
        baseline_results = self.evaluator.evaluate_multiple_datasets(self.datasets)
        results['baseline'] = baseline_results
        
        # Individual dataset evaluation
        results['individual_datasets'] = self._test_individual_datasets()
        
        # Dataset combination evaluation
        if self.config.test_dataset_combinations:
            results['dataset_combinations'] = self._test_dataset_combinations()
        
        # Analysis
        results['analysis'] = self._analyze_dataset_contributions(results)
        
        return results
    
    def _test_individual_datasets(self) -> Dict[str, Any]:
        """Test each dataset individually."""
        
        individual_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            self.logger.info(f"Testing dataset: {dataset_name}")
            
            single_dataset = {dataset_name: dataset}
            dataset_results = self.evaluator.evaluate_multiple_datasets(single_dataset)
            individual_results[dataset_name] = dataset_results
        
        return individual_results
    
    def _test_dataset_combinations(self) -> Dict[str, Any]:
        """Test combinations of datasets."""
        
        combination_results = {}
        dataset_names = list(self.datasets.keys())
        
        # Test pairs of datasets
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset1, dataset2 = dataset_names[i], dataset_names[j]
                combo_name = f"{dataset1}+{dataset2}"
                
                combo_datasets = {
                    dataset1: self.datasets[dataset1],
                    dataset2: self.datasets[dataset2]
                }
                
                combo_results = self.evaluator.evaluate_multiple_datasets(combo_datasets)
                combination_results[combo_name] = combo_results
        
        return combination_results
    
    def _analyze_dataset_contributions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset contributions."""
        
        analysis = {}
        
        # Individual dataset performance
        dataset_performance = {}
        for dataset_name, dataset_results in results['individual_datasets'].items():
            f1_score = dataset_results['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
            dataset_size = len(self.datasets[dataset_name])
            
            dataset_performance[dataset_name] = {
                'f1_score': f1_score,
                'dataset_size': dataset_size,
                'efficiency': f1_score / dataset_size if dataset_size > 0 else 0.0
            }
        
        # Rank datasets
        sorted_datasets = sorted(
            dataset_performance.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        analysis['dataset_ranking'] = [
            {'dataset': name, **metrics} for name, metrics in sorted_datasets
        ]
        
        return analysis


class AblationAnalyzer:
    """Comprehensive ablation analyzer combining all ablation studies."""
    
    def __init__(
        self,
        model,
        datasets: Dict[str, Any],
        config: Union[AblationConfig, Dict[str, Any]] = None
    ):
        """
        Initialize comprehensive ablation analyzer.
        
        Args:
            model: Model to analyze
            datasets: Datasets for analysis
            config: Configuration
        """
        if isinstance(config, dict):
            config = AblationConfig(**config)
        elif config is None:
            config = AblationConfig()
        
        self.model = model
        self.datasets = datasets
        self.config = config
        
        self.logger = get_logger("AblationAnalyzer")
    
    def run_comprehensive_ablation(self) -> Dict[str, Any]:
        """Run comprehensive ablation analysis."""
        
        self.logger.info("Starting comprehensive ablation analysis")
        
        results = {
            'configuration': self.config.__dict__,
            'model_info': self._get_model_info(),
            'dataset_info': self._get_dataset_info()
        }
        
        # Modality ablation (for multimodal models)
        if isinstance(self.model, MultimodalSarcasmModel):
            self.logger.info("Running modality ablation study")
            modality_study = ModalityAblationStudy(self.model, self.datasets, self.config)
            results['modality_ablation'] = modality_study.run_modality_ablation()
        
        # Architecture ablation
        self.logger.info("Running architecture ablation study")
        architecture_study = ArchitectureAblationStudy(self.model, self.datasets, self.config)
        results['architecture_ablation'] = architecture_study.run_architecture_ablation()
        
        # Dataset ablation
        self.logger.info("Running dataset ablation study")
        dataset_study = DatasetAblationStudy(self.model, self.datasets, self.config)
        results['dataset_ablation'] = dataset_study.run_dataset_ablation()
        
        # Comprehensive analysis
        results['comprehensive_analysis'] = self._analyze_comprehensive_results(results)
        
        return results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        info = {
            'model_type': type(self.model).__name__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        
        dataset_info = {}
        total_samples = 0
        
        for dataset_name, dataset in self.datasets.items():
            size = len(dataset)
            total_samples += size
            
            info = {'size': size}
            if hasattr(dataset, 'get_statistics'):
                info['statistics'] = dataset.get_statistics()
            
            dataset_info[dataset_name] = info
        
        dataset_info['total_samples'] = total_samples
        dataset_info['num_datasets'] = len(self.datasets)
        
        return dataset_info
    
    def _analyze_comprehensive_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive ablation results."""
        
        analysis = {
            'summary': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Extract key performance numbers
        baseline_f1 = 0.0
        
        # Try to get baseline from any available ablation study
        for study_name in ['modality_ablation', 'architecture_ablation', 'dataset_ablation']:
            if study_name in results and 'baseline' in results[study_name]:
                baseline_f1 = results[study_name]['baseline']['aggregate_metrics'].get('f1', {}).get('mean', 0.0)
                break
        
        analysis['summary']['baseline_f1'] = baseline_f1
        
        # Key findings from modality ablation
        if 'modality_ablation' in results:
            modality_analysis = results['modality_ablation'].get('analysis', {})
            if 'individual_modality_ranking' in modality_analysis:
                best_modality = modality_analysis['individual_modality_ranking'][0]
                analysis['key_findings'].append(
                    f"Most important individual modality: {best_modality['modality']} "
                    f"(F1: {best_modality['f1_score']:.3f})"
                )
        
        # Key findings from architecture ablation
        if 'architecture_ablation' in results:
            arch_analysis = results['architecture_ablation'].get('analysis', {})
            if 'component_ranking' in arch_analysis:
                most_important_component = arch_analysis['component_ranking'][0]
                analysis['key_findings'].append(
                    f"Most critical architectural component: {most_important_component['component']} "
                    f"(Performance drop: {most_important_component['performance_drop']:.3f})"
                )
        
        # Recommendations
        if baseline_f1 > 0.8:
            analysis['recommendations'].append("Model shows strong performance across ablation studies")
        elif baseline_f1 > 0.6:
            analysis['recommendations'].append("Model shows moderate performance, consider architectural improvements")
        else:
            analysis['recommendations'].append("Model shows weak performance, significant improvements needed")
        
        return analysis
    
    def save_ablation_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        experiment_name: str = "comprehensive_ablation"
    ):
        """Save comprehensive ablation results."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_dir / f"{experiment_name}_ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        
        # Add modality results
        if 'modality_ablation' in results and 'analysis' in results['modality_ablation']:
            modality_ranking = results['modality_ablation']['analysis'].get('individual_modality_ranking', [])
            for item in modality_ranking:
                summary_data.append({
                    'ablation_type': 'modality',
                    'component': item['modality'],
                    'f1_score': item['f1_score'],
                    'relative_contribution': item.get('relative_contribution', 0)
                })
        
        # Add architecture results
        if 'architecture_ablation' in results and 'analysis' in results['architecture_ablation']:
            component_ranking = results['architecture_ablation']['analysis'].get('component_ranking', [])
            for item in component_ranking:
                summary_data.append({
                    'ablation_type': 'architecture',
                    'component': item['component'],
                    'f1_score': item['ablated_f1'],
                    'performance_drop': item['performance_drop']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / f"{experiment_name}_ablation_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Saved ablation results to {output_dir}")
