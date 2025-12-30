"""
Curriculum Learning for Sarcasm Detection
Progressive training based on task difficulty and sample complexity.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import random
from collections import defaultdict

from shared.utils import get_logger, ExperimentLogger
from shared.datasets import create_hardware_aware_dataloader, MultimodalCollator
from ..models import TextSarcasmModel, MultimodalSarcasmModel
from ..utils import SarcasmMetrics
from .train_text import TextSarcasmTrainer
from .train_multimodal import MultimodalSarcasmTrainer

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Curriculum strategy
    curriculum_strategy: str = "difficulty_based"  # difficulty_based, length_based, confidence_based, mixed
    difficulty_metric: str = "text_complexity"  # text_complexity, confidence_score, loss_based
    
    # Curriculum pacing
    curriculum_pacing: str = "linear"  # linear, exponential, root, step
    num_curriculum_steps: int = 5
    samples_per_step: Optional[List[int]] = None  # If None, will be auto-calculated
    
    # Difficulty estimation
    complexity_features: List[str] = field(default_factory=lambda: [
        'text_length', 'sentiment_intensity', 'readability_score',
        'syntactic_complexity', 'irony_markers'
    ])
    
    # Training progression
    start_with_easy_ratio: float = 0.2  # Start with 20% of easiest samples
    final_easy_ratio: float = 0.8  # End with 80% of samples
    add_hard_samples_gradually: bool = True
    
    # Base trainer configuration
    base_trainer_config: Dict[str, Any] = field(default_factory=dict)
    warmup_epochs_per_step: int = 2
    finetune_epochs_per_step: int = 3
    
    # Evaluation and adaptation
    evaluate_curriculum_steps: bool = True
    adapt_curriculum_based_on_performance: bool = True
    performance_threshold: float = 0.7  # F1 score threshold to move to next step
    
    # Logging
    log_curriculum_progression: bool = True
    save_difficulty_scores: bool = True

DATASET_MODALITY_PROFILE = {
    'sarc': 1,
    'sarcasm_headlines': 1,
    'mmsd2': 2,
    'sarcnet': 2,
    'mustard': 4
}

class DifficultyEstimator:
    """Estimates difficulty of sarcasm detection samples."""
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize difficulty estimator.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config
        self.logger = get_logger("DifficultyEstimator")
        
        # Initialize feature extractors
        self._setup_feature_extractors()
        
        # Difficulty scores cache
        self.difficulty_cache = {}
    
    def _setup_feature_extractors(self):
        """Setup feature extractors for difficulty estimation."""
        try:
            # Text complexity analyzers
            import textstat
            self.textstat = textstat
            
            # Sentiment intensity analyzer
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Irony detection patterns
            self.irony_patterns = [
                r'oh\s+(sure|great|wonderful|perfect)',
                r'yeah\s+(right|sure)',
                r'as\s+if',
                r'i\s+bet',
                r'totally',
                r'absolutely\s+(not|never)',
                r'couldn\'t\s+agree\s+more',
                r'sure\s+thing'
            ]
            
            import re
            self.irony_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.irony_patterns]
            
        except ImportError as e:
            self.logger.warning(f"Some difficulty estimation dependencies not available: {e}")
            self.textstat = None
            self.sentiment_analyzer = None
            self.irony_regex = []
    
    def estimate_text_complexity(self, text: str) -> float:
        """
        Estimate text complexity for sarcasm detection.
        
        Args:
            text: Input text
            
        Returns:
            Complexity score (0-1, where 1 is most difficult)
        """
        if not text or not text.strip():
            return 0.0
        
        complexity_scores = []
        
        # Text length complexity
        if 'text_length' in self.config.complexity_features:
            # Shorter texts are often more difficult for sarcasm (subtle cues)
            length_score = max(0, 1 - (len(text.split()) / 50))  # Normalize to 50 words
            complexity_scores.append(length_score)
        
        # Readability complexity
        if 'readability_score' in self.config.complexity_features and self.textstat:
            try:
                flesch_score = self.textstat.flesch_reading_ease(text)
                # Convert Flesch score to difficulty (lower Flesch = higher difficulty)
                readability_score = max(0, (100 - flesch_score) / 100)
                complexity_scores.append(readability_score)
            except:
                pass
        
        # Syntactic complexity
        if 'syntactic_complexity' in self.config.complexity_features:
            # Count complex punctuation patterns
            complex_punct = text.count('!') + text.count('?') + text.count('...') + text.count(';')
            syntactic_score = min(1.0, complex_punct / 5)
            complexity_scores.append(syntactic_score)
        
        # Sentiment intensity (extreme sentiments can be easier to detect)
        if 'sentiment_intensity' in self.config.complexity_features and self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                # Higher compound scores (extreme sentiments) are easier
                sentiment_intensity = abs(sentiment_scores['compound'])
                sentiment_complexity = 1 - sentiment_intensity  # Neutral sentiments are harder
                complexity_scores.append(sentiment_complexity)
            except:
                pass
        
        # Irony markers presence
        if 'irony_markers' in self.config.complexity_features:
            irony_count = sum(1 for pattern in self.irony_regex if pattern.search(text))
            # More explicit irony markers make it easier
            irony_score = max(0, 1 - (irony_count / 3))
            complexity_scores.append(irony_score)
        
        # Average complexity score
        if complexity_scores:
            return np.mean(complexity_scores)
        else:
            # Fallback: random difficulty
            return random.random()
    
    def estimate_sample_difficulty(self, sample: Dict[str, Any]) -> float:
        """
        Estimate difficulty of a complete sample.
        
        Args:
            sample: Sample data
            
        Returns:
            Difficulty score (0-1)
        """
        sample_id = sample.get('id', str(hash(str(sample))))
        
        # Check cache
        if sample_id in self.difficulty_cache:
            return self.difficulty_cache[sample_id]
        
        difficulty_scores = []
        
        # Text complexity
        if 'text' in sample and sample['text']:
            text_difficulty = self.estimate_text_complexity(sample['text'])
            difficulty_scores.append(text_difficulty)
        
        # Dataset-specific difficulty
        dataset_name = sample.get('dataset', 'unknown')
        dataset_difficulty_map = {
            'mustard': 0.8,
            'sarc': 0.6,
            'sarcasm_headlines': 0.4,
            'mmsd2': 0.7,
            'sarcnet': 0.7
        }
        
        if dataset_name in dataset_difficulty_map:
            dataset_difficulty = dataset_difficulty_map[dataset_name]
            difficulty_scores.append(dataset_difficulty)
        else:
            if dataset_name != 'unknown':
                self.logger.warning(f"Unknown dataset '{dataset_name}', using default difficulty of 0.6")
            difficulty_scores.append(0.6)
        
        # Modality-based difficulty (normalized against expected modality profile)
        available_modalities = 0
        for modality in ['text', 'audio', 'image', 'video']:
            if modality in sample and sample[modality] is not None:
                available_modalities += 1
        
        if available_modalities > 0 and dataset_name in DATASET_MODALITY_PROFILE:
            expected_modalities = DATASET_MODALITY_PROFILE[dataset_name]
            modality_ratio = available_modalities / expected_modalities
            modality_difficulty = 1.0 - (modality_ratio - 1.0) * 0.1 if modality_ratio > 1.0 else 1.0 - (1.0 - modality_ratio) * 0.3
            modality_difficulty = max(0.0, min(1.0, modality_difficulty))
            difficulty_scores.append(modality_difficulty)
        
        # Final difficulty score
        final_difficulty = np.mean(difficulty_scores) if difficulty_scores else 0.6
        
        # Cache result
        self.difficulty_cache[sample_id] = final_difficulty
        
        return final_difficulty
    
    def rank_samples_by_difficulty(self, dataset) -> List[Tuple[int, float]]:
        """
        Rank all samples in dataset by difficulty.
        
        Args:
            dataset: Dataset to rank
            
        Returns:
            List of (sample_index, difficulty_score) tuples, sorted by difficulty
        """
        self.logger.info(f"Ranking {len(dataset)} samples by difficulty...")
        
        difficulty_scores = []
        for idx in tqdm(range(len(dataset)), desc="Estimating difficulty"):
            sample = dataset[idx]
            difficulty = self.estimate_sample_difficulty(sample)
            difficulty_scores.append((idx, difficulty))
        
        # Sort by difficulty (easy to hard)
        difficulty_scores.sort(key=lambda x: x[1])
        
        self.logger.info(
            f"Difficulty ranking completed. "
            f"Easiest sample: {difficulty_scores[0][1]:.3f}, "
            f"Hardest sample: {difficulty_scores[-1][1]:.3f}"
        )
        
        return difficulty_scores

class CurriculumScheduler:
    """Manages curriculum progression and sample selection."""
    
    def __init__(self, config: CurriculumConfig, difficulty_ranking: List[Tuple[int, float]]):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum configuration
            difficulty_ranking: List of (sample_index, difficulty_score) tuples
        """
        self.config = config
        self.difficulty_ranking = difficulty_ranking
        self.total_samples = len(difficulty_ranking)
        self.logger = get_logger("CurriculumScheduler")
        
        # Create curriculum steps
        self.curriculum_steps = self._create_curriculum_steps()
        self.current_step = 0
        
        self.logger.info(f"Created curriculum with {len(self.curriculum_steps)} steps")
    
    def _create_curriculum_steps(self) -> List[Dict[str, Any]]:
        """Create curriculum learning steps."""
        steps = []
        
        # Calculate sample counts for each step
        if self.config.samples_per_step is not None:
            step_sample_counts = self.config.samples_per_step
        else:
            # Auto-calculate based on pacing
            start_samples = int(self.total_samples * self.config.start_with_easy_ratio)
            end_samples = int(self.total_samples * self.config.final_easy_ratio)
            
            step_sample_counts = []
            for i in range(self.config.num_curriculum_steps):
                progress = i / (self.config.num_curriculum_steps - 1)
                
                if self.config.curriculum_pacing == "linear":
                    sample_count = start_samples + progress * (end_samples - start_samples)
                elif self.config.curriculum_pacing == "exponential":
                    sample_count = start_samples * (end_samples / start_samples) ** progress
                elif self.config.curriculum_pacing == "root":
                    sample_count = start_samples + (progress ** 0.5) * (end_samples - start_samples)
                elif self.config.curriculum_pacing == "step":
                    sample_count = start_samples if progress < 0.5 else end_samples
                else:
                    sample_count = start_samples + progress * (end_samples - start_samples)
                
                step_sample_counts.append(int(sample_count))
        
        # Create steps
        for i, sample_count in enumerate(step_sample_counts):
            # Select samples for this step
            if self.config.add_hard_samples_gradually:
                # Take easiest samples up to this count
                step_samples = self.difficulty_ranking[:sample_count]
            else:
                # Mix of easy and proportionally hard samples
                easy_count = int(sample_count * 0.7)
                hard_count = sample_count - easy_count
                easy_samples = self.difficulty_ranking[:easy_count]
                hard_samples = self.difficulty_ranking[-hard_count:] if hard_count > 0 else []
                step_samples = easy_samples + hard_samples
            
            step_info = {
                'step_number': i,
                'sample_indices': [idx for idx, _ in step_samples],
                'sample_count': len(step_samples),
                'difficulty_range': (
                    min(diff for _, diff in step_samples),
                    max(diff for _, diff in step_samples)
                ),
                'warmup_epochs': self.config.warmup_epochs_per_step,
                'finetune_epochs': self.config.finetune_epochs_per_step
            }
            
            steps.append(step_info)
            
            self.logger.info(
                f"Step {i}: {step_info['sample_count']} samples, "
                f"difficulty range: {step_info['difficulty_range'][0]:.3f} - {step_info['difficulty_range'][1]:.3f}"
            )
        
        return steps
    
    def get_current_step_data(self) -> Dict[str, Any]:
        """Get data for current curriculum step."""
        if self.current_step >= len(self.curriculum_steps):
            return self.curriculum_steps[-1]  # Return last step
        return self.curriculum_steps[self.current_step]
    
    def advance_step(self) -> bool:
        """
        Advance to next curriculum step.
        
        Returns:
            True if advanced, False if already at final step
        """
        if self.current_step < len(self.curriculum_steps) - 1:
            self.current_step += 1
            self.logger.info(f"Advanced to curriculum step {self.current_step}")
            return True
        return False
    
    def should_advance_step(self, performance_metrics: Dict[str, float]) -> bool:
        """
        Check if should advance to next curriculum step based on performance.
        
        Args:
            performance_metrics: Performance metrics from current step
            
        Returns:
            True if should advance
        """
        if not self.config.adapt_curriculum_based_on_performance:
            return True  # Always advance if not adapting
        
        # Check if performance threshold is met
        f1_score = performance_metrics.get('val_f1', 0.0)
        return f1_score >= self.config.performance_threshold

class CurriculumTrainer:
    """Trainer that implements curriculum learning for sarcasm detection."""
    
    def __init__(
        self,
        model: Union[TextSarcasmModel, MultimodalSarcasmModel],
        config: Union[CurriculumConfig, Dict[str, Any]],
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Model to train
            config: Curriculum configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if isinstance(config, dict):
            config = CurriculumConfig(**config)
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.logger = get_logger("CurriculumTrainer")
        
        if not train_dataset:
            raise ValueError("Training dataset required for curriculum learning")
        
        # Initialize difficulty estimator
        self.difficulty_estimator = DifficultyEstimator(config)
        
        # Rank samples by difficulty
        self.difficulty_ranking = self.difficulty_estimator.rank_samples_by_difficulty(train_dataset)
        
        # Initialize curriculum scheduler
        self.curriculum_scheduler = CurriculumScheduler(config, self.difficulty_ranking)
        
        # Initialize base trainer
        self.base_trainer = self._create_base_trainer()
        
        # Training history
        self.curriculum_history = []
        
        self.logger.info("Initialized curriculum trainer")
    
    def _create_base_trainer(self):
        """Create base trainer for curriculum steps."""
        if isinstance(self.model, MultimodalSarcasmModel):
            from .train_multimodal import MultimodalTrainingConfig, MultimodalSarcasmTrainer
            base_config = MultimodalTrainingConfig(**self.config.base_trainer_config)
            return MultimodalSarcasmTrainer(
                model=self.model,
                config=base_config,
                train_dataset=None,  # Will be set for each step
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset
            )
        else:
            from .train_text import TextTrainingConfig, TextSarcasmTrainer
            base_config = TextTrainingConfig(**self.config.base_trainer_config)
            return TextSarcasmTrainer(
                model=self.model,
                config=base_config,
                train_dataset=None,  # Will be set for each step
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset
            )
    
    def _create_step_dataset(self, step_info: Dict[str, Any]):
        """Create dataset for a curriculum step."""
        from torch.utils.data import Subset
        step_indices = step_info['sample_indices']
        step_dataset = Subset(self.train_dataset, step_indices)
        return step_dataset
    
    def train_curriculum_step(
        self,
        step_info: Dict[str, Any],
        checkpoint_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train model on a single curriculum step.
        
        Args:
            step_info: Information about current step
            checkpoint_dir: Checkpoint directory
            
        Returns:
            Training results for this step
        """
        step_num = step_info['step_number']
        self.logger.info(
            f"Training curriculum step {step_num}: "
            f"{step_info['sample_count']} samples, "
            f"difficulty range: {step_info['difficulty_range']}"
        )
        
        # Create dataset for this step
        step_dataset = self._create_step_dataset(step_info)
        
        # Update base trainer with step dataset
        self.base_trainer.train_dataset = step_dataset
        self.base_trainer._setup_data_loaders()
        
        # Create step checkpoint directory
        step_checkpoint_dir = None
        if checkpoint_dir:
            step_checkpoint_dir = checkpoint_dir / f"step_{step_num}"
            step_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Train on this step
        step_results = self.base_trainer.train(
            checkpoint_dir=step_checkpoint_dir,
            experiment_name=f"curriculum_step_{step_num}"
        )
        
        # Add curriculum-specific information
        step_results['curriculum_step'] = step_num
        step_results['step_info'] = step_info
        step_results['difficulty_range'] = step_info['difficulty_range']
        step_results['sample_count'] = step_info['sample_count']
        
        return step_results
    
    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        resume_from_step: int = 0,
        experiment_name: str = "curriculum_sarcasm_training"
    ) -> Dict[str, Any]:
        """
        Main curriculum training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            resume_from_step: Curriculum step to resume from
            experiment_name: Name for experiment logging
            
        Returns:
            Complete curriculum training results
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment logging
        experiment_logger = ExperimentLogger(
            log_dir=checkpoint_dir or Path("logs"),
            project_name="curriculum_sarcasm_detection",
            experiment_name=experiment_name,
            config={**self.config.__dict__, 'total_samples': len(self.train_dataset)}
        )
        
        # Resume from specific step if requested
        self.curriculum_scheduler.current_step = resume_from_step
        self.logger.info(f"Starting curriculum training from step {resume_from_step}")
        
        # Training loop through curriculum steps
        all_results = []
        while True:
            # Get current step information
            step_info = self.curriculum_scheduler.get_current_step_data()
            
            # Train on current step
            step_results = self.train_curriculum_step(step_info, checkpoint_dir)
            
            # Log step results
            experiment_logger.log_metrics(
                {f"curriculum_step_{step_info['step_number']}_{k}": v
                 for k, v in step_results.items() if isinstance(v, (int, float))},
                step=step_info['step_number']
            )
            
            # Store results
            all_results.append(step_results)
            self.curriculum_history.append(step_results)
            
            # Check if should advance to next step
            if self.config.evaluate_curriculum_steps and step_results.get('final_validation'):
                should_advance = self.curriculum_scheduler.should_advance_step(
                    step_results['final_validation']
                )
                
                if not should_advance:
                    self.logger.info(
                        f"Performance threshold not met for step {step_info['step_number']}, "
                        f"training additional epochs"
                    )
                    # Could implement additional training here
            
            # Try to advance to next step
            if not self.curriculum_scheduler.advance_step():
                self.logger.info("Completed all curriculum steps")
                break
        
        # Final evaluation on complete dataset
        self.logger.info("Performing final evaluation on complete dataset")
        
        # Reset trainer to use full dataset
        self.base_trainer.train_dataset = self.train_dataset
        self.base_trainer._setup_data_loaders()
        
        final_results = {}
        if self.val_dataset:
            final_val_metrics = self.base_trainer.evaluate(self.base_trainer.val_dataloader, "final_val")
            final_results['final_validation'] = final_val_metrics
        
        if self.test_dataset:
            final_test_metrics = self.base_trainer.evaluate(self.base_trainer.test_dataloader, "final_test")
            final_results['final_test'] = final_test_metrics
        
        # Compile complete results
        complete_results = {
            'curriculum_steps': all_results,
            'curriculum_history': self.curriculum_history,
            'final_results': final_results,
            'difficulty_ranking': self.difficulty_ranking,
            'curriculum_config': self.config.__dict__
        }
        
        # Save difficulty scores if requested
        if self.config.save_difficulty_scores and checkpoint_dir:
            difficulty_file = checkpoint_dir / "difficulty_scores.json"
            import json
            with open(difficulty_file, 'w') as f:
                json.dump({
                    'difficulty_ranking': [(int(idx), float(score)) for idx, score in self.difficulty_ranking],
                    'difficulty_cache': {k: float(v) for k, v in self.difficulty_estimator.difficulty_cache.items()}
                }, f, indent=2)
        
        experiment_logger.close()
        
        self.logger.info("Curriculum training completed")
        
        return complete_results
