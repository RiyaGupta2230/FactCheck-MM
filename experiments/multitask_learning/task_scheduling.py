#!/usr/bin/env python3
"""
Task Scheduling for Multitask Learning

Implements various task scheduling strategies for efficient multitask training
including round-robin, proportional sampling, adaptive sampling, and curriculum learning.
"""

import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger


@dataclass 
class SchedulerConfig:
    """Configuration for task scheduling."""
    
    # Scheduling strategy
    strategy: str = "adaptive"  # round_robin, proportional, adaptive, curriculum
    
    # Proportional sampling configuration
    task_weights: Dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0  # For softmax sampling
    
    # Adaptive sampling configuration
    adaptation_rate: float = 0.1
    performance_threshold: float = 0.05  # Minimum improvement to continue focusing
    lookback_window: int = 5  # Number of epochs to look back for performance trends
    
    # Curriculum learning configuration
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    stage_transition_epochs: List[int] = field(default_factory=list)
    
    # General configuration
    warmup_epochs: int = 2
    min_task_probability: float = 0.1  # Minimum probability for any task
    
    # Logging
    log_scheduling_decisions: bool = True


class TaskScheduler:
    """Base task scheduler with multiple scheduling strategies."""
    
    def __init__(self, config: SchedulerConfig, task_names: List[str]):
        """
        Initialize task scheduler.
        
        Args:
            config: Scheduler configuration
            task_names: List of task names
        """
        self.config = config
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Setup logging
        self.logger = get_logger("TaskScheduler")
        
        # Initialize scheduling state
        self.current_epoch = 0
        self.global_step = 0
        self.task_performance_history = {task: [] for task in task_names}
        self.task_step_counts = {task: 0 for task in task_names}
        
        # Initialize task probabilities
        self.task_probabilities = self._initialize_task_probabilities()
        
        # Initialize strategy-specific components
        self._initialize_strategy_components()
        
        self.logger.info(f"Initialized task scheduler with strategy: {config.strategy}")
        self.logger.info(f"Tasks: {task_names}")
    
    def _initialize_task_probabilities(self) -> Dict[str, float]:
        """Initialize task probabilities based on configuration."""
        
        if self.config.strategy == "proportional" and self.config.task_weights:
            # Use configured weights
            total_weight = sum(self.config.task_weights.values())
            probabilities = {task: weight / total_weight for task, weight in self.config.task_weights.items()}
            
            # Fill in missing tasks with equal probability
            missing_tasks = [task for task in self.task_names if task not in probabilities]
            if missing_tasks:
                remaining_prob = 1.0 - sum(probabilities.values())
                prob_per_missing = remaining_prob / len(missing_tasks) if remaining_prob > 0 else 1.0 / len(missing_tasks)
                for task in missing_tasks:
                    probabilities[task] = prob_per_missing
        else:
            # Equal probability for all tasks initially
            prob_per_task = 1.0 / self.num_tasks
            probabilities = {task: prob_per_task for task in self.task_names}
        
        return probabilities
    
    def _initialize_strategy_components(self):
        """Initialize strategy-specific components."""
        
        if self.config.strategy == "round_robin":
            self.round_robin_index = 0
        
        elif self.config.strategy == "curriculum":
            self.current_curriculum_stage = 0
            self._validate_curriculum_config()
        
        elif self.config.strategy == "adaptive":
            self.adaptation_history = []
            self.last_adaptation_epoch = -1
    
    def _validate_curriculum_config(self):
        """Validate curriculum learning configuration."""
        
        if not self.config.curriculum_stages:
            self.logger.warning("Curriculum strategy selected but no stages defined")
            return
        
        if len(self.config.stage_transition_epochs) != len(self.config.curriculum_stages) - 1:
            self.logger.warning("Mismatch between curriculum stages and transition epochs")
    
    def get_next_task(self, global_step: int) -> str:
        """
        Get the next task to train on.
        
        Args:
            global_step: Current global training step
            
        Returns:
            Task name to train on next
        """
        self.global_step = global_step
        
        if self.config.strategy == "round_robin":
            return self._round_robin_scheduling()
        
        elif self.config.strategy == "proportional":
            return self._proportional_scheduling()
        
        elif self.config.strategy == "adaptive":
            return self._adaptive_scheduling()
        
        elif self.config.strategy == "curriculum":
            return self._curriculum_scheduling()
        
        else:
            self.logger.warning(f"Unknown scheduling strategy: {self.config.strategy}, using round-robin")
            return self._round_robin_scheduling()
    
    def _round_robin_scheduling(self) -> str:
        """Round-robin task scheduling."""
        
        task = self.task_names[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % self.num_tasks
        
        self.task_step_counts[task] += 1
        
        if self.config.log_scheduling_decisions and self.global_step % 100 == 0:
            self.logger.debug(f"Round-robin selected: {task}")
        
        return task
    
    def _proportional_scheduling(self) -> str:
        """Proportional (weighted) task scheduling."""
        
        # Apply temperature to probabilities
        if self.config.temperature != 1.0:
            adjusted_probs = {}
            for task, prob in self.task_probabilities.items():
                adjusted_probs[task] = prob ** (1.0 / self.config.temperature)
            
            # Normalize
            total_prob = sum(adjusted_probs.values())
            probabilities = {task: prob / total_prob for task, prob in adjusted_probs.items()}
        else:
            probabilities = self.task_probabilities
        
        # Sample task based on probabilities
        tasks = list(probabilities.keys())
        probs = list(probabilities.values())
        
        task = np.random.choice(tasks, p=probs)
        self.task_step_counts[task] += 1
        
        if self.config.log_scheduling_decisions and self.global_step % 100 == 0:
            prob_str = ", ".join([f"{t}:{p:.3f}" for t, p in probabilities.items()])
            self.logger.debug(f"Proportional selected: {task} (probs: {prob_str})")
        
        return task
    
    def _adaptive_scheduling(self) -> str:
        """Adaptive task scheduling based on performance."""
        
        # During warmup, use round-robin
        if self.current_epoch < self.config.warmup_epochs:
            return self._round_robin_scheduling()
        
        # Update probabilities based on recent performance
        if self.global_step % 100 == 0:  # Update every 100 steps
            self._update_adaptive_probabilities()
        
        # Sample based on current probabilities
        return self._proportional_scheduling()
    
    def _curriculum_scheduling(self) -> str:
        """Curriculum learning task scheduling."""
        
        # Determine current curriculum stage
        current_stage_idx = self._get_current_curriculum_stage()
        
        if current_stage_idx >= len(self.config.curriculum_stages):
            # Past all curriculum stages, use equal probability
            return self._round_robin_scheduling()
        
        # Get tasks for current stage
        current_stage = self.config.curriculum_stages[current_stage_idx]
        stage_tasks = current_stage.get('tasks', self.task_names)
        stage_weights = current_stage.get('weights', {})
        
        # Create probabilities for current stage
        if stage_weights:
            total_weight = sum(stage_weights.values())
            stage_probabilities = {task: stage_weights.get(task, 0) / total_weight for task in stage_tasks}
        else:
            prob_per_task = 1.0 / len(stage_tasks)
            stage_probabilities = {task: prob_per_task for task in stage_tasks}
        
        # Sample from stage tasks
        tasks = list(stage_probabilities.keys())
        probs = list(stage_probabilities.values())
        
        task = np.random.choice(tasks, p=probs)
        self.task_step_counts[task] += 1
        
        if self.config.log_scheduling_decisions and self.global_step % 100 == 0:
            self.logger.debug(f"Curriculum stage {current_stage_idx} selected: {task}")
        
        return task
    
    def _get_current_curriculum_stage(self) -> int:
        """Get current curriculum stage based on epoch."""
        
        stage_idx = 0
        for transition_epoch in self.config.stage_transition_epochs:
            if self.current_epoch >= transition_epoch:
                stage_idx += 1
            else:
                break
        
        return stage_idx
    
    def _update_adaptive_probabilities(self):
        """Update task probabilities based on performance trends."""
        
        if len(self.adaptation_history) < self.config.lookback_window:
            return  # Not enough history
        
        # Calculate performance improvements for each task
        task_improvements = {}
        
        for task in self.task_names:
            if len(self.task_performance_history[task]) >= self.config.lookback_window:
                recent_perf = self.task_performance_history[task][-self.config.lookback_window:]
                
                # Calculate trend (simple linear regression slope)
                x = np.arange(len(recent_perf))
                y = np.array(recent_perf)
                
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    task_improvements[task] = slope
                else:
                    task_improvements[task] = 0.0
            else:
                task_improvements[task] = 0.0
        
        # Adjust probabilities based on improvements
        # Tasks with lower improvement get higher probability (need more attention)
        min_improvement = min(task_improvements.values())
        max_improvement = max(task_improvements.values())
        
        if max_improvement > min_improvement:
            # Normalize improvements to [0, 1] and invert
            normalized_needs = {}
            for task, improvement in task_improvements.items():
                normalized = (improvement - min_improvement) / (max_improvement - min_improvement)
                normalized_needs[task] = 1.0 - normalized  # Invert: lower improvement = higher need
            
            # Update probabilities with adaptation rate
            for task in self.task_names:
                need = normalized_needs[task]
                target_prob = self.config.min_task_probability + need * (1.0 - self.config.min_task_probability * self.num_tasks)
                
                # Smooth update
                current_prob = self.task_probabilities[task]
                new_prob = current_prob * (1 - self.config.adaptation_rate) + target_prob * self.config.adaptation_rate
                self.task_probabilities[task] = new_prob
            
            # Normalize probabilities
            total_prob = sum(self.task_probabilities.values())
            self.task_probabilities = {task: prob / total_prob for task, prob in self.task_probabilities.items()}
        
        if self.config.log_scheduling_decisions:
            prob_str = ", ".join([f"{t}:{p:.3f}" for t, p in self.task_probabilities.items()])
            self.logger.debug(f"Adaptive probabilities updated: {prob_str}")
    
    def update_epoch(self, epoch: int, performance_metrics: Dict[str, float]):
        """
        Update scheduler state at end of epoch.
        
        Args:
            epoch: Current epoch number
            performance_metrics: Performance metrics for each task
        """
        self.current_epoch = epoch
        
        # Update performance history
        for task in self.task_names:
            # Look for task-specific F1 score
            f1_key = f'val_{task}_f1'
            if f1_key in performance_metrics:
                self.task_performance_history[task].append(performance_metrics[f1_key])
            
            # Limit history length
            max_history = self.config.lookback_window * 3
            if len(self.task_performance_history[task]) > max_history:
                self.task_performance_history[task] = self.task_performance_history[task][-max_history:]
        
        # Log epoch scheduling statistics
        if self.config.log_scheduling_decisions:
            self._log_epoch_statistics()
    
    def _log_epoch_statistics(self):
        """Log scheduling statistics for the epoch."""
        
        total_steps = sum(self.task_step_counts.values())
        if total_steps > 0:
            step_percentages = {task: count / total_steps * 100 for task, count in self.task_step_counts.items()}
            
            percentage_str = ", ".join([f"{task}:{pct:.1f}%" for task, pct in step_percentages.items()])
            self.logger.info(f"Epoch {self.current_epoch} task distribution: {percentage_str}")
        
        # Reset step counts for next epoch
        self.task_step_counts = {task: 0 for task in self.task_names}
    
    def get_current_probabilities(self) -> Dict[str, float]:
        """Get current task probabilities."""
        return self.task_probabilities.copy()
    
    def get_scheduling_info(self) -> Dict[str, Any]:
        """Get comprehensive scheduling information."""
        
        info = {
            'strategy': self.config.strategy,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'task_probabilities': self.task_probabilities.copy(),
            'task_step_counts': self.task_step_counts.copy()
        }
        
        if self.config.strategy == "curriculum":
            info['current_curriculum_stage'] = self._get_current_curriculum_stage()
        
        if self.config.strategy == "adaptive":
            # Add performance trends
            trends = {}
            for task in self.task_names:
                if len(self.task_performance_history[task]) >= 2:
                    recent = self.task_performance_history[task][-5:]  # Last 5 epochs
                    x = np.arange(len(recent))
                    y = np.array(recent)
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        trends[task] = slope
                    else:
                        trends[task] = 0.0
                else:
                    trends[task] = 0.0
            
            info['performance_trends'] = trends
        
        return info


class CurriculumBuilder:
    """Helper class for building curriculum learning configurations."""
    
    @staticmethod
    def create_difficulty_based_curriculum(
        tasks: List[str],
        task_difficulties: Dict[str, float],
        num_stages: int = 3,
        stage_epochs: List[int] = None
    ) -> Dict[str, Any]:
        """
        Create curriculum based on task difficulties.
        
        Args:
            tasks: List of task names
            task_difficulties: Difficulty score for each task (0-1, higher = more difficult)
            num_stages: Number of curriculum stages
            stage_epochs: Epochs at which to transition between stages
            
        Returns:
            Curriculum configuration
        """
        
        if stage_epochs is None:
            stage_epochs = [i * 5 for i in range(1, num_stages)]
        
        # Sort tasks by difficulty
        sorted_tasks = sorted(tasks, key=lambda t: task_difficulties.get(t, 0.5))
        
        stages = []
        tasks_per_stage = len(tasks) // num_stages
        
        for stage_idx in range(num_stages):
            start_idx = 0  # Always include easiest tasks
            end_idx = min((stage_idx + 1) * tasks_per_stage, len(tasks))
            
            stage_tasks = sorted_tasks[start_idx:end_idx + tasks_per_stage]
            
            # Higher weight for more difficult tasks in later stages
            weights = {}
            for task in stage_tasks:
                difficulty = task_difficulties.get(task, 0.5)
                if stage_idx == 0:
                    # Early stage: focus on easier tasks
                    weights[task] = max(0.1, 1.0 - difficulty)
                else:
                    # Later stages: more balanced or focus on harder tasks
                    weights[task] = 0.5 + 0.5 * difficulty
            
            stages.append({
                'tasks': stage_tasks,
                'weights': weights,
                'description': f'Stage {stage_idx + 1}: Difficulty-based'
            })
        
        return {
            'strategy': 'curriculum',
            'curriculum_stages': stages,
            'stage_transition_epochs': stage_epochs
        }
    
    @staticmethod
    def create_prerequisite_curriculum(
        task_prerequisites: Dict[str, List[str]],
        stage_epochs: List[int] = None
    ) -> Dict[str, Any]:
        """
        Create curriculum based on task prerequisites.
        
        Args:
            task_prerequisites: Dict mapping tasks to their prerequisite tasks
            stage_epochs: Epochs at which to transition between stages
            
        Returns:
            Curriculum configuration
        """
        
        # Topological sort to determine order
        def topological_sort(prereqs):
            in_degree = {task: 0 for task in prereqs}
            for task, deps in prereqs.items():
                for dep in deps:
                    if dep in in_degree:
                        in_degree[task] += 1
            
            stages = []
            remaining_tasks = set(prereqs.keys())
            
            while remaining_tasks:
                # Find tasks with no dependencies
                current_stage = [task for task in remaining_tasks if in_degree[task] == 0]
                
                if not current_stage:
                    # Circular dependency, add all remaining
                    current_stage = list(remaining_tasks)
                
                stages.append(current_stage)
                
                # Remove current stage tasks and update dependencies
                for task in current_stage:
                    remaining_tasks.remove(task)
                    for other_task in remaining_tasks:
                        if task in prereqs[other_task]:
                            in_degree[other_task] -= 1
            
            return stages
        
        ordered_stages = topological_sort(task_prerequisites)
        
        if stage_epochs is None:
            stage_epochs = [i * 3 for i in range(1, len(ordered_stages))]
        
        curriculum_stages = []
        cumulative_tasks = []
        
        for stage_idx, stage_tasks in enumerate(ordered_stages):
            cumulative_tasks.extend(stage_tasks)
            
            # Equal weights within stage, but include all previous tasks
            weights = {task: 1.0 for task in cumulative_tasks}
            
            curriculum_stages.append({
                'tasks': cumulative_tasks.copy(),
                'weights': weights,
                'description': f'Stage {stage_idx + 1}: Prerequisites-based'
            })
        
        return {
            'strategy': 'curriculum',
            'curriculum_stages': curriculum_stages,
            'stage_transition_epochs': stage_epochs
        }


def main():
    """Example usage of task scheduling."""
    
    tasks = ['sarcasm_detection', 'paraphrasing', 'fact_verification']
    
    # Example 1: Adaptive scheduling
    adaptive_config = SchedulerConfig(
        strategy="adaptive",
        adaptation_rate=0.1,
        lookback_window=5,
        min_task_probability=0.1,
        log_scheduling_decisions=True
    )
    
    scheduler = TaskScheduler(adaptive_config, tasks)
    
    # Simulate training steps
    for step in range(100):
        task = scheduler.get_next_task(step)
        
        if step % 10 == 0:
            print(f"Step {step}: Selected task {task}")
    
    # Simulate epoch update with performance metrics
    mock_metrics = {
        'val_sarcasm_detection_f1': 0.75,
        'val_paraphrasing_f1': 0.65,
        'val_fact_verification_f1': 0.80
    }
    
    scheduler.update_epoch(1, mock_metrics)
    
    print(f"Current probabilities: {scheduler.get_current_probabilities()}")
    
    # Example 2: Curriculum learning
    curriculum_config = CurriculumBuilder.create_difficulty_based_curriculum(
        tasks=tasks,
        task_difficulties={
            'sarcasm_detection': 0.3,
            'paraphrasing': 0.8,
            'fact_verification': 0.6
        },
        num_stages=2,
        stage_epochs=[5]
    )
    
    print(f"Curriculum configuration: {curriculum_config}")


if __name__ == "__main__":
    main()
