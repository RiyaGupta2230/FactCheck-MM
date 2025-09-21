"""
Multitask Learning Module

Joint training across multiple FactCheck-MM tasks with shared representations
and task-specific heads. Includes sophisticated task scheduling strategies.
"""

from .joint_trainer import MultitaskTrainer
from .task_scheduling import TaskScheduler

__all__ = ["MultitaskTrainer", "TaskScheduler"]
