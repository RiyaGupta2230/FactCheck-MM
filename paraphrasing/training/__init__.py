"""
Paraphrasing Training Module

This module provides comprehensive training capabilities for paraphrasing models
including supervised training, reinforcement learning, curriculum learning,
and quality scorer training.
"""

from .train_generation import train_t5, train_bart
from .train_rl import train_rl
from .curriculum_trainer import train_curriculum
from .quality_trainer import train_quality

__all__ = [
    "train_t5",
    "train_bart",
    "train_rl", 
    "train_curriculum",
    "train_quality"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
