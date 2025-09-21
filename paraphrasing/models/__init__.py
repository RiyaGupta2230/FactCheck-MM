"""
Paraphrasing Models Module

This module provides state-of-the-art paraphrasing models including T5, BART,
sarcasm-aware generation, reinforcement learning-based training, and quality scoring.
"""

from .t5_paraphraser import T5Paraphraser
from .bart_paraphraser import BARTParaphraser
from .sarcasm_aware_model import SarcasmAwareParaphraser
from .rl_paraphraser import RLParaphraser
from .quality_scorer import QualityScorer

__all__ = [
    "T5Paraphraser",
    "BARTParaphraser", 
    "SarcasmAwareParaphraser",
    "RLParaphraser",
    "QualityScorer"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
