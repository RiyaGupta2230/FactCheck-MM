"""
Paraphrasing Data Module

This module provides unified data loading capabilities for paraphrasing tasks
in the FactCheck-MM project, supporting ParaNMT-5M, MRPC, and Quora datasets.
"""

from .paranmt_loader import ParaNMTLoader
from .mrpc_loader import MRPCDataset
from .quora_loader import QuoraDataset
from .unified_loader import UnifiedParaphraseDataset

__all__ = [
    "ParaNMTLoader",
    "MRPCDataset", 
    "QuoraDataset",
    "UnifiedParaphraseDataset"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
