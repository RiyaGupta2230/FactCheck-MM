"""
Fact Verification Data Module

This module provides comprehensive data loading capabilities for fact verification tasks,
including individual loaders for FEVER and LIAR datasets, and a unified loader that
combines both datasets for comprehensive fact-checking model training.

Key components:
- FeverDataset: FEVER dataset loader for claim verification with evidence
- LiarDataset: LIAR dataset loader for political fact-checking 
- UnifiedFactDataset: Combined fact-checking dataset with unified label schema

Example Usage:
    >>> from fact_verification.data import FeverDataset, LiarDataset, UnifiedFactDataset
    >>> 
    >>> # Load individual datasets
    >>> fever_data = FeverDataset('train', max_evidence_length=256)
    >>> liar_data = LiarDataset('train', binary_labels=True)
    >>> 
    >>> # Use unified dataset
    >>> unified_data = UnifiedFactDataset('train', use_both_datasets=True, balance_datasets=True)
    >>> print(f"Unified dataset size: {len(unified_data)}")
    >>>
    >>> # Access sample
    >>> sample = unified_data[0]
    >>> print(f"Claim: {sample['claim']}")
    >>> print(f"Label: {sample['label']}")
"""

from .fever_loader import FeverDataset
from .liar_loader import LiarDataset
from .unified_loader import UnifiedFactDataset

__all__ = [
    "FeverDataset",
    "LiarDataset", 
    "UnifiedFactDataset"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
