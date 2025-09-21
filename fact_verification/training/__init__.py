"""
Fact Verification Training Module

This module provides comprehensive training capabilities for fact verification
components including evidence retrieval, fact verification, end-to-end pipeline
training, and domain adaptation strategies.

Key components:
- train_retrieval: Train dense/sparse evidence retrieval models
- train_verification: Train fact verification models with curriculum learning
- train_end_to_end: Joint training of retrieval and verification pipeline
- domain_adaptation: Fine-tune models for specific domains (political, medical, etc.)

Example Usage:
    >>> from fact_verification.training import train_verification
    >>> 
    >>> # Train fact verifier
    >>> train_verification.main([
    ...     "--epochs", "5",
    ...     "--batch_size", "16", 
    ...     "--learning_rate", "2e-5"
    >>> ])
    >>>
    >>> # Train end-to-end pipeline
    >>> from fact_verification.training import train_end_to_end
    >>> train_end_to_end.main(["--joint_training", "--epochs", "3"])
"""

from . import train_retrieval
from . import train_verification  
from . import train_end_to_end
from . import domain_adaptation

__all__ = [
    "train_retrieval",
    "train_verification", 
    "train_end_to_end",
    "domain_adaptation"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
