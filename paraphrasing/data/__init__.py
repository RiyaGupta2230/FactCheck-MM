# paraphrasing_detection/data/__init__.py

"""
Paraphrasing Detection Data Loaders
Research-grade implementations with strict PDF compliance.
"""

from .paranmt_loader import ParaNMTDataset, ParaNMTConfig
from .mrpc_loader import MRPCDataset, MRPCConfig
from .quora_loader import QuoraDataset, QuoraConfig
from .unified_loader import UnifiedParaphraseDataset, UnifiedParaphraseConfig, create_unified_dataloader

__all__ = [
    'ParaNMTDataset',
    'ParaNMTConfig',
    'MRPCDataset',
    'MRPCConfig',
    'QuoraDataset',
    'QuoraConfig',
    'UnifiedParaphraseDataset',
    'UnifiedParaphraseConfig',
    'create_unified_dataloader'
]
