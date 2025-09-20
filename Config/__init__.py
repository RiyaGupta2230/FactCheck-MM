"""
FactCheck-MM Configuration System
Centralized configuration management for the entire pipeline.
"""

from .base_config import BaseConfig
from .model_configs import ModelConfigs
from .dataset_configs import DatasetConfigs
from .training_configs import TrainingConfigs

__all__ = [
    "BaseConfig",
    "ModelConfigs", 
    "DatasetConfigs",
    "TrainingConfigs"
]

def get_config(config_name: str = "default"):
    """
    Get a complete configuration object.
    
    Args:
        config_name (str): Configuration name/profile
        
    Returns:
        Complete configuration object
    """
    base = BaseConfig()
    models = ModelConfigs()
    datasets = DatasetConfigs()
    training = TrainingConfigs()
    
    return {
        "base": base,
        "models": models,
        "datasets": datasets,
        "training": training
    }
