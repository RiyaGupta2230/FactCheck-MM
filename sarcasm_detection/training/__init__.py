# sarcasm_detection/training/__init__.py
"""
Sarcasm Detection Training Modules
Comprehensive training framework for text, multimodal, and ensemble models.
"""

from .train_text import (
    TextSarcasmTrainer,
    RobertaTrainer,
    LSTMTrainer
)
from .train_multimodal import (
    MultimodalSarcasmTrainer,
    MultimodalTrainingConfig
)
from .train_ensemble import (
    EnsembleTrainer,
    EnsembleTrainingConfig
)
from .chunked_trainer import (
    ChunkedTrainer,
    ChunkedTrainingConfig,
    MemoryEfficientTrainer
)
from .curriculum_learning import (
    CurriculumTrainer,
    DifficultyEstimator,
    CurriculumScheduler
)

__all__ = [
    # Text trainers
    "TextSarcasmTrainer",
    "RobertaTrainer", 
    "LSTMTrainer",
    
    # Multimodal trainers
    "MultimodalSarcasmTrainer",
    "MultimodalTrainingConfig",
    
    # Ensemble trainers
    "EnsembleTrainer",
    "EnsembleTrainingConfig",
    
    # Memory-efficient trainers
    "ChunkedTrainer",
    "ChunkedTrainingConfig",
    "MemoryEfficientTrainer",
    
    # Curriculum learning
    "CurriculumTrainer",
    "DifficultyEstimator",
    "CurriculumScheduler"
]

def create_trainer(
    trainer_type: str,
    model,
    config: dict,
    **kwargs
):
    """
    Factory function to create sarcasm detection trainers.
    
    Args:
        trainer_type: Type of trainer
        model: Model to train
        config: Training configuration
        **kwargs: Additional arguments
        
    Returns:
        Trainer instance
    """
    trainer_type = trainer_type.lower()
    
    if trainer_type in ['text', 'roberta', 'lstm']:
        return TextSarcasmTrainer(model, config, **kwargs)
    elif trainer_type == 'multimodal':
        return MultimodalSarcasmTrainer(model, config, **kwargs)
    elif trainer_type == 'ensemble':
        return EnsembleTrainer(model, config, **kwargs)
    elif trainer_type in ['chunked', 'memory_efficient']:
        return ChunkedTrainer(model, config, **kwargs)
    elif trainer_type == 'curriculum':
        return CurriculumTrainer(model, config, **kwargs)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
