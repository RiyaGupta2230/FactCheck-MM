"""
FactCheck-MM Experiments Module

Comprehensive experimental framework for hyperparameter tuning, ablation studies,
multitask learning, and performance benchmarking across all FactCheck-MM tasks.

Modules:
- hyperparameter_tuning: Automated and manual hyperparameter optimization
- ablation_studies: Systematic component removal and analysis
- multitask_learning: Joint training across multiple tasks
- benchmarking: Performance and resource usage benchmarking
"""

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"

# Import key experiment classes for convenience
from .hyperparameter_tuning.optuna_tuning import OptunaHyperparameterTuner
from .hyperparameter_tuning.grid_search import GridSearchTuner
from .ablation_studies.modality_ablation import ModalityAblationStudy
from .ablation_studies.architecture_ablation import ArchitectureAblationStudy
from .ablation_studies.dataset_ablation import DatasetAblationStudy
from .multitask_learning.joint_trainer import MultitaskTrainer
from .multitask_learning.task_scheduling import TaskScheduler
from .benchmarking.speed_benchmarks import SpeedBenchmarker
from .benchmarking.memory_profiling import MemoryProfiler

__all__ = [
    "OptunaHyperparameterTuner",
    "GridSearchTuner",
    "ModalityAblationStudy",
    "ArchitectureAblationStudy", 
    "DatasetAblationStudy",
    "MultitaskTrainer",
    "TaskScheduler",
    "SpeedBenchmarker",
    "MemoryProfiler"
]

# Experiment configuration defaults
EXPERIMENT_CONFIG = {
    "output_dir": "outputs/experiments",
    "log_level": "INFO",
    "save_checkpoints": True,
    "track_metrics": True,
    "use_tensorboard": True,
    "use_wandb": False
}
