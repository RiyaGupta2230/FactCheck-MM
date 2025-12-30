"""
FactCheck-MM Utility Functions
Logging, checkpointing, metrics, and visualization utilities.
"""

from .logging_utils import (
    setup_logging,
    get_logger,
    log_system_info,
    TensorBoardLogger,
    WandBLogger,
    ExperimentLogger
)

from .checkpoint_manager import (
    CheckpointManager,
    ChunkedCheckpointManager,
    ModelState
)

from .metrics import (
    MetricsComputer,
    ClassificationMetrics,
    GenerationMetrics,
    FactVerificationMetrics
)

from .visualization import (
    Visualizer,
    plot_training_curves,
    plot_confusion_matrix,
    plot_attention_heatmap,
    create_results_dashboard
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger", 
    "log_system_info",
    "TensorBoardLogger",
    "WandBLogger",
    
    # Checkpointing
    "CheckpointManager",
    "ChunkedCheckpointManager",
    "ModelState",
    
    # Metrics
    "MetricsComputer",
    "ClassificationMetrics",
    "GenerationMetrics", 
    "FactVerificationMetrics",
    
    # Visualization
    "Visualizer",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_attention_heatmap",
    "create_results_dashboard"
]
