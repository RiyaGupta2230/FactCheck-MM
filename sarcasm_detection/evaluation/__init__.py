# sarcasm_detection/evaluation/__init__.py
"""
Sarcasm Detection Evaluation Framework
Comprehensive evaluation, ablation studies, error analysis, and visualizations.
"""

from .evaluator import (
    SarcasmEvaluator,
    ModelEvaluator,
    DatasetEvaluator,
    CrossDatasetEvaluator
)
from .ablation_study import (
    ModalityAblationStudy,
    ArchitectureAblationStudy,
    DatasetAblationStudy,
    AblationAnalyzer
)
from .error_analysis import (
    ErrorAnalyzer,
    MisclassificationAnalyzer,
    FailureCaseAnalyzer,
    ErrorPatternDetector
)
from .visualizations import (
    SarcasmVisualizer,
    ResultsVisualizer,
    AttentionVisualizer,
    PerformanceVisualizer
)

__all__ = [
    # Core evaluators
    "SarcasmEvaluator",
    "ModelEvaluator", 
    "DatasetEvaluator",
    "CrossDatasetEvaluator",
    
    # Ablation studies
    "ModalityAblationStudy",
    "ArchitectureAblationStudy",
    "DatasetAblationStudy",
    "AblationAnalyzer",
    
    # Error analysis
    "ErrorAnalyzer",
    "MisclassificationAnalyzer",
    "FailureCaseAnalyzer",
    "ErrorPatternDetector",
    
    # Visualizations
    "SarcasmVisualizer",
    "ResultsVisualizer",
    "AttentionVisualizer",
    "PerformanceVisualizer"
]

def create_evaluator(
    evaluator_type: str,
    model,
    datasets=None,
    **kwargs
):
    """
    Factory function to create evaluation components.
    
    Args:
        evaluator_type: Type of evaluator
        model: Model to evaluate
        datasets: Datasets for evaluation
        **kwargs: Additional arguments
        
    Returns:
        Evaluator instance
    """
    evaluator_type = evaluator_type.lower()
    
    if evaluator_type == "model":
        return ModelEvaluator(model, **kwargs)
    elif evaluator_type == "dataset":
        return DatasetEvaluator(model, datasets, **kwargs)
    elif evaluator_type == "cross_dataset":
        return CrossDatasetEvaluator(model, datasets, **kwargs)
    elif evaluator_type == "ablation":
        return AblationAnalyzer(model, datasets, **kwargs)
    elif evaluator_type == "error":
        return ErrorAnalyzer(model, datasets, **kwargs)
    elif evaluator_type == "visualizer":
        return SarcasmVisualizer(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
