"""
Logging Utilities for FactCheck-MM
Rich logging, WandB, and TensorBoard integration.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from datetime import datetime
import json

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        use_rich: Whether to use rich formatting
        format_string: Custom format string
        
    Returns:
        Root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=False
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
    
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info():
    """Log system and hardware information."""
    logger = get_logger("system_info")
    
    if RICH_AVAILABLE:
        console = Console()
        
        # System info table
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Python Version", sys.version.split()[0])
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("CUDA Available", str(torch.cuda.is_available()))
        
        if torch.cuda.is_available():
            table.add_row("CUDA Version", str(torch.version.cuda))
            table.add_row("GPU Count", str(torch.cuda.device_count()))
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                table.add_row(f"GPU {i}", f"{gpu_name} ({gpu_memory:.1f}GB)")
        
        console.print(table)
    else:
        logger.info(f"Python Version: {sys.version.split()[0]}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str,
        enabled: bool = True
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Log directory
            experiment_name: Experiment name
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        
        if self.enabled:
            self.writer = SummaryWriter(log_dir / experiment_name)
            self.logger = get_logger("tensorboard")
            self.logger.info(f"TensorBoard logging to {log_dir / experiment_name}")
        else:
            self.writer = None
            if not TENSORBOARD_AVAILABLE:
                get_logger("tensorboard").warning("TensorBoard not available")
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(name, value, step)
    
    def log_scalars(self, name: str, values: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled and self.writer:
            self.writer.add_scalars(name, values, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if self.enabled and self.writer:
            self.writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: torch.Tensor, step: int):
        """Log image."""
        if self.enabled and self.writer:
            self.writer.add_image(name, image, step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer:
            self.writer.add_text(name, text, step)
    
    def log_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph."""
        if self.enabled and self.writer:
            self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close logger."""
        if self.enabled and self.writer:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logging wrapper."""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize W&B logger.
        
        Args:
            project_name: W&B project name
            experiment_name: Experiment name
            config: Configuration dictionary
            tags: Experiment tags
            notes: Experiment notes
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.logger = get_logger("wandb")
        
        if self.enabled:
            try:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    tags=tags,
                    notes=notes,
                    reinit=True
                )
                self.logger.info(f"W&B logging initialized for project: {project_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize W&B: {e}")
                self.enabled = False
        else:
            if not WANDB_AVAILABLE:
                self.logger.warning("W&B not available")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_artifact(self, artifact_path: Path, artifact_type: str, name: str):
        """Log artifact."""
        if self.enabled:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 1000):
        """Watch model parameters."""
        if self.enabled:
            wandb.watch(model, log_freq=log_freq)
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()


class ExperimentLogger:
    """Combined experiment logger with multiple backends."""
    
    def __init__(
        self,
        log_dir: Path,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Base log directory
            project_name: Project name
            experiment_name: Experiment name
            config: Configuration
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use W&B
            tags: Experiment tags
            notes: Experiment notes
        """
        self.experiment_name = experiment_name
        self.config = config
        self.logger = get_logger("experiment")
        
        # Initialize loggers
        self.tensorboard = TensorBoardLogger(
            log_dir / "tensorboard",
            experiment_name,
            enabled=use_tensorboard
        )
        
        self.wandb = WandBLogger(
            project_name,
            experiment_name,
            config,
            tags=tags,
            notes=notes,
            enabled=use_wandb
        )
        
        # Create experiment directory
        self.experiment_dir = log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Experiment logger initialized: {experiment_name}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """Log metrics to all backends."""
        
        # Add prefix if provided
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics
        
        # Log to TensorBoard
        for name, value in prefixed_metrics.items():
            self.tensorboard.log_scalar(name, value, step)
        
        # Log to W&B
        self.wandb.log(prefixed_metrics, step=step)
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture and parameters."""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        self.logger.info(f"Model info: {model_info}")
        self.wandb.log({"model": model_info})
        
        # Save model architecture
        with open(self.experiment_dir / "model_architecture.txt", "w") as f:
            f.write(str(model))
    
    def save_checkpoint_info(self, checkpoint_path: Path, metrics: Dict[str, float]):
        """Save checkpoint information."""
        
        checkpoint_info = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics
        }
        
        # Append to checkpoint log
        with open(self.experiment_dir / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(checkpoint_info) + "\n")
    
    def close(self):
        """Close all loggers."""
        self.tensorboard.close()
        self.wandb.finish()
        self.logger.info("Experiment logging finished")
