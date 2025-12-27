"""
Base Configuration for FactCheck-MM
Global settings, device configuration, and logging setup.
"""

import os
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    """
    Base configuration class containing global settings.
    """
    
    # Project settings
    project_name: str = "FactCheck-MM"
    version: str = "1.0.0"
    random_seed: int = 42
    
    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cuda_devices: Optional[str] = None
    mixed_precision: bool = True
    compile_model: bool = True  # PyTorch 2.0 compilation
    
    # Memory and performance
    num_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Chunked training for resource-constrained devices (MacBook Air M2)
    enable_chunked_training: bool = False
    chunk_size: int = 1000  # Number of samples per chunk
    memory_limit_gb: float = 7.0  # Memory limit in GB
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    rich_logging: bool = True
    
    # Monitoring and tracking
    use_wandb: bool = True
    use_tensorboard: bool = True
    use_mlflow: bool = False
    
    # Experiment tracking
    experiment_name: str = field(default_factory=lambda: f"factcheck_mm_experiment")
    run_name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._setup_directories()
        self._setup_device()
        self._set_random_seeds()
        self._detect_system_capabilities()
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.checkpoint_dir, self.logs_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        """Configure device settings."""
        if self.device == "cuda":
            if self.cuda_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
            
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory_gb < 6.0:  # Less than 6GB VRAM
                    self.enable_chunked_training = True
                    self.mixed_precision = True
                    print(f"⚠️  Low GPU memory detected ({gpu_memory_gb:.1f}GB). Enabling chunked training.")
        
        elif self.device == "mps":  # Apple Silicon
            self.enable_chunked_training = True
            self.mixed_precision = True
            print("🍎 Apple Silicon detected. Optimizing for M1/M2.")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _detect_system_capabilities(self):
        """Detect and optimize for system capabilities."""
        import psutil
        
        # Memory detection
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        
        if available_memory_gb < 10.0:  # Less than 10GB RAM
            self.enable_chunked_training = True
            self.num_workers = max(1, self.num_workers // 2)
            print(f"⚠️  Limited system memory detected ({available_memory_gb:.1f}GB). Optimizing settings.")
        
        # CPU detection
        cpu_count = os.cpu_count() or 1
        if cpu_count < 4:
            self.num_workers = max(1, cpu_count - 1)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "chunked_training": self.enable_chunked_training
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "cuda_version": torch.version.cuda
            })
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
