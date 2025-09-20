"""
Training Configuration for FactCheck-MM
Hyperparameters, optimization settings, and training strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    name: str = "adamw"  # adamw, adam, sgd, adafactor
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # AdaFactor specific (for T5/BART)
    scale_parameter: bool = True
    relative_step_size: bool = True
    warmup_init: bool = False


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    
    name: str = "cosine_with_restarts"  # linear, cosine, cosine_with_restarts, polynomial
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    
    # Cosine specific
    num_cycles: float = 0.5
    
    # Polynomial specific  
    power: float = 1.0
    
    # Step scheduler
    step_size: int = 10000
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    """Base training configuration."""
    
    # Basic training settings
    num_epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False  # Use bf16 on newer hardware
    tf32: bool = True   # Use TF32 on Ampere GPUs
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Advanced training techniques
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_start: float = 0.8  # Start SWA at 80% of training
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class SarcasmTrainingConfig(TrainingConfig):
    """Training configuration for sarcasm detection."""
    
    num_epochs: int = 15
    batch_size: int = 12  # Reduced for multimodal data
    learning_rate: float = 1e-5
    
    # Sarcasm-specific settings
    class_weights: Optional[List[float]] = None  # Auto-computed from data
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = True
    
    # Multimodal fusion
    modality_dropout: float = 0.1
    cross_modal_attention_dropout: float = 0.1
    
    # Data augmentation
    use_text_augmentation: bool = True
    use_audio_augmentation: bool = True
    use_image_augmentation: bool = True
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "difficulty_based"  # difficulty_based, length_based


@dataclass  
class ParaphrasingTrainingConfig(TrainingConfig):
    """Training configuration for paraphrasing."""
    
    num_epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 3e-5
    
    # Generation-specific
    max_source_length: int = 512
    max_target_length: int = 512
    generation_max_length: int = 512
    generation_num_beams: int = 4
    
    # Reinforcement Learning
    use_rl: bool = True
    rl_start_epoch: int = 3
    rl_learning_rate: float = 1e-6
    ppo_epochs: int = 4
    ppo_clip_range: float = 0.2
    
    # Reward function weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "bleu": 0.25,
        "rouge": 0.25, 
        "bert_score": 0.25,
        "semantic_similarity": 0.25
    })
    
    # Quality filtering
    min_quality_score: float = 0.7
    use_quality_filtering: bool = True


@dataclass
class FactVerificationTrainingConfig(TrainingConfig):
    """Training configuration for fact verification."""
    
    num_epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 2e-5
    
    # Multi-stage training
    stages: List[str] = field(default_factory=lambda: ["retrieval", "verification", "end_to_end"])
    stage_epochs: Dict[str, int] = field(default_factory=lambda: {
        "retrieval": 4,
        "verification": 4, 
        "end_to_end": 4
    })
    
    # Retrieval settings
    retrieval_batch_size: int = 32
    num_hard_negatives: int = 7
    
    # Evidence processing
    max_evidence_length: int = 256
    max_num_evidence: int = 10
    
    # RAG settings
    rag_n_docs: int = 5
    rag_max_combined_length: int = 1024


@dataclass
class ChunkedTrainingConfig:
    """Configuration for chunked training on resource-constrained devices."""
    
    enabled: bool = False
    chunk_size: int = 1000  # Number of samples per chunk
    max_memory_gb: float = 7.0  # Maximum memory usage
    
    # Optimization for low-resource training
    micro_batch_size: int = 2
    gradient_accumulation_multiplier: int = 4
    
    # Checkpoint management
    save_every_chunk: bool = True
    resume_from_chunk: bool = True
    
    # Memory optimization
    use_cpu_offload: bool = True
    use_gradient_checkpointing: bool = True
    empty_cache_frequency: int = 10


@dataclass
class TrainingConfigs:
    """Complete training configuration container."""
    
    # Task-specific configurations
    sarcasm_detection: SarcasmTrainingConfig = field(default_factory=SarcasmTrainingConfig)
    paraphrasing: ParaphrasingTrainingConfig = field(default_factory=ParaphrasingTrainingConfig)
    fact_verification: FactVerificationTrainingConfig = field(default_factory=FactVerificationTrainingConfig)
    
    # Chunked training for MacBook Air M2
    chunked_training: ChunkedTrainingConfig = field(default_factory=ChunkedTrainingConfig)
    
    # Multi-task learning
    multitask_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "task_sampling": "proportional",  # proportional, uniform, temperature
        "temperature": 2.0,
        "gradient_surgery": True,
        "task_balancing_method": "uncertainty_weighting"
    })
    
    # Distributed training
    distributed_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "backend": "nccl",
        "find_unused_parameters": True,
        "gradient_as_bucket_view": True
    })
    
    # Experiment tracking
    tracking_config: Dict[str, Any] = field(default_factory=lambda: {
        "log_model_architecture": True,
        "log_gradients": False,
        "log_parameters": True,
        "log_predictions_frequency": 1000,
        "save_model_checkpoints": True
    })
    
    def get_config_for_task(self, task: str) -> TrainingConfig:
        """Get training configuration for specific task."""
        configs = {
            "sarcasm_detection": self.sarcasm_detection,
            "paraphrasing": self.paraphrasing,
            "fact_verification": self.fact_verification
        }
        
        if task not in configs:
            raise ValueError(f"Unknown task: {task}. Available: {list(configs.keys())}")
        
        return configs[task]
    
    def update_for_device_constraints(self, available_memory_gb: float, gpu_memory_gb: float):
        """Update configurations based on available hardware resources."""
        
        # Enable chunked training for limited resources
        if available_memory_gb < 10.0 or gpu_memory_gb < 6.0:
            self.chunked_training.enabled = True
            
            # Reduce batch sizes
            for config in [self.sarcasm_detection, self.paraphrasing, self.fact_verification]:
                config.batch_size = max(2, config.batch_size // 2)
                config.eval_batch_size = max(4, config.eval_batch_size // 2)
                config.gradient_accumulation_steps *= 2
        
        # Optimize for Apple Silicon
        if gpu_memory_gb == 0:  # Likely CPU or MPS
            for config in [self.sarcasm_detection, self.paraphrasing, self.fact_verification]:
                config.fp16 = False  # MPS doesn't support fp16 well
                config.tf32 = False
                config.dataloader_num_workers = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sarcasm_detection": self.sarcasm_detection.__dict__,
            "paraphrasing": self.paraphrasing.__dict__,
            "fact_verification": self.fact_verification.__dict__,
            "chunked_training": self.chunked_training.__dict__,
            "multitask_config": self.multitask_config,
            "distributed_config": self.distributed_config,
            "tracking_config": self.tracking_config
        }
