"""
Model Configuration for FactCheck-MM
Architecture specifications for sarcasm detection, paraphrasing, and fact verification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class SarcasmDetectionConfig:
    """Configuration for sarcasm detection models."""
    
    # Text encoder
    text_model_name: str = "roberta-large"
    text_max_length: int = 512
    text_dropout: float = 0.1
    
    # Audio encoder (Wav2Vec2)
    audio_model_name: str = "facebook/wav2vec2-large-960h"
    audio_max_length: int = 16000 * 10  # 10 seconds at 16kHz
    audio_dropout: float = 0.1
    
    # Vision encoder (ViT)
    vision_model_name: str = "google/vit-large-patch16-224"
    image_size: int = 224
    vision_dropout: float = 0.1
    
    # Fusion configuration
    fusion_strategy: str = "cross_modal_attention"  # Options: concat, attention, cross_modal_attention
    fusion_hidden_dim: int = 768
    fusion_num_heads: int = 12
    fusion_num_layers: int = 2
    
    # Classification head
    num_classes: int = 2  # Binary classification
    classifier_hidden_dim: int = 512
    classifier_dropout: float = 0.3
    
    # Model ensemble
    use_ensemble: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ["text_only", "multimodal", "cross_modal"])


@dataclass
class ParaphrasingConfig:
    """Configuration for paraphrase generation models."""
    
    # Base generation model
    base_model_name: str = "t5-large"  # or "facebook/bart-large"
    max_input_length: int = 512
    max_target_length: int = 512
    
    # Generation parameters
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Sarcasm-aware conditioning
    sarcasm_conditioning: bool = True
    sarcasm_embed_dim: int = 128
    
    # Reinforcement learning
    use_rl_optimization: bool = True
    rl_algorithm: str = "ppo"  # Options: reinforce, actor_critic, ppo
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic_similarity": 0.4,
        "fluency": 0.3,
        "factual_consistency": 0.3
    })
    
    # Quality scorer
    quality_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: ["simple", "moderate", "complex"])


@dataclass
class FactVerificationConfig:
    """Configuration for fact verification models."""
    
    # Claim detection
    claim_model_name: str = "roberta-large"
    claim_max_length: int = 256
    
    # Evidence retrieval (DPR)
    retrieval_model_name: str = "facebook/dpr-ctx_encoder-multiset-base"
    question_encoder_name: str = "facebook/dpr-question_encoder-multiset-base"
    top_k_evidence: int = 10
    evidence_max_length: int = 256
    
    # Verification model
    verifier_model_name: str = "roberta-large"
    verifier_max_length: int = 512
    
    # RAG configuration
    use_rag: bool = True
    rag_model_name: str = "facebook/rag-sequence-nq"
    
    # Knowledge bases
    knowledge_sources: List[str] = field(default_factory=lambda: ["wikipedia", "wikidata"])
    use_web_search: bool = True
    max_web_results: int = 5
    
    # Multi-hop reasoning
    max_reasoning_hops: int = 3
    reasoning_strategy: str = "iterative"  # Options: single_hop, iterative, graph_based
    
    # Classification
    num_classes: int = 3  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    stance_classes: int = 4  # AGREE, DISAGREE, DISCUSS, UNRELATED


@dataclass
class ModelConfigs:
    """Complete model configuration container."""
    
    sarcasm_detection: SarcasmDetectionConfig = field(default_factory=SarcasmDetectionConfig)
    paraphrasing: ParaphrasingConfig = field(default_factory=ParaphrasingConfig)
    fact_verification: FactVerificationConfig = field(default_factory=FactVerificationConfig)
    
    # Shared multimodal encoder settings
    shared_encoder: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 1024,
        "num_attention_heads": 16,
        "num_layers": 6,
        "intermediate_size": 4096,
        "activation": "gelu",
        "layer_norm_eps": 1e-6,
        "dropout": 0.1
    })
    
    # Multi-task learning
    multitask_learning: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "shared_encoder": True,
        "task_weights": {
            "sarcasm_detection": 1.0,
            "paraphrasing": 1.0,
            "fact_verification": 1.0
        },
        "gradient_surgery": True,  # PCGrad or similar
        "task_balancing": "uncertainty_weighting"
    })
    
    # Model optimization
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        "use_gradient_checkpointing": True,
        "use_flash_attention": True,
        "quantization": None,  # Options: None, "8bit", "4bit"
        "pruning": None,  # Options: None, "magnitude", "structured"
        "distillation": False
    })
    
    def get_model_config(self, task: str) -> Any:
        """Get configuration for specific task."""
        task_configs = {
            "sarcasm_detection": self.sarcasm_detection,
            "paraphrasing": self.paraphrasing,
            "fact_verification": self.fact_verification
        }
        
        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_configs.keys())}")
        
        return task_configs[task]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sarcasm_detection": self.sarcasm_detection.__dict__,
            "paraphrasing": self.paraphrasing.__dict__,
            "fact_verification": self.fact_verification.__dict__,
            "shared_encoder": self.shared_encoder,
            "multitask_learning": self.multitask_learning,
            "optimization": self.optimization
        }
