
# Sarcasm Detection Module

A comprehensive multimodal sarcasm detection system as part of the FactCheck-MM pipeline. This module implements state-of-the-art approaches for detecting sarcasm across text, audio, image, and video modalities using transformer-based architectures and ensemble methods.

## Overview

The sarcasm detection module serves as **Task 1** in the FactCheck-MM pipeline, providing crucial contextual understanding for fact-checking systems. By identifying sarcastic content, the module helps distinguish between literal statements and ironic expressions, which is essential for accurate fact verification and paraphrase generation downstream.

### Key Features

- **Multimodal Architecture**: Supports text, audio, image, and video inputs
- **Multiple Model Types**: Text-only, multimodal, and ensemble approaches
- **7 Dataset Support**: Comprehensive evaluation across diverse sarcasm datasets
- **Production Ready**: Memory-efficient training, chunked loading, and robust evaluation
- **Research Grade**: Advanced metrics, ablation studies, and error analysis

## Module Structure

```
sarcasm_detection/
├── data/                   # Dataset loaders and preprocessing
├── models/                 # Model architectures (text, multimodal, ensemble)
├── training/               # Training scripts and strategies
├── evaluation/             # Comprehensive evaluation and analysis
├── utils/                  # Specialized metrics and data augmentation
└── checkpoints/            # Model checkpoints (created during training)
```

### Submodules

- **`data/`**: Dataset loaders for 7 sarcasm datasets with unified interface
- **`models/`**: RoBERTa-based text models, multimodal fusion architectures, and ensemble methods
- **`training/`**: Trainers for text/multimodal models, chunked loading for memory efficiency, curriculum learning
- **`evaluation/`**: Standard metrics, ablation studies, error analysis, and rich visualizations
- **`utils/`**: Sarcasm-specific metrics, advanced data augmentation, and helper functions
- **`checkpoints/`**: Automatically created directory for storing model checkpoints

## Supported Datasets

| Dataset | Modalities | Size | Description |
|---------|------------|------|-------------|
| **MUStARD** | Text + Audio + Video | 690 | TV show clips with balanced sarcasm samples |
| **MMSD2** | Text + Image | 24K | Multimodal sarcasm with image context |
| **SarcNet** | Text + Image | Variable | Separate modality-specific labels |
| **SARC** | Text | 1.3M | Large-scale Reddit sarcasm corpus |
| **Sarcasm Headlines** | Text | 28.6K | Professional news headlines |
| **Spanish Sarcasm** | Text | 1K+ | Multilingual sarcasm detection |
| **UR-FUNNY** | Text + Audio | 16K | TED Talk humor and sarcasm |

## Model Types

### Text-Only Models
- **RoBERTa-based**: Fine-tuned transformer with attention pooling
- **LSTM Baseline**: Lightweight LSTM with attention for comparison

### Multimodal Models
- **Full Multimodal**: Text + Audio + Image + Video fusion
- **Cross-Modal Attention**: Advanced fusion strategies
- **Progressive Training**: Modality-aware curriculum learning

### Ensemble Methods
- **Voting Ensemble**: Hard/soft voting across model predictions
- **Weighted Ensemble**: Learned weights for model combination
- **Stacking Ensemble**: Meta-learner for prediction aggregation

## Training & Evaluation

### Training Features
- **Memory Efficient**: Chunked loading for MacBook M2 (8GB RAM)
- **Mixed Precision**: FP16 training for faster convergence
- **Curriculum Learning**: Progressive difficulty-based training
- **Multimodal Coordination**: Progressive modality unfreezing

### Evaluation Metrics
- **Standard Metrics**: Accuracy, Precision, Recall, F1-Score
- **Sarcasm-Specific**: Detection rates, subtlety analysis, irony markers
- **Calibration**: Expected/Maximum calibration error, Brier score
- **Multimodal**: Cross-modal agreement, modality contributions

## Usage Examples

### Load Unified Dataset

```
from sarcasm_detection.data import UnifiedSarcasmLoader

# Initialize loader with multiple datasets
loader = UnifiedSarcasmLoader(
    data_dir="data/",
    datasets=['mustard', 'mmsd2', 'sarc', 'headlines'],
    split_ratio={'train': 0.8, 'val': 0.1, 'test': 0.1}
)

# Load datasets
train_dataset, val_dataset, test_dataset = loader.load_datasets()
print(f"Loaded {len(train_dataset)} training samples")
```

### Initialize Multimodal Model

```
from sarcasm_detection.models import MultimodalSarcasmModel

# Model configuration
config = {
    'fusion_strategy': 'cross_modal_attention',
    'modalities': ['text', 'audio', 'image', 'video'],
    'text_hidden_dim': 1024,
    'audio_hidden_dim': 768,
    'image_hidden_dim': 768,
    'video_hidden_dim': 768,
    'fusion_output_dim': 512,
    'num_classes': 2
}

# Initialize model
model = MultimodalSarcasmModel(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Train Model

```
from sarcasm_detection.training import MultimodalSarcasmTrainer
from sarcasm_detection.training import MultimodalTrainingConfig

# Training configuration
train_config = MultimodalTrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=15,
    use_mixed_precision=True,
    progressive_unfreezing=True,
    modality_dropout=0.1
)

# Initialize trainer
trainer = MultimodalSarcasmTrainer(
    model=model,
    config=train_config,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Train model
results = trainer.train(
    checkpoint_dir="checkpoints/multimodal_experiment",
    experiment_name="multimodal_sarcasm_v1"
)
```

### Evaluate Model

```
from sarcasm_detection.evaluation import SarcasmEvaluator

# Initialize evaluator
evaluator = SarcasmEvaluator(model)

# Comprehensive evaluation
evaluation_results = evaluator.evaluate_multiple_datasets({
    'test_mmsd2': test_dataset_mmsd2,
    'test_mustard': test_dataset_mustard,
    'test_sarc': test_dataset_sarc
})

# Print results
for dataset, results in evaluation_results['individual_results'].items():
    f1 = results['metrics']['f1']
    accuracy = results['metrics']['accuracy']
    print(f"{dataset}: F1={f1:.3f}, Accuracy={accuracy:.3f}")
```

### Data Augmentation

```
from sarcasm_detection.utils import SarcasmDataAugmenter, AugmentationConfig

# Configure augmentation
aug_config = AugmentationConfig(
    text_augmentation=True,
    audio_augmentation=True,
    preserve_sarcasm_markers=True,
    enhance_irony_signals=True
)

# Initialize augmenter
augmenter = SarcasmDataAugmenter(aug_config)

# Augment dataset
augmented_dataset = augmenter.augment_dataset(
    dataset=train_dataset,
    target_size_multiplier=1.5
)
print(f"Dataset size increased from {len(train_dataset)} to {len(augmented_dataset)}")
```

### Quick Metrics Computation

```
from sarcasm_detection.utils import compute_sarcasm_metrics

# Compute comprehensive sarcasm metrics
metrics = compute_sarcasm_metrics(
    predictions=model_predictions,
    labels=true_labels,
    probabilities=prediction_probabilities
)

print(f"Sarcasm Detection Rate: {metrics['sarcasm_detection_rate']:.3f}")
print(f"Calibration Error: {metrics['expected_calibration_error']:.3f}")
```

## Hardware Requirements

### Minimum Requirements (CPU Training)
- **RAM**: 8GB+ (with chunked loading)
- **Storage**: 5GB+ for datasets and checkpoints
- **Python**: 3.8+

### Recommended (GPU Training)
- **GPU**: NVIDIA RTX 2050 (4GB) or better
- **RAM**: 16GB+ for full multimodal training
- **Storage**: 20GB+ for full datasets and experiments

### Memory-Constrained Training
- Use `ChunkedTrainer` for MacBook M2 (8GB RAM)
- Enable `use_mixed_precision=True`
- Reduce `batch_size` and increase `gradient_accumulation_steps`

## Installation

```
# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install matplotlib seaborn plotly

# Optional: Audio processing
pip install librosa soundfile audiomentations

# Optional: Image/Video processing  
pip install opencv-python albumentations Pillow

# Optional: Advanced NLP augmentation
pip install nlpaug textblob
```

## Performance Benchmarks

| Model Type | MMSD2 F1 | MUStARD F1 | SARC F1 | Avg F1 |
|------------|----------|------------|---------|--------|
| RoBERTa-Text | 0.742 | 0.681 | 0.856 | 0.760 |
| Multimodal | 0.798 | 0.743 | 0.863 | 0.801 |
| Ensemble | 0.812 | 0.757 | 0.871 | 0.813 |

## References & Credits

### Key Papers
- Castro et al. (2019). "Towards Multimodal Sarcasm Detection"
- Cai et al. (2019). "Multi-Modal Sarcasm Detection in Twitter"
- Liu et al. (2022). "Towards Multi-Modal Sarcasm Detection via Hierarchical Congruity Modeling"

### Dataset Sources
- **MUStARD**: Castro et al., "Towards Multimodal Sarcasm Detection"
- **MMSD2**: Yitao Cai et al., "Multi-Modal Sarcasm Detection in Twitter"
- **SARC**: Khodak et al., "A Large Self-Annotated Corpus for Sarcasm"
- **Sarcasm Headlines**: Misra & Arora, "Sarcasm Detection using Hybrid Neural Network"

### Implementation Credits
- Built on HuggingFace Transformers and PyTorch
- Inspired by state-of-the-art multimodal fusion architectures
- Evaluation framework adapted from sklearn and specialized sarcasm metrics

---

For detailed API documentation, see individual module docstrings. For issues and contributions, please refer to the main FactCheck-MM repository.
