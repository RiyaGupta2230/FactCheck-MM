# FactCheck-MM Paraphrasing Module

## Overview

The paraphrasing module is a critical component of the FactCheck-MM pipeline that focuses on generating semantically equivalent text variations while preserving meaning. Paraphrasing plays a crucial role in enhancing fact verification robustness by:

- **Claim Normalization**: Converting diverse claim formulations into standardized forms for consistent fact checking
- **Sarcasm Mitigation**: Reformulating sarcastic claims into neutral, fact-checkable statements
- **Data Augmentation**: Generating paraphrased training samples to improve model generalization
- **Cross-Modal Consistency**: Ensuring consistent representation across different input modalities

This module leverages three primary datasets: **ParaNMT-5M** (5.5M paraphrase pairs), **MRPC** (Microsoft Research Paraphrase Corpus with 5.8K pairs), and **Quora Question Pairs** (400K+ question pairs), providing diverse training data for robust paraphrase generation across different domains and text types.

The paraphrasing component integrates seamlessly with sarcasm detection (reformulating detected sarcastic content) and fact verification (normalizing claims before evidence retrieval and verification), creating a cohesive multimodal fact-checking pipeline.

## Features

### Implemented Models
- **T5Paraphraser**: Text-to-Text Transfer Transformer based generation with fine-tuned T5-base/large models
- **BARTParaphraser**: BART-based sequence-to-sequence generation with denoising pre-training advantages  
- **SarcasmAwareParaphraser**: Sarcasm-conditioned paraphrasing that generates neutral reformulations of sarcastic content
- **RLParaphraser**: Reinforcement learning enhanced paraphrasing with reward-based optimization
- **QualityScorer**: Multi-metric quality assessment using BLEU, ROUGE, semantic similarity, and fluency scores

### Training Strategies
- **Generation Training**: Standard sequence-to-sequence training with teacher forcing
- **Reinforcement Learning**: Policy gradient optimization using BLEU/ROUGE rewards
- **Curriculum Learning**: Progressive training from simple to complex paraphrase pairs
- **Quality Optimization**: Multi-objective training balancing diversity, fluency, and semantic preservation
- **Cross-Domain Adaptation**: Transfer learning across different text domains and styles

### Evaluation Metrics
- **BLEU Scores**: N-gram overlap precision for generation quality assessment
- **ROUGE Metrics**: Recall-oriented evaluation including ROUGE-1, ROUGE-2, and ROUGE-L
- **METEOR**: Alignment-based metric considering synonyms and paraphrases
- **BERTScore**: Semantic similarity using contextualized embeddings
- **Semantic Similarity**: Cosine similarity in sentence embedding space
- **Diversity Metrics**: Self-BLEU and distinct-n for paraphrase variation measurement

## Usage

### Training Paraphrase Models

```bash
# Train T5-based paraphraser on ParaNMT-5M dataset
python paraphrasing/training/train_generation.py \
    --model t5 \
    --dataset paranmt \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 3e-4

# Train BART-based paraphraser with curriculum learning
python paraphrasing/training/train_generation.py \
    --model bart \
    --dataset mrpc \
    --curriculum_learning \
    --epochs 10 \
    --warmup_steps 1000

# Train sarcasm-aware paraphraser
python paraphrasing/training/train_sarcasm_aware.py \
    --base_model t5 \
    --sarcasm_data_path data/sarcasm_detection/ \
    --paraphrase_data_path data/paraphrasing/

###Reinforcement Learning Training

# Train with RL optimization using BLEU rewards
python paraphrasing/training/train_rl.py \
    --base_model_path checkpoints/t5_paraphraser/ \
    --reward_type bleu \
    --episodes 5000 \
    --learning_rate 1e-5

# Train with multi-metric rewards
python paraphrasing/training/train_rl.py \
    --base_model_path checkpoints/bart_paraphraser/ \
    --reward_type multi \
    --reward_weights 0.4,0.3,0.3 \
    --metrics bleu,rouge,bertscore

###Evaluation and Testing
# Evaluate generation quality metrics
python paraphrasing/evaluation/generation_metrics.py \
    --model_path checkpoints/t5_paraphraser/ \
    --test_data data/paraphrasing/test/ \
    --output_dir results/

# Run comprehensive evaluation
python paraphrasing/evaluation/comprehensive_eval.py \
    --models t5,bart,sarcasm_aware \
    --datasets paranmt,mrpc,quora \
    --metrics all


###Interactive Generation
from paraphrasing.models import T5Paraphraser, QualityScorer

# Initialize paraphraser and quality scorer
paraphraser = T5Paraphraser.from_pretrained('checkpoints/t5_paraphraser/')
scorer = QualityScorer()

# Generate paraphrases
original_text = "Climate change is causing global temperatures to rise"
paraphrases = paraphraser.generate(
    original_text, 
    num_return_sequences=3,
    temperature=0.8
)

# Score quality
for paraphrase in paraphrases:
    scores = scorer.score(original_text, paraphrase)
    print(f"Paraphrase: {paraphrase}")
    print(f"Scores: {scores}")

##Folder Structure
paraphrasing/
├── data/                          # Dataset loading and preprocessing
│   ├── __init__.py               
│   ├── paranmt_dataset.py         # ParaNMT-5M dataset handler
│   ├── mrpc_dataset.py            # MRPC dataset handler  
│   ├── quora_dataset.py           # Quora Question Pairs handler
│   └── unified_dataset.py         # Combined dataset interface
├── models/                        # Paraphrase generation models
│   ├── __init__.py
│   ├── t5_paraphraser.py          # T5-based paraphrasing model
│   ├── bart_paraphraser.py        # BART-based paraphrasing model
│   ├── sarcasm_aware_paraphraser.py # Sarcasm-conditioned paraphrasing
│   ├── rl_paraphraser.py          # Reinforcement learning paraphraser
│   └── quality_scorer.py          # Paraphrase quality assessment
├── training/                      # Training scripts and strategies
│   ├── __init__.py
│   ├── train_generation.py        # Standard generation training
│   ├── train_rl.py               # Reinforcement learning training
│   ├── train_sarcasm_aware.py    # Sarcasm-aware training
│   └── curriculum_trainer.py     # Curriculum learning implementation
├── evaluation/                    # Evaluation metrics and scripts
│   ├── __init__.py
│   ├── generation_metrics.py      # BLEU, ROUGE, METEOR evaluation
│   ├── quality_evaluator.py       # Comprehensive quality assessment
│   ├── diversity_metrics.py       # Paraphrase diversity measurement
│   └── comprehensive_eval.py      # Full evaluation pipeline
├── utils/                         # Utility functions and helpers
│   ├── __init__.py
│   ├── text_processing.py         # Text preprocessing utilities
│   ├── reward_functions.py        # RL reward computation
│   └── data_augmentation.py       # Data augmentation strategies
└── checkpoints/                   # Pre-trained model checkpoints
    ├── t5_paraphraser/           # T5 model checkpoints
    ├── bart_paraphraser/         # BART model checkpoints
    ├── sarcasm_aware/            # Sarcasm-aware model checkpoints
    └── rl_optimized/             # RL-optimized model checkpoints

#Results
| Model           | Dataset    | BLEU-4 | ROUGE-L | METEOR | BERTScore | Semantic Sim |
| --------------- | ---------- | ------ | ------- | ------ | --------- | ------------ |
| T5-base         | ParaNMT-5M | TBD    | TBD     | TBD    | TBD       | TBD          |
| T5-large        | ParaNMT-5M | TBD    | TBD     | TBD    | TBD       | TBD          |
| BART-base       | MRPC       | TBD    | TBD     | TBD    | TBD       | TBD          |
| BART-large      | Quora      | TBD    | TBD     | TBD    | TBD       | TBD          |
| SarcasmAware-T5 | Mixed      | TBD    | TBD     | TBD    | TBD       | TBD          |
| RL-T5           | ParaNMT-5M | TBD    | TBD     | TBD    | TBD       | TBD          |

Note: Actual performance scores will be populated after experimental evaluation

#Downstream Task Performance
| Integration         | Task        | Baseline | With Paraphrasing | Improvement |
| ------------------- | ----------- | -------- | ----------------- | ----------- |
| Sarcasm Detection   | F1 Score    | TBD      | TBD               | TBD         |
| Fact Verification   | Accuracy    | TBD      | TBD               | TBD         |
| Claim Normalization | Consistency | TBD      | TBD               | TBD         |


##References
Datasets

ParaNMT-5M: Wieting, J., & Gimpel, K. (2018). "ParaNMT-50M: Pushing the limits of paraphrastic sentence embeddings with millions of machine translations"

MRPC: Dolan, W. B., & Brockett, C. (2005). "Automatically constructing a corpus of sentential paraphrases"

Quora Question Pairs: First Quora Dataset Release: Question Pairs (2017)

Models and Frameworks

T5: Raffel, C. et al. (2020). "Exploring the limits of transfer learning with a unified text-to-text transformer"

BART: Lewis, M. et al. (2020). "BART: Denoising sequence-to-sequence pre-training for natural language generation"

Hugging Face Transformers: Wolf, T. et al. (2020). "Transformers: State-of-the-art natural language processing"

Evaluation Metrics

BLEU: Papineni, K. et al. (2002). "BLEU: a method for automatic evaluation of machine translation"

ROUGE: Lin, C. Y. (2004). "ROUGE: A package for automatic evaluation of summaries"

BERTScore: Zhang, T. et al. (2020). "BERTScore: Evaluating text generation with BERT"

Future Work
Planned Improvements

Multilingual Paraphrasing: Extend support to multiple languages for global fact-checking applications

Larger Model Integration: Incorporate T5-XL, BART-large, and GPT-based models for enhanced generation quality

Knowledge-Grounded Paraphrasing: Integrate external knowledge bases for factually consistent paraphrase generation

Domain-Specific Adaptation: Specialized models for scientific, political, and medical claim paraphrasing

Real-time Optimization: Streamlined inference for production deployment with reduced latency

Interactive Refinement: User feedback integration for iterative paraphrase quality improvement

Research Directions

Controllable Generation: Fine-grained control over paraphrase characteristics (formality, complexity, style)

Multi-Modal Paraphrasing: Integration with image and video content for comprehensive multimodal fact-checking

Adversarial Robustness: Defense mechanisms against adversarial paraphrase attacks

Zero-Shot Transfer: Cross-domain paraphrasing without domain-specific training data

For questions, issues, or contributions, please refer to the main FactCheck-MM repository documentation or create an issue in the project repository.