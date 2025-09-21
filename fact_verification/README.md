# FactCheck-MM: Fact Verification Module

## Overview

The Fact Verification module is the core of the FactCheck-MM pipeline, responsible for automated fact-checking of textual claims. It determines the veracity of claims by retrieving relevant evidence and performing multi-step reasoning to classify claims as **SUPPORTS**, **REFUTES**, or **NOT_ENOUGH_INFO**.

### Pipeline Stages
1. **Claim Processing**: Normalize and extract factual claims from input text
2. **Evidence Retrieval**: Search for supporting or contradicting evidence from knowledge bases
3. **Stance Detection**: Determine the relationship between claims and evidence
4. **Fact Classification**: Make final veracity judgments based on aggregated evidence

### Dataset Sources
- **FEVER**: Fact Extraction and VERification (185K+ claims/evidence from Wikipedia)
- **LIAR**: 11.2K+ manually labeled short statements from PolitiFact

### Integration
- **Paraphrasing**: Normalizes claim representations for consistent fact-checking
- **Sarcasm Detection**: Handles sarcastic claims by reformulating them into neutral, fact-checkable statements
- **Multimodal Processing**: Supports integration with visual evidence and multimodal claim types


## Features

### Data Processing
- **FEVER Dataset Handler**: Loads Wikipedia-based claims with evidence and annotations
- **LIAR Dataset Handler**: Handles political fact-checking data with rich metadata
- **Unified Dataset Interface**: Combines both datasets with consistent preprocessing
- **Cross-Domain Evaluation**: Tools for evaluating model performance across domains

### Core Models
- **Claim Detector**: Extracts fact-checkable claims using transformer-based classification
- **Evidence Retrievers**:
    - **Dense Retriever**: Semantic retrieval (DPR, ColBERT, sentence transformers, FAISS)
    - **Sparse Retriever**: BM25 and TF-IDF lexical retrieval
    - **Hybrid Retriever**: Combines dense and sparse methods
- **Fact Verifier**: RoBERTa-based classification with RAG integration
- **Stance Detector**: Textual entailment for claim-evidence pairs
- **End-to-End Pipeline**: Integrates all components with configurable flows

### Training Strategies
- **Multi-Task Learning**: Joint training for retrieval, stance, and verification
- **Curriculum Learning**: Progressive training from simple to complex scenarios
- **Domain Adaptation**: Fine-tuning for political, scientific, and medical domains
- **Reinforcement Learning**: Policy-based optimization
- **Knowledge Distillation**: Model compression for efficient deployment

### Evaluation
- **Retrieval Metrics**: Precision@k, Recall@k, MRR, MAP, NDCG
- **Classification Metrics**: Accuracy, macro/micro F1, precision, recall
- **Evidence Quality Analysis**: Measures relevance, diversity, coverage, redundancy
- **Pipeline Evaluation**: End-to-end system performance
- **Error Analysis**: Categorization and visualization of failure cases

### Utilities
- **Claim Processing**: NLP pipeline with NER, coreference, boundary detection
- **Evidence Utilities**: Formatting, deduplication, chunking, multimodal integration
- **Knowledge Base Integration**: Connectors for Wikidata, DBpedia, Wikipedia
- **Text Processing**: Normalization, tokenization, semantic analysis


## Usage

### Training Fact Verification Models

```bash
# Train RoBERTa-based fact verifier on FEVER dataset
python fact_verification/training/train_verification.py \
    --model roberta \
    --dataset fever \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5

# Train with curriculum learning strategy
python fact_verification/training/train_verification.py \
    --model roberta \
    --dataset both \
    --curriculum_learning \
    --epochs 10 \
    --warmup_steps 2000

# Train domain-adapted model for political claims
python fact_verification/training/domain_adaptation.py \
    --source_domain general \
    --target_domain political \
    --adaptation_method adversarial \
    --pretrained_model checkpoints/roberta_verifier/
```

### Training Evidence Retrieval Models

```bash
# Train dense retriever with DPR
python fact_verification/training/train_retrieval.py \
    --model_type dense \
    --encoder_model facebook/dpr-ctx_encoder-single-nq-base \
    --dataset fever \
    --batch_size 32

# Train hybrid retrieval system
python fact_verification/training/train_end_to_end.py \
    --joint_training \
    --dense_weight 0.6 \
    --sparse_weight 0.4 \
    --epochs 8
```

### Running End-to-End Fact Checking

```bash
# Check individual claims using the complete pipeline
python fact_verification/models/end_to_end_model.py \
    --claim "The Eiffel Tower is in Berlin" \
    --enable_evidence_retrieval \
    --enable_stance_detection \
    --top_k_evidence 5

# Batch processing with domain-specific models
python fact_verification/models/end_to_end_model.py \
    --input_file claims.txt \
    --output_file results.json \
    --domain political \
    --confidence_threshold 0.75
```

### Evaluation and Analysis

```bash
# Comprehensive pipeline evaluation
python fact_verification/evaluation/pipeline_evaluation.py \
    --model_path checkpoints/end_to_end_model/ \
    --test_dataset fever \
    --include_evidence_analysis \
    --save_results

# Error analysis with visualization
python fact_verification/evaluation/error_analysis.py \
    --predictions results/predictions.json \
    --ground_truth data/fever/test.json \
    --export_top_errors 50 \
    --generate_visualizations
```

### Interactive Usage

```python
from fact_verification.models import FactCheckPipeline, FactCheckPipelineConfig

# Initialize complete fact-checking pipeline
config = FactCheckPipelineConfig(
    enable_evidence_retrieval=True,
    enable_stance_detection=True,
    enable_fact_verification=True,
    max_evidence_per_claim=5
)

pipeline = FactCheckPipeline.from_pretrained(
    "checkpoints/end_to_end_model/", 
    config=config
)

# Check a claim
result = pipeline.check_fact("COVID-19 vaccines are 95% effective")
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Evidence: {result['evidence'][:2]}")  # Show top 2 evidence pieces
```


## Folder Structure

```text
fact_verification/
├── data/                  # Dataset loading and preprocessing
│   ├── __init__.py
│   ├── fever_dataset.py   # FEVER dataset handler
│   ├── liar_dataset.py    # LIAR dataset handler
│   ├── unified_dataset.py # Combined dataset interface
│   └── data_loaders.py    # Data loading utilities
├── models/                # Core fact verification models
│   ├── __init__.py
│   ├── claim_detector.py
│   ├── evidence_retriever.py
│   ├── fact_verifier.py
│   ├── stance_detector.py
│   └── end_to_end_model.py
├── training/              # Training scripts
│   ├── __init__.py
│   ├── train_retrieval.py
│   ├── train_verification.py
│   ├── train_end_to_end.py
│   └── domain_adaptation.py
├── evaluation/            # Evaluation suite
│   ├── __init__.py
│   ├── fact_check_metrics.py
│   ├── evidence_eval.py
│   ├── pipeline_evaluation.py
│   └── error_analysis.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── claim_processing.py
│   ├── evidence_utils.py
│   └── knowledge_bases.py
├── retrieval/             # Evidence retrieval systems
│   ├── __init__.py
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   └── hybrid_retriever.py
└── checkpoints/           # Pre-trained model checkpoints
    ├── fact_verifier/
    ├── evidence_retriever/
    ├── stance_detector/
    └── end_to_end_model/
```


## Results

### Evidence Retrieval Performance

| Model         | Dataset | Precision@1 | Precision@5 | Recall@5 | MRR | NDCG@5 |
|-------------- | ------- | ----------- | ----------- | -------- | --- | ------- |
| Dense (DPR)   | FEVER   | TBD         | TBD         | TBD      | TBD | TBD     |
| Sparse (BM25) | FEVER   | TBD         | TBD         | TBD      | TBD | TBD     |
| Hybrid        | FEVER   | TBD         | TBD         | TBD      | TBD | TBD     |
| Dense (DPR)   | LIAR    | TBD         | TBD         | TBD      | TBD | TBD     |
| Sparse (BM25) | LIAR    | TBD         | TBD         | TBD      | TBD | TBD     |
| Hybrid        | LIAR    | TBD         | TBD         | TBD      | TBD | TBD     |

### Fact Verification Accuracy

| Model          | Dataset | Accuracy | Macro F1 | Precision | Recall | SUPPORTS F1 | REFUTES F1 | NEI F1 |
|--------------- | ------- | -------- | -------- | --------- | ------ | ----------- | ---------- | ------ |
| RoBERTa-base   | FEVER   | TBD      | TBD      | TBD       | TBD    | TBD         | TBD        | TBD    |
| RoBERTa-large  | FEVER   | TBD      | TBD      | TBD       | TBD    | TBD         | TBD        | TBD    |
| RoBERTa+RAG    | FEVER   | TBD      | TBD      | TBD       | TBD    | TBD         | TBD        | TBD    |
| RoBERTa-base   | LIAR    | TBD      | TBD      | TBD       | TBD    | TBD         | TBD        | TBD    |
| Domain-Adapted | LIAR    | TBD      | TBD      | TBD       | TBD    | TBD         | TBD        | TBD    |

### End-to-End Pipeline Performance

| Pipeline Configuration | Dataset | Accuracy | Pipeline F1 | Avg Processing Time | Evidence Quality |
|----------------------- | ------- | -------- | ----------- | ------------------- | ---------------- |
| Full Pipeline          | FEVER   | TBD      | TBD         | TBD                 | TBD              |
| Full Pipeline          | LIAR    | TBD      | TBD         | TBD                 | TBD              |
| Hybrid Retrieval       | FEVER   | TBD      | TBD         | TBD                 | TBD              |
| Sarcasm-Aware          | Mixed   | TBD      | TBD         | TBD                 | TBD              |

*Note: Actual performance metrics will be populated after comprehensive experimental evaluation across all model configurations and datasets.*


## References

### Datasets
- **FEVER**: Thorne, J. et al. (2018). "FEVER: a Large-scale Dataset for Fact Extraction and VERification." NAACL-HLT 2018
- **LIAR**: Wang, W. Y. (2017). "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." ACL 2017

### Core Models and Architectures
- **RoBERTa**: Liu, Y. et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv preprint
- **RAG**: Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020
- **DPR**: Karpukhin, V. et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP 2020
- **ColBERT**: Khattab, O. & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020

### Retrieval Methods
- **BM25**: Robertson, S. E. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval
- **TF-IDF**: Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management

### Evaluation Frameworks
- **Hugging Face Transformers**: Wolf, T. et al. (2020). "Transformers: State-of-the-art Natural Language Processing." EMNLP 2020
- **FAISS**: Johnson, J. et al. (2019). "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data

## Future Work

### Planned Enhancements
- **Advanced Evidence Ranking**: Graph-based reasoning and multi-hop evidence retrieval
- **Cross-Lingual Fact-Checking**: Support for multiple languages
- **Larger Knowledge Base Integration**: Freebase, ConceptNet, domain-specific graphs
- **Real-Time Processing**: Low-latency fact-checking
- **Explainable AI**: Interpretability tools for model decisions

### Research Directions
- **Multimodal Evidence Integration**: Visual evidence from images, videos, infographics
- **Temporal Reasoning**: Time-sensitive claims and evidence dating
- **Uncertainty Quantification**: Calibrated confidence estimation
- **Adversarial Robustness**: Defenses against adversarial attacks
- **Few-Shot Domain Adaptation**: Rapid adaptation to new domains
- **Human-AI Collaboration**: Interactive systems with expert feedback

### Technical Improvements
- **Model Efficiency**: Knowledge distillation and quantization
- **Continual Learning**: Update models with new evidence
- **Federated Learning**: Privacy-preserving distributed fact-checking
- **Multi-Agent Systems**: Coordinate specialized agents for comprehensive verification

---

For implementation guides, API documentation, and contribution guidelines, see the main FactCheck-MM repository documentation. For issues, feature requests, or research collaboration, create an issue in the project repository.

```
```
