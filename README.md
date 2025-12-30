# FactCheck-MM: Multimodal Fact-Checking with Sarcasm Detection and Paraphrasing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30-blue)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

FactCheck-MM is an end-to-end multimodal fact-checking system that integrates sarcasm detection, paraphrasing, and fact verification using state-of-the-art models across text, audio, image, and video modalities. Designed for both research experimentation and production deployment with a modular, scalable architecture.

## 🚀 Features

### **Multimodal Sarcasm Detection**
- **Text Analysis**: RoBERTa-based transformers with attention mechanisms for contextual sarcasm detection
- **Audio Processing**: Wav2Vec2 feature extraction with prosodic and acoustic analysis for vocal sarcasm cues
- **Visual Recognition**: Vision Transformer (ViT) and CNN-based models for facial expression and gesture analysis
- **Video Understanding**: Temporal fusion networks combining audio-visual features for dynamic sarcasm detection
- **Cross-Modal Integration**: Sophisticated fusion strategies with attention-based alignment and modality weighting

### **Advanced Paraphrasing**
- **T5/BART Generation**: Fine-tuned sequence-to-sequence models for high-quality paraphrase generation
- **Reinforcement Learning Optimization**: Policy gradient methods with BLEU/ROUGE reward functions for enhanced quality
- **Sarcasm-Aware Paraphrasing**: Specialized models for reformulating sarcastic content into neutral fact-checkable statements
- **Quality Assessment**: Multi-metric evaluation including semantic similarity, fluency, and diversity measurements
- **Curriculum Learning**: Progressive training strategies from simple to complex paraphrase patterns

### **Comprehensive Fact Verification**
- **Evidence Retrieval**: 
  - **Dense Retrieval**: DPR and ColBERT for semantic similarity search with FAISS indexing
  - **Sparse Retrieval**: BM25 and TF-IDF for fast lexical matching
  - **Hybrid Fusion**: Configurable combination strategies with reciprocal rank fusion and score normalization
- **Fact Classification**: RoBERTa and RAG-enhanced models for three-way classification (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)
- **Knowledge Base Integration**: Wikidata, DBpedia, and Wikipedia connectors with intelligent caching
- **End-to-End Pipeline**: Integrated claim processing, evidence retrieval, and verification with confidence estimation

### **Unified Architecture**
- **Shared Backbone**: Modular encoder architecture supporting RoBERTa, Wav2Vec2, and ViT models
- **Flexible Training**: Chunked processing for resource-constrained environments (MacBook M2) and full GPU acceleration (RTX 2050)
- **Modular Design**: Independent modules with clear interfaces enabling custom pipeline configurations
- **Scalable Infrastructure**: Docker containerization, FastAPI deployment, and production monitoring tools

## 📁 Project Structure

```

FactCheck-MM/
├── data/
├── shared/
│   ├── backbone/
│   ├── utils/
│   ├── datasets/
│   └── config/
├── sarcasm\_detection/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── paraphrasing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── data/
├── fact\_verification/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   └── retrieval/
├── experiments/
│   ├── ablation\_studies/
│   ├── hyperparameter\_tuning/
│   ├── multitask\_learning/
│   └── analysis/
├── deployment/
│   ├── api/
│   ├── docker/
│   ├── monitoring/
│   └── config/
├── scripts/
├── notebooks/
├── tests/
├── docs/
├── config/
├── requirements.txt
├── setup.py
├── main.py
└── README.md

````

## 🛠️ Installation

### Prerequisites
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support
- **System Requirements**: 
  - Minimum: 8GB RAM, 4GB disk space
  - Recommended: 16GB+ RAM, RTX 2050+ GPU, 50GB+ disk space

### Quick Setup

```bash
git clone https://github.com/your-username/FactCheck-MM.git
cd FactCheck-MM
pip install -r requirements.txt
pip install -e .
python scripts/download_models.py
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
````

### Environment-Specific Setup

**MacBook M2 (CPU-optimized):**

```bash
pip install torch torchvision torchaudio
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
```

**RTX 2050 (GPU-accelerated):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

## 🚀 Usage

### Quick Start

```bash
python main.py --task fact_verification --input "The Eiffel Tower is in Paris."
python main.py --task complete_pipeline --input "Oh sure, vaccines are totally dangerous" --detect_sarcasm --paraphrase
python main.py --task fact_verification --input_file claims.txt --output_file results.json --batch_size 16
```

### Individual Module Usage

**Sarcasm Detection:**

```bash
python sarcasm_detection/models/text_sarcasm_detector.py --input "Great, another meeting!"
python sarcasm_detection/models/multimodal_sarcasm_detector.py --text "This is fine" --image path/to/image.jpg --audio path/to/audio.wav
```

**Paraphrasing:**

```bash
python paraphrasing/models/t5_paraphraser.py --input "Climate change causes global warming" --num_paraphrases 3
python paraphrasing/models/sarcasm_aware_paraphraser.py --input "Oh wonderful, more rain" --neutralize_sarcasm
```

**Fact Verification:**

```bash
python fact_verification/models/end_to_end_model.py --claim "COVID-19 vaccines are 95% effective" --retrieve_evidence --top_k 5
```

## 📊 Datasets

### Sarcasm Detection Datasets

* SARC, MMSD2, MUStARD, SarcNet, Sarcasm Headlines

### Paraphrasing Datasets

* ParaNMT-5M, MRPC, Quora Question Pairs

### Fact Verification Datasets

* FEVER, LIAR

```bash
python scripts/data_download.py --all
python scripts/verify_datasets.py
```

## 🎯 Training

**MacBook M2:**

```bash
python sarcasm_detection/training/train_multimodal.py --dataset mmsd2 --batch_size 8 --gradient_accumulation 4 --use_mps --chunk_size 1000
```

**RTX 2050:**

```bash
python sarcasm_detection/training/train_multimodal.py --dataset mmsd2 --batch_size 32 --epochs 10 --use_cuda --mixed_precision
```

**Multi-Task Learning:**

```bash
python experiments/multitask_learning/train_joint.py --tasks sarcasm_detection paraphrasing fact_verification --shared_encoder roberta-large --task_weights 0.3 0.3 0.4
```

## 📈 Evaluation

**Metrics:**

* Sarcasm Detection: Accuracy, F1, Precision, Recall, Multimodal Fusion
* Paraphrasing: BLEU, ROUGE, METEOR, BERTScore, Diversity
* Fact Verification: Accuracy, Precision\@k, Recall\@k, MRR, F1, Pipeline Metrics

```bash
python scripts/run_evaluation.py --all --output_dir results/
python fact_verification/evaluation/error_analysis.py --predictions results/predictions.json --visualize
```

## 🚀 Deployment

**REST API:**

```bash
cd deployment/api/
python app.py --host 0.0.0.0 --port 8000
```

**Docker:**

```bash
docker build -t factcheck-mm:latest -f deployment/docker/Dockerfile .
docker run -d -p 8000:8000 --name factcheck-mm-api factcheck-mm:latest
```

**Kubernetes:**

```bash
kubectl apply -f deployment/k8s/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

* Datasets: FEVER, LIAR, SARC
* Libraries: Hugging Face Transformers, PyTorch
* Research & Communities: FEVER Workshop, SemEval, ACL/EMNLP

## 📚 References

```bibtex
@inproceedings{thorne2018fever,
  title={FEVER: a Large-scale Dataset for Fact Extraction and VERification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={NAACL-HLT},
  year={2018}
}

@inproceedings{wang2017liar,
  title={"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection},
  author={Wang, William Yang},
  booktitle={ACL},
  year={2017}
}

@inproceedings{khodak2017large,
  title={A Large Self-Annotated Corpus for Sarcasm},
  author={Khodak, Mikhail and Saunshi, Nikunj and Vodrahalli, Kiran},
  booktitle={LREC},
  year={2018}
}

@article{liu2019roberta,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and others},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@inproceedings{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  booktitle={NeurIPS},
  year={2020}
}
```

**For full documentation and tutorials, visit the [docs](docs/) directory or our online documentation site.**

```

