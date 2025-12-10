# Integration Guide: Replacing Mocks with Real Models

This guide explains how to integrate trained FactCheck-MM models into the demo application.

## 🎯 Overview

The demo uses a **graceful fallback pattern**:

```python
try:
    from real_module import real_function
    USE_REAL = True
except ImportError:
    from mock_pipeline import mock_function
    USE_REAL = False
```

## 📂 Expected Repository Structure

```text
FactCheck-MM/
├── sarcasm_detection/
│   ├── __init__.py
│   ├── predict.py                # Contains detect_sarcasm()
│   └── models/
│       ├── text_model.pt
│       ├── audio_model.pt
│       └── fusion_model.pt
├── paraphrasing/
│   ├── __init__.py
│   ├── generate.py               # Contains paraphrase()
│   └── models/
│       └── t5_paraphrase.pt
├── fact_verification/
│   ├── __init__.py
│   ├── verify.py                 # Contains verify_claim()
│   └── models/
│       ├── retriever.pt
│       └── classifier.pt
└── demo/
    └── app.py
```


## 🔌 Required Function Signatures

### 1. Sarcasm Detection

**Module**: `sarcasm_detection/predict.py`

```python
from typing import Optional, Any, Dict

def detect_sarcasm(
    text: Optional[str] = None,
    image: Optional[Any] = None,
    audio: Optional[Any] = None,
    video: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Detect sarcasm across modalities.
    
    Returns:
        {
            'label': str,           # 'sarcastic' or 'not_sarcastic'
            'score': float,         # 0.0 to 1.0
            'modalities_used': dict,
            'attention_weights': dict (optional)
        }
    """
    pass
```

**Model Loading Example**:

```python
import torch
import os
from transformers import RobertaTokenizer, Wav2Vec2Processor
from sarcasm_detection.models import MultimodalSarcasmModel

# Load models (do this once at module level)
MODEL_PATH = os.getenv('SARCASM_MODEL_PATH', './models/sarcasm_fusion.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = MultimodalSarcasmModel.load_from_checkpoint(MODEL_PATH).to(device)
model.eval()
```

### 2. Paraphrasing

**Module**: `paraphrasing/generate.py`

```python
def paraphrase(
    text: str,
    max_length: int = 128,
    num_beams: int = 4,
    temperature: float = 0.7
) -> str:
    """
    Generate literal paraphrase of sarcastic/ambiguous text.

    Returns:
        str: Paraphrased text
    """
    pass
```

**Model Loading Example**:

```python
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_PATH = os.getenv('PARAPHRASE_MODEL_PATH', './models/t5_paraphrase')
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()
```

### 3. Fact Verification

**Module**: `fact_verification/verify.py`

```python
from typing import Dict, Any, List

def verify_claim(
    claim: str,
    top_k: int = 5,
    evidence_sources: List[str] = None
) -> Dict[str, Any]:
    """
    Verify factual claim against knowledge bases.
    
    Returns:
        {
            'verdict': str,         # 'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'
            'confidence': float,
            'evidence': List[Dict],
            'explanation': str
        }
    """
    pass
```

**Model Loading Example**:

```python
import os
from transformers import RobertaForSequenceClassification, DPRQuestionEncoder
from fact_verification.retrieval import FaissRetriever

RETRIEVER_PATH = os.getenv('RETRIEVER_PATH', './models/dpr_retriever')
CLASSIFIER_PATH = os.getenv('CLASSIFIER_PATH', './models/roberta_classifier.pt')

retriever = FaissRetriever(RETRIEVER_PATH)
classifier = RobertaForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
classifier.eval()
```


## 🔧 Environment Variables

Create a `.env` file in the demo directory:

```text
# Model Paths
SARCASM_MODEL_PATH=../sarcasm_detection/models/sarcasm_fusion.pt
PARAPHRASE_MODEL_PATH=../paraphrasing/models/t5_paraphrase
RETRIEVER_PATH=../fact_verification/models/dpr_retriever
CLASSIFIER_PATH=../fact_verification/models/roberta_classifier.pt

# Inference Settings
DEVICE=cuda
BATCH_SIZE=8
MAX_LENGTH=512

# External APIs (optional)
GOOGLE_FACT_CHECK_API_KEY=your_key_here
WIKIDATA_ENDPOINT=https://query.wikidata.org/sparql
```

Load in `app.py` using `python-dotenv`:

```python
from dotenv import load_dotenv

load_dotenv()  # Loads .env file if it exists
```

## ✨ Example: How to Enable Real Models in demo/app.py

Here's a minimal, safe pattern for enabling real models:

```python
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import FactCheck-MM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue with defaults

# Try to import real models, fall back to mocks
try:
    from sarcasm_detection.predict import detect_sarcasm
    from paraphrasing.generate import paraphrase
    from fact_verification.verify import verify_claim
    USE_REAL_MODELS = True
except ImportError:
    from mock_pipeline import detect_sarcasm, paraphrase, verify_claim
    USE_REAL_MODELS = False

# Use in your Flask routes
if USE_REAL_MODELS:
    result = detect_sarcasm(text=user_input)
else:
    result = detect_sarcasm(text=user_input)  # Mock returns consistent format
```

## 📝 Integration Checklist

- [ ] **Step 1**: Train models using `../experiments/train_*.py` scripts
- [ ] **Step 2**: Export model checkpoints to `models/` directories
- [ ] **Step 3**: Create `__init__.py` files in each module
- [ ] **Step 4**: Implement required function signatures
- [ ] **Step 5**: Add environment variable configuration
- [ ] **Step 6**: Test imports: `python -c "from sarcasm_detection.predict import detect_sarcasm"`
- [ ] **Step 7**: Enable "Use repo models" checkbox in demo UI
- [ ] **Step 8**: Verify outputs match expected format

## 🐛 Debugging

### Issue: ImportError

Check Python path:

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

Manually add parent directory:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

### Issue: CUDA Out of Memory

In each predict function (e.g., in `app.py`):

```python
import torch

torch.cuda.empty_cache()
```

Use smaller batch sizes in `.env`:

```text
BATCH_SIZE=2
```

### Issue: Slow Inference

Enable half-precision (in model loading):

```python
model.half()
```

Use TorchScript compilation:

```python
model = torch.jit.script(model)
```

## 🚀 Advanced: GPU Acceleration

For production deployment with GPU, modify `app.py`:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pass device to all functions
sarcasm_result = detect_sarcasm(text, image, audio, video, device=device)
```


## 📊 Performance Benchmarks

Expected inference times (approximate):

| Component | CPU (M2) | GPU (RTX 2050) |
|-----------|----------|----------------|
| Sarcasm Detection | 0.8s | 0.15s |
| Paraphrasing | 1.2s | 0.25s |
| Fact Verification | 2.5s | 0.60s |
| **Total Pipeline** | **4.5s** | **1.0s** |

## 🔐 Security Notes

- Never commit API keys to Git
- Use `.env` files (add to `.gitignore`)
- Validate all user inputs before processing
- Sanitize file uploads (check MIME types, size limits)

## 📞 Support

If integration issues persist:
1. Check main repo `docs/` folder for detailed API docs
2. Review training logs in `experiments/logs/`
3. Open an issue: https://github.com/RiyaGupta2230/FactCheck-MM/issues

---

## ✅ Integration Checklist

- [ ] **Step 1**: Train models using `../experiments/train_*.py` scripts
- [ ] **Step 2**: Export model checkpoints to `models/` directories
- [ ] **Step 3**: Create `__init__.py` files in each module
- [ ] **Step 4**: Implement required function signatures
- [ ] **Step 5**: Add environment variable configuration
- [ ] **Step 6**: Test imports: `python -c "from sarcasm_detection.predict import detect_sarcasm"`
- [ ] **Step 7**: Enable "Use repo models" checkbox in demo UI
- [ ] **Step 8**: Verify outputs match expected format

**Last Updated**: December 2025
