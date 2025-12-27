# Installation Guide

Complete installation instructions for FactCheck-MM across different platforms and use cases.

## Table of Contents

- [Prerequisites](#prerequisites)
- [CPU Installation (MacBook M2)](#cpu-installation-macbook-m2)
- [GPU Installation (RTX 2050)](#gpu-installation-rtx-2050)
- [Docker Installation](#docker-installation)
- [Development Setup](#development-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.10 or 3.11 |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB free | 100GB SSD |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None (CPU mode) | NVIDIA RTX 2050+ |

### Software Dependencies

- **Git**: Version control
- **Python**: 3.10 or higher
- **pip**: Latest version (23.0+)
- **CUDA**: 11.7+ (GPU only)
- **Docker**: 20.10+ (optional, for containerized deployment)

## CPU Installation (MacBook M2)

### Step 1: Clone Repository

Clone the repository
git clone https://github.com/factcheck-mm/FactCheck-MM.git
cd FactCheck-MM

Verify directory structure
ls -la


### Step 2: Create Virtual Environment

Create virtual environment
python3 -m venv venv

Activate virtual environment
source venv/bin/activate # macOS/Linux

Upgrade pip
pip install --upgrade pip setuptools wheel


### Step 3: Install PyTorch (CPU)

Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


### Step 4: Install Project Dependencies

Install all dependencies
pip install -r requirements.txt

Install project in editable mode
pip install -e .


### Step 5: Download NLP Models

Download spaCy models
python -m spacy download en_core_web_sm

Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"


### Step 6: Verify Installation

Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.version}')"

Verify project imports
python -c "from shared.utils.logging_utils import get_logger; print('Success!')"


## GPU Installation (RTX 2050)

### Step 1: CUDA Setup

Check NVIDIA driver
nvidia-smi

Verify CUDA version (should be 11.7+)
nvcc --version


If CUDA is not installed:

Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda



### Step 2: Clone and Setup

Clone repository
git clone https://github.com/factcheck-mm/FactCheck-MM.git
cd FactCheck-MM

Create virtual environment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel


### Step 3: Install PyTorch (GPU)

Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


### Step 4: Install Dependencies

Install all dependencies
pip install -r requirements.txt

Install project
pip install -e .

Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"



### Step 5: Verify GPU

Check CUDA availability in PyTorch
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"

Expected output:
CUDA Available: True
Device: NVIDIA GeForce RTX 2050


## Docker Installation

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)

### Step 1: Install Docker

#### macOS

Install Docker Desktop
brew install --cask docker


#### Linux (Ubuntu)

Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose


### Step 2: Install NVIDIA Docker (GPU Only)

Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Install NVIDIA Docker runtime
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi


### Step 3: Clone and Build

Clone repository
git clone https://github.com/factcheck-mm/FactCheck-MM.git
cd FactCheck-MM

Build Docker image (CPU)
cd deployment/docker
docker-compose build

Or build GPU version
docker-compose build --build-arg target=gpu


### Step 4: Run Container

Start API server (CPU)
docker-compose up -d

Start API server (GPU)
docker-compose --profile gpu up -d

View logs
docker-compose logs -f

Check status
docker-compose ps


### Step 5: Verify Deployment

Health check
curl http://localhost:8000/health

API documentation
open http://localhost:8000/docs


## Development Setup

For active development with hot-reloading:

Clone repository
git clone https://github.com/factcheck-mm/FactCheck-MM.git
cd FactCheck-MM

Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

Install pre-commit hooks
pre-commit install

Install project in editable mode
pip install -e .


### Development Tools

Code formatting
black .

Linting
flake8 .

Type checking
mypy shared/

Run tests
pytest tests/ -v

Run specific test
pytest tests/test_sarcasm_detection.py -v


## Verification

### Test All Components

Run comprehensive verification
python scripts/verify_installation.py


### Test Individual Modules

Test shared utilities
python -c "from shared.utils.logging_utils import get_logger; logger = get_logger('test'); logger.info('Success!')"

Test sarcasm detection
python -c "from sarcasm_detection.models import TextSarcasmDetector; print('Sarcasm module OK')"

Test paraphrasing
python -c "from paraphrasing.models import T5Paraphraser; print('Paraphrasing module OK')"

Test fact verification
python -c "from fact_verification.models import FactCheckPipeline; print('Fact verification module OK')"


### Test API (if deployed)

Health check
curl http://localhost:8000/health

Test sarcasm endpoint
curl -X POST http://localhost:8000/api/v1/sarcasm/predict
-H "Content-Type: application/json"
-d '{"text": "Oh wonderful, more documentation!"}'


## Troubleshooting

### Common Issues

#### Issue: ImportError for shared modules

**Solution**:
Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

Or install in editable mode
pip install -e .


#### Issue: CUDA out of memory (RTX 2050)

**Solution**:
Reduce batch size in training configs
Edit config/training_configs/*.yaml
Set batch_size: 4 or 8
Enable gradient accumulation
Set gradient_accumulation_steps: 4


#### Issue: spaCy model not found

**Solution**:
Download required model
python -m spacy download en_core_web_sm

Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"



#### Issue: Docker build fails

**Solution**:
Clear Docker cache
docker system prune -a

Rebuild without cache
docker-compose build --no-cache

Check Docker logs
docker-compose logs


#### Issue: Port 8000 already in use

**Solution**:
Find process using port
lsof -i :8000

Kill process
kill -9 <PID>

Or use different port
docker-compose up -d -p 8001:8000


### Getting Help

If issues persist:

1. Check existing [GitHub Issues](https://github.com/factcheck-mm/FactCheck-MM/issues)
2. Review logs: `cat logs/factcheck-mm.log`
3. Open a new issue with:
   - System information (`python --version`, `nvidia-smi`, etc.)
   - Full error traceback
   - Steps to reproduce

## Next Steps

After successful installation:

1. Review [Usage Guide](usage.md) for running experiments
2. Explore [API Reference](api_reference.md) for integration
3. Read [Model Architectures](model_architectures.md) for technical details
4. Check [Dataset Information](dataset_info.md) for data understanding

---

**Installation Support**: installation@factcheck-mm.org  
**Last Updated**: December 2024
