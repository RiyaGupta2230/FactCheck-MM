# FactCheck-MM: A Multimodal NLP System

A comprehensive system for sarcasm detection, paraphrase analysis, and fact verification using state-of-the-art NLP models.

## Features

- **Multimodal Sarcasm Detection**: Analyze text, audio, and visual cues
- **Paraphrase Detection**: Compare semantic similarity between sentences
- **Fact Verification**: Evidence-based claim verification
- **Web Interface**: Interactive dashboard for all functionalities

## Quick Start

### 1. Environment Setup

Clone the repository
git clone <your-repo-url>
cd FactCheck-MM

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


### 2. Data Preparation

Ensure your data directory structure matches:
data/
├── fact_verification/
├── multimodal/
└── text/


### 3. Training Models

Train Milestone 1: Sarcasm Detection
python scripts/train_milestone1.py

Train Milestone 2: Paraphrase Detection (coming soon)
python scripts/train_milestone2.py

Train Milestone 3: Fact Verification (coming soon)
python scripts/train_milestone3.py


### 4. Run the Application

python run_app.py


Visit `http://localhost:5000` to access the web interface.

## Project Milestones

- ✅ **Milestone 1**: Multimodal Sarcasm Detection
- 🔄 **Milestone 2**: Paraphrase Detection
- 🔄 **Milestone 3**: Fact Verification
- ✅ **Milestone 4**: Web UI

## Configuration

Edit `config.yaml` to customize:
- Model parameters
- Training settings
- Data paths
- API configuration

## API Endpoints

- `POST /api/sarcasm-detection`: Analyze text for sarcasm
- `POST /api/paraphrase-check`: Compare sentence similarity
- `POST /api/fact-check`: Verify claims
- `GET /api/health`: System health check

## License

MIT License - see LICENSE file for details.
