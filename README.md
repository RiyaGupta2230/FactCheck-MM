# FactCheck-MM: A Multimodal NLP System for Sarcasm Detection and Claim Verification

## 📌 Overview
This project implements **FactCheck-MM**, a modular pipeline for:
1. **Sarcasm Detection** (text & multimodal)
2. **Paraphrasing sarcastic text** into literal meaning
3. **Fact Verification** using FEVER, ClaimBuster, and Google Fact Check APIs
4. **UI Prototype** with React frontend + FastAPI backend
5. **Extensions** like offensive sarcasm detection

---

## 📂 Folder Structure
```
FactCheck-MM/
│── requirements.txt        # Dependencies
│── README.md               # Setup instructions (this file)
│── .gitignore
│
├── data/                   # Datasets
│   ├── isarcasm/
│   ├── fever/
│   └── liar/
│
├── checkpoints/            # Trained models
│   └── m1_roberta_isarcasm/
│
├── src/                    # Source code
│   ├── milestone1/         # Sarcasm detection
│   │   └── text_sarcasm.py
│   ├── milestone2/         # Paraphrasing
│   │   └── paraphrase_t5.py
│   ├── milestone3/         # Fact-checking
│   │   └── fact_verifier.py
│   ├── milestone4/         # Backend API
│   │   └── app.py
│   ├── utils/              # Helper functions
│   │   ├── preprocessing.py
│   │   ├── evaluation.py
│   │   └── config.py
│   └── extensions/         # Extra features
│       └── offensive_classifier.py
│
└── frontend/               # React frontend (later milestone)
    ├── package.json
    ├── src/
    │   └── App.jsx
    └── public/
```

---

## ⚙️ Environment Setup (Windows PowerShell)

```powershell
# Step 1: Create project folder
mkdir FactCheck-MM; cd FactCheck-MM

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate environment
.\venv\Scripts\activate

# Step 4: Upgrade pip
pip install --upgrade pip

# Step 5: Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running Milestone 1 (Text Sarcasm Detection)

### Train Model
```powershell
cd src/milestone1
python text_sarcasm.py --dataset isarcasm --epochs 3 --batch_size 16
```

### Predict with Saved Model
```powershell
python text_sarcasm.py --predict "yeah right, that presentation was *amazing*."
```

Expected output:
```json
{
  "text": "yeah right, that presentation was *amazing*.",
  "sarcastic": true,
  "probs": {
    "not_sarcastic": 0.12,
    "sarcastic": 0.88
  }
}
```

---

## 🔮 Next Milestones
- **Milestone 1.2**: Extend sarcasm detection to **multimodal (text + image)**
- **Milestone 2**: Add **T5 paraphrasing** for sarcastic-to-literal conversion
- **Milestone 3**: Implement **fact verification** using FEVER + APIs
- **Milestone 4**: Build **FastAPI backend + React frontend**
- **Extension**: Offensive sarcasm classifier

---

## ✅ Verification
Run this command to check everything is installed:
```powershell
python -c "import torch, transformers, datasets, evaluate; print('✅ All good!')"
```

---

## 📖 References
- HuggingFace Transformers: https://huggingface.co/transformers/
- iSarcasm Dataset: https://huggingface.co/datasets/isarcasm
- FEVER Dataset: https://fever.ai/resources.html
- ClaimBuster API: https://idir.uta.edu/claimbuster/
