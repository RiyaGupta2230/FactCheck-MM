# Usage Guide

Comprehensive guide for using FactCheck-MM for training, inference, and evaluation.

## Table of Contents

- [Quick Start](#quick-start)
- [Sarcasm Detection](#sarcasm-detection)
- [Paraphrasing](#paraphrasing)
- [Fact Verification](#fact-verification)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Training vs Inference](#training-vs-inference)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Quick Start

### Inference Only (No Training)

Activate environment
source venv/bin/activate

Run sarcasm detection
python sarcasm_detection/models/text_sarcasm_detector.py --input "Oh great, another bug!"

Run paraphrasing
python paraphrasing/models/t5_paraphraser.py --input "Climate change is real" --num_paraphrases 3

Run fact verification
python fact_verification/models/end_to_end_model.py --claim "Paris is the capital of France"


### Using Docker API
Start API server
cd deployment/docker
docker-compose up -d

Test endpoints
curl -X POST http://localhost:8000/api/v1/sarcasm/predict
-H "Content-Type: application/json"
-d '{"text": "Oh wonderful day!"}'



## Sarcasm Detection

### Text-Only Sarcasm Detection

Single prediction
python sarcasm_detection/models/text_sarcasm_detector.py
--input "Oh sure, that makes perfect sense"
--model_path checkpoints/text_sarcasm/

Batch prediction
python sarcasm_detection/models/text_sarcasm_detector.py
--input_file data/test_claims.txt
--output_file results/sarcasm_predictions.json


### Multimodal Sarcasm Detection

from sarcasm_detection.models import MultimodalSarcasmDetector

Initialize detector
detector = MultimodalSarcasmDetector.from_pretrained(
"checkpoints/multimodal_sarcasm/best_model"
)

Text + Audio
result = detector.predict({
"text": "This is just fantastic",
"audio_path": "data/audio/sample.wav",
"modalities": ["text", "audio"]
})

print(f"Is Sarcastic: {result['prediction'] == 1}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Modality Scores: {result['modality_scores']}")


### Training Sarcasm Detection

Train text-only model
python sarcasm_detection/training/train_text.py
--dataset sarc
--model roberta-base
--epochs 5
--batch_size 16
--learning_rate 2e-5

Train multimodal model
python sarcasm_detection/training/train_multimodal.py
--dataset mmsd2
--epochs 10
--batch_size 8
--modalities text audio image
--fusion_method attention

Resume training
python sarcasm_detection/training/train_multimodal.py
--resume_from checkpoints/multimodal_sarcasm/epoch_5.pt
--epochs 10


### Evaluation

Evaluate on test set
python sarcasm_detection/evaluation/comprehensive_evaluation.py
--model_path checkpoints/multimodal_sarcasm/best_model
--dataset mmsd2
--split test
--output_dir results/sarcasm_eval/

Results saved to:
- results/sarcasm_eval/metrics.json
- results/sarcasm_eval/confusion_matrix.png
- results/sarcasm_eval/per_class_performance.csv


## Paraphrasing

### Generate Paraphrases

from paraphrasing.models import T5Paraphraser

Initialize paraphraser
paraphraser = T5Paraphraser.from_pretrained("checkpoints/t5_paraphraser/")

Generate paraphrases
paraphrases = paraphraser.generate(
"Global warming is accelerating",
num_return_sequences=5,
temperature=0.8,
diversity_penalty=0.5
)

for i, paraphrase in enumerate(paraphrases, 1):
print(f"{i}. {paraphrase}")


### Sarcasm-Aware Paraphrasing

from paraphrasing.models import SarcasmAwareParaphraser

Initialize sarcasm-aware paraphraser
paraphraser = SarcasmAwareParaphraser.from_pretrained(
"checkpoints/sarcasm_aware_paraphraser/"
)

Neutralize sarcastic claim
sarcastic_text = "Oh sure, vaccines are totally dangerous"
neutral_paraphrase = paraphraser.neutralize_sarcasm(sarcastic_text)

print(f"Original: {sarcastic_text}")
print(f"Neutralized: {neutral_paraphrase}")

Output: "Vaccines have been proven safe and effective"


### Training Paraphrasing Models

Train T5 paraphraser
python paraphrasing/training/train_generation.py
--model t5-base
--dataset paranmt
--epochs 5
--batch_size 16
--learning_rate 3e-4

Train with curriculum learning
python paraphrasing/training/train_generation.py
--model t5-large
--dataset paranmt
--curriculum_learning
--start_difficulty 0.2
--end_difficulty 1.0
--epochs 10

Train RL-enhanced paraphraser
python paraphrasing/training/train_rl.py
--base_model_path checkpoints/t5_paraphraser/
--reward_type bleu
--episodes 5000
--learning_rate 1e-5


### Evaluation

Evaluate paraphrase quality
python paraphrasing/evaluation/generation_metrics.py
--model_path checkpoints/t5_paraphraser/
--test_data data/paraphrasing/test/mrpc_test.json
--output_dir results/paraphrasing_eval/

Metrics computed:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-1, ROUGE-2, ROUGE-L
- METEOR
- BERTScore
- Semantic Similarity

