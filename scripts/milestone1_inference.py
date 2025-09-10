"""
Production-ready inference for milestone1 sarcasm detection
Supports single predictions, batch processing, and confidence scoring
"""
import torch
import pandas as pd
import numpy as np
import os
import yaml
import json
import argparse
from typing import List, Union, Dict
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from milestone1_feature_engineering import prepare_features_for_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SarcasmInferenceEngine:
    """Production-ready sarcasm detection inference engine"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config = self.load_config(config_path) if config_path else {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model()
        
        logger.info(f"🚀 Inference engine initialized on {self.device}")
        logger.info(f"📁 Model loaded from: {model_path}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load inference configuration"""
        if not config_path or not os.path.exists(config_path):
            return {
                'inference': {
                    'batch_size': 32,
                    'confidence_threshold': 0.5,
                    'max_length': 256
                }
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        
        try:
            # Load tokenizer
            tokenizer = RobertaTokenizerFast.from_pretrained(self.model_path)
            
            # Load model
            model = RobertaForSequenceClassification.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()
            
            logger.info("✅ Model and tokenizer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_texts(self, texts: List[str]) -> Dict:
        """Preprocess texts for inference"""
        
        # Create DataFrame for feature extraction
        df = pd.DataFrame({'text': texts, 'label': [0] * len(texts)})  # Dummy labels
        
        # Extract features
        df_enhanced, feature_columns = prepare_features_for_training(df)
        
        return {
            'texts': df_enhanced['text'].tolist(),
            'features': df_enhanced[feature_columns].values
        }
    
    def predict_single(self, text: str, include_confidence: bool = True) -> Dict:
        """Predict sarcasm for a single text"""
        
        result = self.predict_batch([text], include_confidence)
        return result[0]
    
    def predict_batch(self, texts: List[str], include_confidence: bool = True) -> List[Dict]:
        """Predict sarcasm for a batch of texts"""
        
        if not texts:
            return []
        
        # Preprocess texts
        preprocessed = self.preprocess_texts(texts)
        processed_texts = preprocessed['texts']
        features = preprocessed['features']
        
        results = []
        batch_size = self.config.get('inference', {}).get('batch_size', 32)
        max_length = self.config.get('inference', {}).get('max_length', 256)
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_features = features[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
            
            # Process results
            for j, (pred, probs) in enumerate(zip(predictions, probabilities)):
                result = {
                    'text': texts[i + j],  # Original text
                    'prediction': 'sarcastic' if pred.item() == 1 else 'not_sarcastic',
                    'prediction_label': pred.item(),
                }
                
                if include_confidence:
                    result['confidence'] = probs[pred].item()
                    result['probabilities'] = {
                        'not_sarcastic': probs[0].item(),
                        'sarcastic': probs[1].item()
                    }
                
                results.append(result)
        
        return results
    
    def predict_with_explanations(self, text: str) -> Dict:
        """Predict with feature-based explanations"""
        
        # Get basic prediction
        result = self.predict_single(text, include_confidence=True)
        
        # Extract features for explanation
        df = pd.DataFrame({'text': [text], 'label': [0]})
        df_enhanced, feature_columns = prepare_features_for_training(df)
        
        # Add feature analysis
        features = df_enhanced[feature_columns].iloc[0]
        
        explanations = []
        
        # Analyze key features
        if features.get('exclamation_count', 0) > 0:
            explanations.append(f"Contains {int(features['exclamation_count'])} exclamation marks")
        
        if features.get('sarcasm_indicator_count', 0) > 0:
            explanations.append(f"Contains {int(features['sarcasm_indicator_count'])} sarcasm indicator words")
        
        if features.get('sentiment_polarity', 0) < -0.1:
            explanations.append("Has negative sentiment polarity")
        elif features.get('sentiment_polarity', 0) > 0.1:
            explanations.append("Has positive sentiment polarity")
        
        if features.get('caps_ratio', 0) > 0.1:
            explanations.append(f"Has {features['caps_ratio']:.1%} capital letters")
        
        if features.get('contrast_word_count', 0) > 0:
            explanations.append(f"Contains {int(features['contrast_word_count'])} contrast words")
        
        result['explanations'] = explanations
        result['features'] = {col: features[col] for col in feature_columns[:10]}  # Top 10 features
        
        return result
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to file"""
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Predictions saved to: {output_path}")

def main():
    """Command-line interface for inference"""
    
    parser = argparse.ArgumentParser(description='Sarcasm Detection Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config_path', help='Path to inference config')
    parser.add_argument('--input_text', help='Single text to classify')
    parser.add_argument('--input_file', help='File with texts to classify (one per line)')
    parser.add_argument('--output_file', help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--include_confidence', action='store_true', help='Include confidence scores')
    parser.add_argument('--explain', action='store_true', help='Include explanations')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = SarcasmInferenceEngine(args.model_path, args.config_path)
    
    texts = []
    
    # Get input texts
    if args.input_text:
        texts = [args.input_text]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("🎯 Interactive Sarcasm Detection")
        print("Enter texts to classify (empty line to quit):")
        
        while True:
            text = input("\nText: ").strip()
            if not text:
                break
            
            if args.explain:
                result = engine.predict_with_explanations(text)
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.3f}")
                if result['explanations']:
                    print("Explanations:")
                    for exp in result['explanations']:
                        print(f"  • {exp}")
            else:
                result = engine.predict_single(text, args.include_confidence)
                print(f"\nPrediction: {result['prediction']}")
                if args.include_confidence:
                    print(f"Confidence: {result['confidence']:.3f}")
        
        return
    
    # Batch prediction
    if texts:
        logger.info(f"🔍 Processing {len(texts)} texts...")
        
        if args.explain and len(texts) == 1:
            predictions = [engine.predict_with_explanations(texts[0])]
        else:
            predictions = engine.predict_batch(texts, args.include_confidence)
        
        # Output results
        if args.output_file:
            engine.save_predictions(predictions, args.output_file)
        else:
            for pred in predictions:
                print(f"\nText: {pred['text']}")
                print(f"Prediction: {pred['prediction']}")
                if 'confidence' in pred:
                    print(f"Confidence: {pred['confidence']:.3f}")
                if 'explanations' in pred and pred['explanations']:
                    print("Explanations:")
                    for exp in pred['explanations']:
                        print(f"  • {exp}")

if __name__ == "__main__":
    main()
