"""
Comprehensive evaluation system for milestone1 sarcasm detection
"""
import torch
import pandas as pd
import numpy as np
import os
import glob
import yaml
import json
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
from milestone1_chunk_trainer import EnhancedSarcasmModel, EnhancedSarcasmDataset
from milestone1_feature_engineering import prepare_features_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation suite"""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() or torch.backends.mps.is_available() 
            else 'cpu'
        )
        self.results = {}
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_path, num_features):
        """Load trained model"""
        logger.info(f"📥 Loading model from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model = EnhancedSarcasmModel(
            model_name=self.config['model']['name'],
            num_features=num_features,
            dropout=self.config['model']['dropout']
        )
        
        # Load state dict
        state_dict_path = f"{model_path}.pt"
        if os.path.exists(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def evaluate_on_test_chunks(self, model, tokenizer, feature_columns):
        """Evaluate model on all test chunks"""
        
        # Get test files
        chunks_dir = self.config['data']['chunks_dir']
        test_pattern = self.config['data']['test_pattern']
        test_files = sorted(glob.glob(os.path.join(chunks_dir, test_pattern)))
        
        logger.info(f"🧪 Found {len(test_files)} test chunks")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        chunk_results = []
        
        for test_file in test_files:
            logger.info(f"📊 Evaluating on: {os.path.basename(test_file)}")
            
            # Load test data
            df = pd.read_csv(test_file)
            df_enhanced, _ = prepare_features_for_training(df)
            feature_matrix = df_enhanced[feature_columns]
            
            # Create dataset
            test_dataset = EnhancedSarcasmDataset(
                texts=df_enhanced['text'],
                labels=df_enhanced['label'],
                features=feature_matrix,
                tokenizer=tokenizer,
                max_length=self.config['model']['max_length']
            )
            
            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'] * 2,  # Larger batch for inference
                shuffle=False,
                num_workers=self.config['hardware']['dataloader_num_workers']
            )
            
            # Get predictions
            chunk_preds, chunk_labels, chunk_probs = self.get_predictions(model, test_loader)
            
            # Calculate chunk metrics
            chunk_metrics = self.calculate_metrics(chunk_labels, chunk_preds, chunk_probs)
            chunk_metrics['file'] = test_file
            chunk_metrics['samples'] = len(chunk_labels)
            chunk_results.append(chunk_metrics)
            
            # Accumulate for overall metrics
            all_predictions.extend(chunk_preds)
            all_labels.extend(chunk_labels)
            all_probabilities.extend(chunk_probs)
            
            logger.info(f"✅ Chunk F1: {chunk_metrics['f1']:.4f}, Accuracy: {chunk_metrics['accuracy']:.4f}")
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return {
            'overall_metrics': overall_metrics,
            'chunk_results': chunk_results,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def get_predictions(self, model, dataloader):
        """Get model predictions"""
        predictions = []
        labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    features=batch['features']
                )
                
                # Get predictions and probabilities
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of sarcastic class
        
        return predictions, labels, probabilities
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics"""
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        # Additional metrics
        auc = roc_auc_score(y_true, y_prob)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
        }
    
    def create_visualizations(self, y_true, y_pred, y_prob, output_dir):
        """Create evaluation visualizations"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Sarcastic', 'Sarcastic'],
                   yticklabels=['Not Sarcastic', 'Sarcastic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        probabilities_0 = [y_prob[i] for i in range(len(y_prob)) if y_true[i] == 0]
        probabilities_1 = [y_prob[i] for i in range(len(y_prob)) if y_true[i] == 1]
        
        plt.hist(probabilities_0, bins=30, alpha=0.7, label='Not Sarcastic', color='blue')
        plt.hist(probabilities_1, bins=30, alpha=0.7, label='Sarcastic', color='red')
        plt.xlabel('Predicted Probability (Sarcastic)')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualizations saved to: {output_dir}")
    
    def generate_detailed_report(self, results, output_path):
        """Generate detailed evaluation report"""
        
        report = {
            'model_info': {
                'model_name': self.config['model']['name'],
                'max_length': self.config['model']['max_length'],
                'evaluation_date': pd.Timestamp.now().isoformat()
            },
            'overall_performance': results['overall_metrics'],
            'chunk_performance': results['chunk_results'],
            'analysis': {
                'total_samples': len(results['labels']),
                'sarcastic_samples': sum(results['labels']),
                'non_sarcastic_samples': len(results['labels']) - sum(results['labels']),
                'class_distribution': {
                    'sarcastic_ratio': sum(results['labels']) / len(results['labels']),
                    'balance_ratio': min(sum(results['labels']), len(results['labels']) - sum(results['labels'])) / 
                                   max(sum(results['labels']), len(results['labels']) - sum(results['labels']))
                }
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        metrics = results['overall_metrics']
        logger.info("\n" + "="*50)
        logger.info("🎯 FINAL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"📊 Total Samples: {len(results['labels']):,}")
        logger.info(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"🏆 F1 Score: {metrics['f1']:.4f}")
        logger.info(f"🎱 Precision: {metrics['precision']:.4f}")
        logger.info(f"🔍 Recall: {metrics['recall']:.4f}")
        logger.info(f"📈 AUC: {metrics['auc']:.4f}")
        logger.info("="*50)
        
        return report

def evaluate_milestone1_model(config_path, model_path=None):
    """Main evaluation function"""
    
    evaluator = ModelEvaluator(config_path)
    
    # Find best model if not specified
    if model_path is None:
        model_dir = evaluator.config['paths']['model_dir']
        model_files = glob.glob(os.path.join(model_dir, 'best_model_chunk_*'))
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {model_dir}")
        model_path = max(model_files, key=os.path.getctime).replace('.pt', '')
    
    # Load a sample to get feature dimensions
    chunks_dir = evaluator.config['data']['chunks_dir']
    sample_file = glob.glob(os.path.join(chunks_dir, 'ultimate_train_chunk_001.csv'))[0]
    sample_df = pd.read_csv(sample_file).head(100)  # Small sample for features
    _, feature_columns = prepare_features_for_training(sample_df)
    
    # Load model
    model, tokenizer = evaluator.load_model(model_path, len(feature_columns))
    
    # Evaluate model
    results = evaluator.evaluate_on_test_chunks(model, tokenizer, feature_columns)
    
    # Create output directory
    output_dir = os.path.join(evaluator.config['paths']['output_dir'], 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    evaluator.create_visualizations(
        results['labels'], 
        results['predictions'], 
        results['probabilities'],
        output_dir
    )
    
    # Generate detailed report
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    report = evaluator.generate_detailed_report(results, report_path)
    
    logger.info(f"📋 Evaluation report saved to: {report_path}")
    
    return results, report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python milestone1_evaluator.py <config_path> [model_path]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    evaluate_milestone1_model(config_path, model_path)
