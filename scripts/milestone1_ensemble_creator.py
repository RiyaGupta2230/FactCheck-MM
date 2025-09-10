"""
Ensemble creation and management for milestone1 sarcasm detection
Creates powerful ensembles from multiple trained models for maximum accuracy
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob
import yaml
import json
import argparse
from typing import List, Dict, Optional
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from milestone1_chunk_trainer import EnhancedSarcasmDataset
from milestone1_feature_engineering import prepare_features_for_training
from milestone1_evaluator import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEnsemble(nn.Module):
    """Advanced ensemble model for maximum accuracy"""
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None, 
                 ensemble_method: str = 'weighted_voting'):
        super().__init__()
        
        self.model_paths = model_paths
        self.ensemble_method = ensemble_method
        
        # Load all models
        self.models = nn.ModuleList()
        self.tokenizers = []
        
        for model_path in model_paths:
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        
        # Set model weights
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            assert len(weights) == len(self.models), "Number of weights must match number of models"
            # Normalize weights
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]
        
        # Freeze all models (ensemble doesn't train base models)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        
        logger.info(f"🔗 Ensemble created with {len(self.models)} models")
        logger.info(f"⚖️ Weights: {[f'{w:.3f}' for w in self.weights]}")
        logger.info(f"🎯 Method: {ensemble_method}")
    
    def forward(self, input_ids, attention_mask, features=None, labels=None):
        """Forward pass through ensemble"""
        
        all_logits = []
        
        # Get predictions from each model
        with torch.no_grad():
            for i, model in enumerate(self.models):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                weighted_logits = outputs.logits * self.weights[i]
                all_logits.append(weighted_logits)
        
        if self.ensemble_method == 'weighted_voting':
            # Weighted average of logits
            ensemble_logits = torch.stack(all_logits).sum(dim=0)
        
        elif self.ensemble_method == 'max_voting':
            # Take max confidence prediction
            all_probs = [torch.softmax(logits / self.weights[i], dim=1) for i, logits in enumerate(all_logits)]
            max_indices = torch.stack([probs.max(dim=1)[1] for probs in all_probs])
            # Majority vote
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
        
        else:  # simple average
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
        
        outputs = {'logits': ensemble_logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fn(ensemble_logits, labels)
        
        return outputs

class EnsembleCreator:
    """Creates and manages model ensembles"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def find_trained_models(self, models_dir: str, min_f1: float = 0.85) -> List[str]:
        """Find all trained models that meet quality threshold"""
        
        model_paths = []
        
        # Look for model directories
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            
            if os.path.isdir(item_path):
                # Check if it's a valid model directory
                config_file = os.path.join(item_path, 'config.json')
                tokenizer_file = os.path.join(item_path, 'tokenizer.json')
                
                if os.path.exists(config_file) and os.path.exists(tokenizer_file):
                    # Extract F1 score from directory name if available
                    if 'f1_' in item:
                        try:
                            f1_score = float(item.split('f1_')[1].split('_')[0])
                            if f1_score >= min_f1:
                                model_paths.append(item_path)
                                logger.info(f"✅ Found model: {item} (F1: {f1_score:.3f})")
                        except ValueError:
                            # If F1 can't be extracted, include anyway
                            model_paths.append(item_path)
                            logger.info(f"✅ Found model: {item}")
                    else:
                        model_paths.append(item_path)
                        logger.info(f"✅ Found model: {item}")
        
        logger.info(f"🔍 Found {len(model_paths)} suitable models")
        return model_paths
    
    def evaluate_individual_models(self, model_paths: List[str]) -> List[Dict]:
        """Evaluate each model individually to determine ensemble weights"""
        
        logger.info("📊 Evaluating individual models for ensemble weighting...")
        
        evaluations = []
        
        # Get test data
        chunks_dir = self.config['data']['chunks_dir']
        test_pattern = self.config['data']['test_pattern']
        test_files = sorted(glob.glob(os.path.join(chunks_dir, test_pattern)))
        
        # Load test data
        test_dfs = []
        for test_file in test_files:
            df = pd.read_csv(test_file)
            test_dfs.append(df)
        
        combined_test = pd.concat(test_dfs, ignore_index=True)
        
        # Prepare test data
        test_enhanced, feature_columns = prepare_features_for_training(combined_test)
        
        for model_path in model_paths:
            logger.info(f"🧪 Evaluating: {os.path.basename(model_path)}")
            
            try:
                # Load model
                model = RobertaForSequenceClassification.from_pretrained(model_path)
                tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                
                # Create test dataset
                test_dataset = EnhancedSarcasmDataset(
                    texts=test_enhanced['text'],
                    labels=test_enhanced['label'],
                    features=test_enhanced[feature_columns],
                    tokenizer=tokenizer,
                    max_length=self.config['model']['max_length']
                )
                
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=32, 
                    shuffle=False
                )
                
                # Get predictions
                predictions, labels, probabilities = self.get_model_predictions(model, test_loader)
                
                # Calculate metrics
                accuracy = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions)
                precision, recall, _, _ = precision_recall_fscore_support(labels, predictions, average='binary')
                
                evaluation = {
                    'model_path': model_path,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
                evaluations.append(evaluation)
                
                logger.info(f"✅ {os.path.basename(model_path)}: F1={f1:.3f}, Acc={accuracy:.3f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to evaluate {model_path}: {e}")
        
        return evaluations
    
    def get_model_predictions(self, model, dataloader):
        """Get predictions from a single model"""
        
        predictions = []
        labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of sarcastic class
        
        return predictions, labels, probabilities
    
    def calculate_ensemble_weights(self, evaluations: List[Dict]) -> List[float]:
        """Calculate optimal ensemble weights based on individual performance"""
        
        if self.config.get('ensemble', {}).get('method') == 'equal_weights':
            return [1.0 / len(evaluations)] * len(evaluations)
        
        # Performance-based weighting
        f1_scores = [eval_result['f1'] for eval_result in evaluations]
        
        if self.config.get('ensemble', {}).get('method') == 'f1_weighted':
            # Weight by F1 score
            total_f1 = sum(f1_scores)
            weights = [f1 / total_f1 for f1 in f1_scores]
        
        elif self.config.get('ensemble', {}).get('method') == 'softmax_weighted':
            # Softmax weighting (emphasizes best models more)
            f1_tensor = torch.tensor(f1_scores)
            weights = torch.softmax(f1_tensor * 5, dim=0).tolist()  # Temperature = 5
        
        else:  # Default: equal weights
            weights = [1.0 / len(evaluations)] * len(evaluations)
        
        return weights
    
    def create_ensemble(self, models_dir: str, output_path: str, 
                       max_models: int = 5, min_f1: float = 0.85) -> str:
        """Create ensemble model from best individual models"""
        
        logger.info("🏗️ Creating ensemble model...")
        
        # Find trained models
        model_paths = self.find_trained_models(models_dir, min_f1)
        
        if len(model_paths) == 0:
            raise ValueError("No suitable models found for ensemble")
        
        # Limit number of models
        if len(model_paths) > max_models:
            logger.info(f"📊 Limiting ensemble to top {max_models} models")
            
            # Evaluate models to select best ones
            evaluations = self.evaluate_individual_models(model_paths)
            
            # Sort by F1 score and take top models
            evaluations.sort(key=lambda x: x['f1'], reverse=True)
            selected_evaluations = evaluations[:max_models]
            selected_paths = [eval_result['model_path'] for eval_result in selected_evaluations]
            
            # Calculate weights
            weights = self.calculate_ensemble_weights(selected_evaluations)
            
        else:
            selected_paths = model_paths
            evaluations = self.evaluate_individual_models(selected_paths)
            weights = self.calculate_ensemble_weights(evaluations)
        
        # Create ensemble
        ensemble_method = self.config.get('ensemble', {}).get('method', 'weighted_voting')
        ensemble = ModelEnsemble(selected_paths, weights, ensemble_method)
        ensemble.to(self.device)
        
        # Save ensemble
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        ensemble_info = {
            'model_paths': selected_paths,
            'weights': weights,
            'method': ensemble_method,
            'individual_performance': evaluations,
            'creation_date': pd.Timestamp.now().isoformat()
        }
        
        # Save ensemble model
        torch.save({
            'ensemble_state_dict': ensemble.state_dict(),
            'ensemble_info': ensemble_info,
            'model_paths': selected_paths
        }, output_path)
        
        # Save ensemble info
        info_path = output_path.replace('.pt', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        logger.info(f"🎉 Ensemble created successfully!")
        logger.info(f"📁 Saved to: {output_path}")
        logger.info(f"📊 Info saved to: {info_path}")
        logger.info(f"🔗 Models in ensemble: {len(selected_paths)}")
        logger.info(f"⚖️ Weights: {[f'{w:.3f}' for w in weights]}")
        
        return output_path
    
    def evaluate_ensemble(self, ensemble_path: str) -> Dict:
        """Evaluate ensemble performance"""
        
        logger.info("📈 Evaluating ensemble performance...")
        
        # Load ensemble
        checkpoint = torch.load(ensemble_path, map_location=self.device)
        ensemble_info = checkpoint['ensemble_info']
        model_paths = checkpoint['model_paths']
        
        ensemble = ModelEnsemble(
            model_paths, 
            ensemble_info['weights'], 
            ensemble_info['method']
        )
        ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
        ensemble.to(self.device)
        ensemble.eval()
        
        # Get test data
        chunks_dir = self.config['data']['chunks_dir']
        test_pattern = self.config['data']['test_pattern']
        test_files = sorted(glob.glob(os.path.join(chunks_dir, test_pattern)))
        
        # Load and combine test data
        test_dfs = [pd.read_csv(f) for f in test_files]
        combined_test = pd.concat(test_dfs, ignore_index=True)
        
        # Prepare test data
        test_enhanced, feature_columns = prepare_features_for_training(combined_test)
        
        # Use first tokenizer (they should all be the same)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_paths[0])
        
        # Create test dataset
        test_dataset = EnhancedSarcasmDataset(
            texts=test_enhanced['text'],
            labels=test_enhanced['label'],
            features=test_enhanced[feature_columns],
            tokenizer=tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Get ensemble predictions
        predictions, labels, probabilities = self.get_model_predictions(ensemble, test_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        precision, recall, _, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        results = {
            'ensemble_performance': {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'total_samples': len(labels)
            },
            'individual_performance': ensemble_info['individual_performance'],
            'improvement': {
                'f1_improvement': f1 - max([model['f1'] for model in ensemble_info['individual_performance']]),
                'accuracy_improvement': accuracy - max([model['accuracy'] for model in ensemble_info['individual_performance']])
            }
        }
        
        logger.info("🎯 Ensemble Evaluation Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   F1 Score: {f1:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1 Improvement: +{results['improvement']['f1_improvement']:.4f}")
        
        return results

def main():
    """Command-line interface for ensemble creation"""
    
    parser = argparse.ArgumentParser(description='Create Model Ensemble')
    parser.add_argument('--config_path', required=True, help='Path to configuration file')
    parser.add_argument('--models_dir', required=True, help='Directory containing trained models')
    parser.add_argument('--output_path', required=True, help='Output path for ensemble model')
    parser.add_argument('--max_models', type=int, default=5, help='Maximum models in ensemble')
    parser.add_argument('--min_f1', type=float, default=0.85, help='Minimum F1 score for inclusion')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate ensemble after creation')
    
    args = parser.parse_args()
    
    # Create ensemble
    creator = EnsembleCreator(args.config_path)
    ensemble_path = creator.create_ensemble(
        args.models_dir, 
        args.output_path, 
        args.max_models, 
        args.min_f1
    )
    
    # Evaluate if requested
    if args.evaluate:
        results = creator.evaluate_ensemble(ensemble_path)
        
        # Save evaluation results
        results_path = ensemble_path.replace('.pt', '_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Evaluation results saved to: {results_path}")

if __name__ == "__main__":
    main()
