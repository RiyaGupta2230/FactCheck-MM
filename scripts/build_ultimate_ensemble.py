import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

class UltimateEnsembleBuilder:
    """
    Ultimate ensemble builder for maximum multimodal sarcasm detection performance
    Combines best chunk models using advanced meta-learning techniques
    """
    
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        Path(self.config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Ultimate Ensemble Builder initialized")
    
    def discover_trained_models(self):
        """Discover and evaluate all trained chunk models"""
        models_dir = Path(self.config['paths']['models_dir'])
        model_files = list(models_dir.glob("ultimate_*_chunk_*.pt"))
        
        self.logger.info(f"🔍 Found {len(model_files)} trained chunk models")
        
        if not model_files:
            raise FileNotFoundError("No trained chunk models found! Please train models first.")
        
        # Load and evaluate each model
        model_performances = []
        
        for model_path in tqdm(model_files, desc="📊 Evaluating models"):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extract model information
                model_info = {
                    'path': str(model_path),
                    'chunk_name': checkpoint.get('chunk_name', model_path.stem),
                    'epoch': checkpoint.get('epoch', 0),
                    'metrics': checkpoint.get('metrics', {}),
                    'f1_score': checkpoint.get('metrics', {}).get('f1', 0.0),
                    'accuracy': checkpoint.get('metrics', {}).get('accuracy', 0.0),
                    'precision': checkpoint.get('metrics', {}).get('precision', 0.0),
                    'recall': checkpoint.get('metrics', {}).get('recall', 0.0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'training_samples': checkpoint.get('metrics', {}).get('training_samples', 0)
                }
                
                model_performances.append(model_info)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load model {model_path}: {e}")
                continue
        
        # Sort by F1 score (primary metric)
        model_performances.sort(key=lambda x: x['f1_score'], reverse=True)
        
        self.logger.info(f"✅ Successfully evaluated {len(model_performances)} models")
        
        # Display top models
        self.logger.info("🏆 Top performing models:")
        for i, model in enumerate(model_performances[:10]):
            self.logger.info(f"  {i+1:2d}. {model['chunk_name']:<25} "
                           f"F1: {model['f1_score']:.4f} "
                           f"Acc: {model['accuracy']:.4f}")
        
        return model_performances
    
    def select_best_models(self, model_performances):
        """Select top-k models for ensemble based on multiple criteria"""
        top_k = self.config['ensemble']['top_k_models']
        
        # Multi-criteria selection
        ensemble_models = []
        
        # Primary selection: Top F1 scores
        f1_sorted = sorted(model_performances, key=lambda x: x['f1_score'], reverse=True)
        primary_models = f1_sorted[:top_k]
        
        # Secondary selection: Diversity (different chunks/approaches)
        chunk_names_selected = set()
        diverse_models = []
        
        for model in f1_sorted:
            chunk_base = model['chunk_name'].split('_')[0]  # Get base chunk name
            if (chunk_base not in chunk_names_selected and 
                len(diverse_models) < top_k and
                model['f1_score'] > 0.5):  # Minimum quality threshold
                diverse_models.append(model)
                chunk_names_selected.add(chunk_base)
        
        # Combine primary and diverse selections
        final_models = []
        seen_paths = set()
        
        # Add primary models
        for model in primary_models:
            if model['path'] not in seen_paths:
                final_models.append(model)
                seen_paths.add(model['path'])
        
        # Fill remaining slots with diverse models
        for model in diverse_models:
            if model['path'] not in seen_paths and len(final_models) < top_k:
                final_models.append(model)
                seen_paths.add(model['path'])
        
        # Sort final selection by F1 score
        final_models.sort(key=lambda x: x['f1_score'], reverse=True)
        
        self.logger.info(f"\n🎯 Selected {len(final_models)} models for ensemble:")
        self.logger.info("=" * 70)
        for i, model in enumerate(final_models):
            self.logger.info(f"{i+1:2d}. {model['chunk_name']:<25} "
                           f"F1: {model['f1_score']:.4f} "
                           f"Acc: {model['accuracy']:.4f} "
                           f"Samples: {model.get('training_samples', 0):,}")
        self.logger.info("=" * 70)
        
        return final_models
    
    def load_ensemble_models(self, selected_models):
        """Load selected models for ensemble"""
        from src.milestone1.ultimate_multimodal_model import UltimateMultimodalSarcasmModel
        
        ensemble_models = []
        
        self.logger.info("🔄 Loading ensemble models...")
        
        for i, model_info in enumerate(selected_models):
            try:
                # Load checkpoint
                checkpoint = torch.load(model_info['path'], map_location=self.device)
                
                # Initialize model
                model = UltimateMultimodalSarcasmModel(self.config).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                ensemble_models.append({
                    'model': model,
                    'info': model_info,
                    'weight': model_info['f1_score']  # Use F1 score as initial weight
                })
                
                self.logger.info(f"✅ Loaded model {i+1}: {model_info['chunk_name']}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load model {model_info['path']}: {e}")
                continue
        
        self.logger.info(f"🎉 Successfully loaded {len(ensemble_models)} models for ensemble")
        return ensemble_models
    
    def create_meta_learner(self, ensemble_models):
        """Create meta-learner for optimal ensemble combination"""
        num_models = len(ensemble_models)
        num_classes = 2  # sarcastic, non-sarcastic
        
        class MetaLearner(nn.Module):
            """Advanced meta-learner for ensemble combination"""
            
            def __init__(self, num_models, num_classes, hidden_size=256):
                super().__init__()
                
                # Input: concatenated predictions from all models
                input_size = num_models * num_classes
                
                # Meta-learner network
                self.meta_network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(hidden_size // 4, num_classes)
                )
                
                # Model confidence weights (learnable)
                self.model_weights = nn.Parameter(torch.ones(num_models))
                
                # Attention mechanism for dynamic weighting
                self.attention = nn.MultiheadAttention(
                    embed_dim=num_classes, 
                    num_heads=2, 
                    batch_first=True
                )
            
            def forward(self, model_predictions, use_attention=True):
                """
                Forward pass through meta-learner
                
                Args:
                    model_predictions: List of prediction tensors from ensemble models
                    use_attention: Whether to use attention mechanism
                
                Returns:
                    Final ensemble predictions
                """
                batch_size = model_predictions[0].size(0)
                
                # Apply model weights
                weighted_predictions = []
                model_weights_normalized = F.softmax(self.model_weights, dim=0)
                
                for i, pred in enumerate(model_predictions):
                    weighted_pred = pred * model_weights_normalized[i]
                    weighted_predictions.append(weighted_pred)
                
                # Stack predictions for attention
                stacked_preds = torch.stack(weighted_predictions, dim=1)  # [batch, num_models, num_classes]
                
                if use_attention and len(model_predictions) > 1:
                    # Apply attention mechanism
                    attended_preds, attention_weights = self.attention(
                        stacked_preds, stacked_preds, stacked_preds
                    )
                    # Combine with residual connection
                    combined_preds = attended_preds + stacked_preds
                else:
                    combined_preds = stacked_preds
                
                # Flatten for meta-network
                flattened = combined_preds.flatten(start_dim=1)
                
                # Meta-learning prediction
                meta_output = self.meta_network(flattened)
                
                return meta_output, model_weights_normalized
        
        meta_learner = MetaLearner(num_models, num_classes).to(self.device)
        
        self.logger.info("🧠 Meta-learner created successfully")
        return meta_learner
    
    def train_ensemble(self, ensemble_models, meta_learner, train_dataloader, val_dataloader=None):
        """Train the ensemble meta-learner"""
        self.logger.info("🚂 Training ensemble meta-learner...")
        
        # Optimizer for meta-learner
        optimizer = torch.optim.AdamW(
            meta_learner.parameters(),
            lr=float(self.config['ensemble']['ensemble_lr']),
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['ensemble']['ensemble_epochs']
        )
        
        criterion = nn.CrossEntropyLoss()
        best_ensemble_f1 = 0.0
        
        # Training loop
        for epoch in range(1, self.config['ensemble']['ensemble_epochs'] + 1):
            meta_learner.train()
            
            # Set ensemble models to eval mode
            for ensemble_item in ensemble_models:
                ensemble_item['model'].eval()
            
            epoch_loss = 0.0
            all_predictions = []
            all_labels = []
            
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"🧠 Meta-learning Epoch {epoch}",
                leave=False
            )
            
            for batch in progress_bar:
                # Move batch to device
                batch = self.move_batch_to_device(batch)
                
                # Get predictions from all ensemble models
                model_predictions = []
                with torch.no_grad():
                    for ensemble_item in ensemble_models:
                        model = ensemble_item['model']
                        outputs = model(batch, training=False)
                        logits = outputs['logits']
                        probs = F.softmax(logits, dim=1)
                        model_predictions.append(probs)
                
                # Meta-learner forward pass
                optimizer.zero_grad()
                ensemble_logits, model_weights = meta_learner(model_predictions)
                
                # Calculate loss
                loss = criterion(ensemble_logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Collect metrics
                epoch_loss += loss.item()
                predictions = torch.argmax(ensemble_logits, dim=1).detach().cpu().numpy()
                labels = batch['labels'].detach().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(train_dataloader)
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            self.logger.info(f"Epoch {epoch:2d} - Loss: {avg_loss:.4f}, "
                           f"Acc: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate_ensemble(ensemble_models, meta_learner, val_dataloader)
                val_f1 = val_metrics['f1']
                
                self.logger.info(f"         - Val Acc: {val_metrics['accuracy']:.4f}, "
                               f"Val F1: {val_f1:.4f}")
                
                if val_f1 > best_ensemble_f1:
                    best_ensemble_f1 = val_f1
                    self.save_ensemble(ensemble_models, meta_learner, val_metrics)
        
        self.logger.info(f"🎉 Ensemble training completed! Best F1: {best_ensemble_f1:.4f}")
        return best_ensemble_f1
    
    def evaluate_ensemble(self, ensemble_models, meta_learner, dataloader):
        """Evaluate the trained ensemble"""
        self.logger.info("📊 Evaluating ensemble performance...")
        
        meta_learner.eval()
        for ensemble_item in ensemble_models:
            ensemble_item['model'].eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="🔍 Evaluating"):
                batch = self.move_batch_to_device(batch)
                
                # Get predictions from all ensemble models
                model_predictions = []
                for ensemble_item in ensemble_models:
                    model = ensemble_item['model']
                    outputs = model(batch, training=False)
                    logits = outputs['logits']
                    probs = F.softmax(logits, dim=1)
                    model_predictions.append(probs)
                
                # Meta-learner prediction
                ensemble_logits, _ = meta_learner(model_predictions)
                ensemble_probs = F.softmax(ensemble_logits, dim=1)
                predictions = torch.argmax(ensemble_logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(ensemble_probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        self.logger.info("🎯 Ensemble Evaluation Results:")
        self.logger.info(f"   Accuracy:  {accuracy:.4f}")
        self.logger.info(f"   F1 Score:  {f1:.4f}")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall:    {recall:.4f}")
        
        return metrics
    
    def move_batch_to_device(self, batch):
        """Move batch to device (same as trainer)"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                device_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        device_batch[key][sub_key] = sub_value.to(self.device, non_blocking=True)
                    else:
                        device_batch[key][sub_key] = sub_value
            else:
                device_batch[key] = value
        
        return device_batch
    
    def save_ensemble(self, ensemble_models, meta_learner, metrics):
        """Save the complete ensemble model"""
        ensemble_path = Path(self.config['paths']['final_model'])
        ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare ensemble data
        ensemble_data = {
            'meta_learner_state_dict': meta_learner.state_dict(),
            'ensemble_models_info': [item['info'] for item in ensemble_models],
            'ensemble_metrics': metrics,
            'config': self.config,
            'num_models': len(ensemble_models),
            'model_weights': [item['weight'] for item in ensemble_models],
            'creation_timestamp': datetime.now().isoformat(),
            'ensemble_type': 'meta_learner_with_attention'
        }
        
        # Save ensemble
        torch.save(ensemble_data, ensemble_path)
        
        self.logger.info(f"💾 Ultimate ensemble saved to: {ensemble_path}")
        self.logger.info(f"🎯 Final F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"🎯 Final Accuracy: {metrics['accuracy']:.4f}")
        
        return ensemble_path
    
    def build_complete_ensemble(self):
        """Complete ensemble building pipeline"""
        self.logger.info("🚀 Starting ultimate ensemble building pipeline...")
        
        # Step 1: Discover trained models
        model_performances = self.discover_trained_models()
        
        # Step 2: Select best models
        selected_models = self.select_best_models(model_performances)
        
        # Step 3: Load ensemble models
        ensemble_models = self.load_ensemble_models(selected_models)
        
        # Step 4: Create meta-learner
        meta_learner = self.create_meta_learner(ensemble_models)
        
        # Step 5: Load validation data for ensemble training
        from src.milestone1.enhanced_dataset import create_ultimate_dataloader
        
        # Find a validation chunk (use first available chunk for demo)
        chunks_dir = Path(self.config['paths']['chunks_dir'])
        train_chunks = list(chunks_dir.glob("ultimate_train_chunk_*.csv"))
        test_chunks = list(chunks_dir.glob("ultimate_test_chunk_*.csv"))
        
        if train_chunks and test_chunks:
            # Use first chunk for ensemble training
            train_dataloader = create_ultimate_dataloader(
                train_chunks[0], self.config, split='train', 
                batch_size=self.config['ensemble']['ensemble_batch_size']
            )
            val_dataloader = create_ultimate_dataloader(
                test_chunks[0], self.config, split='test', 
                batch_size=self.config['ensemble']['ensemble_batch_size']
            )
            
            # Step 6: Train ensemble
            best_f1 = self.train_ensemble(ensemble_models, meta_learner, train_dataloader, val_dataloader)
            
            # Step 7: Final evaluation
            final_metrics = self.evaluate_ensemble(ensemble_models, meta_learner, val_dataloader)
            
        else:
            self.logger.warning("⚠️ No training/validation chunks found. Saving ensemble without training.")
            # Just save the ensemble without training
            final_metrics = {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
            self.save_ensemble(ensemble_models, meta_learner, final_metrics)
        
        # Step 8: Generate final report
        self.generate_ensemble_report(selected_models, final_metrics)
        
        self.logger.info("🎉 Ultimate ensemble building completed successfully!")
        return final_metrics
    
    def generate_ensemble_report(self, selected_models, final_metrics):
        """Generate comprehensive ensemble report"""
        report = {
            'ensemble_summary': {
                'creation_date': datetime.now().isoformat(),
                'num_models': len(selected_models),
                'final_accuracy': final_metrics.get('accuracy', 0.0),
                'final_f1_score': final_metrics.get('f1', 0.0),
                'final_precision': final_metrics.get('precision', 0.0),
                'final_recall': final_metrics.get('recall', 0.0)
            },
            'individual_models': selected_models,
            'training_config': self.config,
            'performance_improvement': {
                'best_individual_f1': max(model['f1_score'] for model in selected_models),
                'ensemble_f1': final_metrics.get('f1', 0.0),
                'improvement': final_metrics.get('f1', 0.0) - max(model['f1_score'] for model in selected_models)
            }
        }
        
        # Save report
        report_path = Path(self.config['paths']['models_dir']) / 'ultimate_ensemble_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Ensemble report saved to: {report_path}")


def main():
    """Main function to run ensemble building"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Ensemble Builder')
    parser.add_argument('--config', default='config/milestone1_ultimate_config.yaml', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Build ensemble
    builder = UltimateEnsembleBuilder(args.config)
    final_metrics = builder.build_complete_ensemble()
    
    print(f"\n🎉 ULTIMATE ENSEMBLE COMPLETED!")
    print(f"🎯 Final F1 Score: {final_metrics['f1']:.4f}")
    print(f"🎯 Final Accuracy: {final_metrics['accuracy']:.4f}")
    
    if final_metrics['f1'] >= 0.80:
        print("🥇 OUTSTANDING PERFORMANCE! (F1 >= 80%)")
    elif final_metrics['f1'] >= 0.75:
        print("🥈 EXCELLENT PERFORMANCE! (F1 >= 75%)")
    elif final_metrics['f1'] >= 0.70:
        print("🥉 GOOD PERFORMANCE! (F1 >= 70%)")
    else:
        print("📈 ROOM FOR IMPROVEMENT (Consider hyperparameter tuning)")


if __name__ == "__main__":
    main()
