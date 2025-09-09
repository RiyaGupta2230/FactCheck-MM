import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import sys
import os

class ChunkedDataset(Dataset):
    """Dataset class for loading chunked data"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Clean data
        self.data = self.data.dropna(subset=['text', 'label'])
        self.data = self.data[self.data['text'].astype(str).str.len() > 0]
        
        print(f"Loaded {len(self.data)} samples from {data_path.name}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SarcasmClassifier(nn.Module):
    """Basic sarcasm classifier model"""
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class ModelEnsemble(nn.Module):
    """Ensemble model combining multiple trained models"""
    def __init__(self, model_paths, model_name='bert-base-uncased'):
        super().__init__()
        self.model_name = model_name
        self.models = nn.ModuleList()
        self.model_paths = model_paths
        
        print(f"Loading {len(model_paths)} models for ensemble...")
        
        # Load all models
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/{len(model_paths)}: {Path(model_path).name}")
            checkpoint = torch.load(model_path, map_location='cpu')
            model = SarcasmClassifier(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to eval mode
            self.models.append(model)
        
        self.num_models = len(self.models)
        print(f"Successfully loaded {self.num_models} models")
        
        # Meta-learner for combining predictions
        self.meta_classifier = nn.Sequential(
            nn.Linear(2 * self.num_models, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, input_ids, attention_mask, use_meta_learner=True):
        # Get predictions from all models
        model_outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(input_ids, attention_mask)
                model_outputs.append(torch.softmax(output, dim=1))
        
        # Simple voting ensemble (no meta-learner)
        if not use_meta_learner:
            ensemble_output = torch.stack(model_outputs, dim=0).mean(dim=0)
            return ensemble_output
        
        # Meta-learner approach
        ensemble_input = torch.cat(model_outputs, dim=1)
        final_output = self.meta_classifier(ensemble_input)
        return final_output

class EnsembleBuilder:
    def __init__(self, config_path="config/training_config.yaml"):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def select_best_models(self, results_path='training_results.json', method='loss'):
        """Select top K models based on performance metric"""
        if not Path(results_path).exists():
            print(f"Results file not found: {results_path}")
            print("Please run training first: python scripts/train_milestone3.py")
            return []
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Filter out failed models
        successful_models = {k: v for k, v in results.items() 
                           if isinstance(v, dict) and 'error' not in v and 'avg_loss' in v}
        
        if not successful_models:
            print("No successful models found!")
            return []
        
        print(f"Found {len(successful_models)} successful models")
        
        # Sort by metric (lower loss is better)
        if method == 'loss':
            sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['avg_loss'])
        else:
            # Could add accuracy-based sorting here
            sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['avg_loss'])
        
        # Select top K models
        top_k = min(self.config['ensemble']['top_k_models'], len(sorted_models))
        best_models = sorted_models[:top_k]
        
        model_paths = []
        
        print(f"\nSelected top {top_k} models:")
        print("-" * 60)
        
        for i, (chunk_id, info) in enumerate(best_models):
            model_path = Path(self.config['paths']['models_dir']) / f"model_{chunk_id}.pt"
            if model_path.exists():
                model_paths.append(model_path)
                print(f"{i+1:2d}. {chunk_id:<25} - Loss: {info['avg_loss']:.4f}")
            else:
                print(f"⚠️  Model file not found for {chunk_id}: {model_path}")
        
        print("-" * 60)
        print(f"Total models available for ensemble: {len(model_paths)}")
        
        return model_paths
    
    def create_ensemble(self, model_paths):
        """Create ensemble from selected models"""
        if not model_paths:
            print("No model paths provided!")
            return None
        
        print(f"\nCreating ensemble from {len(model_paths)} models...")
        ensemble = ModelEnsemble(model_paths, self.config['training']['model_name'])
        return ensemble.to(self.device)
    
    def train_meta_learner(self, ensemble, train_data_path):
        """Train the ensemble meta-learner component"""
        print("\nTraining ensemble meta-learner...")
        
        # Create tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(self.config['training']['model_name'])
        train_dataset = ChunkedDataset(train_data_path, tokenizer, self.config['training']['max_length'])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['ensemble']['ensemble_batch_size'], 
            shuffle=True
        )
        
        # Setup training for meta-learner only
        # Convert string learning rate to float
        ensemble_lr = float(self.config['ensemble']['ensemble_lr'])
        optimizer = torch.optim.AdamW(
            ensemble.meta_classifier.parameters(), 
            lr=ensemble_lr
        )

        criterion = nn.CrossEntropyLoss()
        
        num_epochs = self.config['ensemble']['ensemble_epochs']
        
        for epoch in range(num_epochs):
            ensemble.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Meta-learning Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = ensemble(input_ids, attention_mask, use_meta_learner=True)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Meta-learning Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        return ensemble
    
    def evaluate_ensemble(self, ensemble, test_data_path, use_meta_learner=True):
        """Evaluate ensemble model"""
        print(f"\nEvaluating ensemble (meta-learner: {use_meta_learner})...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config['training']['model_name'])
        test_dataset = ChunkedDataset(test_data_path, tokenizer, self.config['training']['max_length'])
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        ensemble.eval()
        predictions, true_labels, probabilities = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = ensemble(input_ids, attention_mask, use_meta_learner=use_meta_learner)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels)
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        print(f"\n{'='*50}")
        print(f"ENSEMBLE EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(f"F1 Score (macro):    {f1_macro:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=['Non-Sarcastic', 'Sarcastic']))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                Non-Sarc  Sarcastic")
        print(f"Actual Non-Sarc    {cm[0,0]:6d}     {cm[0,1]:6d}")
        print(f"Actual Sarcastic   {cm[1,0]:6d}     {cm[1,1]:6d}")
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'f1_macro': f1_macro,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }
    
    def compare_ensemble_methods(self, ensemble, test_data_path):
        """Compare different ensemble methods"""
        print(f"\n{'='*60}")
        print(f"COMPARING ENSEMBLE METHODS")
        print(f"{'='*60}")
        
        # Evaluate with simple voting
        results_voting = self.evaluate_ensemble(ensemble, test_data_path, use_meta_learner=False)
        
        # Evaluate with meta-learner
        results_meta = self.evaluate_ensemble(ensemble, test_data_path, use_meta_learner=True)
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Simple Voting:    Accuracy: {results_voting['accuracy']:.4f}, F1: {results_voting['f1_weighted']:.4f}")
        print(f"Meta-learner:     Accuracy: {results_meta['accuracy']:.4f}, F1: {results_meta['f1_weighted']:.4f}")
        
        # Return best method
        if results_meta['f1_weighted'] > results_voting['f1_weighted']:
            print(f"🏆 Meta-learner performs better!")
            return results_meta, True
        else:
            print(f"🏆 Simple voting performs better!")
            return results_voting, False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ensemble model from trained chunks")
    parser.add_argument("--config", default="config/training_config.yaml", help="Config file path")
    parser.add_argument("--results", default="training_results.json", help="Training results file")
    parser.add_argument("--no-meta-learning", action="store_true", help="Skip meta-learner training")
    parser.add_argument("--compare", action="store_true", help="Compare ensemble methods")
    args = parser.parse_args()
    
    try:
        builder = EnsembleBuilder(args.config)
        
        # Step 1: Select best models
        print("Step 1: Selecting best models...")
        best_model_paths = builder.select_best_models(args.results)
        
        if not best_model_paths:
            print("❌ No models available for ensemble creation!")
            return
        
        # Step 2: Create ensemble
        print("\nStep 2: Creating ensemble...")
        ensemble = builder.create_ensemble(best_model_paths)
        
        if ensemble is None:
            print("❌ Failed to create ensemble!")
            return
        
        # Step 3: Train meta-learner (optional)
        if not args.no_meta_learning:
            train_data = Path(builder.config['datasets']['sarc']['test_balanced'])
            if train_data.exists():
                print("\nStep 3: Training ensemble meta-learner...")
                ensemble = builder.train_meta_learner(ensemble, train_data)
            else:
                print(f"⚠️  Training data not found: {train_data}")
        
        # Step 4: Evaluate ensemble
        test_data = Path(builder.config['datasets']['sarc']['test_unbalanced'])
        if test_data.exists():
            if args.compare:
                print("\nStep 4: Comparing ensemble methods...")
                best_results, use_meta = builder.compare_ensemble_methods(ensemble, test_data)
            else:
                print("\nStep 4: Evaluating ensemble...")
                best_results = builder.evaluate_ensemble(ensemble, test_data)
                use_meta = not args.no_meta_learning
        else:
            print(f"⚠️  Test data not found: {test_data}")
            best_results = {}
            use_meta = False
        
        # Step 5: Save ensemble model
        print("\nStep 5: Saving ensemble model...")
        final_model_path = Path(builder.config['paths']['final_model'])
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'ensemble_state_dict': ensemble.state_dict(),
            'model_paths': [str(p) for p in best_model_paths],
            'config': builder.config,
            'results': best_results,
            'use_meta_learner': use_meta,
            'num_models': len(best_model_paths)
        }, final_model_path)
        
        print(f"✅ Ensemble model saved to: {final_model_path}")
        print(f"📊 Model combines {len(best_model_paths)} chunk models")
        
        if best_results:
            print(f"🎯 Final Performance: Accuracy={best_results['accuracy']:.4f}, F1={best_results['f1_weighted']:.4f}")
        
        print("\n🎉 Ensemble building completed successfully!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
