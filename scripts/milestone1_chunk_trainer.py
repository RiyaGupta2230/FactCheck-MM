"""
Chunk-based training system for sarcasm detection
Optimized for your prepared chunks and maximum accuracy
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
import os
import glob
import yaml
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import json
from milestone1_feature_engineering import prepare_features_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSarcasmDataset(Dataset):
    """Enhanced dataset with linguistic features"""
    
    def __init__(self, texts, labels, features, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels  
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get linguistic features
        feature_vector = torch.tensor(
            self.features.iloc[idx].values, 
            dtype=torch.float32
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': feature_vector,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EnhancedSarcasmModel(nn.Module):
    """Enhanced model with feature fusion"""
    
    def __init__(self, model_name, num_features, num_labels=2, dropout=0.15):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # Feature fusion
        hidden_size = self.text_encoder.config.hidden_size
        self.feature_projection = nn.Linear(num_features, hidden_size // 4)
        self.fusion_layer = nn.Linear(hidden_size + hidden_size // 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, features, labels=None):
        # Get text embeddings
        text_outputs = self.text_encoder.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embedding = text_outputs.pooler_output
        
        # Project features
        feature_embedding = torch.relu(self.feature_projection(features))
        
        # Fuse embeddings
        combined = torch.cat([text_embedding, feature_embedding], dim=1)
        fused = torch.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fn(logits, labels)
            
        return outputs

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_on_chunk(model, dataloader, optimizer, scheduler, device):
    """Train model on a single chunk"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs['loss']
            total_loss += loss.item()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_chunk_based_model(config_path):
    """Main training function for chunk-based learning"""
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"🚀 Starting chunk-based training: {config['project']['name']}")
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() or torch.backends.mps.is_available() else 'cpu')
    logger.info(f"📱 Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Get chunk files
    chunks_dir = config['data']['chunks_dir']
    train_pattern = config['data']['train_pattern']
    train_files = sorted(glob.glob(os.path.join(chunks_dir, train_pattern)))
    
    logger.info(f"📁 Found {len(train_files)} training chunks")
    
    # Training results storage
    training_history = []
    best_f1 = 0
    
    # Process each chunk
    for chunk_idx, chunk_file in enumerate(train_files):
        logger.info(f"\n🔄 Processing chunk {chunk_idx + 1}/{len(train_files)}: {os.path.basename(chunk_file)}")
        
        # Load and prepare chunk data
        df = pd.read_csv(chunk_file)
        logger.info(f"📊 Loaded {len(df)} samples")
        
        # Extract features
        df_enhanced, feature_columns = prepare_features_for_training(df)
        feature_matrix = df_enhanced[feature_columns]
        
        # Create dataset
        dataset = EnhancedSarcasmDataset(
            texts=df_enhanced['text'],
            labels=df_enhanced['label'],
            features=feature_matrix,
            tokenizer=tokenizer,
            max_length=config['model']['max_length']
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['hardware']['dataloader_num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
        
        # Initialize model (fresh for each chunk or continue from best)
        if chunk_idx == 0:
            model = EnhancedSarcasmModel(
                model_name=config['model']['name'],
                num_features=len(feature_columns),
                dropout=config['model']['dropout']
            )
        
        model.to(device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        total_steps = len(dataloader) * config['training']['epochs_per_chunk']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Train on chunk
        chunk_metrics = []
        
        for epoch in range(config['training']['epochs_per_chunk']):
            logger.info(f"🏃‍♂️ Epoch {epoch + 1}/{config['training']['epochs_per_chunk']}")
            
            metrics = train_on_chunk(model, dataloader, optimizer, scheduler, device)
            chunk_metrics.append(metrics)
            
            logger.info(f"📈 Epoch {epoch + 1} - Loss: {metrics['loss']:.4f}, "
                       f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save chunk results
        chunk_result = {
            'chunk_idx': chunk_idx,
            'chunk_file': chunk_file,
            'samples': len(df),
            'final_metrics': chunk_metrics[-1]
        }
        training_history.append(chunk_result)
        
        # Save model if best F1
        if chunk_metrics[-1]['f1'] > best_f1:
            best_f1 = chunk_metrics[-1]['f1']
            
            # Create output directory
            os.makedirs(config['paths']['model_dir'], exist_ok=True)
            
            # Save model
            model_path = os.path.join(config['paths']['model_dir'], f'best_model_chunk_{chunk_idx}')
            torch.save(model.state_dict(), f"{model_path}.pt")
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"💾 Saved new best model with F1: {best_f1:.4f}")
    
    # Save training history
    history_path = os.path.join(config['paths']['output_dir'], 'training_history.json')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"🎉 Training completed! Best F1: {best_f1:.4f}")
    logger.info(f"📋 Training history saved to: {history_path}")
    
    return training_history

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python milestone1_chunk_trainer.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    train_chunk_based_model(config_path)
