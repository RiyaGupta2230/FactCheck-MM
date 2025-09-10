"""
Ultimate training script for milestone1 sarcasm detection
Optimized for maximum accuracy with your prepared chunks
"""
import os
import yaml
import torch
import pandas as pd
import numpy as np
import glob
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW 
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
from milestone1_feature_engineering import prepare_features_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateSarcasmDataset(Dataset):
    """Enhanced dataset with linguistic features for ultimate performance"""
    
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

class UltimateTrainer:
    """Ultimate training class for maximum accuracy"""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() or torch.backends.mps.is_available() 
            else 'cpu'
        )
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for training"""
        log_dir = os.path.join(self.config['paths']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'ultimate_training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train_ultimate_model(self):
        """Train the ultimate sarcasm detection model"""
        
        logger.info("🚀 Starting Ultimate Sarcasm Detection Training")
        logger.info(f"🎯 Target F1: {self.config['project']['target_f1']}")
        logger.info(f"📱 Device: {self.device}")
        
        # Initialize tokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained(self.config['model']['name'])
        
        # Get all training chunks
        chunks_dir = self.config['data']['chunks_dir']
        train_pattern = self.config['data']['train_pattern']
        train_files = sorted(glob.glob(os.path.join(chunks_dir, train_pattern)))
        
        logger.info(f"📁 Found {len(train_files)} training chunks")
        
        # Training results storage
        training_results = []
        best_f1 = 0
        
        # Initialize model (will be reused across chunks)
        model = RobertaForSequenceClassification.from_pretrained(
            self.config['model']['name'], 
            num_labels=2
        )
        model.to(self.device)
        
        # Process each chunk
        for chunk_idx, chunk_file in enumerate(train_files):
            logger.info(f"\n🔄 Processing chunk {chunk_idx + 1}/{len(train_files)}")
            logger.info(f"📄 File: {os.path.basename(chunk_file)}")
            
            # Load and prepare chunk data
            df = pd.read_csv(chunk_file)
            logger.info(f"📊 Loaded {len(df)} samples")
            
            # Extract enhanced features
            df_enhanced, feature_columns = prepare_features_for_training(df)
            
            # Create dataset
            dataset = UltimateSarcasmDataset(
                texts=df_enhanced['text'],
                labels=df_enhanced['label'],
                features=df_enhanced[feature_columns],
                tokenizer=tokenizer,
                max_length=self.config['model']['max_length']
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['hardware']['dataloader_num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
            # Setup optimizer and scheduler for this chunk
            optimizer = AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            total_steps = len(dataloader) * self.config['training']['epochs_per_chunk']
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['training']['warmup_steps'],
                num_training_steps=total_steps
            )
            
            # Train on this chunk
            chunk_results = self.train_on_chunk(
                model, dataloader, optimizer, scheduler, chunk_idx + 1
            )
            
            training_results.append({
                'chunk_idx': chunk_idx + 1,
                'chunk_file': chunk_file,
                'samples': len(df),
                'results': chunk_results
            })
            
            # Save best model
            final_f1 = chunk_results[-1]['f1']
            if final_f1 > best_f1:
                best_f1 = final_f1
                self.save_model(model, tokenizer, chunk_idx + 1, final_f1)
                logger.info(f"💾 New best model saved! F1: {final_f1:.4f}")
        
        # Save training history
        self.save_training_history(training_results)
        
        logger.info("\n🎉 Ultimate training completed!")
        logger.info(f"🏆 Best F1 Score: {best_f1:.4f}")
        
        return training_results
    
    def train_on_chunk(self, model, dataloader, optimizer, scheduler, chunk_num):
        """Train model on a single chunk"""
        
        model.train()
        epoch_results = []
        
        for epoch in range(self.config['training']['epochs_per_chunk']):
            total_loss = 0
            all_predictions = []
            all_labels = []
            
            pbar = tqdm(
                dataloader, 
                desc=f"Chunk {chunk_num}, Epoch {epoch + 1}"
            )
            
            for batch in pbar:
                optimizer.zero_grad()
                
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Collect metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Calculate epoch metrics
            epoch_loss = total_loss / len(dataloader)
            epoch_accuracy = accuracy_score(all_labels, all_predictions)
            epoch_f1 = f1_score(all_labels, all_predictions)
            
            epoch_result = {
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_accuracy,
                'f1': epoch_f1
            }
            
            epoch_results.append(epoch_result)
            
            logger.info(
                f"Chunk {chunk_num}, Epoch {epoch + 1}: "
                f"Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, F1: {epoch_f1:.4f}"
            )
        
        return epoch_results
    
    def save_model(self, model, tokenizer, chunk_idx, f1_score):
        """Save the model"""
        
        model_dir = self.config['paths']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        # Save with chunk and score info
        model_name = f"best_model_chunk_{chunk_idx}_f1_{f1_score:.4f}"
        model_path = os.path.join(model_dir, model_name)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Also save as latest best
        latest_path = os.path.join(model_dir, "latest_best")
        model.save_pretrained(latest_path)
        tokenizer.save_pretrained(latest_path)
    
    def save_training_history(self, results):
        """Save training history"""
        
        output_dir = self.config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        history_file = os.path.join(output_dir, 'ultimate_training_history.json')
        
        with open(history_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Training history saved: {history_file}")

def main():
    """Main training function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python milestone1_train_ultimate.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    trainer = UltimateTrainer(config_path)
    results = trainer.train_ultimate_model()
    
    print("✅ Ultimate training completed successfully!")
    print(f"📊 Processed {len(results)} chunks")
    
if __name__ == "__main__":
    main()
