import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score
import yaml
from tqdm import tqdm
import sys

# Import chunking logic from src/milestone3/chunk_datasets
sys.path.append('src')
from milestone3.chunk_datasets import DatasetChunker

class ChunkedMultimodalDataset(Dataset):
    def __init__(self, chunk_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_path = chunk_path

        # Load dataset
        if isinstance(chunk_path, str):
            chunk_path = Path(chunk_path)
        if chunk_path.suffix == '.csv':
            self.data = pd.read_csv(chunk_path)
        else:
            with open(chunk_path, 'r') as f:
                self.data = json.load(f)
                self.data = pd.DataFrame(self.data)

        self.data = self.data.dropna(subset=['text', 'label'])
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        
        print(f"Loaded {len(self.texts)} samples from {chunk_path.name}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
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

class MultiTaskModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class ChunkTrainer:
    def __init__(self, config_path="config/training_config.yaml"):
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Convert string values to proper numeric types
        self.config['training']['learning_rate'] = float(self.config['training']['learning_rate'])
        self.config['training']['batch_size'] = int(self.config['training']['batch_size'])
        self.config['training']['epochs_per_chunk'] = int(self.config['training']['epochs_per_chunk'])
        self.config['training']['max_length'] = int(self.config['training']['max_length'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model_name = self.config['training']['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        Path(self.config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)

        
    def train_chunk(self, chunk_path, chunk_id):
        print(f"\n{'='*60}")
        print(f"TRAINING CHUNK: {chunk_id}")
        print(f"File: {chunk_path}")
        print(f"{'='*60}")
        
        model = MultiTaskModel(self.model_name).to(self.device)
        dataset = ChunkedMultimodalDataset(chunk_path, self.tokenizer, self.config['training']['max_length'])
        
        if len(dataset) == 0:
            print(f"❌ No valid data in chunk {chunk_id}, skipping...")
            return None, float('inf')
            
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        optimizer = AdamW(model.parameters(), lr=self.config['training']['learning_rate'])
        num_epochs = self.config['training']['epochs_per_chunk']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_epochs*len(dataloader)
        )
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        
        avg_total_loss = total_loss / num_epochs
        
        # Save model
        model_path = Path(self.config['paths']['models_dir']) / f"model_{chunk_id}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'chunk_path': str(chunk_path),
            'chunk_id': chunk_id,
            'avg_loss': avg_total_loss,
            'model_name': self.model_name,
            'config': self.config
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Final average loss: {avg_total_loss:.4f}")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        
        return model_path, avg_total_loss
    
    def train_all_chunks(self):
        chunks_dir = Path(self.config['paths']['chunks_dir'])
        chunk_files = list(chunks_dir.glob('*.csv'))
        
        if not chunk_files:
            print("No chunk files found! Creating chunks first...")
            chunker = DatasetChunker()
            chunker.create_chunks()
            chunk_files = list(chunks_dir.glob('*.csv'))
            
        print(f"Found {len(chunk_files)} chunk files")
        results = {}
        
        for i, chunk_file in enumerate(chunk_files, 1):
            chunk_id = chunk_file.stem
            print(f"\nProcessing chunk {i}/{len(chunk_files)}: {chunk_id}")
            
            try:
                model_path, avg_loss = self.train_chunk(chunk_file, chunk_id)
                
                if model_path is not None:
                    results[chunk_id] = {
                        'model_path': str(model_path),
                        'avg_loss': avg_loss,
                        'chunk_file': str(chunk_file)
                    }
                    print(f"✅ Completed chunk {chunk_id} successfully")
                else:
                    results[chunk_id] = {'error': 'Empty or invalid chunk data'}
                    print(f"❌ Skipped chunk {chunk_id} due to invalid data")
                    
            except Exception as e:
                print(f"❌ Error training chunk {chunk_id}: {e}")
                results[chunk_id] = {'error': str(e)}
        
        # Save results
        results_path = 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunk_files)}")
        print(f"Successfully trained: {len([k for k,v in results.items() if 'error' not in v])}")
        print(f"Failed: {len([k for k,v in results.items() if 'error' in v])}")
        print(f"Results saved to: {results_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Train multimodal sarcasm detection chunked models")
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Config file path')
    parser.add_argument('--chunked', action='store_true', help='Run chunked training (recommended)')
    parser.add_argument('--single_chunk', type=str, help='Train only a specific chunk (filename without .csv)')
    
    args = parser.parse_args()
    
    if args.chunked:
        trainer = ChunkTrainer(args.config)
        
        if args.single_chunk:
            # Train single chunk
            chunk_path = Path(trainer.config['paths']['chunks_dir']) / f"{args.single_chunk}.csv"
            if chunk_path.exists():
                trainer.train_chunk(chunk_path, args.single_chunk)
            else:
                print(f"❌ Chunk file not found: {chunk_path}")
        else:
            # Train all chunks
            results = trainer.train_all_chunks()
            
            if results:
                successful_models = [k for k, v in results.items() if 'error' not in v]
                print(f"\n📊 Training completed!")
                print(f"✅ Successful models: {len(successful_models)}")
                print(f"📁 Results saved to: training_results.json")
                print(f"🚀 Next step: python -m src.milestone3.ensemble_builder")
            else:
                print("\n❌ No models were trained successfully.")
    else:
        print("💡 Use --chunked flag for chunked training pipeline:")
        print("   python scripts/train_milestone3.py --chunked")

if __name__ == "__main__":
    main()
