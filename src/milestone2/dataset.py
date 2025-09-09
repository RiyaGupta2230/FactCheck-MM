import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
from transformers import RobertaTokenizer, T5Tokenizer
from datasets import load_dataset

class MRPCDataset(Dataset):
    """Microsoft Research Paraphrase Corpus dataset"""
    def __init__(self, split='train', tokenizer_name='roberta-base', max_length=128):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_mrpc_data(split)
        
    def load_mrpc_data(self, split):
        """Load MRPC from Hugging Face datasets"""
        try:
            # Load from GLUE benchmark
            dataset = load_dataset("nyu-mll/glue", "mrpc", split=split)
            return dataset
        except:
            # Fallback to sample data
            return self.create_sample_data(split)
    
    def create_sample_data(self, split):
        """Create sample paraphrase data for testing"""
        if split == 'train':
            return [
                {"sentence1": "The company is located in New York", 
                 "sentence2": "The firm is based in NYC", "label": 1},
                {"sentence1": "I love machine learning", 
                 "sentence2": "Machine learning fascinates me", "label": 1},
                {"sentence1": "The weather is nice today", 
                 "sentence2": "Programming is challenging", "label": 0},
                {"sentence1": "She works at Microsoft", 
                 "sentence2": "She is employed by Microsoft Corporation", "label": 1},
                {"sentence1": "The meeting starts at 9 AM", 
                 "sentence2": "Our meeting begins at nine in the morning", "label": 1},
                {"sentence1": "Python is a programming language", 
                 "sentence2": "The sky is blue today", "label": 0},
                {"sentence1": "I need to buy groceries", 
                 "sentence2": "I have to purchase food items", "label": 1},
                {"sentence1": "The project deadline is tomorrow", 
                 "sentence2": "We must finish the project by tomorrow", "label": 1},
            ] * 100  # Repeat for more training data
        else:
            return [
                {"sentence1": "The meeting is scheduled for tomorrow", 
                 "sentence2": "Tomorrow we have a meeting planned", "label": 1},
                {"sentence1": "Artificial intelligence is advancing", 
                 "sentence2": "The car is red", "label": 0},
                {"sentence1": "I enjoy reading books", 
                 "sentence2": "Reading books is my hobby", "label": 1},
                {"sentence1": "The train arrives at noon", 
                 "sentence2": "The train reaches at 12 PM", "label": 1},
            ] * 25
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        sentence1 = str(item['sentence1'])
        sentence2 = str(item['sentence2'])
        label = int(item['label'])
        
        # Tokenize sentence pair (RoBERTa format)
        encoding = self.tokenizer(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SiameseParaphraseDataset(Dataset):
    """Dataset for Siamese network paraphrase detection"""
    def __init__(self, data_path, tokenizer_name='roberta-base', max_length=128):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load custom paraphrase data"""
        if not os.path.exists(data_path):
            # Create sample data
            return [
                {"sentence1": "The weather is beautiful", "sentence2": "It's a lovely day", "label": 1},
                {"sentence1": "I am going to the store", "sentence2": "I'm heading to the shop", "label": 1},
                {"sentence1": "The cat is sleeping", "sentence2": "The dog is running", "label": 0},
            ] * 200
            
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        sentence1 = str(item['sentence1'])
        sentence2 = str(item['sentence2'])
        label = int(item['label'])
        
        # Tokenize sentences separately
        encoding1 = self.tokenizer(
            sentence1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(),
            'attention_mask_1': encoding1['attention_mask'].squeeze(),
            'input_ids_2': encoding2['input_ids'].squeeze(),
            'attention_mask_2': encoding2['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ParaphraseGenerationDataset(Dataset):
    """Dataset for training T5 paraphrase generation"""
    def __init__(self, data_path, tokenizer_name='t5-small', max_length=128):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_generation_data(data_path)
    
    def load_generation_data(self, data_path):
        """Load paraphrase generation data"""
        # Sample sarcastic to literal pairs
        return [
            {"input": "Oh great, another meeting", "target": "I am not pleased about another meeting"},
            {"input": "Perfect timing for this to happen", "target": "This happened at an inconvenient time"},
            {"input": "I just love working overtime", "target": "I do not enjoy working overtime"},
            {"input": "Wonderful weather for a picnic", "target": "The weather is not suitable for a picnic"},
            {"input": "Amazing customer service", "target": "The customer service was poor"},
        ] * 200  # Repeat for training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = f"paraphrase: {item['input']}"
        target_text = item['target']
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def create_paraphrase_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0):
    """Create DataLoader for paraphrase dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
