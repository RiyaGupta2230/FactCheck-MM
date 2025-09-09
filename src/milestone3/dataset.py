import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

class FEVERDataset(Dataset):
    """FEVER (Fact Extraction and VERification) dataset"""
    def __init__(self, split='train', tokenizer_name='microsoft/deberta-v3-base', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_fever_data(split)
        
    def load_fever_data(self, split):
        """Load FEVER dataset from Hugging Face"""
        try:
            # Load FEVER dataset
            dataset = load_dataset("fever", "v1.0", split=split)
            return dataset
        except:
            # Fallback to sample data
            return self.create_sample_fever_data(split)
    
    def create_sample_fever_data(self, split):
        """Create sample fact verification data"""
        if split == 'train':
            return [
                {"claim": "Climate change is caused by human activities", "label": "SUPPORTS"},
                {"claim": "The Earth is flat", "label": "REFUTES"},
                {"claim": "Vaccines cause autism", "label": "REFUTES"},
                {"claim": "Water boils at 100°C at sea level", "label": "SUPPORTS"},
                {"claim": "Artificial intelligence will replace all jobs by 2025", "label": "NOT ENOUGH INFO"},
                {"claim": "Regular exercise improves cardiovascular health", "label": "SUPPORTS"},
                {"claim": "The Moon landing was faked", "label": "REFUTES"},
                {"claim": "Chocolate is made from cocoa beans", "label": "SUPPORTS"},
                {"claim": "All birds can fly", "label": "REFUTES"},
                {"claim": "The Pacific Ocean is the largest ocean", "label": "SUPPORTS"},
            ] * 500  # Repeat for more training data
        else:
            return [
                {"claim": "The sun rises in the east", "label": "SUPPORTS"},
                {"claim": "Humans have 12 fingers", "label": "REFUTES"}, 
                {"claim": "There are secret alien bases on Mars", "label": "NOT ENOUGH INFO"},
                {"claim": "Photosynthesis occurs in plants", "label": "SUPPORTS"},
            ] * 125
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        claim = str(item['claim'])
        label_str = item['label']
        
        # Map labels to integers
        label_map = {
            'SUPPORTS': 0,
            'REFUTES': 1, 
            'NOT ENOUGH INFO': 2,
            'NOT_ENOUGH_INFO': 2
        }
        
        label = label_map.get(label_str, 2)  # Default to NOT_ENOUGH_INFO
        
        # Tokenize claim
        encoding = self.tokenizer(
            claim,
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

class LIARDataset(Dataset):
    """LIAR dataset for political fact-checking"""
    def __init__(self, data_path, tokenizer_name='microsoft/deberta-v3-base', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_liar_data(data_path)
        
    def load_liar_data(self, data_path):
        """Load LIAR dataset"""
        if not os.path.exists(data_path):
            # Create sample political fact-checking data
            return [
                {"statement": "Unemployment rate is at historic lows", "label": "mostly-true"},
                {"statement": "Crime has increased by 300% this year", "label": "false"},
                {"statement": "The economy is doing well", "label": "half-true"},
                {"statement": "We spend more on defense than education", "label": "true"},
                {"statement": "All immigrants are criminals", "label": "pants-fire"},
            ] * 1000
            
        # Load actual LIAR data if available
        with open(data_path, 'r') as f:
            data = []
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    data.append({
                        'statement': parts[1],
                        'label': parts[0]
                    })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        statement = str(item['statement'])
        label_str = item['label']
        
        # Map LIAR labels to 3 classes
        label_map = {
            'true': 0, 'mostly-true': 0,  # SUPPORTS
            'false': 1, 'pants-fire': 1, 'barely-true': 1,  # REFUTES
            'half-true': 2  # NOT_ENOUGH_INFO
        }
        
        label = label_map.get(label_str, 2)
        
        encoding = self.tokenizer(
            statement,
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

class ClaimEvidenceDataset(Dataset):
    """Dataset for claim-evidence pairs"""
    def __init__(self, data_path, tokenizer_name='microsoft/deberta-v3-base', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_claim_evidence_data(data_path)
    
    def load_claim_evidence_data(self, data_path):
        """Load claim-evidence pairs"""
        # Sample claim-evidence pairs
        return [
            {
                "claim": "Climate change is real",
                "evidence": "Multiple scientific studies show rising global temperatures due to greenhouse gases",
                "label": "SUPPORTS"
            },
            {
                "claim": "The Earth is 6000 years old", 
                "evidence": "Geological and astronomical evidence shows Earth is approximately 4.5 billion years old",
                "label": "REFUTES"
            },
            {
                "claim": "Eating carrots improves night vision",
                "evidence": "Carrots contain beta-carotene which is converted to vitamin A, supporting eye health",
                "label": "SUPPORTS"
            }
        ] * 1000  # Expand for training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        claim = str(item['claim'])
        evidence = str(item['evidence'])
        label_str = item['label']
        
        # Combine claim and evidence
        combined_text = f"Claim: {claim} [SEP] Evidence: {evidence}"
        
        label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2}
        label = label_map.get(label_str, 2)
        
        encoding = self.tokenizer(
            combined_text,
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

def create_fact_verification_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0):
    """Create DataLoader for fact verification dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
