"""
Data utilities for milestone1 sarcasm detection
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import logging

logger = logging.getLogger(__name__)

def load_chunks(chunks_dir, pattern):
    """Load multiple chunk files"""
    files = sorted(glob.glob(os.path.join(chunks_dir, pattern)))
    logger.info(f"Found {len(files)} chunk files")
    
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} samples")
    
    return combined_df

def balance_dataset(df, target_column='label', method='oversample'):
    """Balance dataset classes"""
    
    class_counts = df[target_column].value_counts()
    logger.info(f"Original class distribution: {dict(class_counts)}")
    
    if method == 'oversample':
        # Oversample minority class
        max_count = class_counts.max()
        balanced_dfs = []
        
        for class_val in class_counts.index:
            class_df = df[df[target_column] == class_val]
            if len(class_df) < max_count:
                # Oversample with replacement
                oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
    elif method == 'undersample':
        # Undersample majority class
        min_count = class_counts.min()
        balanced_dfs = []
        
        for class_val in class_counts.index:
            class_df = df[df[target_column] == class_val]
            sampled = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    else:
        balanced_df = df
    
    new_class_counts = balanced_df[target_column].value_counts()
    logger.info(f"Balanced class distribution: {dict(new_class_counts)}")
    
    return balanced_df

def create_stratified_split(df, test_size=0.2, stratify_column='label', random_state=42):
    """Create stratified train/validation split"""
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_column],
        random_state=random_state
    )
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def prepare_text_for_tokenization(text):
    """Prepare text for optimal tokenization"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Handle special tokens
    text = text.replace('[URL]', '<URL>')
    text = text.replace('[MENTION]', '<MENTION>')
    text = text.replace('[HASHTAG]', '<HASHTAG>')
    
    return text

class ChunkDataLoader:
    """Efficient data loader for chunk-based training"""
    
    def __init__(self, chunks_dir, pattern, batch_size=16, shuffle=True):
        self.chunks_dir = chunks_dir
        self.pattern = pattern
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_files = sorted(glob.glob(os.path.join(chunks_dir, pattern)))
        
    def __iter__(self):
        for chunk_file in self.chunk_files:
            df = pd.read_csv(chunk_file)
            
            if self.shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
            
            # Yield batches
            for i in range(0, len(df), self.batch_size):
                batch_df = df.iloc[i:i+self.batch_size]
                yield batch_df, os.path.basename(chunk_file)
    
    def __len__(self):
        return len(self.chunk_files)

def calculate_dataset_statistics(df):
    """Calculate comprehensive dataset statistics"""
    
    stats = {
        'total_samples': len(df),
        'class_distribution': dict(df['label'].value_counts()),
        'text_length_stats': {
            'mean': df['text'].str.len().mean(),
            'median': df['text'].str.len().median(),
            'std': df['text'].str.len().std(),
            'min': df['text'].str.len().min(),
            'max': df['text'].str.len().max()
        },
        'word_count_stats': {
            'mean': df['text'].str.split().str.len().mean(),
            'median': df['text'].str.split().str.len().median(),
            'std': df['text'].str.split().str.len().std()
        }
    }
    
    # Calculate class balance ratio
    class_counts = list(stats['class_distribution'].values())
    stats['class_balance_ratio'] = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 0
    
    return stats
