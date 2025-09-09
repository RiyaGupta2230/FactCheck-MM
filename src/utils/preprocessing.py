import torch
import numpy as np
import pandas as pd
import re
import nltk
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import librosa
import cv2
from PIL import Image

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip and lowercase
        text = text.strip().lower()
        
        return text
    
    def tokenize(self, text, return_tensors='pt'):
        """Tokenize text using BERT tokenizer"""
        cleaned_text = self.clean_text(text)
        
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        
        return encoding
    
    def tokenize_pair(self, text1, text2, return_tensors='pt'):
        """Tokenize pair of texts"""
        cleaned_text1 = self.clean_text(text1)
        cleaned_text2 = self.clean_text(text2)
        
        encoding = self.tokenizer(
            cleaned_text1,
            cleaned_text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        
        return encoding

class MultimodalPreprocessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        
    def process_visual_features(self, visual_path):
        """Process visual features from file"""
        if isinstance(visual_path, str) and visual_path.endswith('.npy'):
            features = np.load(visual_path)
        else:
            # If it's an image path, extract features (placeholder)
            features = np.random.randn(2048)  # Replace with actual feature extraction
        
        return torch.tensor(features, dtype=torch.float32)
    
    def process_audio_features(self, audio_path, sr=22050, n_mfcc=13):
        """Process audio features from file"""
        if isinstance(audio_path, str):
            if audio_path.endswith('.npy'):
                features = np.load(audio_path)
            else:
                # Load and process audio file
                try:
                    y, sr = librosa.load(audio_path, sr=sr)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    features = mfcc.T  # Shape: (time_steps, n_mfcc)
                except:
                    # Fallback to random features
                    features = np.random.randn(100, n_mfcc)
        else:
            features = np.random.randn(100, 13)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def process_text(self, text):
        """Process text using text preprocessor"""
        return self.text_preprocessor.tokenize(text)
