import torch
import json
import logging
import os
from datetime import datetime
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"factcheck_mm_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_model(model, tokenizer, save_path):
    """Save model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    
    # Save tokenizer if provided
    if tokenizer:
        tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")

def load_model(model_class, model_path, device='cpu'):
    """Load model from path"""
    model = model_class.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model

def calculate_metrics(y_true, y_pred, labels=None):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'classification_report': classification_report(y_true, y_pred, target_names=labels),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(data, filepath):
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
