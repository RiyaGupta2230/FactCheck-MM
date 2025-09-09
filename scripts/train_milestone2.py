import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.milestone2.models import RoBERTaParaphraseDetector, SiameseRoBERTa, ParaphraseGenerator
from src.milestone2.dataset import MRPCDataset, SiameseParaphraseDataset
from src.milestone1.trainer import MultimodalTrainer  # Reuse existing trainer
from src.utils.helpers import setup_logging, set_seed, get_device
from src.utils.config import config

def train_paraphrase_detection():
    """Train RoBERTa paraphrase detection model"""
    set_seed(42)
    device = get_device()
    logger = setup_logging()
    
    logger.info(f"Using device: {device}")
    logger.info("Starting Milestone 2: Paraphrase Detection Training")
    
    # Load datasets
    print("Loading paraphrase detection datasets...")
    train_dataset = MRPCDataset(split='train', tokenizer_name='roberta-base')
    val_dataset = MRPCDataset(split='validation', tokenizer_name='roberta-base')
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = RoBERTaParaphraseDetector(model_name='roberta-base', num_classes=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        device=device,
        config=config.get('training')
    )
    
    # Train model
    save_path = 'checkpoints/paraphrase_detection/'
    num_epochs = config.get('training.num_epochs', 5)
    
    train_losses, val_accuracies = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    logger.info("Paraphrase detection training completed successfully!")
    logger.info(f"Final validation accuracy: {max(val_accuracies):.4f}")

def main():
    train_paraphrase_detection()

if __name__ == "__main__":
    main()
