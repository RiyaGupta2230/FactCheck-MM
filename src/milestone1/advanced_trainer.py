import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import yaml
import os
import json
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb
from datetime import datetime
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

class UltimateRTXTrainer:
    """
    Ultimate RTX-optimized trainer for multimodal sarcasm detection
    Implements all advanced training strategies for maximum 85%+ F1 accuracy
    """
    
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize device and GPU optimizations
        self.setup_device()
        
        # Initialize model
        self.setup_model()
        
        # Initialize training components
        self.setup_training_components()
        
        # Initialize logging and monitoring
        self.setup_monitoring()
        
        # Training state
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        self.logger.info("🚀 Ultimate RTX Trainer initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(self.config['paths']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'ultimate_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device with RTX 2050 optimizations"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            self.logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"🔋 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # RTX optimizations
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TensorFloat-32 for RTX cards
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory management
            if self.config['hardware'].get('cuda_memory_fraction', 1.0) < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config['hardware']['cuda_memory_fraction']
                )
        else:
            self.logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_model(self):
        """Initialize the ultimate multimodal model"""
        from src.milestone1.ultimate_multimodal_model import UltimateMultimodalSarcasmModel
        
        self.model = UltimateMultimodalSarcasmModel(self.config).to(self.device)
        
        # Model compilation for PyTorch 2.0 (RTX optimization)
        if self.config['hardware'].get('compile_model', False):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self.logger.info("✅ Model compiled for maximum performance")
            except Exception as e:
                self.logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 Total parameters: {total_params:,}")
        self.logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    
    def setup_training_components(self):
        """Initialize optimizer, scheduler, and other training components"""
        # Optimizer with advanced settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced learning rate scheduler
        scheduler_type = self.config['training'].get('scheduler', 'cosine_annealing_warm_restarts')
        if scheduler_type == 'cosine_annealing_warm_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['training']['epochs_per_chunk'],
                T_mult=2,
                eta_min=1e-8
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config['hardware']['mixed_precision'] else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
    
    def setup_monitoring(self):
        """Initialize monitoring and logging"""
        # Create directories
        Path(self.config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
        
        # Wandb setup
        if self.config['logging'].get('wandb', {}).get('enabled', False):
            try:
                wandb.init(
                    project=self.config['logging']['wandb']['project'],
                    entity=self.config['logging']['wandb'].get('entity'),
                    name=f"ultimate-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config,
                    tags=self.config['logging']['wandb'].get('tags', [])
                )
                wandb.watch(self.model, log="all", log_freq=100)
                self.logger.info("✅ Wandb monitoring initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Wandb initialization failed: {e}")
    
    def train_ultimate_chunk(self, chunk_name, train_dataloader, val_dataloader=None):
        """
        Train on a single chunk with all ultimate optimizations
        
        Args:
            chunk_name: Name identifier for the chunk
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
        
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"🎯 Starting ultimate training for chunk: {chunk_name}")
        self.logger.info(f"📊 Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            self.logger.info(f"📊 Validation samples: {len(val_dataloader.dataset)}")
        
        epochs = self.config['training']['epochs_per_chunk']
        chunk_best_f1 = 0.0
        chunk_best_metrics = {}
        
        for epoch in range(1, epochs + 1):
            self.logger.info(f"\n📅 Epoch {epoch}/{epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation phase
            if val_dataloader:
                val_metrics = self.validate_epoch(val_dataloader, epoch)
            else:
                val_metrics = train_metrics  # Use training metrics if no validation
            
            # Scheduler step
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.log_epoch_metrics(epoch, train_metrics, val_metrics, chunk_name)
            
            # Save best model
            if val_metrics['f1'] > chunk_best_f1:
                chunk_best_f1 = val_metrics['f1']
                chunk_best_metrics = val_metrics.copy()
                self.save_checkpoint(chunk_name, epoch, val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if (self.config['training'].get('early_stopping', False) and 
                self.patience_counter >= self.config['training'].get('patience', 5)):
                self.logger.info(f"🛑 Early stopping triggered after {epoch} epochs")
                break
        
        # Final chunk results
        final_results = {
            'chunk_name': chunk_name,
            'best_f1': chunk_best_f1,
            'best_accuracy': chunk_best_metrics.get('accuracy', 0.0),
            'best_precision': chunk_best_metrics.get('precision', 0.0),
            'best_recall': chunk_best_metrics.get('recall', 0.0),
            'total_epochs': epoch,
            'training_samples': len(train_dataloader.dataset),
            'validation_samples': len(val_dataloader.dataset) if val_dataloader else 0
        }
        
        self.logger.info(f"✅ Chunk {chunk_name} completed - Best F1: {chunk_best_f1:.4f}")
        return final_results
    
    def train_epoch(self, dataloader, epoch):
        """Single training epoch with advanced optimizations"""
        self.model.train()
        
        # Metrics tracking
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        auxiliary_losses = {}
        
        # Progress tracking
        progress_bar = tqdm(
            dataloader, 
            desc=f"🚂 Training Epoch {epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self.move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['hardware']['mixed_precision']):
                outputs = self.model(batch, training=True)
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Collect metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # Predictions and labels
            predictions = torch.argmax(outputs['logits'], dim=1).detach().cpu().numpy()
            labels = batch['labels'].detach().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            # Track auxiliary losses if available
            for key, value in outputs.items():
                if 'loss' in key.lower() and key != 'loss':
                    if key not in auxiliary_losses:
                        auxiliary_losses[key] = []
                    auxiliary_losses[key].append(value.item())
            
            # Update progress bar
            current_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Add auxiliary losses to metrics
        for key, values in auxiliary_losses.items():
            metrics[f'aux_{key}'] = np.mean(values)
        
        return metrics
    
    def validate_epoch(self, dataloader, epoch):
        """Validation epoch with comprehensive evaluation"""
        self.model.eval()
        
        # Metrics tracking
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        domain_predictions = {}
        language_predictions = {}
        
        # Progress tracking
        progress_bar = tqdm(
            dataloader, 
            desc=f"🔍 Validation Epoch {epoch}",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                batch = self.move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch, training=False)
                loss = outputs['loss']
                
                # Collect metrics
                epoch_loss += loss.item()
                
                # Predictions and probabilities
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(batch['labels'].detach().cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())
                
                # Domain-specific metrics (if available)
                if 'domain_ids' in batch:
                    for domain_id, pred, label in zip(
                        batch['domain_ids'].cpu().numpy(),
                        predictions.cpu().numpy(),
                        batch['labels'].cpu().numpy()
                    ):
                        if domain_id not in domain_predictions:
                            domain_predictions[domain_id] = {'preds': [], 'labels': []}
                        domain_predictions[domain_id]['preds'].append(pred)
                        domain_predictions[domain_id]['labels'].append(label)
                
                # Language-specific metrics (if available)
                if 'language_ids' in batch:
                    for lang_id, pred, label in zip(
                        batch['language_ids'].cpu().numpy(),
                        predictions.cpu().numpy(),
                        batch['labels'].cpu().numpy()
                    ):
                        if lang_id not in language_predictions:
                            language_predictions[lang_id] = {'preds': [], 'labels': []}
                        language_predictions[lang_id]['preds'].append(pred)
                        language_predictions[lang_id]['labels'].append(label)
                
                # Update progress bar
                current_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({'val_loss': f'{current_loss:.4f}'})
        
        # Calculate comprehensive metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        # Domain-specific metrics
        if domain_predictions:
            domain_metrics = {}
            for domain_id, data in domain_predictions.items():
                if len(data['labels']) > 0:
                    domain_acc = accuracy_score(data['labels'], data['preds'])
                    domain_f1 = f1_score(data['labels'], data['preds'], average='weighted')
                    domain_metrics[f'domain_{domain_id}'] = {
                        'accuracy': domain_acc,
                        'f1': domain_f1
                    }
            metrics['domain_metrics'] = domain_metrics
        
        # Language-specific metrics
        if language_predictions:
            language_metrics = {}
            for lang_id, data in language_predictions.items():
                if len(data['labels']) > 0:
                    lang_acc = accuracy_score(data['labels'], data['preds'])
                    lang_f1 = f1_score(data['labels'], data['preds'], average='weighted')
                    language_metrics[f'language_{lang_id}'] = {
                        'accuracy': lang_acc,
                        'f1': lang_f1
                    }
            metrics['language_metrics'] = language_metrics
        
        return metrics
    
    def move_batch_to_device(self, batch):
        """Move batch tensors to device with error handling"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                # Handle nested dictionaries (like text_inputs, image_inputs)
                device_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        device_batch[key][sub_key] = sub_value.to(self.device, non_blocking=True)
                    else:
                        device_batch[key][sub_key] = sub_value
            else:
                device_batch[key] = value
        
        return device_batch
    
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics, chunk_name):
        """Log comprehensive epoch metrics"""
        # Console logging
        self.logger.info(f"📊 Epoch {epoch} Results:")
        self.logger.info(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Acc: {train_metrics['accuracy']:.4f}, "
                        f"F1: {train_metrics['f1']:.4f}")
        self.logger.info(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}, "
                        f"F1: {val_metrics['f1']:.4f}")
        
        # Wandb logging
        if self.config['logging'].get('wandb', {}).get('enabled', False):
            log_dict = {
                'epoch': epoch,
                'chunk_name': chunk_name,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'learning_rate': train_metrics['learning_rate']
            }
            
            # Add domain/language metrics if available
            if 'domain_metrics' in val_metrics:
                log_dict.update({f"val_{k}": v for k, v in val_metrics['domain_metrics'].items()})
            
            if 'language_metrics' in val_metrics:
                log_dict.update({f"val_{k}": v for k, v in val_metrics['language_metrics'].items()})
            
            # Add auxiliary losses
            for key, value in train_metrics.items():
                if key.startswith('aux_'):
                    log_dict[f'train_{key}'] = value
            
            wandb.log(log_dict)
        
        # Update training history
        self.training_history.append({
            'epoch': epoch,
            'chunk_name': chunk_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_checkpoint(self, chunk_name, epoch, metrics):
        """Save model checkpoint with comprehensive metadata"""
        checkpoint_dir = Path(self.config['paths']['models_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filename = f"ultimate_{chunk_name}_epoch_{epoch}_f1_{metrics['f1']:.4f}_{timestamp}.pt"
        checkpoint_path = checkpoint_dir / checkpoint_filename
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'chunk_name': chunk_name,
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest best model
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            best_model_path = checkpoint_dir / f"ultimate_best_model_{chunk_name}.pt"
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"🏆 New best model saved: {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            if 'metrics' in checkpoint:
                self.best_f1 = checkpoint['metrics'].get('f1', 0.0)
                self.best_accuracy = checkpoint['metrics'].get('accuracy', 0.0)
            
            self.logger.info(f"✅ Checkpoint loaded successfully from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Error loading checkpoint: {e}")
            return None
    
    def get_training_summary(self):
        """Get comprehensive training summary"""
        summary = {
            'total_epochs_trained': len(self.training_history),
            'best_f1_score': self.best_f1,
            'best_accuracy': self.best_accuracy,
            'training_duration': len(self.training_history) * self.config['training'].get('epochs_per_chunk', 1),
            'model_parameters': {
                'total': sum(p.numel() for p in self.model.parameters()),
                'trainable': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'hardware_used': {
                'device': str(self.device),
                'mixed_precision': self.config['hardware']['mixed_precision'],
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
            },
            'config_snapshot': self.config
        }
        
        return summary


def create_trainer_from_config(config_path):
    """Factory function to create trainer from config file"""
    return UltimateRTXTrainer(config_path)

if __name__ == "__main__":
    # Example usage
    config_path = "config/milestone1_ultimate_config.yaml"
    trainer = UltimateRTXTrainer(config_path)
    
    print("✅ Ultimate RTX Trainer initialized successfully!")
    print("🚀 Ready for maximum accuracy multimodal sarcasm detection training!")
