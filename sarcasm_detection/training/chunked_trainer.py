# sarcasm_detection/training/chunked_trainer.py
"""
Chunked Training for Memory-Constrained Devices
Specialized trainer for MacBook Air M2 with 8GB RAM using chunked data loading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

from shared.utils import (
    get_logger, CheckpointManager, MetricsComputer,
    ExperimentLogger
)
from shared.datasets import ChunkedDataLoader, MultimodalCollator
from ..models import TextSarcasmModel, MultimodalSarcasmModel
from ..utils import SarcasmMetrics


@dataclass
class ChunkedTrainingConfig:
    """Configuration for chunked training on memory-constrained devices."""
    
    # Memory management
    max_memory_gb: float = 7.0  # MacBook Air M2 safe limit
    chunk_size: int = 1000  # Samples per chunk
    batch_size: int = 4  # Smaller batches for memory efficiency
    gradient_accumulation_steps: int = 4  # Compensate for small batch size
    
    # Model settings
    max_length: int = 256  # Shorter sequences to save memory
    use_mixed_precision: bool = True  # FP16 for memory efficiency
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_epochs: int = 15  # More epochs due to smaller effective batch size
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    adam_epsilon: float = 1e-8
    
    # Memory optimization techniques
    use_gradient_checkpointing: bool = True
    empty_cache_every_n_steps: int = 50
    force_gc_every_n_steps: int = 100
    
    # Chunked-specific settings
    shuffle_chunks: bool = True
    cache_chunks: bool = False  # Disable caching to save memory
    overlap_chunks: bool = True  # Overlap chunk processing with training
    
    # Regularization
    dropout_rate: float = 0.1
    early_stopping_patience: int = 5
    
    # Logging and checkpointing
    save_every: int = 2  # Save more frequently due to memory constraints
    eval_every: int = 1
    log_every: int = 100
    
    # Device settings
    device: str = "auto"
    pin_memory: bool = False  # Disable to save memory


class MemoryMonitor:
    """Memory usage monitoring for chunked training."""
    
    def __init__(self, max_memory_gb: float = 7.0):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_gb: Maximum allowed memory usage in GB
        """
        self.max_memory_gb = max_memory_gb
        self.logger = get_logger("MemoryMonitor")
        
        # Track memory statistics
        self.memory_history = []
        self.peak_memory = 0.0
        self.oom_warnings = 0
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        # System memory
        memory_info = psutil.virtual_memory()
        system_memory_gb = (memory_info.total - memory_info.available) / (1024**3)
        
        # GPU memory (if available)
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        
        # Process memory
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        return {
            'system_memory_gb': system_memory_gb,
            'gpu_memory_gb': gpu_memory_gb,
            'process_memory_gb': process_memory_gb,
            'total_memory_gb': system_memory_gb + gpu_memory_gb
        }
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if memory usage is safe, False if approaching limit
        """
        memory_info = self.get_current_memory_usage()
        total_memory = memory_info['total_memory_gb']
        
        # Update statistics
        self.memory_history.append(total_memory)
        if total_memory > self.peak_memory:
            self.peak_memory = total_memory
        
        # Check if approaching limit
        if total_memory > self.max_memory_gb * 0.9:  # 90% threshold
            self.oom_warnings += 1
            self.logger.warning(
                f"High memory usage: {total_memory:.2f}GB / {self.max_memory_gb}GB "
                f"(Warning #{self.oom_warnings})"
            )
            return False
        
        return True
    
    def force_memory_cleanup(self):
        """Force aggressive memory cleanup."""
        self.logger.info("Forcing memory cleanup...")
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Additional cleanup
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        
        memory_after = self.get_current_memory_usage()
        self.logger.info(
            f"Memory cleanup completed: {collected} objects collected, "
            f"current usage: {memory_after['total_memory_gb']:.2f}GB"
        )


class ChunkedTrainer:
    """Memory-efficient trainer using chunked data loading."""
    
    def __init__(
        self,
        model: Union[TextSarcasmModel, MultimodalSarcasmModel],
        config: Union[ChunkedTrainingConfig, Dict[str, Any]],
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        """
        Initialize chunked trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if isinstance(config, dict):
            config = ChunkedTrainingConfig(**config)
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger = get_logger("ChunkedTrainer")
        
        # Setup device
        if config.device == "auto":
            # Prefer CPU for very memory-constrained scenarios
            if psutil.virtual_memory().total / (1024**3) < 8:
                self.device = torch.device("cpu")
                self.logger.warning("Using CPU due to low system memory")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if config.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Initialize training components
        self._setup_optimization()
        self._setup_loss_function()
        self._setup_metrics()
        self._setup_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.current_chunk = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'memory_usage': [],
            'chunk_info': []
        }
        
        self.logger.info(
            f"Initialized chunked trainer on {self.device} "
            f"with {config.chunk_size} samples per chunk, "
            f"batch size {config.batch_size}, "
            f"max memory {config.max_memory_gb}GB"
        )
    
    def _setup_optimization(self):
        """Setup memory-efficient optimization."""
        # Use smaller learning rate due to small batch size
        adjusted_lr = self.config.learning_rate * (self.config.batch_size / 16)  # Scale from base batch size
        
        # Optimizer with memory-efficient settings
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=adjusted_lr,
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_epsilon
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=adjusted_lr,
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_epsilon
            )
        
        # Scheduler
        if self.train_dataset:
            # Calculate total steps considering chunked loading
            chunks_per_epoch = (len(self.train_dataset) + self.config.chunk_size - 1) // self.config.chunk_size
            steps_per_chunk = self.config.chunk_size // self.config.batch_size
            total_steps = chunks_per_epoch * steps_per_chunk * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            if self.config.scheduler.lower() == "linear":
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=warmup_steps
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps
                )
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_loss_function(self):
        """Setup loss function."""
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_metrics(self):
        """Setup metrics computation."""
        self.metrics_computer = SarcasmMetrics(
            task_name="chunked_sarcasm_detection",
            num_classes=2
        )
    
    def _setup_data_loaders(self):
        """Setup chunked data loaders."""
        from shared.preprocessing import TextProcessor
        
        # Use shorter max length to save memory
        text_processor = TextProcessor(max_length=self.config.max_length)
        
        collate_fn = MultimodalCollator(
            text_processor=text_processor.tokenizer,
            max_length=self.config.max_length
        )
        
        # Chunked training data loader
        if self.train_dataset:
            self.train_dataloader = ChunkedDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size,
                shuffle=self.config.shuffle_chunks,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=0,  # Single-threaded for memory efficiency
                pin_memory=self.config.pin_memory,
                max_memory_gb=self.config.max_memory_gb
            )
        
        # Regular data loaders for validation and test (smaller, so no chunking needed)
        if self.val_dataset:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False
            )
        
        if self.test_dataset:
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with chunked loading."""
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        chunks_processed = 0
        
        # Iterate through chunks
        for chunk_idx, chunk_data in self.train_dataloader:
            chunk_start_time = time.time()
            
            # Check memory before processing chunk
            if not self.memory_monitor.check_memory_usage():
                self.memory_monitor.force_memory_cleanup()
                
                # If still high memory usage, skip this chunk
                if not self.memory_monitor.check_memory_usage():
                    self.logger.warning(f"Skipping chunk {chunk_idx} due to memory constraints")
                    continue
            
            # Process chunk
            chunk_loss = self._train_chunk(chunk_data, chunk_idx)
            
            epoch_loss += chunk_loss
            chunks_processed += 1
            
            # Memory cleanup after chunk
            if chunk_idx % 2 == 0:  # Every other chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log chunk progress
            chunk_time = time.time() - chunk_start_time
            memory_info = self.memory_monitor.get_current_memory_usage()
            
            self.logger.debug(
                f"Chunk {chunk_idx} completed in {chunk_time:.2f}s - "
                f"Loss: {chunk_loss:.4f} - "
                f"Memory: {memory_info['total_memory_gb']:.2f}GB"
            )
            
            # Update chunk history
            self.training_history['chunk_info'].append({
                'epoch': self.current_epoch,
                'chunk': chunk_idx,
                'loss': chunk_loss,
                'memory_gb': memory_info['total_memory_gb'],
                'processing_time': chunk_time
            })
        
        avg_loss = epoch_loss / chunks_processed if chunks_processed > 0 else 0.0
        
        # Log memory statistics
        memory_stats = self.memory_monitor.get_current_memory_usage()
        self.training_history['memory_usage'].append(memory_stats)
        
        return {
            'train_loss': avg_loss,
            'chunks_processed': chunks_processed,
            'peak_memory_gb': self.memory_monitor.peak_memory,
            'memory_warnings': self.memory_monitor.oom_warnings
        }
    
    def _train_chunk(self, chunk_dataloader, chunk_idx: int) -> float:
        """Train on a single chunk of data."""
        chunk_loss = 0.0
        chunk_steps = 0
        
        progress_bar = tqdm(
            chunk_dataloader, 
            desc=f"Epoch {self.current_epoch}, Chunk {chunk_idx}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Extract inputs based on model type
                if isinstance(self.model, MultimodalSarcasmModel):
                    logits = self._multimodal_forward(batch)
                else:
                    # Text-only model
                    text_inputs = {k: v for k, v in batch.items() 
                                 if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                    logits = self.model(**text_inputs)
                
                # Compute loss
                loss = self.criterion(logits, batch['labels'])
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            chunk_loss += loss.item() * self.config.gradient_accumulation_steps
            chunk_steps += 1
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Periodic memory cleanup
                if self.global_step % self.config.empty_cache_every_n_steps == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if self.global_step % self.config.force_gc_every_n_steps == 0:
                    gc.collect()
            
            # Update progress bar
            if self.global_step % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_gb = self.memory_monitor.get_current_memory_usage()['total_memory_gb']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'mem': f'{memory_gb:.1f}GB'
                })
            
            # Emergency memory check
            if not self.memory_monitor.check_memory_usage():
                self.logger.warning("Emergency memory cleanup during batch processing")
                self.memory_monitor.force_memory_cleanup()
        
        return chunk_loss / chunk_steps if chunk_steps > 0 else 0.0
    
    def _multimodal_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for multimodal model."""
        text_input = batch.get('text')
        audio_input = batch.get('audio_features') 
        image_input = batch.get('image_features')
        video_input = batch.get('video_features')
        
        return self.model(
            text=text_input,
            audio=audio_input,
            image=image_input,
            video=video_input
        )
    
    def evaluate(self, dataloader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate model with memory-efficient processing."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                # Check memory before each batch
                if not self.memory_monitor.check_memory_usage():
                    self.memory_monitor.force_memory_cleanup()
                
                batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if isinstance(self.model, MultimodalSarcasmModel):
                    logits = self._multimodal_forward(batch)
                else:
                    text_inputs = {k: v for k, v in batch.items() 
                                 if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                    logits = self.model(**text_inputs)
                
                # Compute loss
                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
                
                # Periodic cleanup during evaluation
                if num_batches % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = self.metrics_computer.compute_classification_metrics(
            predictions=all_predictions,
            labels=all_labels,
            mode=split_name
        )
        
        metrics[f'{split_name}_loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        resume_from_checkpoint: Optional[str] = None,
        experiment_name: str = "chunked_sarcasm_training"
    ) -> Dict[str, Any]:
        """
        Main chunked training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            experiment_name: Name for experiment logging
            
        Returns:
            Training results
        """
        if not self.train_dataset:
            raise ValueError("Training dataset not provided")
        
        # Setup checkpointing and logging
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Use chunked checkpoint manager
            from shared.utils import ChunkedCheckpointManager
            self.checkpoint_manager = ChunkedCheckpointManager(
                save_dir=checkpoint_dir,
                max_checkpoints=3,
                monitor_metric="val_f1",
                mode="max",
                chunk_size=self.config.chunk_size
            )
        
        self.experiment_logger = ExperimentLogger(
            log_dir=checkpoint_dir or Path("logs"),
            project_name="chunked_sarcasm_detection",
            experiment_name=experiment_name,
            config=self.config.__dict__
        )
        
        self.logger.info(f"Starting chunked training for {self.config.num_epochs} epochs")
        
        # Training loop
        best_val_score = self.best_val_score
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Initial memory cleanup
            self.memory_monitor.force_memory_cleanup()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_dataset and epoch % self.config.eval_every == 0:
                # Clean memory before validation
                self.memory_monitor.force_memory_cleanup()
                val_metrics = self.evaluate(self.val_dataloader, "val")
                
                # Check for improvement
                current_val_score = val_metrics.get('val_f1', 0.0)
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    patience_counter = 0
                    
                    # Save best checkpoint
                    if hasattr(self, 'checkpoint_manager'):
                        self._save_checkpoint(val_metrics, is_best=True)
                else:
                    patience_counter += 1
            
            # Combine and log metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in epoch_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            # Log metrics
            self.experiment_logger.log_metrics(epoch_metrics, step=epoch)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            memory_info = self.memory_monitor.get_current_memory_usage()
            
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics.get('train_loss', 0):.4f} - "
                f"Val F1: {val_metrics.get('val_f1', 0):.4f} - "
                f"Memory: {memory_info['total_memory_gb']:.2f}GB - "
                f"Chunks: {train_metrics.get('chunks_processed', 0)}"
            )
            
            # Save regular checkpoint
            if hasattr(self, 'checkpoint_manager') and epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch_metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # End-of-epoch memory cleanup
            self.memory_monitor.force_memory_cleanup()
        
        # Final evaluation
        final_results = {'training_history': self.training_history}
        
        if self.val_dataset:
            self.memory_monitor.force_memory_cleanup()
            final_val_metrics = self.evaluate(self.val_dataloader, "final_val")
            final_results['final_validation'] = final_val_metrics
        
        if self.test_dataset:
            self.memory_monitor.force_memory_cleanup()
            final_test_metrics = self.evaluate(self.test_dataloader, "test")
            final_results['final_test'] = final_test_metrics
        
        # Add memory statistics to results
        final_results['memory_statistics'] = {
            'peak_memory_gb': self.memory_monitor.peak_memory,
            'total_memory_warnings': self.memory_monitor.oom_warnings,
            'average_memory_gb': np.mean([m['total_memory_gb'] for m in self.training_history['memory_usage']]),
            'memory_history': self.training_history['memory_usage']
        }
        
        self.experiment_logger.close()
        self.logger.info("Chunked training completed")
        
        return final_results
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save chunked training checkpoint."""
        if not hasattr(self, 'checkpoint_manager'):
            return
        
        from shared.utils import ModelState
        
        model_state = ModelState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            epoch=self.current_epoch,
            step=self.global_step,
            best_metric=self.best_val_score,
            config=self.config.__dict__,
            metadata={
                'model_type': 'chunked_sarcasm',
                'chunk_size': self.config.chunk_size,
                'max_memory_gb': self.config.max_memory_gb,
                'peak_memory_gb': self.memory_monitor.peak_memory
            }
        )
        
        self.checkpoint_manager.save_chunk_checkpoint(
            model_state=model_state,
            chunk_idx=self.current_chunk,
            chunk_metrics=metrics,
            samples_processed=self.global_step * self.config.batch_size,
            is_best=is_best
        )


class MemoryEfficientTrainer(ChunkedTrainer):
    """Alias for ChunkedTrainer with additional memory optimizations."""
    
    def __init__(self, *args, **kwargs):
        # Force most aggressive memory settings
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            kwargs['config'].update({
                'use_gradient_checkpointing': True,
                'empty_cache_every_n_steps': 25,
                'force_gc_every_n_steps': 50,
                'batch_size': min(kwargs['config'].get('batch_size', 4), 2),
                'gradient_accumulation_steps': max(kwargs['config'].get('gradient_accumulation_steps', 4), 8),
                'max_length': min(kwargs['config'].get('max_length', 256), 128),
                'pin_memory': False,
                'cache_chunks': False
            })
        
        super().__init__(*args, **kwargs)
        
        # Additional memory optimizations
        self._setup_memory_optimizations()
    
    def _setup_memory_optimizations(self):
        """Apply additional memory optimizations."""
        # Enable memory-efficient attention if available
        if hasattr(self.model, 'enable_memory_efficient_attention'):
            self.model.enable_memory_efficient_attention()
        
        # Reduce PyTorch memory allocation
        torch.backends.cuda.max_split_size_mb = 128
        
        self.logger.info("Applied additional memory optimizations for MemoryEfficientTrainer")
