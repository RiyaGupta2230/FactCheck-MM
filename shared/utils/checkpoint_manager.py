"""
Checkpoint Management for FactCheck-MM
Handles saving/loading checkpoints with support for chunked training.
"""

import torch
import pickle
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import gc

from .logging_utils import get_logger


@dataclass
class ModelState:
    """Container for complete model state."""
    
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    epoch: int = 0
    step: int = 0
    best_metric: Optional[float] = None
    metrics_history: Optional[Dict[str, List[float]]] = None
    config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding tensor data."""
        result = asdict(self)
        # Remove heavy tensor data for metadata
        result.pop('model_state_dict', None)
        result.pop('optimizer_state_dict', None) 
        result.pop('scheduler_state_dict', None)
        return result


class CheckpointManager:
    """
    Advanced checkpoint manager with automatic cleanup and versioning.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = "loss",
        mode: str = "min",
        prefix: str = "checkpoint",
        use_compression: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only best checkpoints
            monitor_metric: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for metric monitoring
            prefix: Checkpoint file prefix
            use_compression: Whether to compress checkpoints
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.prefix = prefix
        self.use_compression = use_compression
        
        self.logger = get_logger("CheckpointManager")
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_history = []
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        
        # Load existing checkpoint history
        self._load_history()
        
        self.logger.info(f"Checkpoint manager initialized: {save_dir}")
    
    def _load_history(self):
        """Load checkpoint history from metadata."""
        history_file = self.save_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.checkpoint_history = history_data.get('checkpoints', [])
                    self.best_metric_value = history_data.get('best_metric_value', 
                                                           self.best_metric_value)
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint history: {e}")
    
    def _save_history(self):
        """Save checkpoint history metadata."""
        history_file = self.save_dir / "checkpoint_history.json"
        history_data = {
            'checkpoints': self.checkpoint_history,
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _is_better_metric(self, current_value: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def _generate_checkpoint_path(
        self,
        epoch: int,
        step: int,
        is_best: bool = False,
        suffix: str = ""
    ) -> Path:
        """Generate checkpoint file path."""
        
        if is_best:
            filename = f"{self.prefix}_best{suffix}.pth"
        else:
            filename = f"{self.prefix}_epoch{epoch:03d}_step{step:06d}{suffix}.pth"
        
        return self.save_dir / filename
    
    def save_checkpoint(
        self,
        model_state: Union[ModelState, Dict[str, Any]],
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: Optional[bool] = None,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model_state: Model state or ModelState object
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        # Convert to ModelState if needed
        if isinstance(model_state, dict):
            model_state = ModelState(**model_state)
        
        # Update state info
        model_state.epoch = epoch
        model_state.step = step
        
        # Check if this is the best checkpoint
        if metrics and self.monitor_metric in metrics:
            metric_value = metrics[self.monitor_metric]
            if is_best is None:
                is_best = self._is_better_metric(metric_value)
            
            if is_best:
                self.best_metric_value = metric_value
                model_state.best_metric = metric_value
        
        is_best = is_best or False
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model_state.model_state_dict,
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'config': model_state.config,
            'model_config': model_state.model_config,
            'metadata': model_state.metadata or {},
            'save_timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'is_best': is_best
        }
        
        if save_optimizer and model_state.optimizer_state_dict:
            checkpoint_data['optimizer_state_dict'] = model_state.optimizer_state_dict
        
        if save_scheduler and model_state.scheduler_state_dict:
            checkpoint_data['scheduler_state_dict'] = model_state.scheduler_state_dict
        
        if additional_data:
            checkpoint_data['additional_data'] = additional_data
        
        # Generate file path
        checkpoint_path = self._generate_checkpoint_path(epoch, step, is_best)
        
        # Save checkpoint
        try:
            if self.use_compression:
                # Use torch.save with compression
                torch.save(checkpoint_data, checkpoint_path, 
                          _use_new_zipfile_serialization=True)
            else:
                torch.save(checkpoint_data, checkpoint_path)
            
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update history
            checkpoint_info = {
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'metrics': metrics or {},
                'is_best': is_best,
                'timestamp': datetime.now().isoformat(),
                'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # Save best checkpoint separately
            if is_best:
                best_path = self._generate_checkpoint_path(epoch, step, is_best=True)
                shutil.copy2(checkpoint_path, best_path)
                self.logger.info(f"Saved best checkpoint: {best_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save history
            self._save_history()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        load_latest: bool = False,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path
            load_best: Whether to load best checkpoint
            load_latest: Whether to load latest checkpoint
            map_location: Device to map tensors to
            
        Returns:
            Loaded checkpoint data
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.get_best_checkpoint_path()
            elif load_latest:
                checkpoint_path = self.get_latest_checkpoint_path()
            else:
                raise ValueError("Must specify checkpoint_path or set load_best/load_latest")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=map_location
            )
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self._generate_checkpoint_path(0, 0, is_best=True)
        if best_path.exists():
            return best_path
        
        # Fallback to history
        best_checkpoints = [cp for cp in self.checkpoint_history if cp.get('is_best')]
        if best_checkpoints:
            return Path(best_checkpoints[-1]['path'])
        
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if self.checkpoint_history:
            return Path(self.checkpoint_history[-1]['path'])
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on max_checkpoints setting."""
        if not self.save_best_only and len(self.checkpoint_history) > self.max_checkpoints:
            # Keep best checkpoint and recent ones
            non_best_checkpoints = [cp for cp in self.checkpoint_history if not cp.get('is_best')]
            
            if len(non_best_checkpoints) > self.max_checkpoints - 1:  # -1 for best
                # Remove oldest non-best checkpoints
                to_remove = non_best_checkpoints[:-self.max_checkpoints + 1]
                
                for checkpoint_info in to_remove:
                    checkpoint_path = Path(checkpoint_info['path'])
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        self.logger.debug(f"Removed old checkpoint: {checkpoint_path}")
                    
                    self.checkpoint_history.remove(checkpoint_info)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Delete specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
            # Remove from history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if Path(cp['path']) != checkpoint_path
            ]
            
            self._save_history()
            self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
    
    def cleanup_all(self):
        """Delete all checkpoints."""
        for checkpoint_info in self.checkpoint_history:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        
        self.checkpoint_history.clear()
        self._save_history()
        self.logger.info("All checkpoints deleted")


class ChunkedCheckpointManager(CheckpointManager):
    """
    Extended checkpoint manager for chunked training on resource-constrained devices.
    """
    
    def __init__(self, *args, chunk_size: int = 1000, **kwargs):
        """
        Initialize chunked checkpoint manager.
        
        Args:
            chunk_size: Number of samples per chunk
            *args, **kwargs: Parent class arguments
        """
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.chunk_history = []
    
    def save_chunk_checkpoint(
        self,
        model_state: Union[ModelState, Dict[str, Any]],
        chunk_idx: int,
        chunk_metrics: Dict[str, float],
        samples_processed: int,
        **kwargs
    ) -> Path:
        """
        Save checkpoint for a specific chunk.
        
        Args:
            model_state: Model state
            chunk_idx: Current chunk index
            chunk_metrics: Metrics for this chunk
            samples_processed: Total samples processed so far
            **kwargs: Additional arguments for save_checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Calculate pseudo-epoch and step
        pseudo_epoch = samples_processed // (self.chunk_size * 10)  # Arbitrary scaling
        pseudo_step = samples_processed
        
        # Add chunk information to metadata
        if isinstance(model_state, ModelState):
            if model_state.metadata is None:
                model_state.metadata = {}
            model_state.metadata.update({
                'chunk_idx': chunk_idx,
                'samples_processed': samples_processed,
                'chunk_size': self.chunk_size
            })
        
        # Save with chunk suffix
        checkpoint_path = self.save_checkpoint(
            model_state=model_state,
            epoch=pseudo_epoch,
            step=pseudo_step,
            metrics=chunk_metrics,
            additional_data={'chunk_info': {
                'chunk_idx': chunk_idx,
                'samples_processed': samples_processed,
                'chunk_metrics': chunk_metrics
            }},
            **kwargs
        )
        
        # Update chunk history
        chunk_info = {
            'chunk_idx': chunk_idx,
            'checkpoint_path': str(checkpoint_path),
            'samples_processed': samples_processed,
            'metrics': chunk_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.chunk_history.append(chunk_info)
        self.current_chunk = chunk_idx
        
        return checkpoint_path
    
    def load_chunk_checkpoint(self, chunk_idx: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for specific chunk.
        
        Args:
            chunk_idx: Chunk index to load
            
        Returns:
            Loaded checkpoint data or None
        """
        # Find checkpoint for chunk
        chunk_info = None
        for info in self.chunk_history:
            if info['chunk_idx'] == chunk_idx:
                chunk_info = info
                break
        
        if chunk_info is None:
            return None
        
        return self.load_checkpoint(chunk_info['checkpoint_path'])
    
    def resume_chunked_training(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Resume chunked training from last checkpoint.
        
        Returns:
            Tuple of (next_chunk_idx, checkpoint_data)
        """
        if not self.chunk_history:
            return 0, None
        
        # Get last chunk
        last_chunk_info = self.chunk_history[-1]
        last_checkpoint = self.load_checkpoint(last_chunk_info['checkpoint_path'])
        
        next_chunk_idx = last_chunk_info['chunk_idx'] + 1
        
        self.logger.info(
            f"Resuming chunked training from chunk {next_chunk_idx} "
            f"(processed {last_chunk_info['samples_processed']} samples)"
        )
        
        return next_chunk_idx, last_checkpoint
