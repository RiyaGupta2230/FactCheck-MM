#!/usr/bin/env python3
"""
Memory Profiling for FactCheck-MM Models

Comprehensive GPU/CPU memory profiling using PyTorch hooks, NVML for GPU monitoring,
and system memory tracking. Provides detailed per-batch memory consumption analysis.
"""

import sys
import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict, deque
import psutil
import gc
import tracemalloc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging

# Optional GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


@dataclass
class MemoryProfileConfig:
    """Configuration for memory profiling."""
    
    # Profiling modes
    profile_gpu: bool = True
    profile_cpu: bool = True
    profile_system_memory: bool = True
    
    # GPU profiling settings
    gpu_monitoring_interval: float = 0.1  # seconds
    track_peak_memory: bool = True
    track_memory_fragments: bool = True
    
    # CPU profiling settings
    use_tracemalloc: bool = True
    tracemalloc_limit: int = 10  # Number of frames to trace
    
    # Hook settings
    register_forward_hooks: bool = True
    register_backward_hooks: bool = True
    hook_recursion_limit: int = 100
    
    # Batch profiling
    profile_batch_loading: bool = True
    profile_preprocessing: bool = True
    profile_inference: bool = True
    profile_loss_computation: bool = True
    profile_backward_pass: bool = True
    
    # Output settings
    save_detailed_traces: bool = True
    create_memory_timeline: bool = True
    generate_summary_stats: bool = True
    
    # Resource limits
    max_profiling_time: float = 3600.0  # 1 hour max
    memory_alert_threshold_gb: float = 14.0  # Alert if memory usage exceeds


class MemoryHook:
    """PyTorch hook for tracking memory usage during forward/backward passes."""
    
    def __init__(self, profiler, module_name: str, hook_type: str):
        """
        Initialize memory hook.
        
        Args:
            profiler: Parent memory profiler
            module_name: Name of the module being hooked
            hook_type: Type of hook ('forward' or 'backward')
        """
        self.profiler = profiler
        self.module_name = module_name
        self.hook_type = hook_type
        self.call_count = 0
    
    def __call__(self, module, input_data, output_data=None):
        """Hook function called during forward/backward pass."""
        
        if self.call_count > self.profiler.config.hook_recursion_limit:
            return  # Prevent infinite recursion
        
        self.call_count += 1
        
        try:
            # Record memory usage
            memory_info = self.profiler._capture_memory_snapshot()
            
            # Add hook-specific information
            memory_info.update({
                'module_name': self.module_name,
                'hook_type': self.hook_type,
                'call_count': self.call_count,
                'timestamp': time.time()
            })
            
            self.profiler._record_memory_event(f"{self.hook_type}_hook_{self.module_name}", memory_info)
            
        except Exception as e:
            self.profiler.logger.debug(f"Memory hook error for {self.module_name}: {e}")
        
        finally:
            self.call_count -= 1


class GPUMemoryMonitor:
    """Continuous GPU memory monitoring using NVML."""
    
    def __init__(self, profiler, monitoring_interval: float = 0.1):
        """
        Initialize GPU memory monitor.
        
        Args:
            profiler: Parent memory profiler
            monitoring_interval: Monitoring interval in seconds
        """
        self.profiler = profiler
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_data = []
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                self.nvml_initialized = True
            except Exception as e:
                self.profiler.logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def start_monitoring(self):
        """Start continuous GPU monitoring."""
        if not self.nvml_initialized:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.profiler.logger.debug("Started GPU memory monitoring")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.profiler.logger.debug("Stopped GPU memory monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring:
            try:
                timestamp = time.time()
                gpu_info = []
                
                for i, handle in enumerate(self.handles):
                    # Get memory information
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get utilization
                    try:
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util_info.gpu
                        memory_util = util_info.memory
                    except:
                        gpu_util = None
                        memory_util = None
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW
                    except:
                        power = None
                    
                    gpu_data = {
                        'gpu_id': i,
                        'timestamp': timestamp,
                        'memory_used_mb': mem_info.used / (1024**2),
                        'memory_free_mb': mem_info.free / (1024**2),
                        'memory_total_mb': mem_info.total / (1024**2),
                        'memory_utilization_percent': (mem_info.used / mem_info.total) * 100,
                        'gpu_utilization_percent': gpu_util,
                        'memory_bandwidth_utilization_percent': memory_util,
                        'temperature_c': temp,
                        'power_usage_w': power / 1000 if power else None
                    }
                    
                    gpu_info.append(gpu_data)
                
                self.gpu_data.append({
                    'timestamp': timestamp,
                    'gpus': gpu_info
                })
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.profiler.logger.error(f"GPU monitoring error: {e}")
                break
    
    def get_monitoring_data(self) -> List[Dict[str, Any]]:
        """Get collected monitoring data."""
        return self.gpu_data.copy()


class MemoryProfiler:
    """Comprehensive memory profiler for FactCheck-MM models."""
    
    def __init__(
        self,
        config: MemoryProfileConfig,
        output_dir: str = "outputs/experiments/memory_profiling"
    ):
        """
        Initialize memory profiler.
        
        Args:
            config: Memory profiling configuration
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("MemoryProfiler")
        
        # Profiling state
        self.profiling_active = False
        self.start_time = None
        
        # Data storage
        self.memory_events = []
        self.memory_timeline = []
        self.batch_profiles = []
        self.module_hooks = {}
        
        # Monitoring components
        if self.config.profile_gpu:
            self.gpu_monitor = GPUMemoryMonitor(self, config.gpu_monitoring_interval)
        else:
            self.gpu_monitor = None
        
        # System info
        self.system_info = self._get_system_info()
        
        # Tracemalloc setup
        if self.config.use_tracemalloc and self.config.profile_cpu:
            tracemalloc.start(self.config.tracemalloc_limit)
            self.tracemalloc_enabled = True
        else:
            self.tracemalloc_enabled = False
        
        self.logger.info("Initialized memory profiler")
        self.logger.info(f"GPU monitoring: {self.gpu_monitor is not None}")
        self.logger.info(f"CPU tracing: {self.tracemalloc_enabled}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            
            # GPU details
            gpu_details = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_details.append({
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multiprocessor_count': props.multiprocessor_count,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
            info['gpu_details'] = gpu_details
        
        return info
    
    def start_profiling(self):
        """Start memory profiling."""
        
        if self.profiling_active:
            self.logger.warning("Profiling already active")
            return
        
        self.profiling_active = True
        self.start_time = time.time()
        
        # Clear previous data
        self.memory_events.clear()
        self.memory_timeline.clear()
        self.batch_profiles.clear()
        
        # Start GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
        
        # Initial memory snapshot
        initial_memory = self._capture_memory_snapshot()
        self._record_memory_event("profiling_start", initial_memory)
        
        self.logger.info("Started memory profiling")
    
    def stop_profiling(self):
        """Stop memory profiling."""
        
        if not self.profiling_active:
            return
        
        # Final memory snapshot
        final_memory = self._capture_memory_snapshot()
        self._record_memory_event("profiling_end", final_memory)
        
        # Stop GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
        
        self.profiling_active = False
        
        profiling_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"Stopped memory profiling after {profiling_time:.2f}s")
    
    def register_model_hooks(self, model: nn.Module, model_name: str = "model"):
        """Register memory hooks on model modules."""
        
        if not self.config.register_forward_hooks and not self.config.register_backward_hooks:
            return
        
        self.logger.debug(f"Registering hooks for model: {model_name}")
        
        hook_count = 0
        
        for name, module in model.named_modules():
            # Skip certain modules to reduce noise
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                continue
            
            full_name = f"{model_name}.{name}" if name else model_name
            
            # Register forward hook
            if self.config.register_forward_hooks:
                forward_hook = MemoryHook(self, full_name, "forward")
                handle = module.register_forward_hook(forward_hook)
                self.module_hooks[f"forward_{full_name}"] = handle
                hook_count += 1
            
            # Register backward hook
            if self.config.register_backward_hooks:
                backward_hook = MemoryHook(self, full_name, "backward")
                handle = module.register_backward_hook(backward_hook)
                self.module_hooks[f"backward_{full_name}"] = handle
                hook_count += 1
        
        self.logger.debug(f"Registered {hook_count} hooks")
    
    def unregister_hooks(self):
        """Unregister all hooks."""
        
        for hook_name, handle in self.module_hooks.items():
            handle.remove()
        
        self.module_hooks.clear()
        self.logger.debug("Unregistered all hooks")
    
    def profile_batch(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        batch_idx: int,
        compute_loss: bool = True,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Profile memory usage for a single batch.
        
        Args:
            model: Model to profile
            batch: Input batch
            batch_idx: Batch index
            compute_loss: Whether to compute loss
            loss_fn: Loss function to use
            
        Returns:
            Batch profiling results
        """
        
        if not self.profiling_active:
            self.start_profiling()
        
        batch_start_time = time.time()
        
        # Pre-batch memory
        pre_batch_memory = self._capture_memory_snapshot()
        self._record_memory_event(f"batch_{batch_idx}_start", pre_batch_memory)
        
        # Data loading phase
        if self.config.profile_batch_loading:
            data_loading_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_data_loaded", data_loading_memory)
        
        # Preprocessing phase
        if self.config.profile_preprocessing:
            # Move batch to device (simulating preprocessing)
            device = next(model.parameters()).device
            batch_on_device = self._move_batch_to_device(batch, device)
            
            preprocessing_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_preprocessed", preprocessing_memory)
        else:
            device = next(model.parameters()).device
            batch_on_device = self._move_batch_to_device(batch, device)
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        if self.config.profile_inference:
            forward_start_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_forward_start", forward_start_memory)
        
        # Run forward pass
        outputs = model(batch_on_device)
        
        if self.config.profile_inference:
            forward_end_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_forward_end", forward_end_memory)
        
        # Loss computation
        loss = None
        if compute_loss and loss_fn is not None:
            if self.config.profile_loss_computation:
                loss_start_memory = self._capture_memory_snapshot()
                self._record_memory_event(f"batch_{batch_idx}_loss_start", loss_start_memory)
            
            # Extract targets from batch
            targets = self._extract_targets(batch_on_device)
            if targets is not None:
                loss = loss_fn(outputs, targets)
            
            if self.config.profile_loss_computation:
                loss_end_memory = self._capture_memory_snapshot()
                self._record_memory_event(f"batch_{batch_idx}_loss_end", loss_end_memory)
        
        # Backward pass
        if loss is not None and self.config.profile_backward_pass:
            backward_start_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_backward_start", backward_start_memory)
            
            loss.backward()
            
            backward_end_memory = self._capture_memory_snapshot()
            self._record_memory_event(f"batch_{batch_idx}_backward_end", backward_end_memory)
        
        # Post-batch memory
        post_batch_memory = self._capture_memory_snapshot()
        self._record_memory_event(f"batch_{batch_idx}_end", post_batch_memory)
        
        # Calculate memory differences
        batch_profile = {
            'batch_idx': batch_idx,
            'batch_time': time.time() - batch_start_time,
            'pre_batch_memory': pre_batch_memory,
            'post_batch_memory': post_batch_memory,
            'memory_increase': self._calculate_memory_increase(pre_batch_memory, post_batch_memory),
            'peak_memory_during_batch': self._get_peak_memory_since_event(f"batch_{batch_idx}_start")
        }
        
        # Add batch size information
        batch_profile['batch_size'] = self._estimate_batch_size(batch)
        
        self.batch_profiles.append(batch_profile)
        
        # Memory cleanup
        self._cleanup_memory()
        
        return batch_profile
    
    def _capture_memory_snapshot(self) -> Dict[str, Any]:
        """Capture comprehensive memory snapshot."""
        
        snapshot = {
            'timestamp': time.time(),
            'system_memory': self._get_system_memory_info(),
        }
        
        # PyTorch memory info
        if torch.cuda.is_available():
            snapshot['cuda_memory'] = self._get_cuda_memory_info()
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            snapshot['mps_memory'] = self._get_mps_memory_info()
        
        # CPU memory tracing
        if self.tracemalloc_enabled:
            snapshot['cpu_trace'] = self._get_tracemalloc_info()
        
        return snapshot
    
    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'free_gb': memory.free / (1024**3),
            'percent_used': memory.percent,
            'buffers_gb': getattr(memory, 'buffers', 0) / (1024**3),
            'cached_gb': getattr(memory, 'cached', 0) / (1024**3)
        }
    
    def _get_cuda_memory_info(self) -> Dict[str, Any]:
        """Get CUDA memory information."""
        
        cuda_info = {}
        
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                memory_stats = torch.cuda.memory_stats(device_id)
                
                cuda_info[f'device_{device_id}'] = {
                    'allocated_current_mb': torch.cuda.memory_allocated(device_id) / (1024**2),
                    'reserved_current_mb': torch.cuda.memory_reserved(device_id) / (1024**2),
                    'allocated_peak_mb': torch.cuda.max_memory_allocated(device_id) / (1024**2),
                    'reserved_peak_mb': torch.cuda.max_memory_reserved(device_id) / (1024**2),
                    'num_alloc_retries': memory_stats.get('num_alloc_retries', 0),
                    'num_ooms': memory_stats.get('num_ooms', 0),
                    'active_blocks': memory_stats.get('active.all.current', 0),
                    'inactive_blocks': memory_stats.get('inactive_split.all.current', 0)
                }
        
        return cuda_info
    
    def _get_mps_memory_info(self) -> Dict[str, float]:
        """Get MPS memory information."""
        
        try:
            allocated = torch.mps.current_allocated_memory()
            driver_allocated = torch.mps.driver_allocated_memory()
            
            return {
                'allocated_mb': allocated / (1024**2),
                'driver_allocated_mb': driver_allocated / (1024**2)
            }
        except:
            return {'error': 'MPS memory info not available'}
    
    def _get_tracemalloc_info(self) -> Dict[str, Any]:
        """Get tracemalloc information."""
        
        if not tracemalloc.is_tracing():
            return {'error': 'Tracemalloc not active'}
        
        try:
            current, peak = tracemalloc.get_traced_memory()
            
            # Get top statistics
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Convert to serializable format
            top_allocations = []
            for stat in top_stats[:10]:  # Top 10 allocations
                top_allocations.append({
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / (1024**2),
                    'count': stat.count
                })
            
            return {
                'current_mb': current / (1024**2),
                'peak_mb': peak / (1024**2),
                'top_allocations': top_allocations
            }
        except Exception as e:
            return {'error': f'Tracemalloc error: {e}'}
    
    def _record_memory_event(self, event_name: str, memory_info: Dict[str, Any]):
        """Record a memory event."""
        
        event = {
            'event_name': event_name,
            'memory_info': memory_info
        }
        
        self.memory_events.append(event)
        
        # Check for memory alerts
        self._check_memory_alerts(memory_info)
    
    def _check_memory_alerts(self, memory_info: Dict[str, Any]):
        """Check for memory usage alerts."""
        
        # System memory alert
        system_memory = memory_info.get('system_memory', {})
        used_gb = system_memory.get('used_gb', 0)
        
        if used_gb > self.config.memory_alert_threshold_gb:
            self.logger.warning(f"High system memory usage: {used_gb:.2f} GB")
        
        # GPU memory alert
        cuda_memory = memory_info.get('cuda_memory', {})
        for device, device_info in cuda_memory.items():
            if isinstance(device_info, dict):
                allocated_mb = device_info.get('allocated_current_mb', 0)
                if allocated_mb > 12000:  # > 12GB
                    self.logger.warning(f"High GPU memory usage on {device}: {allocated_mb:.1f} MB")
    
    def _calculate_memory_increase(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        """Calculate memory increase between two snapshots."""
        
        increase = {}
        
        # System memory increase
        before_sys = before.get('system_memory', {})
        after_sys = after.get('system_memory', {})
        
        increase['system_memory_mb'] = (after_sys.get('used_gb', 0) - before_sys.get('used_gb', 0)) * 1024
        
        # CUDA memory increase
        before_cuda = before.get('cuda_memory', {})
        after_cuda = after.get('cuda_memory', {})
        
        for device in after_cuda:
            if device in before_cuda:
                before_allocated = before_cuda[device].get('allocated_current_mb', 0)
                after_allocated = after_cuda[device].get('allocated_current_mb', 0)
                increase[f'{device}_memory_mb'] = after_allocated - before_allocated
        
        return increase
    
    def _get_peak_memory_since_event(self, event_name: str) -> Dict[str, float]:
        """Get peak memory usage since a specific event."""
        
        # Find the event
        event_found = False
        peak_memory = {}
        
        for event in reversed(self.memory_events):
            if event['event_name'] == event_name:
                event_found = True
                break
            
            # Track peak values
            memory_info = event['memory_info']
            
            # System memory peak
            sys_used = memory_info.get('system_memory', {}).get('used_gb', 0)
            peak_memory['peak_system_memory_gb'] = max(peak_memory.get('peak_system_memory_gb', 0), sys_used)
            
            # CUDA memory peak
            cuda_memory = memory_info.get('cuda_memory', {})
            for device, device_info in cuda_memory.items():
                if isinstance(device_info, dict):
                    allocated = device_info.get('allocated_current_mb', 0)
                    peak_key = f'peak_{device}_memory_mb'
                    peak_memory[peak_key] = max(peak_memory.get(peak_key, 0), allocated)
        
        return peak_memory if event_found else {}
    
    def _move_batch_to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Move batch to specified device."""
        
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def _extract_targets(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract target labels from batch."""
        
        target_keys = ['labels', 'label', 'targets', 'target']
        
        for key in target_keys:
            if key in batch:
                return batch[key]
        
        return None
    
    def _estimate_batch_size(self, batch: Dict[str, Any]) -> int:
        """Estimate batch size from batch data."""
        
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value.shape[0]
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, torch.Tensor) and sub_value.dim() > 0:
                        return sub_value.shape[0]
        
        return 1  # Default
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        
        # Python garbage collection
        gc.collect()
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # MPS cache cleanup
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory profiling report."""
        
        self.logger.info("Generating memory profiling report")
        
        # Analyze memory events
        analysis = self._analyze_memory_events()
        
        # Analyze batch profiles
        batch_analysis = self._analyze_batch_profiles()
        
        # Get GPU monitoring data
        gpu_data = self.gpu_monitor.get_monitoring_data() if self.gpu_monitor else []
        
        # Compile report
        report = {
            'profiling_config': self.config.__dict__,
            'system_info': self.system_info,
            'profiling_duration': time.time() - self.start_time if self.start_time else 0,
            'total_memory_events': len(self.memory_events),
            'total_batches_profiled': len(self.batch_profiles),
            'memory_analysis': analysis,
            'batch_analysis': batch_analysis,
            'gpu_monitoring_data': gpu_data,
            'recommendations': self._generate_recommendations(analysis, batch_analysis)
        }
        
        # Save detailed data if requested
        if self.config.save_detailed_traces:
            report['detailed_memory_events'] = self.memory_events
            report['detailed_batch_profiles'] = self.batch_profiles
        
        return report
    
    def _analyze_memory_events(self) -> Dict[str, Any]:
        """Analyze memory events for patterns."""
        
        analysis = {
            'peak_memory_usage': {},
            'memory_growth_rate': {},
            'memory_fragmentation': {},
            'memory_leaks': []
        }
        
        if not self.memory_events:
            return analysis
        
        # Analyze peak memory usage
        peak_system = 0
        peak_gpu = {}
        
        for event in self.memory_events:
            memory_info = event['memory_info']
            
            # System memory
            sys_used = memory_info.get('system_memory', {}).get('used_gb', 0)
            peak_system = max(peak_system, sys_used)
            
            # GPU memory
            cuda_memory = memory_info.get('cuda_memory', {})
            for device, device_info in cuda_memory.items():
                if isinstance(device_info, dict):
                    allocated = device_info.get('allocated_current_mb', 0)
                    peak_gpu[device] = max(peak_gpu.get(device, 0), allocated)
        
        analysis['peak_memory_usage'] = {
            'system_memory_gb': peak_system,
            'gpu_memory_mb': peak_gpu
        }
        
        # Analyze memory growth
        if len(self.memory_events) > 1:
            first_event = self.memory_events[0]['memory_info']
            last_event = self.memory_events[-1]['memory_info']
            
            first_sys = first_event.get('system_memory', {}).get('used_gb', 0)
            last_sys = last_event.get('system_memory', {}).get('used_gb', 0)
            
            analysis['memory_growth_rate']['system_memory_gb'] = last_sys - first_sys
        
        return analysis
    
    def _analyze_batch_profiles(self) -> Dict[str, Any]:
        """Analyze batch profiling results."""
        
        analysis = {
            'average_batch_memory_increase': {},
            'memory_per_sample': {},
            'memory_efficiency': {},
            'problematic_batches': []
        }
        
        if not self.batch_profiles:
            return analysis
        
        # Calculate averages
        total_system_increase = 0
        total_gpu_increase = {}
        total_samples = 0
        
        for profile in self.batch_profiles:
            memory_increase = profile.get('memory_increase', {})
            batch_size = profile.get('batch_size', 1)
            
            # System memory increase
            sys_increase = memory_increase.get('system_memory_mb', 0)
            total_system_increase += sys_increase
            total_samples += batch_size
            
            # GPU memory increase
            for key, value in memory_increase.items():
                if 'device_' in key and key.endswith('_memory_mb'):
                    total_gpu_increase[key] = total_gpu_increase.get(key, 0) + value
        
        # Calculate averages
        num_batches = len(self.batch_profiles)
        
        analysis['average_batch_memory_increase'] = {
            'system_memory_mb': total_system_increase / num_batches,
            **{k: v / num_batches for k, v in total_gpu_increase.items()}
        }
        
        if total_samples > 0:
            analysis['memory_per_sample'] = {
                'system_memory_mb': total_system_increase / total_samples,
                **{k: v / total_samples for k, v in total_gpu_increase.items()}
            }
        
        # Identify problematic batches
        avg_system = analysis['average_batch_memory_increase']['system_memory_mb']
        
        for profile in self.batch_profiles:
            memory_increase = profile.get('memory_increase', {})
            sys_increase = memory_increase.get('system_memory_mb', 0)
            
            if sys_increase > avg_system * 2:  # More than 2x average
                analysis['problematic_batches'].append({
                    'batch_idx': profile['batch_idx'],
                    'memory_increase_mb': sys_increase,
                    'batch_size': profile.get('batch_size', 1)
                })
        
        return analysis
    
    def _generate_recommendations(self, memory_analysis: Dict[str, Any], batch_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Check peak memory usage
        peak_system = memory_analysis.get('peak_memory_usage', {}).get('system_memory_gb', 0)
        system_total = self.system_info.get('memory_total_gb', 16)
        
        if peak_system > system_total * 0.8:
            recommendations.append("High system memory usage detected. Consider reducing batch size or using gradient checkpointing.")
        
        # Check GPU memory
        peak_gpu = memory_analysis.get('peak_memory_usage', {}).get('gpu_memory_mb', {})
        for device, peak_mb in peak_gpu.items():
            if peak_mb > 10000:  # > 10GB
                recommendations.append(f"High GPU memory usage on {device}. Consider mixed precision training or smaller models.")
        
        # Check memory per sample
        memory_per_sample = batch_analysis.get('memory_per_sample', {})
        system_per_sample = memory_per_sample.get('system_memory_mb', 0)
        
        if system_per_sample > 100:  # > 100MB per sample
            recommendations.append("High memory usage per sample. Check for memory leaks or inefficient data processing.")
        
        # Check for problematic batches
        problematic = batch_analysis.get('problematic_batches', [])
        if len(problematic) > len(self.batch_profiles) * 0.1:  # > 10% of batches
            recommendations.append("Many batches have unusually high memory usage. Consider data preprocessing optimization.")
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal for the current configuration.")
        
        return recommendations
    
    def save_report(self, filename: str = "memory_profile_report.json"):
        """Save memory profiling report to file."""
        
        report = self.generate_report()
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create text summary
        summary_path = self.output_dir / filename.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self._create_text_summary(report))
        
        self.logger.info(f"Memory profiling report saved to: {report_path}")
        
        return report_path
    
    def _create_text_summary(self, report: Dict[str, Any]) -> str:
        """Create text summary of memory profiling report."""
        
        lines = []
        lines.append("=" * 60)
        lines.append("FACTCHECK-MM MEMORY PROFILING REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # System info
        lines.append("SYSTEM INFORMATION:")
        lines.append("-" * 30)
        sys_info = report['system_info']
        lines.append(f"Total system memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
        lines.append(f"CPU cores: {sys_info.get('cpu_count', 0)}")
        
        if 'gpu_details' in sys_info:
            for i, gpu in enumerate(sys_info['gpu_details']):
                lines.append(f"GPU {i}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
        
        lines.append("")
        
        # Profiling summary
        lines.append("PROFILING SUMMARY:")
        lines.append("-" * 30)
        lines.append(f"Profiling duration: {report['profiling_duration']:.2f} seconds")
        lines.append(f"Total memory events: {report['total_memory_events']}")
        lines.append(f"Batches profiled: {report['total_batches_profiled']}")
        lines.append("")
        
        # Peak memory usage
        memory_analysis = report.get('memory_analysis', {})
        peak_usage = memory_analysis.get('peak_memory_usage', {})
        
        lines.append("PEAK MEMORY USAGE:")
        lines.append("-" * 30)
        lines.append(f"System memory: {peak_usage.get('system_memory_gb', 0):.2f} GB")
        
        gpu_memory = peak_usage.get('gpu_memory_mb', {})
        for device, memory_mb in gpu_memory.items():
            lines.append(f"{device}: {memory_mb:.1f} MB")
        
        lines.append("")
        
        # Batch analysis
        batch_analysis = report.get('batch_analysis', {})
        memory_per_sample = batch_analysis.get('memory_per_sample', {})
        
        lines.append("MEMORY EFFICIENCY:")
        lines.append("-" * 30)
        system_per_sample = memory_per_sample.get('system_memory_mb', 0)
        lines.append(f"System memory per sample: {system_per_sample:.2f} MB")
        
        for device_key, memory_mb in memory_per_sample.items():
            if 'device_' in device_key:
                lines.append(f"{device_key} per sample: {memory_mb:.2f} MB")
        
        lines.append("")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 30)
        for i, rec in enumerate(recommendations):
            lines.append(f"{i+1}. {rec}")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_profiling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_profiling()
        self.unregister_hooks()
        
        if self.tracemalloc_enabled:
            tracemalloc.stop()


def main():
    """Example usage of memory profiler."""
    
    # Configuration
    config = MemoryProfileConfig(
        profile_gpu=True,
        profile_cpu=True,
        register_forward_hooks=True,
        register_backward_hooks=False,  # Reduce noise
        gpu_monitoring_interval=0.5,
        create_memory_timeline=True
    )
    
    # Create mock model
    from tests.fixtures.mock_models import MockSarcasmModel
    
    model = MockSarcasmModel()
    model.train()
    
    # Create profiler
    profiler = MemoryProfiler(config)
    
    # Example profiling session
    with profiler:
        # Register hooks
        profiler.register_model_hooks(model, "sarcasm_model")
        
        # Simulate training batches
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for batch_idx in range(5):
            # Create mock batch
            batch = {
                'input_ids': torch.randint(0, 1000, (8, 128)),
                'attention_mask': torch.ones(8, 128),
                'labels': torch.randint(0, 2, (8,))
            }
            
            # Profile batch
            batch_profile = profiler.profile_batch(
                model, batch, batch_idx, 
                compute_loss=True, loss_fn=loss_fn
            )
            
            print(f"Batch {batch_idx} memory increase: {batch_profile['memory_increase']}")
    
    # Generate and save report
    report_path = profiler.save_report()
    print(f"Memory profiling report saved to: {report_path}")


if __name__ == "__main__":
    main()
