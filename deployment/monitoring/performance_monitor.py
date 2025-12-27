#!/usr/bin/env python3
"""
Performance Monitoring for FactCheck-MM Inference

Tracks runtime performance metrics including latency, throughput, and resource
usage with minimal overhead and thread-safe implementation.

Example Usage:
    >>> from deployment.monitoring import PerformanceMonitor
    >>> 
    >>> monitor = PerformanceMonitor()
    >>> 
    >>> # Track inference timing
    >>> monitor.start_timer("fact_verification")
    >>> # ... run fact verification ...
    >>> monitor.end_timer("fact_verification")
    >>> 
    >>> # Log inference with batch size
    >>> monitor.log_inference("sarcasm_detection", batch_size=8)
    >>> 
    >>> # Get performance summary
    >>> summary = monitor.get_performance_summary()
    >>> print(f"Average latency: {summary['sarcasm_detection']['avg_latency_ms']:.2f}ms")
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict, deque
import time
import threading
import json
from dataclasses import dataclass, field, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics of a single task."""
    
    task_name: str
    total_requests: int = 0
    total_inference_time: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    batch_sizes: List[int] = field(default_factory=list)
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_name": self.task_name,
            "total_requests": self.total_requests,
            "total_inference_time": self.total_inference_time,
            "min_latency": self.min_latency if self.min_latency != float('inf') else None,
            "max_latency": self.max_latency if self.max_latency > 0 else None,
            "error_count": self.error_count,
            "recent_latencies_count": len(self.latencies)
        }


class PerformanceMonitor:
    """
    Thread-safe performance monitor for tracking inference metrics.
    
    Tracks per-task metrics including latency, throughput, and batch sizes
    with minimal overhead suitable for production environments.
    """
    
    def __init__(
        self,
        logger: Optional[Any] = None,
        enable_tensorboard: bool = False,
        enable_wandb: bool = False,
        max_history: int = 1000
    ):
        """
        Initialize performance monitor.
        
        Args:
            logger: Optional logger instance
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_wandb: Whether to enable Weights & Biases logging
            max_history: Maximum number of recent metrics to keep
        """
        self.logger = logger or get_logger("PerformanceMonitor")
        
        # Thread-safe metrics storage
        self._lock = threading.Lock()
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics(task_name="unknown")
        )
        
        # Active timers (thread-local storage)
        self._active_timers = threading.local()
        
        # Configuration
        self.max_history = max_history
        self.enable_tensorboard = enable_tensorboard
        self.enable_wandb = enable_wandb
        
        # Start time for uptime tracking
        self.start_time = time.time()
        
        # Initialize optional integrations
        self._tensorboard_writer = None
        self._wandb_run = None
        
        if enable_tensorboard:
            self._init_tensorboard()
        
        if enable_wandb:
            self._init_wandb()
        
        self.logger.info(
            f"PerformanceMonitor initialized "
            f"(TensorBoard: {enable_tensorboard}, W&B: {enable_wandb})"
        )
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = Path("logs/tensorboard") / datetime.now().strftime("%Y%m%d-%H%M%S")
            self._tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging enabled: {log_dir}")
            
        except ImportError:
            self.logger.warning("TensorBoard not available (torch not installed)")
            self.enable_tensorboard = False
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            self._wandb_run = wandb.init(
                project="factcheck-mm",
                name=f"monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags=["monitoring", "performance"]
            )
            self.logger.info("W&B logging enabled")
            
        except ImportError:
            self.logger.warning("W&B not available (wandb not installed)")
            self.enable_wandb = False
        except Exception as e:
            self.logger.warning(f"W&B initialization failed: {e}")
            self.enable_wandb = False
    
    def start_timer(self, task_name: str) -> str:
        """
        Start timing for a task.
        
        Args:
            task_name: Name of the task to time
        
        Returns:
            Timer ID for this specific timing session
        """
        if not hasattr(self._active_timers, 'timers'):
            self._active_timers.timers = {}
        
        timer_id = f"{task_name}_{time.time()}"
        self._active_timers.timers[timer_id] = {
            "task_name": task_name,
            "start_time": time.time()
        }
        
        return timer_id
    
    def end_timer(
        self,
        task_name: Optional[str] = None,
        timer_id: Optional[str] = None,
        success: bool = True
    ) -> float:
        """
        End timing for a task and record metrics.
        
        Args:
            task_name: Name of the task (if not using timer_id)
            timer_id: Specific timer ID from start_timer
            success: Whether the task completed successfully
        
        Returns:
            Elapsed time in seconds
        """
        if not hasattr(self._active_timers, 'timers'):
            self.logger.warning("No active timers found")
            return 0.0
        
        # Find the timer
        if timer_id and timer_id in self._active_timers.timers:
            timer_info = self._active_timers.timers.pop(timer_id)
            task_name = timer_info["task_name"]
            start_time = timer_info["start_time"]
        elif task_name:
            # Find most recent timer for this task
            matching_timers = [
                (tid, info) for tid, info in self._active_timers.timers.items()
                if info["task_name"] == task_name
            ]
            
            if not matching_timers:
                self.logger.warning(f"No active timer found for task: {task_name}")
                return 0.0
            
            # Use most recent timer
            timer_id, timer_info = matching_timers[-1]
            self._active_timers.timers.pop(timer_id)
            start_time = timer_info["start_time"]
        else:
            self.logger.warning("Must provide either task_name or timer_id")
            return 0.0
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Record metrics
        with self._lock:
            if task_name not in self._metrics:
                self._metrics[task_name] = PerformanceMetrics(task_name=task_name)
            
            metrics = self._metrics[task_name]
            
            if success:
                metrics.total_requests += 1
                metrics.total_inference_time += elapsed_time
                metrics.latencies.append(elapsed_time)
                metrics.timestamps.append(time.time())
                metrics.min_latency = min(metrics.min_latency, elapsed_time)
                metrics.max_latency = max(metrics.max_latency, elapsed_time)
            else:
                metrics.error_count += 1
        
        # Log to external systems
        if self.enable_tensorboard and self._tensorboard_writer:
            self._tensorboard_writer.add_scalar(
                f"{task_name}/latency",
                elapsed_time * 1000,  # Convert to ms
                metrics.total_requests
            )
        
        if self.enable_wandb and self._wandb_run:
            import wandb
            wandb.log({
                f"{task_name}/latency_ms": elapsed_time * 1000,
                f"{task_name}/success": success
            })
        
        return elapsed_time
    
    def log_inference(
        self,
        task_name: str,
        batch_size: int = 1,
        latency: Optional[float] = None,
        success: bool = True
    ):
        """
        Log inference metrics without using timer.
        
        Args:
            task_name: Name of the task
            batch_size: Batch size for this inference
            latency: Optional pre-calculated latency
            success: Whether inference was successful
        """
        with self._lock:
            if task_name not in self._metrics:
                self._metrics[task_name] = PerformanceMetrics(task_name=task_name)
            
            metrics = self._metrics[task_name]
            
            if success and latency is not None:
                metrics.total_requests += 1
                metrics.total_inference_time += latency
                metrics.latencies.append(latency)
                metrics.timestamps.append(time.time())
                metrics.batch_sizes.append(batch_size)
                metrics.min_latency = min(metrics.min_latency, latency)
                metrics.max_latency = max(metrics.max_latency, latency)
            elif not success:
                metrics.error_count += 1
            
            if batch_size > 1:
                metrics.batch_sizes.append(batch_size)
    
    def get_performance_summary(
        self,
        task_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary for one or all tasks.
        
        Args:
            task_name: Optional specific task name to get summary for
        
        Returns:
            Performance summary dictionary
        """
        with self._lock:
            if task_name:
                # Summary for specific task
                if task_name not in self._metrics:
                    return {
                        "task_name": task_name,
                        "error": "No metrics available for this task"
                    }
                
                return self._compute_task_summary(self._metrics[task_name])
            else:
                # Summary for all tasks
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": time.time() - self.start_time,
                    "tasks": {}
                }
                
                for task_name, metrics in self._metrics.items():
                    summary["tasks"][task_name] = self._compute_task_summary(metrics)
                
                # Overall statistics
                total_requests = sum(
                    m.total_requests for m in self._metrics.values()
                )
                total_errors = sum(
                    m.error_count for m in self._metrics.values()
                )
                
                summary["overall"] = {
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
                    "active_tasks": len(self._metrics)
                }
                
                return summary
    
    def _compute_task_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Compute summary statistics for a task's metrics."""
        
        if metrics.total_requests == 0:
            return {
                "task_name": metrics.task_name,
                "total_requests": 0,
                "message": "No requests recorded"
            }
        
        # Calculate average latency
        avg_latency = metrics.total_inference_time / metrics.total_requests
        
        # Calculate throughput
        if len(metrics.timestamps) > 1:
            time_span = metrics.timestamps[-1] - metrics.timestamps[0]
            throughput = len(metrics.timestamps) / time_span if time_span > 0 else 0
        else:
            throughput = 0
        
        # Calculate percentiles from recent latencies
        recent_latencies = list(metrics.latencies)
        if recent_latencies:
            sorted_latencies = sorted(recent_latencies)
            p50_idx = len(sorted_latencies) // 2
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            
            p50_latency = sorted_latencies[p50_idx]
            p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        else:
            p50_latency = p95_latency = p99_latency = None
        
        # Average batch size
        avg_batch_size = (
            sum(metrics.batch_sizes) / len(metrics.batch_sizes)
            if metrics.batch_sizes else 1.0
        )
        
        summary = {
            "task_name": metrics.task_name,
            "total_requests": metrics.total_requests,
            "error_count": metrics.error_count,
            "error_rate_percent": (
                metrics.error_count / (metrics.total_requests + metrics.error_count) * 100
                if (metrics.total_requests + metrics.error_count) > 0 else 0
            ),
            "latency": {
                "avg_ms": round(avg_latency * 1000, 2),
                "min_ms": round(metrics.min_latency * 1000, 2) if metrics.min_latency != float('inf') else None,
                "max_ms": round(metrics.max_latency * 1000, 2) if metrics.max_latency > 0 else None,
                "p50_ms": round(p50_latency * 1000, 2) if p50_latency else None,
                "p95_ms": round(p95_latency * 1000, 2) if p95_latency else None,
                "p99_ms": round(p99_latency * 1000, 2) if p99_latency else None
            },
            "throughput": {
                "requests_per_second": round(throughput, 2),
                "avg_batch_size": round(avg_batch_size, 2)
            }
        }
        
        return summary
    
    def reset_metrics(self, task_name: Optional[str] = None):
        """
        Reset metrics for one or all tasks.
        
        Args:
            task_name: Optional specific task to reset, or None for all
        """
        with self._lock:
            if task_name:
                if task_name in self._metrics:
                    self._metrics[task_name] = PerformanceMetrics(task_name=task_name)
                    self.logger.info(f"Reset metrics for task: {task_name}")
            else:
                self._metrics.clear()
                self.start_time = time.time()
                self.logger.info("Reset all metrics")
    
    def export_metrics_json(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            output_path: Optional output file path
        
        Returns:
            JSON string of metrics
        """
        summary = self.get_performance_summary()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Metrics exported to {output_path}")
        
        return json.dumps(summary, indent=2)
    
    def print_summary(self, task_name: Optional[str] = None):
        """
        Print human-readable performance summary.
        
        Args:
            task_name: Optional specific task to print summary for
        """
        summary = self.get_performance_summary(task_name)
        
        print("\n" + "=" * 70)
        print("FactCheck-MM Performance Summary")
        print("=" * 70)
        
        if task_name:
            self._print_task_summary(summary)
        else:
            print(f"Uptime: {summary['uptime_seconds']:.1f}s")
            print(f"Total Requests: {summary['overall']['total_requests']}")
            print(f"Total Errors: {summary['overall']['total_errors']}")
            print(f"Error Rate: {summary['overall']['error_rate']:.2f}%")
            print(f"Active Tasks: {summary['overall']['active_tasks']}")
            print("\nPer-Task Metrics:")
            print("-" * 70)
            
            for task_name, task_summary in summary['tasks'].items():
                print(f"\n{task_name}:")
                self._print_task_summary(task_summary, indent=2)
        
        print("=" * 70 + "\n")
    
    def _print_task_summary(self, summary: Dict[str, Any], indent: int = 0):
        """Print formatted task summary."""
        prefix = " " * indent
        
        if "message" in summary:
            print(f"{prefix}{summary['message']}")
            return
        
        print(f"{prefix}Requests: {summary['total_requests']}")
        print(f"{prefix}Errors: {summary['error_count']} ({summary['error_rate_percent']:.2f}%)")
        
        latency = summary['latency']
        print(f"{prefix}Latency (ms):")
        print(f"{prefix}  Average: {latency['avg_ms']}")
        print(f"{prefix}  Min: {latency['min_ms']}")
        print(f"{prefix}  Max: {latency['max_ms']}")
        if latency['p95_ms']:
            print(f"{prefix}  P95: {latency['p95_ms']}")
            print(f"{prefix}  P99: {latency['p99_ms']}")
        
        throughput = summary['throughput']
        print(f"{prefix}Throughput: {throughput['requests_per_second']:.2f} req/s")
        print(f"{prefix}Avg Batch Size: {throughput['avg_batch_size']:.1f}")
    
    def close(self):
        """Clean up resources."""
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
        
        if self._wandb_run:
            import wandb
            wandb.finish()
        
        self.logger.info("PerformanceMonitor closed")


if __name__ == "__main__":
    # Demo usage
    print("FactCheck-MM Performance Monitor Demo\n")
    
    monitor = PerformanceMonitor()
    
    # Simulate some inference tasks
    print("Simulating inference tasks...\n")
    
    for i in range(10):
        # Sarcasm detection
        timer_id = monitor.start_timer("sarcasm_detection")
        time.sleep(0.05 + 0.01 * i)  # Simulate varying latency
        monitor.end_timer(timer_id=timer_id)
        
        # Fact verification
        timer_id = monitor.start_timer("fact_verification")
        time.sleep(0.1 + 0.02 * i)
        monitor.end_timer(timer_id=timer_id)
        
        # Paraphrasing with batch
        monitor.log_inference("paraphrasing", batch_size=4, latency=0.08)
    
    # Print summary
    monitor.print_summary()
    
    # Export to JSON
    json_output = monitor.export_metrics_json()
    print("JSON Export:")
    print(json_output)
