#!/usr/bin/env python3
"""
Speed Benchmarking for FactCheck-MM Models

Comprehensive inference speed benchmarking for text-only, multimodal, 
and ensemble models across different batch sizes and sequence lengths.
"""

import sys
import os
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging


@dataclass
class BenchmarkConfig:
    """Configuration for speed benchmarking."""
    
    # Test configurations
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 50
    
    # Model types to benchmark
    benchmark_text_only: bool = True
    benchmark_multimodal: bool = True
    benchmark_ensemble: bool = False
    
    # Hardware configurations
    test_cpu: bool = True
    test_gpu: bool = True
    test_mixed_precision: bool = True
    
    # Data configurations
    use_synthetic_data: bool = True
    use_real_data: bool = False
    real_data_samples: int = 1000
    
    # Output configuration
    save_detailed_results: bool = True
    create_visualizations: bool = True
    
    # Resource monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_gpu: bool = True


class SpeedBenchmarker:
    """Comprehensive speed benchmarker for FactCheck-MM models."""
    
    def __init__(
        self,
        config: BenchmarkConfig,
        models: Dict[str, nn.Module],
        data_loaders: Optional[Dict[str, DataLoader]] = None,
        output_dir: str = "outputs/experiments/benchmarking"
    ):
        """
        Initialize speed benchmarker.
        
        Args:
            config: Benchmark configuration
            models: Dictionary of models to benchmark
            data_loaders: Optional real data loaders
            output_dir: Output directory for results
        """
        self.config = config
        self.models = models
        self.data_loaders = data_loaders or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("SpeedBenchmarker")
        
        # Device setup
        self.available_devices = self._detect_available_devices()
        
        # Results storage
        self.benchmark_results = {}
        
        # System info
        self.system_info = self._get_system_info()
        
        self.logger.info(f"Initialized speed benchmarker")
        self.logger.info(f"Available devices: {self.available_devices}")
        self.logger.info(f"Models to benchmark: {list(models.keys())}")
    
    def _detect_available_devices(self) -> List[str]:
        """Detect available devices for benchmarking."""
        
        devices = []
        
        # Always have CPU
        if self.config.test_cpu:
            devices.append('cpu')
        
        # Check for CUDA
        if self.config.test_gpu and torch.cuda.is_available():
            devices.append('cuda')
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Check for MPS (Apple Silicon)
        if self.config.test_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
            self.logger.info("MPS (Apple Silicon) available")
        
        return devices
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'platform': sys.platform
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive speed benchmarks."""
        
        self.logger.info("Starting speed benchmarking")
        benchmark_start_time = time.time()
        
        # Benchmark each model on each device
        for model_name, model in self.models.items():
            self.logger.info(f"Benchmarking model: {model_name}")
            
            model_results = {}
            
            for device in self.available_devices:
                self.logger.info(f"  Testing on device: {device}")
                
                device_results = self._benchmark_model_on_device(model, model_name, device)
                model_results[device] = device_results
            
            self.benchmark_results[model_name] = model_results
        
        benchmark_time = time.time() - benchmark_start_time
        
        # Compile final results
        final_results = {
            'benchmark_time': benchmark_time,
            'system_info': self.system_info,
            'config': self.config.__dict__,
            'results': self.benchmark_results,
            'summary': self._create_benchmark_summary()
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info(f"Benchmarking completed in {benchmark_time:.2f}s")
        
        return final_results
    
    def _benchmark_model_on_device(self, model: nn.Module, model_name: str, device: str) -> Dict[str, Any]:
        """Benchmark a specific model on a specific device."""
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        device_results = {}
        
        # Test different configurations
        for batch_size in self.config.batch_sizes:
            batch_results = {}
            
            for seq_length in self.config.sequence_lengths:
                self.logger.debug(f"    Batch size: {batch_size}, Sequence length: {seq_length}")
                
                try:
                    # Test with and without mixed precision
                    config_results = {}
                    
                    # Standard precision
                    standard_results = self._benchmark_configuration(
                        model, model_name, device, batch_size, seq_length, use_mixed_precision=False
                    )
                    config_results['standard'] = standard_results
                    
                    # Mixed precision (if supported)
                    if self.config.test_mixed_precision and device == 'cuda':
                        mixed_results = self._benchmark_configuration(
                            model, model_name, device, batch_size, seq_length, use_mixed_precision=True
                        )
                        config_results['mixed_precision'] = mixed_results
                    
                    batch_results[f'seq_{seq_length}'] = config_results
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.warning(f"OOM for batch_size={batch_size}, seq_length={seq_length} on {device}")
                        batch_results[f'seq_{seq_length}'] = {'error': 'out_of_memory'}
                    else:
                        self.logger.error(f"Error benchmarking: {e}")
                        batch_results[f'seq_{seq_length}'] = {'error': str(e)}
                
                # Clear cache
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            device_results[f'batch_{batch_size}'] = batch_results
        
        return device_results
    
    def _benchmark_configuration(
        self, 
        model: nn.Module, 
        model_name: str, 
        device: str, 
        batch_size: int, 
        seq_length: int,
        use_mixed_precision: bool = False
    ) -> Dict[str, Any]:
        """Benchmark a specific configuration."""
        
        # Create synthetic input data
        input_data = self._create_synthetic_input(model_name, batch_size, seq_length, device)
        
        # Warmup runs
        for _ in range(self.config.num_warmup_runs):
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        _ = model(input_data)
                else:
                    _ = model(input_data)
        
        # Synchronize if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark runs
        inference_times = []
        memory_usage = []
        cpu_usage = []
        
        for run_idx in range(self.config.num_benchmark_runs):
            # Monitor system resources before
            if self.config.monitor_cpu:
                cpu_before = psutil.cpu_percent()
            
            if self.config.monitor_memory and device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated()
            
            # Time the inference
            start_time = time.perf_counter()
            
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = model(input_data)
                else:
                    output = model(input_data)
            
            # Synchronize if using CUDA
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            
            # Monitor system resources after
            if self.config.monitor_memory and device == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_memory / (1024**2))  # Convert to MB
            
            if self.config.monitor_cpu:
                cpu_after = psutil.cpu_percent()
                cpu_usage.append(cpu_after)
        
        # Calculate statistics
        results = {
            'mean_inference_time_ms': statistics.mean(inference_times),
            'std_inference_time_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'min_inference_time_ms': min(inference_times),
            'max_inference_time_ms': max(inference_times),
            'median_inference_time_ms': statistics.median(inference_times),
            'throughput_samples_per_sec': batch_size * 1000 / statistics.mean(inference_times),
            'throughput_tokens_per_sec': batch_size * seq_length * 1000 / statistics.mean(inference_times),
            'all_inference_times': inference_times,
            'batch_size': batch_size,
            'sequence_length': seq_length,
            'use_mixed_precision': use_mixed_precision,
            'num_runs': self.config.num_benchmark_runs
        }
        
        # Add memory statistics if available
        if memory_usage:
            results.update({
                'mean_memory_usage_mb': statistics.mean(memory_usage),
                'peak_memory_usage_mb': max(memory_usage),
                'memory_per_sample_mb': statistics.mean(memory_usage) / batch_size
            })
        
        # Add CPU statistics if available
        if cpu_usage:
            results.update({
                'mean_cpu_usage_percent': statistics.mean(cpu_usage),
                'peak_cpu_usage_percent': max(cpu_usage)
            })
        
        return results
    
    def _create_synthetic_input(self, model_name: str, batch_size: int, seq_length: int, device: str) -> Dict[str, Any]:
        """Create synthetic input data for benchmarking."""
        
        input_data = {}
        
        # Common text inputs
        input_data['input_ids'] = torch.randint(1, 30522, (batch_size, seq_length), device=device)
        input_data['attention_mask'] = torch.ones(batch_size, seq_length, device=device)
        
        # Add multimodal inputs if needed
        if 'multimodal' in model_name.lower():
            # Audio features
            input_data['audio'] = torch.randn(batch_size, 100, 128, device=device)
            
            # Image features
            input_data['image'] = torch.randn(batch_size, 3, 224, 224, device=device)
            
            # Video features (if applicable)
            if 'video' in model_name.lower():
                input_data['video'] = torch.randn(batch_size, 16, 3, 224, 224, device=device)
        
        # Task-specific inputs
        if 'paraphrase' in model_name.lower():
            input_data['decoder_input_ids'] = torch.randint(1, 30522, (batch_size, seq_length), device=device)
        
        if 'fact' in model_name.lower():
            input_data['evidence_ids'] = torch.randint(1, 30522, (batch_size, seq_length), device=device)
            input_data['evidence_attention_mask'] = torch.ones(batch_size, seq_length, device=device)
        
        return input_data
    
    def _create_benchmark_summary(self) -> Dict[str, Any]:
        """Create a summary of benchmark results."""
        
        summary = {
            'models_tested': list(self.benchmark_results.keys()),
            'devices_tested': self.available_devices,
            'configurations_tested': len(self.config.batch_sizes) * len(self.config.sequence_lengths),
            'fastest_configurations': {},
            'throughput_comparison': {},
            'memory_efficiency': {}
        }
        
        # Find fastest configurations
        for model_name, model_results in self.benchmark_results.items():
            fastest_config = None
            fastest_time = float('inf')
            
            for device, device_results in model_results.items():
                for batch_config, batch_results in device_results.items():
                    for seq_config, seq_results in batch_results.items():
                        if isinstance(seq_results, dict) and 'standard' in seq_results:
                            standard_results = seq_results['standard']
                            if 'mean_inference_time_ms' in standard_results:
                                inference_time = standard_results['mean_inference_time_ms']
                                if inference_time < fastest_time:
                                    fastest_time = inference_time
                                    fastest_config = {
                                        'device': device,
                                        'batch_config': batch_config,
                                        'seq_config': seq_config,
                                        'inference_time_ms': inference_time,
                                        'throughput_samples_per_sec': standard_results.get('throughput_samples_per_sec', 0)
                                    }
            
            if fastest_config:
                summary['fastest_configurations'][model_name] = fastest_config
        
        # Throughput comparison
        for model_name, model_results in self.benchmark_results.items():
            throughputs = []
            
            for device, device_results in model_results.items():
                for batch_config, batch_results in device_results.items():
                    for seq_config, seq_results in batch_results.items():
                        if isinstance(seq_results, dict) and 'standard' in seq_results:
                            standard_results = seq_results['standard']
                            if 'throughput_samples_per_sec' in standard_results:
                                throughputs.append(standard_results['throughput_samples_per_sec'])
            
            if throughputs:
                summary['throughput_comparison'][model_name] = {
                    'max_throughput': max(throughputs),
                    'mean_throughput': statistics.mean(throughputs),
                    'min_throughput': min(throughputs)
                }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        
        # Save detailed results
        if self.config.save_detailed_results:
            detailed_file = self.output_dir / "speed_benchmark_detailed.json"
            with open(detailed_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Save summary results
        summary_file = self.output_dir / "speed_benchmark_summary.json"
        summary_data = {
            'system_info': results['system_info'],
            'summary': results['summary'],
            'benchmark_time': results['benchmark_time']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Create text report
        report = self._create_text_report(results)
        report_file = self.output_dir / "speed_benchmark_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(results)
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _create_text_report(self, results: Dict[str, Any]) -> str:
        """Create human-readable text report."""
        
        lines = []
        lines.append("=" * 60)
        lines.append("FACTCHECK-MM SPEED BENCHMARK REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # System info
        lines.append("SYSTEM INFORMATION:")
        lines.append("-" * 30)
        sys_info = results['system_info']
        lines.append(f"CPU: {sys_info.get('cpu_count', 'Unknown')} cores")
        lines.append(f"Memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
        lines.append(f"Python: {sys_info.get('python_version', 'Unknown')}")
        lines.append(f"PyTorch: {sys_info.get('torch_version', 'Unknown')}")
        
        if 'gpu_name' in sys_info:
            lines.append(f"GPU: {sys_info['gpu_name']} ({sys_info.get('gpu_memory_gb', 0):.1f} GB)")
        
        lines.append("")
        
        # Summary
        summary = results['summary']
        lines.append("BENCHMARK SUMMARY:")
        lines.append("-" * 30)
        lines.append(f"Models tested: {len(summary['models_tested'])}")
        lines.append(f"Devices tested: {', '.join(summary['devices_tested'])}")
        lines.append(f"Configurations tested: {summary['configurations_tested']}")
        lines.append(f"Total benchmark time: {results['benchmark_time']:.2f}s")
        lines.append("")
        
        # Fastest configurations
        if 'fastest_configurations' in summary:
            lines.append("FASTEST CONFIGURATIONS:")
            lines.append("-" * 30)
            for model, config in summary['fastest_configurations'].items():
                lines.append(f"{model}:")
                lines.append(f"  Device: {config['device']}")
                lines.append(f"  Configuration: {config['batch_config']}, {config['seq_config']}")
                lines.append(f"  Inference time: {config['inference_time_ms']:.2f}ms")
                lines.append(f"  Throughput: {config['throughput_samples_per_sec']:.1f} samples/sec")
                lines.append("")
        
        # Throughput comparison
        if 'throughput_comparison' in summary:
            lines.append("THROUGHPUT COMPARISON:")
            lines.append("-" * 30)
            for model, throughput in summary['throughput_comparison'].items():
                lines.append(f"{model}:")
                lines.append(f"  Max throughput: {throughput['max_throughput']:.1f} samples/sec")
                lines.append(f"  Mean throughput: {throughput['mean_throughput']:.1f} samples/sec")
                lines.append("")
        
        return "\n".join(lines)
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Create visualization plots."""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract data for plotting
            models = list(results['results'].keys())
            devices = self.available_devices
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Inference time by batch size
            for model_idx, (model_name, model_results) in enumerate(results['results'].items()):
                for device in devices:
                    if device in model_results:
                        batch_sizes = []
                        inference_times = []
                        
                        device_results = model_results[device]
                        for batch_config, batch_results in device_results.items():
                            if 'seq_256' in batch_results and 'standard' in batch_results['seq_256']:
                                batch_size = int(batch_config.replace('batch_', ''))
                                time_ms = batch_results['seq_256']['standard'].get('mean_inference_time_ms', 0)
                                
                                batch_sizes.append(batch_size)
                                inference_times.append(time_ms)
                        
                        if batch_sizes:
                            label = f"{model_name} ({device})"
                            axes[0, 0].plot(batch_sizes, inference_times, marker='o', label=label)
            
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Inference Time (ms)')
            axes[0, 0].set_title('Inference Time vs Batch Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: Throughput by batch size
            for model_idx, (model_name, model_results) in enumerate(results['results'].items()):
                for device in devices:
                    if device in model_results:
                        batch_sizes = []
                        throughputs = []
                        
                        device_results = model_results[device]
                        for batch_config, batch_results in device_results.items():
                            if 'seq_256' in batch_results and 'standard' in batch_results['seq_256']:
                                batch_size = int(batch_config.replace('batch_', ''))
                                throughput = batch_results['seq_256']['standard'].get('throughput_samples_per_sec', 0)
                                
                                batch_sizes.append(batch_size)
                                throughputs.append(throughput)
                        
                        if batch_sizes:
                            label = f"{model_name} ({device})"
                            axes[0, 1].plot(batch_sizes, throughputs, marker='s', label=label)
            
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Throughput (samples/sec)')
            axes[0, 1].set_title('Throughput vs Batch Size')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot 3: Memory usage by batch size (if available)
            memory_data_available = False
            for model_idx, (model_name, model_results) in enumerate(results['results'].items()):
                if 'cuda' in model_results:
                    batch_sizes = []
                    memory_usage = []
                    
                    device_results = model_results['cuda']
                    for batch_config, batch_results in device_results.items():
                        if 'seq_256' in batch_results and 'standard' in batch_results['seq_256']:
                            standard_results = batch_results['seq_256']['standard']
                            if 'mean_memory_usage_mb' in standard_results:
                                batch_size = int(batch_config.replace('batch_', ''))
                                memory_mb = standard_results['mean_memory_usage_mb']
                                
                                batch_sizes.append(batch_size)
                                memory_usage.append(memory_mb)
                                memory_data_available = True
                    
                    if batch_sizes:
                        axes[1, 0].plot(batch_sizes, memory_usage, marker='^', label=model_name)
            
            if memory_data_available:
                axes[1, 0].set_xlabel('Batch Size')
                axes[1, 0].set_ylabel('Memory Usage (MB)')
                axes[1, 0].set_title('Memory Usage vs Batch Size')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'Memory data not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Plot 4: Device comparison
            device_throughputs = {device: [] for device in devices}
            
            for model_name, model_results in results['results'].items():
                for device in devices:
                    if device in model_results:
                        device_results = model_results[device]
                        
                        # Get best throughput for this device
                        best_throughput = 0
                        for batch_config, batch_results in device_results.items():
                            for seq_config, seq_results in batch_results.items():
                                if isinstance(seq_results, dict) and 'standard' in seq_results:
                                    throughput = seq_results['standard'].get('throughput_samples_per_sec', 0)
                                    best_throughput = max(best_throughput, throughput)
                        
                        device_throughputs[device].append(best_throughput)
            
            # Create bar plot
            x_pos = np.arange(len(devices))
            mean_throughputs = [np.mean(device_throughputs[device]) if device_throughputs[device] else 0 
                               for device in devices]
            
            axes[1, 1].bar(x_pos, mean_throughputs, color=['skyblue', 'lightcoral', 'lightgreen'][:len(devices)])
            axes[1, 1].set_xlabel('Device')
            axes[1, 1].set_ylabel('Mean Best Throughput (samples/sec)')
            axes[1, 1].set_title('Device Performance Comparison')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(devices)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / "speed_benchmark_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualizations saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")


def main():
    """Example usage of speed benchmarker."""
    
    # Create mock models for testing
    from tests.fixtures.mock_models import MockSarcasmModel, MockMultimodalSarcasmModel
    
    models = {
        'text_sarcasm': MockSarcasmModel(),
        'multimodal_sarcasm': MockMultimodalSarcasmModel()
    }
    
    # Configuration (smaller for example)
    config = BenchmarkConfig(
        batch_sizes=[1, 4, 8],
        sequence_lengths=[128, 256],
        num_warmup_runs=3,
        num_benchmark_runs=10,
        benchmark_text_only=True,
        benchmark_multimodal=True
    )
    
    # Run benchmarks
    benchmarker = SpeedBenchmarker(config, models)
    results = benchmarker.run_benchmarks()
    
    print("Speed benchmarking completed!")
    print(f"Models tested: {results['summary']['models_tested']}")
    print(f"Devices tested: {results['summary']['devices_tested']}")


if __name__ == "__main__":
    main()
