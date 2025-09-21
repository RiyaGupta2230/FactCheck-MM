#!/usr/bin/env python3
"""
Training Profiler for FactCheck-MM
Profiles memory, GPU usage, and performance across different devices.
"""

import sys
import os
import time
import psutil
import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.logging_utils import get_logger


@dataclass
class ProfileConfig:
    """Configuration for training profiling."""
    
    # Profiling settings
    profile_memory: bool = True
    profile_gpu: bool = True
    profile_cpu: bool = True
    profile_disk: bool = True
    
    # Sampling
    sampling_interval: float = 1.0  # seconds
    max_duration: float = 3600.0    # 1 hour max
    
    # Model testing
    test_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    test_sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    
    # Output
    save_plots: bool = True
    save_data: bool = True
    output_dir: str = "profiling_results"


class SystemProfiler:
    """System resource profiler."""
    
    def __init__(self, config: ProfileConfig):
        """Initialize system profiler."""
        self.config = config
        self.logger = get_logger("SystemProfiler")
        
        # Profiling state
        self.profiling = False
        self.profile_data = defaultdict(list)
        self.start_time = None
        
        # Check available monitoring capabilities
        self.capabilities = self._check_capabilities()
        
        self.logger.info(f"Initialized profiler with capabilities: {list(self.capabilities.keys())}")
    
    def _check_capabilities(self) -> Dict[str, bool]:
        """Check what system monitoring capabilities are available."""
        capabilities = {
            'cpu': True,
            'memory': True,
            'disk': True,
            'gpu_nvidia': False,
            'mps': False
        }
        
        # Check NVIDIA GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            capabilities['gpu_nvidia'] = True
            self.logger.info("✅ NVIDIA GPU monitoring available")
        except:
            self.logger.info("⚠️ NVIDIA GPU monitoring not available")
        
        # Check MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                capabilities['mps'] = True
                self.logger.info("✅ MPS monitoring available")
        except:
            pass
        
        return capabilities
    
    def start_profiling(self):
        """Start continuous profiling."""
        if self.profiling:
            self.logger.warning("Profiling already started")
            return
        
        self.profiling = True
        self.start_time = time.time()
        self.profile_data = defaultdict(list)
        
        # Start profiling thread
        self.profile_thread = threading.Thread(target=self._profile_loop)
        self.profile_thread.daemon = True
        self.profile_thread.start()
        
        self.logger.info("🔍 Started system profiling")
    
    def stop_profiling(self):
        """Stop profiling."""
        if not self.profiling:
            return
        
        self.profiling = False
        
        if hasattr(self, 'profile_thread'):
            self.profile_thread.join(timeout=5)
        
        self.logger.info("⏹️ Stopped system profiling")
    
    def _profile_loop(self):
        """Main profiling loop."""
        while self.profiling:
            try:
                timestamp = time.time() - self.start_time
                
                # CPU and Memory
                if self.config.profile_cpu:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.profile_data['cpu_percent'].append((timestamp, cpu_percent))
                
                if self.config.profile_memory:
                    memory = psutil.virtual_memory()
                    self.profile_data['memory_percent'].append((timestamp, memory.percent))
                    self.profile_data['memory_gb'].append((timestamp, memory.used / (1024**3)))
                
                # Disk I/O
                if self.config.profile_disk:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.profile_data['disk_read_mb'].append((timestamp, disk_io.read_bytes / (1024**2)))
                        self.profile_data['disk_write_mb'].append((timestamp, disk_io.write_bytes / (1024**2)))
                
                # GPU monitoring
                if self.config.profile_gpu:
                    self._profile_gpu(timestamp)
                
                # Check duration limit
                if timestamp > self.config.max_duration:
                    self.logger.warning(f"Profiling stopped - max duration reached ({self.config.max_duration}s)")
                    break
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Profiling error: {e}")
                break
        
        self.profiling = False
    
    def _profile_gpu(self, timestamp: float):
        """Profile GPU usage."""
        
        # NVIDIA GPU
        if self.capabilities['gpu_nvidia']:
            try:
                import pynvml
                
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_percent = (mem_info.used / mem_info.total) * 100
                    gpu_memory_gb = mem_info.used / (1024**3)
                    
                    self.profile_data[f'gpu_{i}_memory_percent'].append((timestamp, gpu_memory_percent))
                    self.profile_data[f'gpu_{i}_memory_gb'].append((timestamp, gpu_memory_gb))
                    
                    # GPU utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.profile_data[f'gpu_{i}_utilization'].append((timestamp, util.gpu))
                    except:
                        pass
            
            except Exception as e:
                self.logger.debug(f"NVIDIA GPU profiling error: {e}")
        
        # MPS (Apple Silicon)
        if self.capabilities['mps']:
            try:
                import torch
                
                if torch.backends.mps.is_available():
                    # MPS doesn't provide detailed stats like NVIDIA
                    allocated = torch.mps.current_allocated_memory() / (1024**3)
                    self.profile_data['mps_allocated_gb'].append((timestamp, allocated))
            
            except Exception as e:
                self.logger.debug(f"MPS profiling error: {e}")
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of profiling data."""
        summary = {
            'duration': time.time() - self.start_time if self.start_time else 0,
            'samples': len(self.profile_data.get('cpu_percent', [])),
            'metrics': {}
        }
        
        # Calculate statistics for each metric
        for metric, data_points in self.profile_data.items():
            if data_points:
                values = [point[1] for point in data_points]
                summary['metrics'][metric] = {
                    'mean': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'final': values[-1] if values else 0
                }
        
        return summary
    
    def save_profile_data(self, output_path: Path):
        """Save profiling data to file."""
        data_to_save = {
            'config': self.config.__dict__,
            'capabilities': self.capabilities,
            'profile_data': dict(self.profile_data),
            'summary': self.get_profile_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved profile data to: {output_path}")
    
    def create_profile_plots(self, output_dir: Path):
        """Create visualization plots from profiling data."""
        if not self.profile_data:
            self.logger.warning("No profile data to plot")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CPU and Memory plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        if 'cpu_percent' in self.profile_data:
            timestamps, values = zip(*self.profile_data['cpu_percent'])
            ax1.plot(timestamps, values, 'b-', linewidth=2)
            ax1.set_title('CPU Usage (%)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('CPU %')
            ax1.grid(True)
        
        # Memory Usage
        if 'memory_gb' in self.profile_data:
            timestamps, values = zip(*self.profile_data['memory_gb'])
            ax2.plot(timestamps, values, 'r-', linewidth=2)
            ax2.set_title('Memory Usage (GB)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Memory (GB)')
            ax2.grid(True)
        
        # GPU Memory (if available)
        gpu_plotted = False
        for key in self.profile_data:
            if 'gpu_0_memory_gb' in key or 'mps_allocated_gb' in key:
                timestamps, values = zip(*self.profile_data[key])
                ax3.plot(timestamps, values, 'g-', linewidth=2)
                ax3.set_title('GPU Memory Usage (GB)')
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('GPU Memory (GB)')
                ax3.grid(True)
                gpu_plotted = True
                break
        
        if not gpu_plotted:
            ax3.text(0.5, 0.5, 'No GPU Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('GPU Memory Usage (GB)')
        
        # GPU Utilization (if available)
        if 'gpu_0_utilization' in self.profile_data:
            timestamps, values = zip(*self.profile_data['gpu_0_utilization'])
            ax4.plot(timestamps, values, 'm-', linewidth=2)
            ax4.set_title('GPU Utilization (%)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('GPU %')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No GPU Utilization Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('GPU Utilization (%)')
        
        plt.tight_layout()
        
        plot_path = output_dir / 'system_profile.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Saved profile plot to: {plot_path}")


class ModelProfiler:
    """Profiler for FactCheck-MM models."""
    
    def __init__(self, config: ProfileConfig):
        """Initialize model profiler."""
        self.config = config
        self.logger = get_logger("ModelProfiler")
        
        # Initialize system profiler
        self.system_profiler = SystemProfiler(config)
        
        # Results storage
        self.profiling_results = {}
    
    def profile_sarcasm_model(
        self, 
        device: str = 'cpu',
        model_type: str = 'multimodal'
    ) -> Dict[str, Any]:
        """Profile sarcasm detection model performance."""
        
        self.logger.info(f"📊 Profiling {model_type} sarcasm model on {device}")
        
        results = {
            'device': device,
            'model_type': model_type,
            'batch_tests': {},
            'system_profile': None
        }
        
        try:
            import torch
            
            # Setup device
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                device = 'cpu'
            
            # Load model
            if model_type == 'multimodal':
                from sarcasm_detection.models import MultimodalSarcasmModel
                model_config = {
                    'modalities': ['text', 'audio', 'image'],
                    'fusion_strategy': 'cross_modal_attention',
                    'text_hidden_dim': 768,
                    'audio_hidden_dim': 512,
                    'image_hidden_dim': 512,
                    'fusion_output_dim': 256,
                    'num_classes': 2
                }
                model = MultimodalSarcasmModel(model_config)
            else:
                from sarcasm_detection.models import RobertaSarcasmModel
                model_config = {'num_classes': 2}
                model = RobertaSarcasmModel(model_config)
            
            model.to(device)
            model.eval()
            
            # Start system profiling
            self.system_profiler.start_profiling()
            
            # Test different batch sizes
            for batch_size in self.config.test_batch_sizes:
                batch_results = self._profile_batch_size(model, batch_size, device, model_type)
                results['batch_tests'][batch_size] = batch_results
            
            # Stop system profiling
            self.system_profiler.stop_profiling()
            results['system_profile'] = self.system_profiler.get_profile_summary()
            
        except Exception as e:
            self.logger.error(f"Model profiling failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _profile_batch_size(
        self,
        model,
        batch_size: int,
        device: str,
        model_type: str
    ) -> Dict[str, Any]:
        """Profile model with specific batch size."""
        
        import torch
        
        results = {
            'batch_size': batch_size,
            'inference_times': [],
            'memory_usage': {},
            'success': False
        }
        
        try:
            # Create sample inputs
            seq_len = 256
            
            if model_type == 'multimodal':
                inputs = {
                    'input_ids': torch.randint(0, 30522, (batch_size, seq_len)).to(device),
                    'attention_mask': torch.ones(batch_size, seq_len).to(device),
                    'audio_features': torch.randn(batch_size, 100, 512).to(device),
                    'image_features': torch.randn(batch_size, 196, 512).to(device)
                }
            else:
                inputs = {
                    'input_ids': torch.randint(0, 30522, (batch_size, seq_len)).to(device),
                    'attention_mask': torch.ones(batch_size, seq_len).to(device)
                }
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(3):
                    _ = model(**inputs)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Timed runs
            num_runs = 10
            
            for i in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(**inputs)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                results['inference_times'].append(inference_time)
            
            # Memory usage
            if device == 'cuda' and torch.cuda.is_available():
                results['memory_usage']['allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                results['memory_usage']['cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
            
            # Calculate statistics
            times = results['inference_times']
            results['avg_inference_ms'] = sum(times) / len(times)
            results['min_inference_ms'] = min(times)
            results['max_inference_ms'] = max(times)
            results['throughput_samples_per_sec'] = (batch_size * 1000) / results['avg_inference_ms']
            
            results['success'] = True
            
            self.logger.info(
                f"Batch {batch_size}: {results['avg_inference_ms']:.2f}ms avg, "
                f"{results['throughput_samples_per_sec']:.1f} samples/sec"
            )
            
        except Exception as e:
            self.logger.error(f"Batch size {batch_size} failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def compare_devices(self) -> Dict[str, Any]:
        """Compare model performance across available devices."""
        
        self.logger.info("🔄 Comparing device performance...")
        
        comparison_results = {
            'devices_tested': [],
            'results': {},
            'recommendations': []
        }
        
        # Test available devices
        devices_to_test = []
        
        import torch
        
        # Always test CPU
        devices_to_test.append('cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        # Test MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices_to_test.append('mps')
        
        comparison_results['devices_tested'] = devices_to_test
        
        # Profile each device
        for device in devices_to_test:
            self.logger.info(f"Testing device: {device}")
            
            device_results = {}
            
            # Test text model
            device_results['text_model'] = self.profile_sarcasm_model(device, 'text')
            
            # Test multimodal model (if not CPU - might be too slow)
            if device != 'cpu':
                device_results['multimodal_model'] = self.profile_sarcasm_model(device, 'multimodal')
            
            comparison_results['results'][device] = device_results
        
        # Generate recommendations
        comparison_results['recommendations'] = self._generate_device_recommendations(comparison_results)
        
        return comparison_results
    
    def _generate_device_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate device recommendations based on profiling results."""
        
        recommendations = []
        
        devices_tested = comparison_results['devices_tested']
        results = comparison_results['results']
        
        if 'cuda' in devices_tested and 'cuda' in results:
            cuda_results = results['cuda'].get('text_model', {})
            if cuda_results.get('batch_tests', {}):
                recommendations.append("✅ CUDA available - recommended for training")
                recommendations.append("🎮 Use RTX 2050 for multimodal training with batch sizes 4-8")
        
        if 'mps' in devices_tested and 'mps' in results:
            recommendations.append("🍎 MPS available - good for text-only models")
            recommendations.append("💡 MacBook M2 suitable for prototyping and text baselines")
        
        if len(devices_tested) == 1 and devices_tested[0] == 'cpu':
            recommendations.append("⚠️ Only CPU available - consider GPU for better performance")
            recommendations.append("🐌 CPU training will be slow - use small batch sizes")
        
        return recommendations
    
    def save_results(self, output_dir: Path, experiment_name: str = "model_profile"):
        """Save profiling results."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save profiling data
        if self.profiling_results:
            results_path = output_dir / f"{experiment_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.profiling_results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Saved profiling results to: {results_path}")
        
        # Save system profile data
        profile_data_path = output_dir / f"{experiment_name}_system_profile.json"
        self.system_profiler.save_profile_data(profile_data_path)
        
        # Create plots
        if self.config.save_plots:
            self.system_profiler.create_profile_plots(output_dir)


def main():
    """Main entry point for profiling."""
    parser = argparse.ArgumentParser(description="FactCheck-MM Training Profiler")
    
    # Profiling mode
    parser.add_argument('--mode', choices=['system', 'model', 'compare'], default='compare',
                       help='Profiling mode')
    
    # Model settings
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='auto',
                       help='Device to profile')
    parser.add_argument('--model-type', choices=['text', 'multimodal'], default='multimodal',
                       help='Model type to profile')
    
    # Profiling settings
    parser.add_argument('--duration', type=float, default=300,
                       help='Max profiling duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Sampling interval in seconds')
    
    # Output
    parser.add_argument('--output-dir', default='profiling_results',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', default='training_profile',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Create profiling config
    config = ProfileConfig(
        sampling_interval=args.interval,
        max_duration=args.duration,
        output_dir=args.output_dir
    )
    
    try:
        # Initialize profiler
        profiler = ModelProfiler(config)
        
        if args.mode == 'system':
            # System-only profiling
            profiler.system_profiler.start_profiling()
            
            print(f"🔍 System profiling started for {args.duration}s...")
            print("Press Ctrl+C to stop early")
            
            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                print("\n⏹️ Profiling stopped by user")
            
            profiler.system_profiler.stop_profiling()
            
        elif args.mode == 'model':
            # Model profiling
            device = args.device
            if device == 'auto':
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            
            results = profiler.profile_sarcasm_model(device, args.model_type)
            profiler.profiling_results = {f'{args.model_type}_{device}': results}
            
        elif args.mode == 'compare':
            # Device comparison
            results = profiler.compare_devices()
            profiler.profiling_results = results
            
            # Print recommendations
            print("\n🎯 Device Recommendations:")
            for rec in results.get('recommendations', []):
                print(f"   {rec}")
        
        # Save results
        output_dir = Path(args.output_dir)
        profiler.save_results(output_dir, args.experiment_name)
        
        print(f"\n✅ Profiling completed!")
        print(f"📂 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
