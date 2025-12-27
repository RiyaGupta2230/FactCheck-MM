#!/usr/bin/env python3
"""
System and Application Health Checks for FactCheck-MM

Provides comprehensive health monitoring for production deployments including
system resources, GPU status, model availability, and API responsiveness.

Example Usage:
    >>> from deployment.monitoring import HealthChecker
    >>> 
    >>> health_checker = HealthChecker()
    >>> 
    >>> # Check system health
    >>> system_health = health_checker.check_system_health()
    >>> print(f"CPU: {system_health['cpu_percent']}%")
    >>> 
    >>> # Get comprehensive health report
    >>> report = health_checker.get_health_report()
    >>> if report['status'] == 'healthy':
    ...     print("All systems operational")
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import platform

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger


class HealthChecker:
    """
    Comprehensive health checker for FactCheck-MM deployment.
    
    Monitors system resources, GPU availability, model status, and provides
    structured health reports suitable for API consumption and logging.
    """
    
    def __init__(
        self,
        logger: Optional[Any] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize health checker.
        
        Args:
            logger: Optional logger instance
            thresholds: Optional custom thresholds for health checks
                       (cpu_warning, cpu_critical, memory_warning, memory_critical, etc.)
        """
        self.logger = logger or get_logger("HealthChecker")
        
        # Default thresholds (percentage)
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 75.0,
            "memory_critical": 90.0,
            "disk_warning": 80.0,
            "disk_critical": 95.0,
            "gpu_memory_warning": 80.0,
            "gpu_memory_critical": 95.0
        }
        
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Check available monitoring capabilities
        self.capabilities = {
            "psutil": PSUTIL_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "cuda": TORCH_AVAILABLE and torch.cuda.is_available()
        }
        
        self.logger.info(f"HealthChecker initialized. Capabilities: {self.capabilities}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check system resource health (CPU, memory, disk).
        
        Returns:
            Dictionary containing system health metrics with status levels
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, system health check limited")
            return {
                "status": "unknown",
                "message": "psutil not available",
                "available": False
            }
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Determine CPU status
            if cpu_percent >= self.thresholds["cpu_critical"]:
                cpu_status = "critical"
            elif cpu_percent >= self.thresholds["cpu_warning"]:
                cpu_status = "warning"
            else:
                cpu_status = "healthy"
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            memory_used_gb = memory.used / (1024 ** 3)
            
            # Determine memory status
            if memory_percent >= self.thresholds["memory_critical"]:
                memory_status = "critical"
            elif memory_percent >= self.thresholds["memory_warning"]:
                memory_status = "warning"
            else:
                memory_status = "healthy"
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)
            disk_used_gb = disk.used / (1024 ** 3)
            
            # Determine disk status
            if disk_percent >= self.thresholds["disk_critical"]:
                disk_status = "critical"
            elif disk_percent >= self.thresholds["disk_warning"]:
                disk_status = "warning"
            else:
                disk_status = "healthy"
            
            # Overall system status
            statuses = [cpu_status, memory_status, disk_status]
            if "critical" in statuses:
                overall_status = "critical"
            elif "warning" in statuses:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            health_data = {
                "status": overall_status,
                "available": True,
                "cpu": {
                    "percent": round(cpu_percent, 2),
                    "status": cpu_status,
                    "count_physical": cpu_count,
                    "count_logical": cpu_count_logical,
                    "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None
                },
                "memory": {
                    "percent": round(memory_percent, 2),
                    "status": memory_status,
                    "total_gb": round(memory_total_gb, 2),
                    "used_gb": round(memory_used_gb, 2),
                    "available_gb": round(memory_available_gb, 2)
                },
                "disk": {
                    "percent": round(disk_percent, 2),
                    "status": disk_status,
                    "total_gb": round(disk_total_gb, 2),
                    "used_gb": round(disk_used_gb, 2),
                    "free_gb": round(disk_free_gb, 2)
                },
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                }
            }
            
            self.logger.debug(f"System health check: {overall_status}")
            return health_data
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return {
                "status": "error",
                "available": False,
                "error": str(e)
            }
    
    def check_gpu_health(self) -> Dict[str, Any]:
        """
        Check GPU availability and health (CUDA/RTX 2050).
        
        Returns:
            Dictionary containing GPU health metrics
        """
        if not TORCH_AVAILABLE:
            self.logger.debug("PyTorch not available, GPU health check skipped")
            return {
                "status": "not_available",
                "available": False,
                "message": "PyTorch not available"
            }
        
        if not torch.cuda.is_available():
            self.logger.debug("CUDA not available, running on CPU")
            return {
                "status": "not_available",
                "available": False,
                "cuda_available": False,
                "device": "cpu",
                "message": "CUDA not available (CPU mode)"
            }
        
        try:
            # GPU device information
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # GPU memory metrics
            memory_allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            memory_free = memory_total - memory_allocated
            memory_percent = (memory_allocated / memory_total) * 100
            
            # Determine GPU memory status
            if memory_percent >= self.thresholds["gpu_memory_critical"]:
                gpu_status = "critical"
            elif memory_percent >= self.thresholds["gpu_memory_warning"]:
                gpu_status = "warning"
            else:
                gpu_status = "healthy"
            
            # CUDA version and capabilities
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            
            # GPU compute capability
            device_properties = torch.cuda.get_device_properties(current_device)
            compute_capability = f"{device_properties.major}.{device_properties.minor}"
            
            gpu_data = {
                "status": gpu_status,
                "available": True,
                "cuda_available": True,
                "device": "cuda",
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "memory": {
                    "total_gb": round(memory_total, 2),
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "free_gb": round(memory_free, 2),
                    "percent": round(memory_percent, 2),
                    "status": gpu_status
                },
                "cuda_version": cuda_version,
                "cudnn_version": cudnn_version,
                "compute_capability": compute_capability,
                "multi_processor_count": device_properties.multi_processor_count
            }
            
            self.logger.debug(f"GPU health check: {gpu_status} ({device_name})")
            return gpu_data
            
        except Exception as e:
            self.logger.error(f"GPU health check failed: {e}")
            return {
                "status": "error",
                "available": False,
                "cuda_available": torch.cuda.is_available(),
                "error": str(e)
            }
    
    def check_model_health(
        self,
        model_registry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check model availability without loading them.
        
        Args:
            model_registry: Optional dictionary of model paths/metadata
                           Format: {"model_name": {"path": "...", "loaded": bool}}
        
        Returns:
            Dictionary containing model health status
        """
        if model_registry is None:
            # Default model paths
            model_registry = {
                "sarcasm_detection": {
                    "path": "checkpoints/multimodal_sarcasm/",
                    "loaded": False
                },
                "paraphrasing": {
                    "path": "checkpoints/t5_paraphraser/",
                    "loaded": False
                },
                "fact_verification": {
                    "path": "checkpoints/fact_verification/",
                    "loaded": False
                }
            }
        
        try:
            model_statuses = {}
            all_healthy = True
            
            for model_name, model_info in model_registry.items():
                model_path = Path(model_info.get("path", ""))
                
                # Check if model directory exists
                exists = model_path.exists() and model_path.is_dir()
                
                # Check for model files (without loading)
                has_config = False
                has_weights = False
                
                if exists:
                    # Look for common model files
                    config_files = list(model_path.glob("config.json")) or \
                                 list(model_path.glob("*.json"))
                    has_config = len(config_files) > 0
                    
                    weight_files = list(model_path.glob("*.pth")) or \
                                 list(model_path.glob("*.pt")) or \
                                 list(model_path.glob("pytorch_model.bin")) or \
                                 list(model_path.glob("*.safetensors"))
                    has_weights = len(weight_files) > 0
                
                # Determine model status
                if exists and has_config and has_weights:
                    status = "available"
                elif exists:
                    status = "incomplete"
                    all_healthy = False
                else:
                    status = "missing"
                    all_healthy = False
                
                model_statuses[model_name] = {
                    "status": status,
                    "path": str(model_path),
                    "exists": exists,
                    "has_config": has_config,
                    "has_weights": has_weights,
                    "loaded": model_info.get("loaded", False)
                }
            
            overall_status = "healthy" if all_healthy else "degraded"
            
            result = {
                "status": overall_status,
                "models": model_statuses,
                "total_models": len(model_statuses),
                "available_models": sum(1 for m in model_statuses.values() if m["status"] == "available"),
                "missing_models": sum(1 for m in model_statuses.values() if m["status"] == "missing")
            }
            
            self.logger.debug(f"Model health check: {overall_status}")
            return result
            
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_api_health(
        self,
        endpoint_url: str = "http://localhost:8000/health"
    ) -> Dict[str, Any]:
        """
        Check API responsiveness (optional, for deployed systems).
        
        Args:
            endpoint_url: API health endpoint URL
        
        Returns:
            Dictionary containing API health status
        """
        try:
            import requests
            
            start_time = time.time()
            response = requests.get(endpoint_url, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                status = "healthy"
            elif 200 <= response.status_code < 300:
                status = "warning"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "available": True,
                "response_code": response.status_code,
                "response_time_ms": round(response_time * 1000, 2),
                "endpoint": endpoint_url
            }
            
        except ImportError:
            self.logger.debug("requests library not available for API health check")
            return {
                "status": "unknown",
                "available": False,
                "message": "requests library not available"
            }
        except Exception as e:
            self.logger.warning(f"API health check failed: {e}")
            return {
                "status": "error",
                "available": False,
                "error": str(e),
                "endpoint": endpoint_url
            }
    
    def get_health_report(
        self,
        include_api_check: bool = False,
        api_endpoint: Optional[str] = None,
        model_registry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive health report.
        
        Args:
            include_api_check: Whether to include API health check
            api_endpoint: Optional API endpoint URL for health check
            model_registry: Optional model registry for model health check
        
        Returns:
            Complete health report dictionary with all metrics
        """
        self.logger.info("Generating comprehensive health report")
        
        start_time = time.time()
        
        # Collect all health metrics
        system_health = self.check_system_health()
        gpu_health = self.check_gpu_health()
        model_health = self.check_model_health(model_registry)
        
        # Optional API health check
        api_health = None
        if include_api_check:
            endpoint = api_endpoint or "http://localhost:8000/health"
            api_health = self.check_api_health(endpoint)
        
        # Determine overall status
        all_statuses = [
            system_health.get("status", "unknown"),
            gpu_health.get("status", "unknown") if gpu_health.get("available") else "healthy",
            model_health.get("status", "unknown")
        ]
        
        if api_health:
            all_statuses.append(api_health.get("status", "unknown"))
        
        if "critical" in all_statuses or "error" in all_statuses:
            overall_status = "critical"
        elif "warning" in all_statuses or "degraded" in all_statuses:
            overall_status = "degraded"
        elif "unhealthy" in all_statuses:
            overall_status = "unhealthy"
        else:
            overall_status = "healthy"
        
        # Compile report
        report = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
            "capabilities": self.capabilities,
            "system": system_health,
            "gpu": gpu_health,
            "models": model_health,
            "thresholds": self.thresholds
        }
        
        if api_health:
            report["api"] = api_health
        
        # Add summary
        report["summary"] = {
            "overall_status": overall_status,
            "system_healthy": system_health.get("status") in ["healthy", "warning"],
            "gpu_available": gpu_health.get("available", False),
            "models_available": model_health.get("available_models", 0),
            "models_missing": model_health.get("missing_models", 0)
        }
        
        self.logger.info(f"Health report generated: {overall_status}")
        return report
    
    def get_health_summary(self) -> str:
        """
        Get human-readable health summary string.
        
        Returns:
            Formatted health summary string
        """
        report = self.get_health_report()
        
        lines = [
            "=" * 60,
            "FactCheck-MM Health Report",
            "=" * 60,
            f"Overall Status: {report['status'].upper()}",
            f"Timestamp: {report['timestamp']}",
            "",
            "System Resources:",
        ]
        
        if report['system'].get('available'):
            sys_data = report['system']
            lines.extend([
                f"  CPU: {sys_data['cpu']['percent']}% ({sys_data['cpu']['status']})",
                f"  Memory: {sys_data['memory']['percent']}% ({sys_data['memory']['used_gb']:.1f}GB / {sys_data['memory']['total_gb']:.1f}GB)",
                f"  Disk: {sys_data['disk']['percent']}% ({sys_data['disk']['free_gb']:.1f}GB free)"
            ])
        
        lines.append("")
        lines.append("GPU Status:")
        
        if report['gpu'].get('available'):
            gpu_data = report['gpu']
            lines.extend([
                f"  Device: {gpu_data['device_name']}",
                f"  Memory: {gpu_data['memory']['percent']}% ({gpu_data['memory']['allocated_gb']:.1f}GB / {gpu_data['memory']['total_gb']:.1f}GB)",
                f"  CUDA Version: {gpu_data['cuda_version']}"
            ])
        else:
            lines.append(f"  Status: {report['gpu'].get('message', 'Not available')}")
        
        lines.append("")
        lines.append("Models:")
        
        model_data = report['models']
        for model_name, model_info in model_data.get('models', {}).items():
            lines.append(f"  {model_name}: {model_info['status'].upper()}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("FactCheck-MM Health Checker Demo\n")
    
    health_checker = HealthChecker()
    
    # Print comprehensive health summary
    print(health_checker.get_health_summary())
    
    # Get detailed report
    report = health_checker.get_health_report()
    
    # Print JSON report
    import json
    print("\nDetailed Health Report (JSON):")
    print(json.dumps(report, indent=2))
