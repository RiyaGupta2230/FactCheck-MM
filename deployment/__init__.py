"""
FactCheck-MM Deployment Module

Production deployment infrastructure for the FactCheck-MM multimodal fact-checking system.
Provides REST API serving, health monitoring, and model export utilities for production environments.

Submodules:
    api: FastAPI-based REST API for serving inference requests
        - Sarcasm detection endpoints
        - Paraphrasing generation endpoints
        - Fact verification endpoints
        - Health checks and status monitoring
    
    monitoring: Runtime health checks and performance monitoring
        - System resource monitoring (CPU, RAM, disk, GPU)
        - Model availability checks
        - Performance metrics tracking (latency, throughput)
        - Real-time monitoring dashboards
    
    scripts: Production utility scripts
        - Environment setup automation
        - Model export and optimization
        - Deployment preparation tools

Example Usage:
    >>> # Import API application
    >>> from deployment.api import app
    >>> 
    >>> # Import monitoring components
    >>> from deployment.monitoring import HealthChecker, PerformanceMonitor
    >>> 
    >>> # Initialize health checker
    >>> health_checker = HealthChecker()
    >>> health_report = health_checker.get_health_report()
    >>> print(f"System Status: {health_report['status']}")
    >>> 
    >>> # Initialize performance monitor
    >>> perf_monitor = PerformanceMonitor()
    >>> perf_monitor.start_timer("inference")
    >>> # ... run inference ...
    >>> perf_monitor.end_timer("inference")

Deployment Options:
    - Docker: Containerized deployment with docker-compose
    - Bare Metal: Direct deployment with uvicorn
    - Kubernetes: Scalable orchestration with k8s manifests

Hardware Support:
    - CPU: Intel/AMD x86_64, Apple Silicon (M1/M2)
    - GPU: NVIDIA CUDA 11.7+ (RTX 2050, RTX 3060+)

For detailed deployment instructions, see:
    - deployment/docker/README.md (Docker deployment)
    - deployment/api/README.md (API documentation)
    - deployment/monitoring/README.md (Monitoring setup)
"""

# Import submodules for convenient access
from . import api
from . import monitoring
from . import scripts

__all__ = [
    "api",
    "monitoring",
    "scripts"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"

# Deployment metadata
__deployment_targets__ = ["docker", "kubernetes", "bare-metal"]
__supported_devices__ = ["cpu", "cuda", "mps"]
__api_framework__ = "FastAPI"
__container_runtime__ = "Docker"
