"""
FactCheck-MM Deployment Monitoring Module

Provides runtime monitoring capabilities for production deployments including:
- System health checks (CPU, RAM, disk, GPU)
- Application health monitoring (model availability, API responsiveness)
- Performance tracking (latency, throughput, resource usage)
- Real-time metrics collection and reporting

This module is designed for both development (MacBook M2) and production
(RTX 2050 GPU) environments with minimal overhead.

Example Usage:
    >>> from deployment.monitoring import HealthChecker, PerformanceMonitor
    >>> 
    >>> # Health checking
    >>> health = HealthChecker()
    >>> report = health.get_health_report()
    >>> print(f"System Status: {report['status']}")
    >>> 
    >>> # Performance monitoring
    >>> perf = PerformanceMonitor()
    >>> perf.start_timer("fact_verification")
    >>> # ... run inference ...
    >>> perf.end_timer("fact_verification")
    >>> summary = perf.get_performance_summary()
"""

from .health_checks import HealthChecker
from .performance_monitor import PerformanceMonitor

__all__ = [
    "HealthChecker",
    "PerformanceMonitor"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
