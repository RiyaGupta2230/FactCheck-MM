"""
FactCheck-MM Production API Module

This module provides a production-ready REST API for serving multimodal fact-checking
services including sarcasm detection, paraphrasing, and fact verification.

Key components:
- FastAPI application with async endpoints
- Pydantic models for request/response validation
- Lazy model loading for memory efficiency
- Comprehensive error handling and logging
- Health monitoring and status endpoints

Example Usage:
    >>> from deployment.api import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

from .app import create_app, app
from .models import (
    SarcasmRequest, SarcasmResponse,
    ParaphraseRequest, ParaphraseResponse,
    FactRequest, FactResponse,
    StatusResponse, HealthResponse
)
from .endpoints import router

__all__ = [
    "create_app",
    "app", 
    "router",
    "SarcasmRequest",
    "SarcasmResponse",
    "ParaphraseRequest", 
    "ParaphraseResponse",
    "FactRequest",
    "FactResponse",
    "StatusResponse",
    "HealthResponse"
]

__version__ = "1.0.0"
__author__ = "FactCheck-MM Team"
