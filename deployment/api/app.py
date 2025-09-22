#!/usr/bin/env python3
"""
FastAPI Application for FactCheck-MM Production API

Main application setup with middleware, error handling, health checks,
and production-ready configuration for serving multimodal fact-checking services.

Example Usage:
    >>> from deployment.api.app import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import psutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from .endpoints import router
from .models import HealthResponse, ErrorResponse

# Initialize logging
setup_logging()
logger = get_logger("FactCheck-MM-App")

# Global application state
app_start_time = time.time()
health_status = {"status": "starting"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    
    Handles model preloading, resource initialization, and cleanup.
    """
    logger.info("Starting FactCheck-MM API...")
    
    # Startup tasks
    try:
        # Update health status
        health_status["status"] = "healthy"
        health_status["startup_time"] = time.time()
        
        # Optional: Preload critical models
        # This can be enabled in production for faster first requests
        if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
            logger.info("Preloading models...")
            # Add model preloading logic here
            pass
        
        logger.info("FactCheck-MM API startup completed")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        raise
    
    finally:
        # Shutdown tasks
        logger.info("Shutting down FactCheck-MM API...")
        health_status["status"] = "shutting_down"
        
        # Cleanup resources
        # Add cleanup logic here if needed
        
        logger.info("FactCheck-MM API shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application instance.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app with metadata
    app = FastAPI(
        title="FactCheck-MM API",
        description="Production API for multimodal fact-checking with sarcasm detection and paraphrasing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Add direct health endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Simple health check endpoint for load balancers and monitoring.
        
        Returns basic health status without detailed system information.
        """
        try:
            # Perform basic health checks
            checks = {
                "api_responsive": True,
                "memory_usage": psutil.virtual_memory().percent < 90,
                "disk_space": psutil.disk_usage('/').percent < 90
            }
            
            # Check if any critical checks failed
            all_healthy = all(checks.values())
            
            status = "healthy" if all_healthy else "degraded"
            
            # Override with global status if unhealthy
            if health_status.get("status") == "unhealthy":
                status = "unhealthy"
                checks["startup"] = False
            
            return HealthResponse(
                status=status,
                checks=checks,
                details={
                    "uptime_seconds": time.time() - app_start_time,
                    "version": "1.0.0"
                } if status == "healthy" else None
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                checks={"health_check": False},
                details={"error": str(e)}
            )
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "FactCheck-MM API",
            "version": "1.0.0",
            "status": "running",
            "uptime_seconds": time.time() - app_start_time,
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "status": "/api/v1/status",
                "sarcasm": "/api/v1/sarcasm/predict",
                "paraphrase": "/api/v1/paraphrase/generate",
                "fact_verification": "/api/v1/fact/verify"
            }
        }
    
    return app


def setup_middleware(app: FastAPI):
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests with timing information."""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s - "
            f"{request.method} {request.url.path}"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def setup_exception_handlers(app: FastAPI):
    """Configure global exception handlers."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured error responses."""
        
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url.path}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                details={
                    "status_code": exc.status_code,
                    "path": str(request.url.path),
                    "method": request.method
                }
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with detailed information."""
        
        logger.warning(f"Validation error: {exc.errors()} - {request.method} {request.url.path}")
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                details={
                    "validation_errors": exc.errors(),
                    "path": str(request.url.path),
                    "method": request.method
                }
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with error logging."""
        
        logger.error(f"Unexpected error: {str(exc)} - {request.method} {request.url.path}")
        logger.error(f"Exception type: {type(exc).__name__}")
        
        # Don't expose internal error details in production
        if os.getenv("DEBUG", "false").lower() == "true":
            error_details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "path": str(request.url.path),
                "method": request.method
            }
        else:
            error_details = {
                "path": str(request.url.path),
                "method": request.method
            }
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details=error_details
            ).dict()
        )


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    # Run application
    logger.info(f"Starting FactCheck-MM API on {host}:{port}")
    
    if workers > 1:
        # Multi-worker setup for production
        uvicorn.run(
            "deployment.api.app:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        # Single worker for development
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            reload=os.getenv("RELOAD", "false").lower() == "true"
        )
