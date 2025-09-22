#!/usr/bin/env python3
"""
FastAPI Endpoints for FactCheck-MM Production API

Implements REST endpoints for sarcasm detection, paraphrasing, and fact verification
with comprehensive error handling, logging, and performance monitoring.

Example Usage:
    >>> from deployment.api.endpoints import router
    >>> from fastapi import FastAPI
    >>> 
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import traceback
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
import psutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger
from .models import (
    SarcasmRequest, SarcasmResponse,
    ParaphraseRequest, ParaphraseResponse, ParaphraseItem,
    FactRequest, FactResponse, EvidenceItem,
    StatusResponse, ModelStatus, HealthResponse, ErrorResponse
)

# Initialize logger
logger = get_logger("FactCheck-MM-API")

# Global model storage for lazy loading
_models: Dict[str, Any] = {}
_model_load_times: Dict[str, float] = {}
_request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "response_times": []
}

# Initialize router
router = APIRouter()


class ModelManager:
    """Manages lazy loading and caching of ML models."""
    
    def __init__(self):
        self.models = {}
        self.load_times = {}
        self.logger = get_logger("ModelManager")
    
    async def get_sarcasm_model(self, model_version: Optional[str] = None):
        """Load and return sarcasm detection model."""
        model_key = f"sarcasm_{model_version or 'default'}"
        
        if model_key not in self.models:
            self.logger.info(f"Loading sarcasm model: {model_key}")
            start_time = time.time()
            
            try:
                # Import here to avoid circular imports and enable lazy loading
                from sarcasm_detection.models import MultimodalSarcasmDetector
                
                # Load model with appropriate configuration
                model = MultimodalSarcasmDetector.from_pretrained(
                    "checkpoints/multimodal_sarcasm/best_model"
                )
                model.eval()
                
                self.models[model_key] = model
                self.load_times[model_key] = time.time() - start_time
                
                self.logger.info(f"Sarcasm model loaded in {self.load_times[model_key]:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to load sarcasm model: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Sarcasm model unavailable: {str(e)}"
                )
        
        return self.models[model_key]
    
    async def get_paraphrase_model(self, model_type: str = "t5"):
        """Load and return paraphrasing model."""
        model_key = f"paraphrase_{model_type}"
        
        if model_key not in self.models:
            self.logger.info(f"Loading paraphrase model: {model_key}")
            start_time = time.time()
            
            try:
                if model_type == "t5":
                    from paraphrasing.models import T5Paraphraser
                    model = T5Paraphraser.from_pretrained("checkpoints/t5_paraphraser/")
                elif model_type == "bart":
                    from paraphrasing.models import BARTParaphraser
                    model = BARTParaphraser.from_pretrained("checkpoints/bart_paraphraser/")
                elif model_type == "sarcasm_aware":
                    from paraphrasing.models import SarcasmAwareParaphraser
                    model = SarcasmAwareParaphraser.from_pretrained("checkpoints/sarcasm_aware_paraphraser/")
                else:
                    raise ValueError(f"Unknown paraphrase model type: {model_type}")
                
                model.eval()
                
                self.models[model_key] = model
                self.load_times[model_key] = time.time() - start_time
                
                self.logger.info(f"Paraphrase model loaded in {self.load_times[model_key]:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to load paraphrase model: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Paraphrase model unavailable: {str(e)}"
                )
        
        return self.models[model_key]
    
    async def get_fact_verification_model(self, domain: Optional[str] = None):
        """Load and return fact verification model."""
        model_key = f"fact_verification_{domain or 'general'}"
        
        if model_key not in self.models:
            self.logger.info(f"Loading fact verification model: {model_key}")
            start_time = time.time()
            
            try:
                from fact_verification.models import FactCheckPipeline, FactCheckPipelineConfig
                
                # Configure pipeline based on domain
                config = FactCheckPipelineConfig(
                    enable_evidence_retrieval=True,
                    enable_stance_detection=True,
                    enable_fact_verification=True,
                    max_evidence_per_claim=10
                )
                
                # Load domain-specific model if available
                if domain and domain != "general":
                    model_path = f"checkpoints/fact_verification/{domain}_model/"
                else:
                    model_path = "checkpoints/fact_verification/general_model/"
                
                model = FactCheckPipeline.from_pretrained(model_path, config=config)
                
                self.models[model_key] = model
                self.load_times[model_key] = time.time() - start_time
                
                self.logger.info(f"Fact verification model loaded in {self.load_times[model_key]:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to load fact verification model: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Fact verification model unavailable: {str(e)}"
                )
        
        return self.models[model_key]
    
    def get_model_status(self) -> Dict[str, ModelStatus]:
        """Get status of all loaded models."""
        status = {}
        
        for model_key, model in self.models.items():
            # Estimate memory usage (simplified)
            try:
                num_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
                memory_mb = num_params * 4 / (1024 * 1024)  # Assume 4 bytes per parameter
            except:
                num_params = None
                memory_mb = None
            
            status[model_key] = ModelStatus(
                name=model_key,
                version="1.0.0",  # This would come from model metadata in production
                loaded=True,
                load_time=None,  # Would track actual load time
                memory_usage_mb=memory_mb,
                num_parameters=num_params
            )
        
        return status


# Global model manager instance
model_manager = ModelManager()


async def get_request_id() -> str:
    """Generate unique request ID for tracking."""
    return str(uuid.uuid4())


async def track_request_stats(start_time: float):
    """Update request statistics."""
    response_time = time.time() - start_time
    _request_stats["response_times"].append(response_time)
    _request_stats["active_requests"] = max(0, _request_stats["active_requests"] - 1)
    
    # Keep only last 1000 response times for memory efficiency
    if len(_request_stats["response_times"]) > 1000:
        _request_stats["response_times"] = _request_stats["response_times"][-1000:]


@router.post(
    "/sarcasm/predict",
    response_model=SarcasmResponse,
    summary="Detect Sarcasm",
    description="Analyze text (and optionally audio/visual content) for sarcastic intent using multimodal models"
)
async def predict_sarcasm(
    request: SarcasmRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id)
) -> SarcasmResponse:
    """
    Detect sarcasm in multimodal input using state-of-the-art models.
    
    Supports text-only and multimodal analysis with configurable confidence thresholds
    and detailed explanations of the prediction process.
    """
    start_time = time.time()
    
    try:
        _request_stats["total_requests"] += 1
        _request_stats["active_requests"] += 1
        
        logger.info(f"Processing sarcasm request {request_id}: {request.text[:50]}...")
        
        # Load model
        model = await model_manager.get_sarcasm_model(request.model_version)
        
        # Prepare input based on modalities
        model_input = {
            "text": request.text,
            "modalities": request.modalities
        }
        
        # Add multimodal inputs if specified
        if "audio" in request.modalities and request.audio_url:
            model_input["audio_path"] = request.audio_url
        
        if "image" in request.modalities and request.image_url:
            model_input["image_path"] = request.image_url
            
        if "video" in request.modalities and request.video_url:
            model_input["video_path"] = request.video_url
        
        # Run prediction
        prediction = await asyncio.get_event_loop().run_in_executor(
            None, model.predict_batch, [model_input]
        )
        
        result = prediction[0]  # Get first (and only) result
        
        # Prepare response
        response = SarcasmResponse(
            is_sarcastic=result["prediction"] == 1,
            confidence=float(result["confidence"]),
            prediction_details=result.get("modality_scores") if request.include_confidence else None,
            explanation=result.get("explanation") if request.include_explanation else None,
            processing_time=time.time() - start_time,
            model_version=request.model_version or "multimodal-v1.0"
        )
        
        # Update stats in background
        background_tasks.add_task(track_request_stats, start_time)
        
        logger.info(f"Sarcasm prediction completed for {request_id}: {response.is_sarcastic} ({response.confidence:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Sarcasm prediction failed for {request_id}: {e}")
        logger.error(traceback.format_exc())
        
        background_tasks.add_task(track_request_stats, start_time)
        
        raise HTTPException(
            status_code=500,
            detail=f"Sarcasm prediction failed: {str(e)}"
        )


@router.post(
    "/paraphrase/generate",
    response_model=ParaphraseResponse,
    summary="Generate Paraphrases",
    description="Generate high-quality paraphrases using T5, BART, or sarcasm-aware models with quality scoring"
)
async def generate_paraphrases(
    request: ParaphraseRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id)
) -> ParaphraseResponse:
    """
    Generate multiple paraphrases for input text with quality assessment.
    
    Supports various generation strategies including diversity control,
    sarcasm neutralization, and comprehensive quality metrics.
    """
    start_time = time.time()
    
    try:
        _request_stats["total_requests"] += 1
        _request_stats["active_requests"] += 1
        
        logger.info(f"Processing paraphrase request {request_id}: {request.text[:50]}...")
        
        # Load model
        model = await model_manager.get_paraphrase_model(request.model_type)
        
        # Prepare generation parameters
        generation_params = {
            "num_return_sequences": request.num_paraphrases,
            "temperature": request.temperature,
            "diversity_penalty": request.diversity_penalty,
            "max_length": request.max_length,
            "min_length": request.min_length
        }
        
        # Generate paraphrases
        paraphrases = await asyncio.get_event_loop().run_in_executor(
            None, model.generate, request.text, generation_params
        )
        
        # Process paraphrases and compute quality scores
        paraphrase_items = []
        quality_scores = []
        
        for i, paraphrase_text in enumerate(paraphrases):
            quality_score = None
            semantic_sim = None
            fluency_score = None
            diversity_score = None
            
            if request.include_quality_scores:
                # Compute quality metrics (simplified for demo)
                try:
                    from paraphrasing.models import QualityScorer
                    scorer = QualityScorer()
                    
                    scores = await asyncio.get_event_loop().run_in_executor(
                        None, scorer.score, request.text, paraphrase_text
                    )
                    
                    quality_score = scores.get("overall_quality", 0.8)
                    semantic_sim = scores.get("semantic_similarity", 0.85)
                    fluency_score = scores.get("fluency", 0.9)
                    diversity_score = scores.get("diversity", 0.7)
                    
                    quality_scores.append(quality_score)
                    
                except Exception as e:
                    logger.warning(f"Quality scoring failed: {e}")
            
            paraphrase_items.append(ParaphraseItem(
                text=paraphrase_text,
                quality_score=quality_score,
                semantic_similarity=semantic_sim,
                fluency_score=fluency_score,
                diversity_score=diversity_score
            ))
        
        # Prepare response
        response = ParaphraseResponse(
            original_text=request.text,
            paraphrases=paraphrase_items,
            average_quality=sum(quality_scores) / len(quality_scores) if quality_scores else None,
            processing_time=time.time() - start_time,
            model_version=f"{request.model_type}-v1.0"
        )
        
        # Update stats in background
        background_tasks.add_task(track_request_stats, start_time)
        
        logger.info(f"Paraphrase generation completed for {request_id}: {len(paraphrase_items)} paraphrases")
        
        return response
        
    except Exception as e:
        logger.error(f"Paraphrase generation failed for {request_id}: {e}")
        logger.error(traceback.format_exc())
        
        background_tasks.add_task(track_request_stats, start_time)
        
        raise HTTPException(
            status_code=500,
            detail=f"Paraphrase generation failed: {str(e)}"
        )


@router.post(
    "/fact/verify",
    response_model=FactResponse,
    summary="Verify Facts",
    description="Verify factual claims using evidence retrieval and multi-step reasoning with confidence estimation"
)
async def verify_fact(
    request: FactRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id)
) -> FactResponse:
    """
    Verify factual claims using comprehensive evidence retrieval and reasoning.
    
    Supports domain-specific models, configurable evidence sources,
    and detailed explanations of the verification process.
    """
    start_time = time.time()
    
    try:
        _request_stats["total_requests"] += 1
        _request_stats["active_requests"] += 1
        
        logger.info(f"Processing fact verification request {request_id}: {request.claim[:50]}...")
        
        # Load model
        model = await model_manager.get_fact_verification_model(request.domain)
        
        # Run fact verification
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            model.check_fact,
            request.claim,
            {
                "retrieve_evidence": request.retrieve_evidence,
                "max_evidence": request.max_evidence,
                "evidence_sources": request.evidence_sources,
                "include_stance": request.include_stance,
                "include_explanations": request.include_explanation
            }
        )
        
        # Process evidence
        evidence_items = []
        if request.retrieve_evidence and result.get("evidence"):
            for evidence in result["evidence"][:request.max_evidence]:
                evidence_items.append(EvidenceItem(
                    text=evidence.get("text", ""),
                    source=evidence.get("source", "Unknown"),
                    url=evidence.get("url"),
                    relevance_score=evidence.get("relevance_score", 0.5),
                    stance=evidence.get("stance"),
                    stance_confidence=evidence.get("stance_confidence")
                ))
        
        # Prepare response
        response = FactResponse(
            claim=request.claim,
            verdict=result["verdict"],
            confidence=float(result["confidence"]),
            evidence=evidence_items,
            explanation=result.get("explanation") if request.include_explanation else None,
            reasoning_steps=result.get("reasoning_steps") if request.include_explanation else None,
            processing_time=time.time() - start_time,
            model_version=f"fact-verification-{request.domain or 'general'}-v1.0"
        )
        
        # Update stats in background
        background_tasks.add_task(track_request_stats, start_time)
        
        logger.info(f"Fact verification completed for {request_id}: {response.verdict} ({response.confidence:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Fact verification failed for {request_id}: {e}")
        logger.error(traceback.format_exc())
        
        background_tasks.add_task(track_request_stats, start_time)
        
        raise HTTPException(
            status_code=500,
            detail=f"Fact verification failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="System Status",
    description="Get comprehensive system status including model status, performance metrics, and resource usage"
)
async def get_status() -> StatusResponse:
    """
    Get comprehensive system status and performance metrics.
    
    Returns information about loaded models, request statistics,
    system resources, and overall service health.
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                system_info.update({
                    "gpu_available": True,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total_gb": gpu_memory,
                    "gpu_memory_used_gb": gpu_memory_used,
                    "gpu_memory_usage_percent": (gpu_memory_used / gpu_memory) * 100
                })
            else:
                system_info["gpu_available"] = False
        except ImportError:
            system_info["gpu_available"] = False
        
        # Calculate average response time
        avg_response_time = 0.0
        if _request_stats["response_times"]:
            avg_response_time = sum(_request_stats["response_times"]) / len(_request_stats["response_times"])
        
        # Get model status
        model_status = model_manager.get_model_status()
        
        response = StatusResponse(
            version="1.0.0",
            uptime_seconds=time.time() - start_time,  # This would be tracked from app start
            models=model_status,
            total_requests=_request_stats["total_requests"],
            active_requests=_request_stats["active_requests"],
            average_response_time=avg_response_time,
            system_info=system_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Status endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


# Track application start time for uptime calculation
start_time = time.time()
