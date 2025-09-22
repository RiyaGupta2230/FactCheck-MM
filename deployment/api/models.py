#!/usr/bin/env python3
"""
Pydantic Data Models for FactCheck-MM API

Defines request and response models for all API endpoints with comprehensive
validation, documentation, and type safety.

Example Usage:
    >>> from deployment.api.models import SarcasmRequest, SarcasmResponse
    >>> 
    >>> # Create request model
    >>> request = SarcasmRequest(
    ...     text="Oh great, another meeting!",
    ...     modalities=["text", "audio"],
    ...     include_confidence=True
    ... )
    >>> 
    >>> # Validate response
    >>> response = SarcasmResponse(
    ...     is_sarcastic=True,
    ...     confidence=0.89,
    ...     prediction_details={"text_score": 0.85, "audio_score": 0.93}
    ... )
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum


class ModalityType(str, Enum):
    """Supported modality types for multimodal processing."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class SarcasmRequest(BaseModel):
    """
    Request model for sarcasm detection endpoint.
    
    Supports multimodal inputs with configurable processing options.
    """
    
    # Required text input
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Input text to analyze for sarcasm",
        example="Oh wonderful, another rainy day!"
    )
    
    # Multimodal inputs (optional)
    audio_url: Optional[str] = Field(
        None,
        description="URL or path to audio file for multimodal analysis",
        example="https://example.com/audio.wav"
    )
    
    image_url: Optional[str] = Field(
        None,
        description="URL or path to image file for multimodal analysis", 
        example="https://example.com/image.jpg"
    )
    
    video_url: Optional[str] = Field(
        None,
        description="URL or path to video file for multimodal analysis",
        example="https://example.com/video.mp4"
    )
    
    # Processing options
    modalities: List[ModalityType] = Field(
        default=["text"],
        description="List of modalities to use for analysis",
        example=["text", "audio"]
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in response"
    )
    
    include_explanation: bool = Field(
        default=False,
        description="Whether to include model explanation/reasoning"
    )
    
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to use (optional)",
        example="multimodal-v1.2"
    )
    
    @validator('modalities')
    def validate_modalities(cls, v):
        """Ensure modalities list is not empty and contains valid types."""
        if not v:
            raise ValueError("At least one modality must be specified")
        return v
    
    @root_validator
    def validate_multimodal_inputs(cls, values):
        """Ensure required inputs are provided for specified modalities."""
        modalities = values.get('modalities', [])
        
        if ModalityType.AUDIO in modalities and not values.get('audio_url'):
            raise ValueError("audio_url required when audio modality is specified")
        
        if ModalityType.IMAGE in modalities and not values.get('image_url'):
            raise ValueError("image_url required when image modality is specified")
            
        if ModalityType.VIDEO in modalities and not values.get('video_url'):
            raise ValueError("video_url required when video modality is specified")
        
        return values


class SarcasmResponse(BaseModel):
    """Response model for sarcasm detection results."""
    
    is_sarcastic: bool = Field(
        ...,
        description="Whether the input is predicted to be sarcastic"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction",
        example=0.87
    )
    
    prediction_details: Optional[Dict[str, float]] = Field(
        None,
        description="Per-modality confidence scores",
        example={"text": 0.85, "audio": 0.89, "combined": 0.87}
    )
    
    explanation: Optional[str] = Field(
        None,
        description="Human-readable explanation of the prediction",
        example="High sarcasm probability due to positive words with negative prosodic features"
    )
    
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        example=0.234
    )
    
    model_version: str = Field(
        ..., 
        description="Version of the model used for prediction",
        example="multimodal-sarcasm-v1.2"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the prediction"
    )


class ParaphraseRequest(BaseModel):
    """Request model for paraphrase generation endpoint."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text to paraphrase",
        example="Climate change is causing global temperatures to rise"
    )
    
    num_paraphrases: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of paraphrases to generate",
        example=3
    )
    
    diversity_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Penalty for generating similar paraphrases (0=no penalty, 1=max diversity)",
        example=0.5
    )
    
    min_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum length of generated paraphrases",
        example=10
    )
    
    max_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum length of generated paraphrases",
        example=100
    )
    
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature for generation (higher=more creative)",
        example=0.8
    )
    
    neutralize_sarcasm: bool = Field(
        default=False,
        description="Whether to neutralize sarcastic tone in paraphrases"
    )
    
    include_quality_scores: bool = Field(
        default=True,
        description="Whether to include quality scores for each paraphrase"
    )
    
    model_type: str = Field(
        default="t5",
        description="Model type to use for generation",
        example="t5",
        regex="^(t5|bart|sarcasm_aware)$"
    )


class ParaphraseItem(BaseModel):
    """Individual paraphrase item with quality metrics."""
    
    text: str = Field(
        ...,
        description="Generated paraphrase text"
    )
    
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall quality score for the paraphrase"
    )
    
    semantic_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity to original text"
    )
    
    fluency_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fluency/grammaticality score"
    )
    
    diversity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Diversity score compared to other paraphrases"
    )


class ParaphraseResponse(BaseModel):
    """Response model for paraphrase generation results."""
    
    original_text: str = Field(
        ...,
        description="Original input text"
    )
    
    paraphrases: List[ParaphraseItem] = Field(
        ...,
        description="List of generated paraphrases with quality scores"
    )
    
    average_quality: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average quality score across all paraphrases"
    )
    
    processing_time: float = Field(
        ...,
        description="Processing time in seconds"
    )
    
    model_version: str = Field(
        ...,
        description="Version of the model used for generation"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the generation"
    )


class FactRequest(BaseModel):
    """Request model for fact verification endpoint."""
    
    claim: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Factual claim to verify",
        example="The Eiffel Tower is located in Paris, France"
    )
    
    retrieve_evidence: bool = Field(
        default=True,
        description="Whether to retrieve supporting evidence"
    )
    
    max_evidence: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of evidence pieces to retrieve"
    )
    
    evidence_sources: List[str] = Field(
        default=["wikipedia", "wikidata"],
        description="Sources to search for evidence",
        example=["wikipedia", "wikidata", "dbpedia"]
    )
    
    include_stance: bool = Field(
        default=True,
        description="Whether to include stance detection results"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for high-confidence predictions"
    )
    
    domain: Optional[str] = Field(
        default=None,
        description="Domain-specific model to use (political, scientific, general)",
        example="political"
    )
    
    include_explanation: bool = Field(
        default=True,
        description="Whether to include detailed explanation of the verdict"
    )


class EvidenceItem(BaseModel):
    """Individual evidence item with metadata."""
    
    text: str = Field(
        ...,
        description="Evidence text content"
    )
    
    source: str = Field(
        ...,
        description="Source of the evidence",
        example="Wikipedia"
    )
    
    url: Optional[str] = Field(
        None,
        description="URL or identifier for the evidence source"
    )
    
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score to the claim"
    )
    
    stance: Optional[str] = Field(
        None,
        description="Stance of evidence towards claim (supports/refutes/neutral)",
        example="supports"
    )
    
    stance_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in stance prediction"
    )


class FactResponse(BaseModel):
    """Response model for fact verification results."""
    
    claim: str = Field(
        ...,
        description="Original claim that was verified"
    )
    
    verdict: str = Field(
        ...,
        description="Fact-checking verdict",
        example="SUPPORTS",
        regex="^(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)$"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the verdict"
    )
    
    evidence: List[EvidenceItem] = Field(
        default=[],
        description="Retrieved evidence supporting the verdict"
    )
    
    explanation: Optional[str] = Field(
        None,
        description="Human-readable explanation of the verdict",
        example="Multiple reliable sources confirm this claim with high confidence"
    )
    
    reasoning_steps: Optional[List[str]] = Field(
        None,
        description="Step-by-step reasoning process",
        example=[
            "Retrieved 5 relevant evidence pieces",
            "3 pieces support the claim",
            "1 piece is neutral", 
            "Overall evidence strongly supports the claim"
        ]
    )
    
    processing_time: float = Field(
        ...,
        description="Processing time in seconds"
    )
    
    model_version: str = Field(
        ...,
        description="Version of the fact verification model used"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the verification"
    )


class ModelStatus(BaseModel):
    """Status information for a loaded model."""
    
    name: str = Field(
        ...,
        description="Model name/identifier"
    )
    
    version: str = Field(
        ...,
        description="Model version"
    )
    
    loaded: bool = Field(
        ...,
        description="Whether the model is currently loaded in memory"
    )
    
    load_time: Optional[datetime] = Field(
        None,
        description="When the model was loaded"
    )
    
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Approximate memory usage in MB"
    )
    
    num_parameters: Optional[int] = Field(
        None,
        ge=0,
        description="Number of model parameters"
    )


class StatusResponse(BaseModel):
    """Response model for system status endpoint."""
    
    service_name: str = Field(
        default="FactCheck-MM API",
        description="Name of the service"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Service uptime in seconds"
    )
    
    models: Dict[str, ModelStatus] = Field(
        ...,
        description="Status of all available models"
    )
    
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of requests processed"
    )
    
    active_requests: int = Field(
        ...,
        ge=0,
        description="Number of currently active requests"
    )
    
    average_response_time: float = Field(
        ...,
        ge=0.0,
        description="Average response time in seconds"
    )
    
    system_info: Dict[str, Any] = Field(
        ...,
        description="System information (CPU, memory, GPU)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Current timestamp"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Overall health status",
        example="healthy",
        regex="^(healthy|degraded|unhealthy)$"
    )
    
    checks: Dict[str, bool] = Field(
        ...,
        description="Individual health check results",
        example={
            "database": True,
            "models_loaded": True,
            "disk_space": True,
            "memory_usage": True
        }
    )
    
    details: Optional[Dict[str, str]] = Field(
        None,
        description="Additional details about health status"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        ...,
        description="Error type or category"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier for debugging"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
