#!/usr/bin/env python3
"""
End-to-End Fact Checking Pipeline

Implements complete fact-checking pipeline combining claim detection, evidence 
retrieval, stance detection, and fact verification into a unified system with
comprehensive logging and evaluation capabilities.

Example Usage:
    >>> from fact_verification.models import FactCheckPipeline
    >>> 
    >>> # Initialize complete pipeline
    >>> pipeline = FactCheckPipeline()
    >>> 
    >>> # Check a single claim
    >>> result = pipeline.check_fact("COVID-19 vaccines are 95% effective")
    >>> print(f"Verdict: {result['verdict']}, Confidence: {result['confidence']:.3f}")
    >>> 
    >>> # Process raw text with claim detection
    >>> text = "Some people think vaccines are dangerous. However, studies show vaccines are safe and effective."
    >>> results = pipeline.check_document(text)
    >>> for claim_result in results:
    ...     print(f"Claim: {claim_result['claim']}")
    ...     print(f"Verdict: {claim_result['verdict']}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import json
import time
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.utils.logging_utils import get_logger
from .claim_detector import ClaimDetector, ClaimDetectorConfig
from .evidence_retriever import EvidenceRetriever, EvidenceRetrieverConfig
from .stance_detector import StanceDetector, StanceDetectorConfig
from .fact_verifier import FactVerifier, FactVerifierConfig

# Optional imports for logging and visualization
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class FactCheckPipelineConfig:
    """Configuration for end-to-end fact checking pipeline."""
    
    # Pipeline components
    enable_claim_detection: bool = True
    enable_evidence_retrieval: bool = True
    enable_stance_detection: bool = True
    enable_fact_verification: bool = True
    
    # Component configurations
    claim_detector_config: Optional[ClaimDetectorConfig] = None
    evidence_retriever_config: Optional[EvidenceRetrieverConfig] = None
    stance_detector_config: Optional[StanceDetectorConfig] = None
    fact_verifier_config: Optional[FactVerifierConfig] = None
    
    # Pipeline parameters
    max_evidence_per_claim: int = 5
    evidence_aggregation_strategy: str = "top_k"  # "top_k", "threshold", "diverse"
    confidence_threshold: float = 0.5
    
    # Multi-task learning
    enable_multitask_learning: bool = False
    shared_encoder: bool = False
    joint_training: bool = False
    
    # Pipeline optimization
    use_caching: bool = True
    cache_evidence: bool = True
    cache_stance_results: bool = True
    parallel_processing: bool = False
    
    # Output configuration
    return_intermediate_results: bool = True
    return_evidence_details: bool = True
    return_attention_weights: bool = False
    detailed_explanations: bool = True
    
    # Performance monitoring
    enable_timing: bool = True
    enable_logging: bool = True
    log_predictions: bool = True
    
    # Model paths (for loading pre-trained components)
    claim_detector_path: Optional[str] = None
    evidence_retriever_path: Optional[str] = None
    stance_detector_path: Optional[str] = None
    fact_verifier_path: Optional[str] = None
    
    # Experiment tracking
    experiment_name: str = "fact_check_pipeline"
    enable_wandb: bool = False
    enable_tensorboard: bool = False
    log_dir: str = "logs/fact_checking"


class FactCheckPipeline(BaseMultimodalModel):
    """
    End-to-end fact checking pipeline.
    
    Combines claim detection, evidence retrieval, stance detection, and 
    fact verification into a unified system for comprehensive automated
    fact checking with detailed explanations and intermediate results.
    """
    
    def __init__(self, config: Optional[FactCheckPipelineConfig] = None):
        """
        Initialize fact checking pipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__()
        
        self.config = config or FactCheckPipelineConfig()
        self.logger = get_logger("FactCheckPipeline")
        
        # Initialize pipeline components
        self.claim_detector = None
        self.evidence_retriever = None
        self.stance_detector = None
        self.fact_verifier = None
        
        self._initialize_components()
        
        # Caching for performance
        self.evidence_cache = {} if self.config.use_caching else None
        self.stance_cache = {} if self.config.cache_stance_results else None
        
        # Performance tracking
        self.pipeline_stats = {
            'total_processed': 0,
            'claims_detected': 0,
            'evidence_retrieved': 0,
            'stances_detected': 0,
            'facts_verified': 0,
            'processing_times': []
        }
        
        # Initialize logging
        self._setup_logging()
        
        self.logger.info("Initialized FactCheckPipeline")
        self.logger.info(f"Components: Claim Detection={self.claim_detector is not None}, "
                        f"Evidence Retrieval={self.evidence_retriever is not None}, "
                        f"Stance Detection={self.stance_detector is not None}, "
                        f"Fact Verification={self.fact_verifier is not None}")
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        
        # Claim Detector
        if self.config.enable_claim_detection:
            if self.config.claim_detector_path:
                self.claim_detector = ClaimDetector.from_pretrained(self.config.claim_detector_path)
            else:
                claim_config = self.config.claim_detector_config or ClaimDetectorConfig()
                self.claim_detector = ClaimDetector(claim_config)
        
        # Evidence Retriever
        if self.config.enable_evidence_retrieval:
            if self.config.evidence_retriever_path:
                self.evidence_retriever = EvidenceRetriever.from_pretrained(self.config.evidence_retriever_path)
            else:
                retriever_config = self.config.evidence_retriever_config or EvidenceRetrieverConfig()
                self.evidence_retriever = EvidenceRetriever(retriever_config)
        
        # Shared encoder for multi-task learning
        shared_encoder = None
        if self.config.enable_multitask_learning and self.config.shared_encoder:
            if self.fact_verifier:
                shared_encoder = self.fact_verifier.roberta
            elif self.stance_detector:
                shared_encoder = self.stance_detector.roberta
        
        # Stance Detector
        if self.config.enable_stance_detection:
            if self.config.stance_detector_path:
                self.stance_detector = StanceDetector.from_pretrained(self.config.stance_detector_path)
            else:
                stance_config = self.config.stance_detector_config or StanceDetectorConfig()
                self.stance_detector = StanceDetector(stance_config, shared_encoder=shared_encoder)
        
        # Fact Verifier
        if self.config.enable_fact_verification:
            if self.config.fact_verifier_path:
                self.fact_verifier = FactVerifier.from_pretrained(self.config.fact_verifier_path)
            else:
                verifier_config = self.config.fact_verifier_config or FactVerifierConfig()
                self.fact_verifier = FactVerifier(verifier_config)
    
    def _setup_logging(self):
        """Setup experiment tracking and logging."""
        
        self.wandb_run = None
        self.tensorboard_writer = None
        
        if self.config.enable_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project="factcheck-mm",
                    name=self.config.experiment_name,
                    config=self.config.__dict__
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
        
        if self.config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                log_dir = Path(self.config.log_dir) / self.config.experiment_name
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir)
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
    
    def detect_claims(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect claims in text.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for claim detection
            
        Returns:
            List of detected claims with positions and confidence
        """
        if not self.claim_detector:
            # Return entire text as a single claim if detector not available
            return [{'text': text, 'confidence': 1.0, 'start_char': 0, 'end_char': len(text)}]
        
        start_time = time.time()
        
        claims = self.claim_detector.detect_claims(text, threshold=threshold)
        
        processing_time = time.time() - start_time
        self.pipeline_stats['claims_detected'] += len(claims)
        
        if self.config.enable_logging:
            self.logger.info(f"Detected {len(claims)} claims in {processing_time:.3f}s")
        
        return claims
    
    def retrieve_evidence(
        self, 
        claim: str, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for a claim.
        
        Args:
            claim: Claim text
            top_k: Number of evidence pieces to retrieve
            
        Returns:
            List of retrieved evidence with scores
        """
        if not self.evidence_retriever:
            return []
        
        top_k = top_k or self.config.max_evidence_per_claim
        
        # Check cache
        if self.evidence_cache and claim in self.evidence_cache:
            return self.evidence_cache[claim][:top_k]
        
        start_time = time.time()
        
        evidence = self.evidence_retriever.retrieve(claim, top_k=top_k)
        
        processing_time = time.time() - start_time
        self.pipeline_stats['evidence_retrieved'] += len(evidence)
        
        # Cache results
        if self.evidence_cache:
            self.evidence_cache[claim] = evidence
        
        if self.config.enable_logging:
            self.logger.info(f"Retrieved {len(evidence)} evidence pieces in {processing_time:.3f}s")
        
        return evidence
    
    def detect_stance(
        self, 
        claim: str, 
        evidence: str
    ) -> Dict[str, Any]:
        """
        Detect stance between claim and evidence.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            
        Returns:
            Stance detection result
        """
        if not self.stance_detector:
            # Return neutral stance if detector not available
            return {
                'stance': 'NEUTRAL',
                'confidence': 0.5,
                'probabilities': {'SUPPORT': 0.33, 'AGAINST': 0.33, 'NEUTRAL': 0.34}
            }
        
        # Check cache
        cache_key = f"{claim}|{evidence}"
        if self.stance_cache and cache_key in self.stance_cache:
            return self.stance_cache[cache_key]
        
        start_time = time.time()
        
        result = self.stance_detector.detect_stance(claim, evidence)
        
        processing_time = time.time() - start_time
        self.pipeline_stats['stances_detected'] += 1
        
        # Cache results
        if self.stance_cache:
            self.stance_cache[cache_key] = result
        
        if self.config.enable_logging:
            self.logger.debug(f"Detected stance in {processing_time:.3f}s: {result['stance']}")
        
        return result
    
    def verify_fact(
        self, 
        claim: str, 
        evidence: List[str]
    ) -> Dict[str, Any]:
        """
        Verify fact using claim and evidence.
        
        Args:
            claim: Claim text
            evidence: List of evidence texts
            
        Returns:
            Fact verification result
        """
        if not self.fact_verifier:
            # Return uncertain verdict if verifier not available
            return {
                'label': 'NOT_ENOUGH_INFO',
                'confidence': 0.5,
                'probabilities': {'SUPPORTS': 0.33, 'REFUTES': 0.33, 'NOT_ENOUGH_INFO': 0.34}
            }
        
        start_time = time.time()
        
        result = self.fact_verifier.verify(
            claim, 
            evidence,
            return_attention=self.config.return_attention_weights
        )
        
        processing_time = time.time() - start_time
        self.pipeline_stats['facts_verified'] += 1
        
        if self.config.enable_logging:
            self.logger.info(f"Verified fact in {processing_time:.3f}s: {result['label']}")
        
        return result
    
    def _aggregate_evidence(
        self, 
        evidence_list: List[Dict[str, Any]], 
        claim: str
    ) -> List[Dict[str, Any]]:
        """
        Aggregate evidence based on configured strategy.
        
        Args:
            evidence_list: List of evidence with scores
            claim: Original claim text
            
        Returns:
            Filtered and aggregated evidence list
        """
        if not evidence_list:
            return []
        
        if self.config.evidence_aggregation_strategy == "top_k":
            return evidence_list[:self.config.max_evidence_per_claim]
        
        elif self.config.evidence_aggregation_strategy == "threshold":
            threshold = self.config.confidence_threshold
            return [ev for ev in evidence_list if ev.get('score', 0) >= threshold][:self.config.max_evidence_per_claim]
        
        elif self.config.evidence_aggregation_strategy == "diverse":
            # Simple diversity: select evidence with different stance patterns
            diverse_evidence = []
            seen_keywords = set()
            
            for evidence in evidence_list:
                evidence_words = set(evidence['text'].lower().split())
                
                # Check for diversity
                if not seen_keywords or len(evidence_words & seen_keywords) < len(evidence_words) * 0.7:
                    diverse_evidence.append(evidence)
                    seen_keywords.update(evidence_words)
                    
                    if len(diverse_evidence) >= self.config.max_evidence_per_claim:
                        break
            
            return diverse_evidence
        
        else:
            return evidence_list[:self.config.max_evidence_per_claim]
    
    def check_fact(
        self, 
        claim: str,
        include_evidence_details: Optional[bool] = None,
        include_stance_details: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Complete fact checking pipeline for a single claim.
        
        Args:
            claim: Claim text to verify
            include_evidence_details: Whether to include evidence details
            include_stance_details: Whether to include stance details
            
        Returns:
            Complete fact checking result with verdict and explanations
        """
        pipeline_start = time.time()
        
        include_evidence = include_evidence_details if include_evidence_details is not None else self.config.return_evidence_details
        include_stance = include_stance_details if include_stance_details is not None else self.config.return_intermediate_results
        
        result = {
            'claim': claim,
            'verdict': 'NOT_ENOUGH_INFO',
            'confidence': 0.0,
            'explanation': '',
            'processing_time': 0.0
        }
        
        try:
            # Step 1: Evidence Retrieval
            evidence_list = []
            if self.config.enable_evidence_retrieval:
                evidence_list = self.retrieve_evidence(claim)
                evidence_list = self._aggregate_evidence(evidence_list, claim)
            
            if not evidence_list:
                result.update({
                    'verdict': 'NOT_ENOUGH_INFO',
                    'confidence': 0.1,
                    'explanation': 'No relevant evidence found for the claim.'
                })
                return result
            
            # Step 2: Stance Detection (optional)
            stance_results = []
            if self.config.enable_stance_detection:
                for evidence in evidence_list:
                    stance_result = self.detect_stance(claim, evidence['text'])
                    evidence['stance'] = stance_result
                    stance_results.append(stance_result)
            
            # Step 3: Fact Verification
            if self.config.enable_fact_verification:
                evidence_texts = [ev['text'] for ev in evidence_list]
                verification_result = self.verify_fact(claim, evidence_texts)
                
                result.update({
                    'verdict': verification_result['label'],
                    'confidence': verification_result['confidence'],
                    'probabilities': verification_result.get('probabilities', {})
                })
                
                # Add attention weights if available
                if self.config.return_attention_weights and 'attention_weights' in verification_result:
                    result['attention_weights'] = verification_result['attention_weights']
            
            # Generate explanation
            if self.config.detailed_explanations:
                result['explanation'] = self._generate_explanation(
                    claim, evidence_list, stance_results, result
                )
            
            # Add intermediate results
            if self.config.return_intermediate_results:
                result['intermediate_results'] = {
                    'evidence_count': len(evidence_list),
                    'avg_evidence_score': np.mean([ev.get('score', 0) for ev in evidence_list]),
                    'stance_distribution': self._get_stance_distribution(stance_results) if stance_results else {}
                }
            
            # Add evidence details
            if include_evidence:
                result['evidence'] = evidence_list
            
            # Add stance details
            if include_stance:
                result['stance_results'] = stance_results
        
        except Exception as e:
            self.logger.error(f"Error in fact checking pipeline: {e}")
            result.update({
                'verdict': 'ERROR',
                'confidence': 0.0,
                'explanation': f'Pipeline error: {str(e)}'
            })
        
        # Record timing
        result['processing_time'] = time.time() - pipeline_start
        self.pipeline_stats['processing_times'].append(result['processing_time'])
        self.pipeline_stats['total_processed'] += 1
        
        # Log results
        if self.config.log_predictions:
            self._log_prediction(result)
        
        return result
    
    def check_document(
        self, 
        text: str,
        claim_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Check facts in a document by first detecting claims.
        
        Args:
            text: Document text to analyze
            claim_threshold: Confidence threshold for claim detection
            
        Returns:
            List of fact checking results for detected claims
        """
        # Step 1: Claim Detection
        claims = self.detect_claims(text, threshold=claim_threshold)
        
        if not claims:
            return []
        
        # Step 2: Fact Check Each Claim
        results = []
        
        for claim_info in claims:
            claim_text = claim_info['text']
            
            # Skip very short claims
            if len(claim_text.split()) < 3:
                continue
            
            # Fact check the claim
            fact_result = self.check_fact(claim_text)
            
            # Add claim detection information
            fact_result['claim_detection'] = {
                'confidence': claim_info.get('confidence', 1.0),
                'start_char': claim_info.get('start_char', 0),
                'end_char': claim_info.get('end_char', len(claim_text)),
                'type': claim_info.get('type', 'classification')
            }
            
            results.append(fact_result)
        
        return results
    
    def _generate_explanation(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]],
        stance_results: List[Dict[str, Any]],
        verification_result: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for the verdict."""
        
        verdict = verification_result['verdict']
        confidence = verification_result['confidence']
        
        explanation_parts = []
        
        # Verdict summary
        if verdict == 'SUPPORTS':
            explanation_parts.append(f"The claim is SUPPORTED by available evidence (confidence: {confidence:.1%}).")
        elif verdict == 'REFUTES':
            explanation_parts.append(f"The claim is REFUTED by available evidence (confidence: {confidence:.1%}).")
        else:
            explanation_parts.append(f"There is NOT ENOUGH INFO to verify the claim (confidence: {confidence:.1%}).")
        
        # Evidence summary
        if evidence_list:
            explanation_parts.append(f"Based on analysis of {len(evidence_list)} pieces of evidence:")
            
            # Stance distribution
            if stance_results:
                stance_counts = {}
                for stance_result in stance_results:
                    stance = stance_result['stance']
                    stance_counts[stance] = stance_counts.get(stance, 0) + 1
                
                stance_summary = []
                for stance, count in stance_counts.items():
                    stance_summary.append(f"{count} {stance.lower()}")
                
                explanation_parts.append(f"Evidence stance: {', '.join(stance_summary)}.")
            
            # Top evidence
            if len(evidence_list) > 0:
                top_evidence = evidence_list[0]
                explanation_parts.append(f"Key evidence: \"{top_evidence['text'][:100]}...\"")
        
        return " ".join(explanation_parts)
    
    def _get_stance_distribution(self, stance_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get distribution of stance results."""
        
        if not stance_results:
            return {}
        
        stance_counts = {}
        for result in stance_results:
            stance = result['stance']
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
        
        total = len(stance_results)
        return {stance: count / total for stance, count in stance_counts.items()}
    
    def _log_prediction(self, result: Dict[str, Any]):
        """Log prediction to experiment tracking systems."""
        
        if self.wandb_run:
            self.wandb_run.log({
                'verdict': result['verdict'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'evidence_count': result.get('intermediate_results', {}).get('evidence_count', 0)
            })
        
        if self.tensorboard_writer:
            step = self.pipeline_stats['total_processed']
            self.tensorboard_writer.add_scalar('confidence', result['confidence'], step)
            self.tensorboard_writer.add_scalar('processing_time', result['processing_time'], step)
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        stats = self.pipeline_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = np.sum(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        
        # Component statistics
        stats['components'] = {
            'claim_detector': self.claim_detector is not None,
            'evidence_retriever': self.evidence_retriever is not None,
            'stance_detector': self.stance_detector is not None,
            'fact_verifier': self.fact_verifier is not None
        }
        
        # Cache statistics
        if self.evidence_cache:
            stats['evidence_cache_size'] = len(self.evidence_cache)
        
        if self.stance_cache:
            stats['stance_cache_size'] = len(self.stance_cache)
        
        return stats
    
    def save_pipeline(self, save_directory: str):
        """Save complete pipeline."""
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline configuration
        with open(save_path / "pipeline_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save individual components
        if self.claim_detector:
            self.claim_detector.save_pretrained(save_path / "claim_detector")
        
        if self.evidence_retriever:
            self.evidence_retriever.save_pretrained(save_path / "evidence_retriever")
        
        if self.stance_detector:
            self.stance_detector.save_pretrained(save_path / "stance_detector")
        
        if self.fact_verifier:
            self.fact_verifier.save_pretrained(save_path / "fact_verifier")
        
        # Save pipeline statistics
        with open(save_path / "pipeline_stats.json", 'w') as f:
            json.dump(self.get_pipeline_statistics(), f, indent=2)
        
        self.logger.info(f"Pipeline saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[FactCheckPipelineConfig] = None
    ) -> 'FactCheckPipeline':
        """Load pipeline from saved components."""
        
        model_path = Path(model_path)
        
        # Load pipeline configuration
        if config is None:
            config_path = model_path / "pipeline_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = FactCheckPipelineConfig(**config_dict)
            else:
                config = FactCheckPipelineConfig()
        
        # Update component paths
        if (model_path / "claim_detector").exists():
            config.claim_detector_path = str(model_path / "claim_detector")
        
        if (model_path / "evidence_retriever").exists():
            config.evidence_retriever_path = str(model_path / "evidence_retriever")
        
        if (model_path / "stance_detector").exists():
            config.stance_detector_path = str(model_path / "stance_detector")
        
        if (model_path / "fact_verifier").exists():
            config.fact_verifier_path = str(model_path / "fact_verifier")
        
        return cls(config)
    
    def __del__(self):
        """Cleanup resources."""
        
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.finish()


    _config_class = FactCheckPipelineConfig

    @classmethod
    def _build_from_config(cls, config: FactCheckPipelineConfig):
        return cls(config)


def main():
    """Example usage of FactCheckPipeline."""
    
    # Initialize pipeline
    config = FactCheckPipelineConfig(
        enable_claim_detection=True,
        enable_evidence_retrieval=True,
        enable_stance_detection=True,
        enable_fact_verification=True,
        return_evidence_details=True,
        detailed_explanations=True
    )
    
    pipeline = FactCheckPipeline(config)
    
    print("=== Fact Checking Pipeline Example ===")
    print(f"Components active: {pipeline.get_pipeline_statistics()['components']}")
    
    # Test single claim verification
    test_claims = [
        "COVID-19 vaccines are highly effective against severe illness",
        "The Earth is flat and space agencies are lying",
        "Regular exercise can improve mental health and cognitive function"
    ]
    
    print("\n=== Single Claim Verification ===")
    for i, claim in enumerate(test_claims, 1):
        print(f"\n--- Claim {i} ---")
        print(f"Claim: {claim}")
        
        result = pipeline.check_fact(claim)
        
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        
        if result['explanation']:
            print(f"Explanation: {result['explanation']}")
        
        if 'evidence' in result:
            print(f"Evidence pieces: {len(result['evidence'])}")
            for j, ev in enumerate(result['evidence'][:2]):
                print(f"  {j+1}. {ev['text'][:80]}... (score: {ev.get('score', 0):.3f})")
    
    # Test document processing
    print("\n=== Document Processing ===")
    test_document = (
        "There are many misconceptions about vaccines. Some people believe vaccines cause autism, "
        "but this has been thoroughly debunked by scientific research. COVID-19 vaccines have "
        "been shown to be highly effective in preventing severe illness and death. Climate change "
        "is another important issue that affects global health."
    )
    
    print(f"Document: {test_document[:100]}...")
    
    doc_results = pipeline.check_document(test_document)
    
    print(f"\nDetected and verified {len(doc_results)} claims:")
    for i, result in enumerate(doc_results, 1):
        print(f"{i}. \"{result['claim'][:50]}...\" -> {result['verdict']} ({result['confidence']:.3f})")
    
    # Pipeline statistics
    print(f"\n=== Pipeline Statistics ===")
    stats = pipeline.get_pipeline_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    # Test pipeline serialization
    print("\n=== Pipeline Serialization ===")
    save_path = "test_fact_check_pipeline"
    pipeline.save_pipeline(save_path)
    print(f"Pipeline saved to {save_path}")
    
    # Load pipeline
    try:
        loaded_pipeline = FactCheckPipeline.from_pretrained(save_path)
        print("Pipeline loaded successfully")
        
        # Test loaded pipeline
        test_result = loaded_pipeline.check_fact("Test claim for loaded pipeline")
        print(f"Loaded pipeline test: {test_result['verdict']}")
        
    except Exception as e:
        print(f"Pipeline loading failed: {e}")


if __name__ == "__main__":
    main()
# ==================== ALIAS ====================
# Alias for main.py compatibility
FactVerificationModel = FactCheckPipeline
