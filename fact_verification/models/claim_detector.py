#!/usr/bin/env python3
"""
Claim Detection Model for Fact Verification

Implements RoBERTa-based claim detector that identifies factual claims in text
with support for span extraction and binary classification for automated 
fact-checking pipelines.

Example Usage:
    >>> from fact_verification.models import ClaimDetector
    >>> 
    >>> # Initialize detector
    >>> detector = ClaimDetector()
    >>> 
    >>> # Detect claims in text
    >>> text = "The weather is nice today. COVID-19 vaccines are 95% effective."
    >>> claims = detector.detect_claims(text)
    >>> for claim in claims:
    ...     print(f"Claim: {claim['text']}, Confidence: {claim['confidence']:.3f}")
    >>>
    >>> # Binary classification
    >>> is_claim = detector.is_claim("Vaccines prevent disease.")
    >>> print(f"Is claim: {is_claim}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import numpy as np
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.preprocessing.text_processor import TextProcessor, TextProcessorConfig
from shared.utils.logging_utils import get_logger


@dataclass
class ClaimDetectorConfig:
    """Configuration for claim detection model."""
    
    # Model architecture
    model_name: str = "roberta-base"
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # Classification head
    num_classes: int = 2  # claim / not-claim
    classifier_layers: List[int] = field(default_factory=lambda: [768, 256, 2])
    
    # Text processing
    max_sequence_length: int = 512
    max_claim_length: int = 128
    sentence_overlap: int = 32  # For sliding window
    
    # Span extraction
    enable_span_extraction: bool = True
    span_threshold: float = 0.5
    max_span_length: int = 100
    min_span_length: int = 5
    
    # Training parameters
    label_smoothing: float = 0.1
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Performance
    batch_size: int = 16
    gradient_checkpointing: bool = False


class ClaimDetector(BaseMultimodalModel):
    """
    RoBERTa-based claim detector for identifying factual claims in text.
    
    Supports both binary classification (claim/not-claim) and span extraction
    to identify specific claim boundaries within longer texts.
    """
    
    def __init__(self, config: Optional[ClaimDetectorConfig] = None):
        """
        Initialize claim detector.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or ClaimDetectorConfig()
        self.logger = get_logger("ClaimDetector")
        
        # Load pre-trained RoBERTa
        self.roberta_config = RobertaConfig.from_pretrained(self.config.model_name)
        self.roberta = RobertaModel.from_pretrained(
            self.config.model_name,
            config=self.roberta_config
        )
        
        # Initialize tokenizer and text processor
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
        processor_config = TextProcessorConfig(
            model_name=self.config.model_name,
            max_length=self.config.max_sequence_length
        )
        self.text_processor = TextProcessor(processor_config)
        
        # Classification head for binary claim detection
        self.classifier = self._build_classifier()
        
        # Span extraction head (optional)
        if self.config.enable_span_extraction:
            self.span_start_classifier = nn.Linear(self.config.hidden_size, 1)
            self.span_end_classifier = nn.Linear(self.config.hidden_size, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Enable gradient checkpointing if specified
        if self.config.gradient_checkpointing:
            self.roberta.gradient_checkpointing_enable()
        
        self.logger.info(f"Initialized ClaimDetector with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _build_classifier(self) -> nn.Module:
        """Build multi-layer classification head."""
        
        layers = []
        layer_sizes = self.config.classifier_layers
        
        for i in range(len(layer_sizes) - 1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU() if i < len(layer_sizes) - 2 else nn.Identity(),
                nn.Dropout(self.config.dropout_rate) if i < len(layer_sizes) - 2 else nn.Identity()
            ])
        
        # Remove last identity layers
        return nn.Sequential(*[layer for layer in layers if not isinstance(layer, nn.Identity)])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        span_start_positions: Optional[torch.Tensor] = None,
        span_end_positions: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for claim detection.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            labels: Binary labels for claim detection (optional)
            span_start_positions: Start positions for span extraction (optional)
            span_end_positions: End positions for span extraction (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        # RoBERTa encoding
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention,
            return_dict=True
        )
        
        sequence_output = roberta_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = roberta_outputs.pooler_output       # [batch_size, hidden_size]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)
        
        # Binary classification
        classification_logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        outputs = {
            'classification_logits': classification_logits,
            'sequence_output': sequence_output,
            'pooled_output': pooled_output
        }
        
        # Span extraction (if enabled)
        if self.config.enable_span_extraction:
            start_logits = self.span_start_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_len]
            end_logits = self.span_end_classifier(sequence_output).squeeze(-1)      # [batch_size, seq_len]
            
            # Mask out special tokens for span prediction
            if attention_mask is not None:
                start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
                end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
            
            outputs.update({
                'start_logits': start_logits,
                'end_logits': end_logits
            })
        
        # Include attention weights if requested
        if return_attention and roberta_outputs.attentions is not None:
            outputs['attentions'] = roberta_outputs.attentions
        
        # Compute losses if labels provided
        total_loss = 0.0
        
        if labels is not None:
            # Classification loss
            classification_loss = self._compute_classification_loss(classification_logits, labels)
            outputs['classification_loss'] = classification_loss
            total_loss += classification_loss
        
        if (self.config.enable_span_extraction and 
            span_start_positions is not None and 
            span_end_positions is not None):
            # Span extraction loss
            span_loss = self._compute_span_loss(
                start_logits, end_logits,
                span_start_positions, span_end_positions,
                attention_mask
            )
            outputs['span_loss'] = span_loss
            total_loss += span_loss
        
        if total_loss > 0:
            outputs['loss'] = total_loss
        
        return outputs
    
    def _compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss with optional focal loss."""
        
        if self.config.focal_loss_alpha > 0:
            # Focal loss for handling class imbalance
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.config.focal_loss_alpha * (1 - pt) ** self.config.focal_loss_gamma * ce_loss
            return focal_loss.mean()
        else:
            # Standard cross-entropy with label smoothing
            if self.config.label_smoothing > 0:
                return self._label_smoothing_cross_entropy(logits, labels)
            else:
                return F.cross_entropy(logits, labels)
    
    def _label_smoothing_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss with label smoothing."""
        
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)
        
        eps = self.config.label_smoothing
        loss = (1 - eps) * nll_loss + eps * smooth_loss
        
        return loss.mean()
    
    def _compute_span_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute span extraction loss."""
        
        # Clamp positions to valid range
        seq_len = start_logits.size(1)
        start_positions = start_positions.clamp(0, seq_len - 1)
        end_positions = end_positions.clamp(0, seq_len - 1)
        
        # Compute losses
        start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=-1)
        end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=-1)
        
        return (start_loss + end_loss) / 2.0
    
    def is_claim(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict[str, Union[bool, float]]:
        """
        Determine if a given text contains a factual claim.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for claim classification
            
        Returns:
            Dictionary with classification result and confidence
        """
        self.eval()
        
        with torch.no_grad():
            # Process text
            inputs = self.text_processor.process_text(text)
            
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0).to(next(self.parameters()).device)
            
            # Forward pass
            outputs = self.forward(**inputs)
            
            # Get prediction
            logits = outputs['classification_logits']
            probabilities = F.softmax(logits, dim=-1)
            confidence = probabilities[0, 1].item()  # Probability of being a claim
            
            is_claim_result = confidence > threshold
            
            return {
                'is_claim': is_claim_result,
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
    
    def detect_claims(
        self,
        text: str,
        threshold: float = 0.5,
        use_sliding_window: bool = True,
        extract_spans: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Detect claims in a longer text using sliding window approach.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for claim detection
            use_sliding_window: Whether to use sliding window for long texts
            extract_spans: Whether to extract claim spans (uses config default if None)
            
        Returns:
            List of detected claims with positions and confidence scores
        """
        if extract_spans is None:
            extract_spans = self.config.enable_span_extraction
        
        self.eval()
        
        # Split text into sentences for better claim boundary detection
        sentences = self._split_into_sentences(text)
        
        claims = []
        
        with torch.no_grad():
            if use_sliding_window and len(text) > self.config.max_sequence_length:
                claims.extend(self._detect_claims_sliding_window(text, threshold, extract_spans))
            else:
                # Process entire text at once
                claims.extend(self._detect_claims_single(text, threshold, extract_spans))
        
        # Post-process and deduplicate claims
        claims = self._post_process_claims(claims)
        
        return claims
    
    def _detect_claims_single(
        self,
        text: str,
        threshold: float,
        extract_spans: bool
    ) -> List[Dict[str, Any]]:
        """Detect claims in a single text segment."""
        
        # Process text
        inputs = self.text_processor.process_text(text)
        
        # Move to device
        device = next(self.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].unsqueeze(0).to(device)
        
        # Forward pass
        outputs = self.forward(**inputs, return_attention=True)
        
        # Classification result
        logits = outputs['classification_logits']
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities[0, 1].item()
        
        claims = []
        
        if confidence > threshold:
            claim_info = {
                'text': text,
                'confidence': confidence,
                'start_char': 0,
                'end_char': len(text),
                'type': 'classification'
            }
            
            # Add span extraction if enabled
            if extract_spans and 'start_logits' in outputs:
                spans = self._extract_spans(
                    outputs['start_logits'][0],
                    outputs['end_logits'][0],
                    inputs['input_ids'][0],
                    text,
                    threshold
                )
                
                if spans:
                    # Replace with extracted spans
                    claims.extend(spans)
                else:
                    claims.append(claim_info)
            else:
                claims.append(claim_info)
        
        return claims
    
    def _detect_claims_sliding_window(
        self,
        text: str,
        threshold: float,
        extract_spans: bool
    ) -> List[Dict[str, Any]]:
        """Detect claims using sliding window approach for long texts."""
        
        claims = []
        
        # Calculate window parameters
        window_size = self.config.max_sequence_length - 2  # Account for special tokens
        step_size = window_size - self.config.sentence_overlap
        
        for start in range(0, len(text), step_size):
            end = min(start + window_size, len(text))
            window_text = text[start:end]
            
            # Detect claims in this window
            window_claims = self._detect_claims_single(window_text, threshold, extract_spans)
            
            # Adjust positions to global coordinates
            for claim in window_claims:
                claim['start_char'] += start
                claim['end_char'] += start
                claim['global_text'] = text[claim['start_char']:claim['end_char']]
            
            claims.extend(window_claims)
            
            # Break if we've reached the end
            if end >= len(text):
                break
        
        return claims
    
    def _extract_spans(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        input_ids: torch.Tensor,
        original_text: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Extract claim spans from logits."""
        
        # Convert logits to probabilities
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        
        # Find potential spans
        potential_spans = []
        
        for start_idx in range(len(start_probs)):
            if start_probs[start_idx] > threshold:
                for end_idx in range(start_idx, min(start_idx + self.config.max_span_length, len(end_probs))):
                    if end_probs[end_idx] > threshold and end_idx - start_idx >= self.config.min_span_length:
                        span_score = (start_probs[start_idx] + end_probs[end_idx]) / 2
                        potential_spans.append({
                            'start_token': start_idx,
                            'end_token': end_idx,
                            'score': span_score.item()
                        })
        
        # Convert token spans to character spans
        spans = []
        for span in potential_spans:
            try:
                # Decode token span to text
                span_tokens = input_ids[span['start_token']:span['end_token'] + 1]
                span_text = self.tokenizer.decode(span_tokens, skip_special_tokens=True)
                
                if span_text.strip():
                    # Find character positions in original text
                    start_char = original_text.find(span_text.strip())
                    if start_char >= 0:
                        end_char = start_char + len(span_text.strip())
                        
                        spans.append({
                            'text': span_text.strip(),
                            'confidence': span['score'],
                            'start_char': start_char,
                            'end_char': end_char,
                            'type': 'span_extraction'
                        })
            except Exception as e:
                self.logger.warning(f"Error extracting span: {e}")
                continue
        
        # Sort by confidence and return top spans
        spans.sort(key=lambda x: x['confidence'], reverse=True)
        return spans[:5]  # Return top 5 spans
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing."""
        
        # Simple sentence splitting (could be enhanced with NLTK/spaCy)
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _post_process_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process and deduplicate detected claims."""
        
        # Remove duplicates based on text similarity
        unique_claims = []
        
        for claim in claims:
            is_duplicate = False
            
            for existing_claim in unique_claims:
                # Check for significant overlap
                overlap_ratio = self._compute_text_overlap(claim['text'], existing_claim['text'])
                if overlap_ratio > 0.8:  # 80% overlap threshold
                    # Keep the one with higher confidence
                    if claim['confidence'] > existing_claim['confidence']:
                        unique_claims.remove(existing_claim)
                        unique_claims.append(claim)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
        
        # Sort by confidence
        unique_claims.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_claims
    
    def _compute_text_overlap(self, text1: str, text2: str) -> float:
        """Compute overlap ratio between two texts."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration."""
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save configuration
        import json
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ClaimDetectorConfig] = None
    ) -> 'ClaimDetector':
        """Load model from pretrained checkpoint."""
        
        model_path = Path(model_path)
        
        # Load configuration
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = ClaimDetectorConfig(**config_dict)
            else:
                config = ClaimDetectorConfig()
        
        # Create model instance
        model = cls(config)
        
        # Load model weights
        model_weights_path = model_path / "pytorch_model.bin"
        if model_weights_path.exists():
            state_dict = torch.load(model_weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        return {
            'model_type': 'ClaimDetector',
            'base_model': self.config.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'max_sequence_length': self.config.max_sequence_length,
            'num_classes': self.config.num_classes,
            'span_extraction_enabled': self.config.enable_span_extraction
        }


def main():
    """Example usage of ClaimDetector."""
    
    # Initialize detector
    config = ClaimDetectorConfig(
        max_sequence_length=256,
        enable_span_extraction=True
    )
    
    detector = ClaimDetector(config)
    
    print("=== Claim Detection Example ===")
    print(f"Model info: {detector.get_model_info()}")
    
    # Test texts
    test_texts = [
        "The weather is nice today.",
        "COVID-19 vaccines are 95% effective against severe illness.",
        "I think chocolate ice cream tastes better than vanilla.",
        "The Earth orbits around the Sun once every 365.25 days.",
        "This is a really long paragraph that contains multiple sentences. Some of them might be factual claims that can be verified. Others might just be opinions or observations. The goal is to identify which sentences contain verifiable factual claims."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")
        
        # Binary classification
        result = detector.is_claim(text)
        print(f"Is claim: {result['is_claim']} (confidence: {result['confidence']:.3f})")
        
        # Claim detection with spans (if enabled)
        if config.enable_span_extraction:
            claims = detector.detect_claims(text, threshold=0.3)
            if claims:
                print("Detected claims:")
                for j, claim in enumerate(claims):
                    print(f"  {j+1}. {claim['text'][:50]}... (conf: {claim['confidence']:.3f})")
            else:
                print("No claims detected")
    
    # Test model serialization
    print("\n=== Model Serialization Test ===")
    save_path = "test_claim_detector"
    detector.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    loaded_detector = ClaimDetector.from_pretrained(save_path)
    print("Model loaded successfully")
    
    # Test loaded model
    test_result = loaded_detector.is_claim("The sky is blue.")
    print(f"Loaded model test: {test_result}")


if __name__ == "__main__":
    main()
