#!/usr/bin/env python3
"""
Stance Detection Model for Fact Verification

Implements RoBERTa-based stance detector that determines whether evidence
supports, opposes, or is neutral toward a given claim. Designed for 
multi-task learning integration with fact verification systems.

Example Usage:
    >>> from fact_verification.models import StanceDetector
    >>> 
    >>> # Initialize stance detector
    >>> detector = StanceDetector()
    >>> 
    >>> # Detect stance between claim and evidence
    >>> claim = "Climate change is caused by human activities"
    >>> evidence = "Scientific consensus shows greenhouse gas emissions drive climate change"
    >>> result = detector.detect_stance(claim, evidence)
    >>> print(f"Stance: {result['stance']}, Confidence: {result['confidence']:.3f}")
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
import json
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.preprocessing.text_processor import TextProcessor, TextProcessorConfig
from shared.utils.logging_utils import get_logger


@dataclass
class StanceDetectorConfig:
    """Configuration for stance detection model."""
    
    # Model architecture
    model_name: str = "roberta-base"
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # Classification
    num_classes: int = 3  # SUPPORT, AGAINST, NEUTRAL
    class_names: List[str] = field(default_factory=lambda: ["SUPPORT", "AGAINST", "NEUTRAL"])
    classifier_layers: List[int] = field(default_factory=lambda: [768, 256, 3])
    
    # Text processing
    max_sequence_length: int = 512
    max_claim_length: int = 128
    max_evidence_length: int = 384
    claim_evidence_separator: str = " </s> "
    
    # Multi-task learning
    enable_multitask: bool = False
    shared_encoder: bool = True  # Share encoder with FactVerifier
    task_specific_layers: int = 2
    
    # Contrastive learning
    enable_contrastive: bool = False
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    
    # Training parameters
    label_smoothing: float = 0.1
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    
    # Attention and interpretability
    return_attention_weights: bool = True
    cross_attention: bool = False  # Enable cross-attention between claim and evidence
    
    # Performance
    gradient_checkpointing: bool = False


class CrossAttentionModule(nn.Module):
    """Cross-attention module for claim-evidence interaction."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # Attention layers
        self.claim_query = nn.Linear(hidden_size, hidden_size)
        self.evidence_key = nn.Linear(hidden_size, hidden_size)
        self.evidence_value = nn.Linear(hidden_size, hidden_size)
        
        self.evidence_query = nn.Linear(hidden_size, hidden_size)
        self.claim_key = nn.Linear(hidden_size, hidden_size)
        self.claim_value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        claim_embeddings: torch.Tensor,  # [batch, claim_len, hidden]
        evidence_embeddings: torch.Tensor,  # [batch, evidence_len, hidden]
        claim_mask: Optional[torch.Tensor] = None,
        evidence_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention between claim and evidence.
        
        Returns:
            Claim-attended and evidence-attended representations
        """
        # Claim attending to evidence
        claim_attended = self._apply_attention(
            self.claim_query(claim_embeddings),
            self.evidence_key(evidence_embeddings),
            self.evidence_value(evidence_embeddings),
            evidence_mask
        )
        
        # Evidence attending to claim
        evidence_attended = self._apply_attention(
            self.evidence_query(evidence_embeddings),
            self.claim_key(claim_embeddings),
            self.claim_value(claim_embeddings),
            claim_mask
        )
        
        # Residual connection and layer norm
        claim_attended = self.layer_norm(claim_embeddings + self.dropout(claim_attended))
        evidence_attended = self.layer_norm(evidence_embeddings + self.dropout(evidence_attended))
        
        return claim_attended, evidence_attended
    
    def _apply_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-head attention."""
        
        def reshape_for_scores(x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        
        query_layer = reshape_for_scores(query)
        key_layer = reshape_for_scores(key)
        value_layer = reshape_for_scores(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Apply mask if provided
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class StanceDetector(BaseMultimodalModel):
    """
    RoBERTa-based stance detection model.
    
    Determines whether evidence SUPPORTS, opposes (AGAINST), or is NEUTRAL
    toward a given claim, with support for multi-task learning and 
    contrastive learning approaches.
    """
    
    def __init__(
        self, 
        config: Optional[StanceDetectorConfig] = None,
        shared_encoder: Optional[nn.Module] = None
    ):
        """
        Initialize stance detector.
        
        Args:
            config: Model configuration
            shared_encoder: Optional shared encoder for multi-task learning
        """
        super().__init__()
        
        self.config = config or StanceDetectorConfig()
        self.logger = get_logger("StanceDetector")
        
        # Initialize encoder (shared or dedicated)
        if shared_encoder is not None and self.config.shared_encoder:
            self.roberta = shared_encoder
            self.logger.info("Using shared encoder for multi-task learning")
        else:
            # Load pre-trained RoBERTa
            self.roberta_config = RobertaConfig.from_pretrained(self.config.model_name)
            self.roberta_config.output_attentions = self.config.return_attention_weights
            
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
        
        # Cross-attention module (optional)
        if self.config.cross_attention:
            self.cross_attention = CrossAttentionModule(self.config.hidden_size)
        
        # Task-specific layers for multi-task learning
        if self.config.enable_multitask:
            self.task_specific_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_size,
                    nhead=8,
                    dropout=self.config.dropout_rate
                ) for _ in range(self.config.task_specific_layers)
            ])
        
        # Classification head
        self.classifier = self._build_classifier()
        
        # Contrastive learning components
        if self.config.enable_contrastive:
            self.contrastive_head = nn.Linear(self.config.hidden_size, 256)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Class weights for loss computation
        if self.config.class_weights:
            self.register_buffer('class_weights', torch.tensor(self.config.class_weights))
        else:
            self.class_weights = None
        
        # Enable gradient checkpointing if specified
        if self.config.gradient_checkpointing:
            self.roberta.gradient_checkpointing_enable()
        
        self.logger.info(f"Initialized StanceDetector with {sum(p.numel() for p in self.parameters()):,} parameters")
    
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
        
        return nn.Sequential(*[layer for layer in layers if not isinstance(layer, nn.Identity)])
    
    def _prepare_claim_evidence_input(
        self, 
        claim: str, 
        evidence: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare claim-evidence pair input."""
        
        # Create claim-evidence pair
        combined_text = claim + self.config.claim_evidence_separator + evidence
        
        # Process with text processor
        inputs = self.text_processor.process_text(
            combined_text,
            max_length=self.config.max_sequence_length,
            truncation=True
        )
        
        return inputs
    
    def _separate_claim_evidence_embeddings(
        self,
        sequence_output: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate claim and evidence embeddings from combined sequence."""
        
        # Find separator token positions
        sep_token_id = self.tokenizer.sep_token_id
        separator_positions = []
        
        for batch_idx in range(input_ids.size(0)):
            sep_positions = (input_ids[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 1:  # Use second </s> as separator
                separator_positions.append(sep_positions[1].item())
            else:
                # Fallback: use middle of sequence
                separator_positions.append(input_ids.size(1) // 2)
        
        # Extract claim and evidence embeddings
        claim_embeddings = []
        evidence_embeddings = []
        claim_masks = []
        evidence_masks = []
        
        for batch_idx, sep_pos in enumerate(separator_positions):
            claim_emb = sequence_output[batch_idx, 1:sep_pos]  # Skip [CLS]
            evidence_emb = sequence_output[batch_idx, sep_pos+1:-1]  # Skip [SEP] and [PAD]
            
            claim_mask = attention_mask[batch_idx, 1:sep_pos]
            evidence_mask = attention_mask[batch_idx, sep_pos+1:-1]
            
            # Pad to consistent lengths
            max_claim_len = max(self.config.max_claim_length, claim_emb.size(0))
            max_evidence_len = max(self.config.max_evidence_length, evidence_emb.size(0))
            
            if claim_emb.size(0) < max_claim_len:
                padding = torch.zeros(
                    max_claim_len - claim_emb.size(0), 
                    claim_emb.size(1),
                    device=claim_emb.device,
                    dtype=claim_emb.dtype
                )
                claim_emb = torch.cat([claim_emb, padding], dim=0)
                
                mask_padding = torch.zeros(
                    max_claim_len - claim_mask.size(0),
                    device=claim_mask.device,
                    dtype=claim_mask.dtype
                )
                claim_mask = torch.cat([claim_mask, mask_padding], dim=0)
            
            if evidence_emb.size(0) < max_evidence_len:
                padding = torch.zeros(
                    max_evidence_len - evidence_emb.size(0),
                    evidence_emb.size(1),
                    device=evidence_emb.device,
                    dtype=evidence_emb.dtype
                )
                evidence_emb = torch.cat([evidence_emb, padding], dim=0)
                
                mask_padding = torch.zeros(
                    max_evidence_len - evidence_mask.size(0),
                    device=evidence_mask.device,
                    dtype=evidence_mask.dtype
                )
                evidence_mask = torch.cat([evidence_mask, mask_padding], dim=0)
            
            claim_embeddings.append(claim_emb)
            evidence_embeddings.append(evidence_emb)
            claim_masks.append(claim_mask)
            evidence_masks.append(evidence_mask)
        
        return (
            torch.stack(claim_embeddings),
            torch.stack(evidence_embeddings),
            torch.stack(claim_masks),
            torch.stack(evidence_masks)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for stance detection.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            labels: Ground truth labels (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        return_attn = return_attention if return_attention is not None else self.config.return_attention_weights
        
        # RoBERTa encoding
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attn,
            return_dict=True
        )
        
        sequence_output = roberta_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = roberta_outputs.pooler_output       # [batch_size, hidden_size]
        
        # Apply task-specific layers for multi-task learning
        if self.config.enable_multitask:
            for layer in self.task_specific_layers:
                sequence_output = layer(sequence_output.transpose(0, 1)).transpose(0, 1)
            
            # Re-compute pooled output
            pooled_output = sequence_output[:, 0]  # [CLS] token
        
        # Cross-attention between claim and evidence (optional)
        if self.config.cross_attention:
            claim_emb, evidence_emb, claim_mask, evidence_mask = self._separate_claim_evidence_embeddings(
                sequence_output, input_ids, attention_mask
            )
            
            claim_attended, evidence_attended = self.cross_attention(
                claim_emb, evidence_emb, claim_mask, evidence_mask
            )
            
            # Pool cross-attended representations
            claim_pooled = claim_attended.mean(dim=1)
            evidence_pooled = evidence_attended.mean(dim=1)
            
            # Combine representations
            pooled_output = torch.cat([claim_pooled, evidence_pooled], dim=-1)
            pooled_output = nn.Linear(pooled_output.size(-1), self.config.hidden_size).to(pooled_output.device)(pooled_output)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        outputs = {
            'logits': logits,
            'sequence_output': sequence_output,
            'pooled_output': pooled_output
        }
        
        # Contrastive learning embeddings (optional)
        if self.config.enable_contrastive:
            contrastive_embeddings = self.contrastive_head(pooled_output)
            contrastive_embeddings = F.normalize(contrastive_embeddings, p=2, dim=-1)
            outputs['contrastive_embeddings'] = contrastive_embeddings
        
        # Include attention weights if requested
        if return_attn and roberta_outputs.attentions is not None:
            # Average attention across heads for interpretability
            attention_weights = torch.stack([
                layer_attn.mean(dim=1) for layer_attn in roberta_outputs.attentions[-4:]
            ], dim=1)
            outputs['attention_weights'] = attention_weights
        
        # Compute loss if labels provided
        if labels is not None:
            classification_loss = self._compute_classification_loss(logits, labels)
            outputs['classification_loss'] = classification_loss
            
            total_loss = classification_loss
            
            # Add contrastive loss if enabled
            if self.config.enable_contrastive and 'contrastive_embeddings' in outputs:
                contrastive_loss = self._compute_contrastive_loss(
                    outputs['contrastive_embeddings'], labels
                )
                outputs['contrastive_loss'] = contrastive_loss
                total_loss += contrastive_loss
            
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
            ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.config.focal_loss_alpha * (1 - pt) ** self.config.focal_loss_gamma * ce_loss
            return focal_loss.mean()
        else:
            # Standard cross-entropy with label smoothing
            if self.config.label_smoothing > 0:
                return self._label_smoothing_cross_entropy(logits, labels)
            else:
                return F.cross_entropy(logits, labels, weight=self.class_weights)
    
    def _label_smoothing_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss with label smoothing."""
        
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, weight=self.class_weights, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)
        
        eps = self.config.label_smoothing
        loss = (1 - eps) * nll_loss + eps * smooth_loss
        
        return loss.mean()
    
    def _compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for similar stance learning."""
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.config.contrastive_temperature
        
        # Create positive/negative masks
        batch_size = labels.size(0)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        negative_mask = 1 - positive_mask
        
        # Remove self-similarities
        positive_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        positive_sum = torch.sum(exp_sim * positive_mask, dim=1)
        negative_sum = torch.sum(exp_sim * negative_mask, dim=1)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        
        return loss.mean()
    
    def detect_stance(
        self,
        claim: str,
        evidence: str,
        return_attention: bool = False,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Detect stance between claim and evidence.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            return_attention: Whether to return attention weights
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with stance detection results
        """
        self.eval()
        
        with torch.no_grad():
            # Prepare inputs
            inputs = self._prepare_claim_evidence_input(claim, evidence)
            
            # Move to device
            device = next(self.parameters()).device
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = self.forward(**inputs, return_attention=return_attention)
            
            # Get predictions
            logits = outputs['logits']
            predicted_class = torch.argmax(logits, dim=-1).item()
            predicted_stance = self.config.class_names[predicted_class]
            
            result = {
                'stance': predicted_stance,
                'class_id': predicted_class,
                'claim': claim,
                'evidence': evidence
            }
            
            # Add probabilities if requested
            if return_probabilities:
                probabilities = F.softmax(logits, dim=-1)[0]
                result['probabilities'] = {
                    name: prob.item() 
                    for name, prob in zip(self.config.class_names, probabilities)
                }
                result['confidence'] = probabilities[predicted_class].item()
            
            # Add attention weights if requested
            if return_attention and 'attention_weights' in outputs:
                result['attention_weights'] = outputs['attention_weights'][0].cpu().numpy()
            
            return result
    
    def detect_stance_batch(
        self,
        claims: List[str],
        evidence_list: List[str],
        return_attention: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect stance for multiple claim-evidence pairs.
        
        Args:
            claims: List of claim texts
            evidence_list: List of evidence texts
            return_attention: Whether to return attention weights
            
        Returns:
            List of stance detection results
        """
        if len(claims) != len(evidence_list):
            raise ValueError("Number of claims must match number of evidence texts")
        
        self.eval()
        
        results = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            batch_size = 8
            
            for i in range(0, len(claims), batch_size):
                batch_claims = claims[i:i + batch_size]
                batch_evidence = evidence_list[i:i + batch_size]
                
                # Prepare batch inputs
                batch_inputs = []
                for claim, evidence in zip(batch_claims, batch_evidence):
                    inputs = self._prepare_claim_evidence_input(claim, evidence)
                    batch_inputs.append(inputs)
                
                # Stack batch
                batch_input_ids = torch.stack([inp['input_ids'] for inp in batch_inputs])
                batch_attention_mask = torch.stack([inp['attention_mask'] for inp in batch_inputs])
                
                # Move to device
                device = next(self.parameters()).device
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                
                # Forward pass
                outputs = self.forward(
                    batch_input_ids, 
                    batch_attention_mask,
                    return_attention=return_attention
                )
                
                # Process results
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                
                for j, (claim, evidence) in enumerate(zip(batch_claims, batch_evidence)):
                    predicted_class = predictions[j].item()
                    predicted_stance = self.config.class_names[predicted_class]
                    
                    result = {
                        'stance': predicted_stance,
                        'class_id': predicted_class,
                        'claim': claim,
                        'evidence': evidence,
                        'probabilities': {
                            name: probabilities[j, k].item()
                            for k, name in enumerate(self.config.class_names)
                        },
                        'confidence': probabilities[j, predicted_class].item()
                    }
                    
                    # Add attention if requested
                    if return_attention and 'attention_weights' in outputs:
                        result['attention_weights'] = outputs['attention_weights'][j].cpu().numpy()
                    
                    results.append(result)
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration."""
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"StanceDetector saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[StanceDetectorConfig] = None
    ) -> 'StanceDetector':
        """Load model from pretrained checkpoint."""
        
        model_path = Path(model_path)
        
        # Load configuration
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = StanceDetectorConfig(**config_dict)
            else:
                config = StanceDetectorConfig()
        
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
            'model_type': 'StanceDetector',
            'base_model': self.config.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'max_sequence_length': self.config.max_sequence_length,
            'cross_attention_enabled': self.config.cross_attention,
            'multitask_enabled': self.config.enable_multitask,
            'contrastive_enabled': self.config.enable_contrastive
        }


def main():
    """Example usage of StanceDetector."""
    
    # Initialize stance detector
    config = StanceDetectorConfig(
        max_sequence_length=256,
        cross_attention=False,
        enable_contrastive=False
    )
    
    detector = StanceDetector(config)
    
    print("=== Stance Detection Example ===")
    print(f"Model info: {detector.get_model_info()}")
    
    # Test cases
    test_cases = [
        {
            "claim": "Climate change is caused by human activities",
            "evidence": "Scientific consensus shows greenhouse gas emissions drive climate change",
            "expected": "SUPPORT"
        },
        {
            "claim": "Vaccines are dangerous and cause autism",
            "evidence": "Multiple large-scale studies have found no link between vaccines and autism",
            "expected": "AGAINST"
        },
        {
            "claim": "Regular exercise improves mental health",
            "evidence": "Weather patterns affect seasonal mood changes in some individuals",
            "expected": "NEUTRAL"
        },
        {
            "claim": "The Earth is flat",
            "evidence": "Satellite images clearly show Earth is spherical",
            "expected": "AGAINST"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Claim: {case['claim']}")
        print(f"Evidence: {case['evidence']}")
        print(f"Expected stance: {case['expected']}")
        
        # Detect stance
        result = detector.detect_stance(case['claim'], case['evidence'])
        
        print(f"\nDetection Result:")
        print(f"  Stance: {result['stance']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print("  Probabilities:")
        for stance, prob in result['probabilities'].items():
            print(f"    {stance}: {prob:.3f}")
    
    # Test batch detection
    print("\n=== Batch Stance Detection ===")
    claims = [case['claim'] for case in test_cases]
    evidence = [case['evidence'] for case in test_cases]
    
    batch_results = detector.detect_stance_batch(claims, evidence)
    
    for i, result in enumerate(batch_results):
        expected = test_cases[i]['expected']
        print(f"{i+1}. {result['claim'][:40]}... -> {result['stance']} "
              f"({result['confidence']:.3f}) [Expected: {expected}]")
    
    # Test model serialization
    print("\n=== Model Serialization Test ===")
    save_path = "test_stance_detector"
    detector.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    loaded_detector = StanceDetector.from_pretrained(save_path)
    print("Model loaded successfully")
    
    # Test loaded model
    test_result = loaded_detector.detect_stance("Test claim", "Test evidence")
    print(f"Loaded model test: {test_result['stance']}")


if __name__ == "__main__":
    main()
