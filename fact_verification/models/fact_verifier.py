#!/usr/bin/env python3
"""
Fact Verification Model

Implements RoBERTa-based fact verifier that determines the veracity of claims
given retrieved evidence, with support for interpretable attention weights
and optional RAG-style verification approaches.

Example Usage:
    >>> from fact_verification.models import FactVerifier
    >>> 
    >>> # Initialize verifier
    >>> verifier = FactVerifier()
    >>> 
    >>> # Verify claim with evidence
    >>> claim = "COVID-19 vaccines are 95% effective"
    >>> evidence = ["Clinical trials show vaccine efficacy of 90-95%"]
    >>> result = verifier.verify(claim, evidence)
    >>> print(f"Verdict: {result['label']}, Confidence: {result['confidence']:.3f}")
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
class FactVerifierConfig:
    """Configuration for fact verification model."""
    
    # Model architecture
    model_name: str = "roberta-base"
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # Classification
    num_classes: int = 3  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    class_names: List[str] = field(default_factory=lambda: ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
    classifier_layers: List[int] = field(default_factory=lambda: [768, 512, 3])
    
    # Text processing
    max_sequence_length: int = 512
    max_claim_length: int = 128
    max_evidence_length: int = 384
    evidence_separator: str = " [SEP] "
    claim_evidence_separator: str = " </s> "
    
    # Evidence aggregation
    max_evidence_pieces: int = 5
    evidence_aggregation: str = "concatenate"  # "concatenate", "attention", "hierarchical"
    
    # Attention and interpretability
    return_attention_weights: bool = True
    attention_heads_to_save: int = 4  # Number of attention heads to save
    
    # RAG-style extensions
    enable_rag: bool = False
    rag_retrieval_dim: int = 256
    rag_generation_max_length: int = 64
    
    # Training parameters
    label_smoothing: float = 0.1
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    
    # Knowledge base integration
    use_knowledge_base: bool = False
    knowledge_base_path: Optional[str] = None
    
    # Performance
    gradient_checkpointing: bool = False


class AttentionAggregator(nn.Module):
    """Attention-based evidence aggregation module."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        claim_embedding: torch.Tensor, 
        evidence_embeddings: torch.Tensor,
        evidence_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate evidence using attention mechanism.
        
        Args:
            claim_embedding: [batch_size, hidden_size]
            evidence_embeddings: [batch_size, num_evidence, hidden_size]
            evidence_mask: [batch_size, num_evidence]
            
        Returns:
            Aggregated evidence embedding and attention weights
        """
        batch_size, num_evidence, hidden_size = evidence_embeddings.shape
        
        # Reshape for multi-head attention
        def reshape_for_scores(x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        
        # Use claim as query, evidence as key/value
        query_layer = reshape_for_scores(self.query(claim_embedding.unsqueeze(1)))  # [batch, heads, 1, head_size]
        key_layer = reshape_for_scores(self.key(evidence_embeddings))  # [batch, heads, num_ev, head_size]
        value_layer = reshape_for_scores(self.value(evidence_embeddings))  # [batch, heads, num_ev, head_size]
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Apply evidence mask if provided
        if evidence_mask is not None:
            attention_mask = evidence_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, num_ev]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch, heads, 1, head_size]
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch, 1, hidden_size]
        
        return context_layer.squeeze(1), attention_probs.squeeze(2)  # [batch, hidden_size], [batch, heads, num_ev]


class FactVerifier(BaseMultimodalModel):
    """
    RoBERTa-based fact verification model.
    
    Determines whether evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO
    for a given claim, with interpretable attention mechanisms.
    """
    
    def __init__(self, config: Optional[FactVerifierConfig] = None):
        """
        Initialize fact verifier.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or FactVerifierConfig()
        self.logger = get_logger("FactVerifier")
        
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
        
        # Classification head
        self.classifier = self._build_classifier()
        
        # Evidence aggregation components
        if self.config.evidence_aggregation == "attention":
            self.evidence_aggregator = AttentionAggregator(self.config.hidden_size)
        elif self.config.evidence_aggregation == "hierarchical":
            self.evidence_encoder = RobertaModel.from_pretrained(self.config.model_name)
            self.claim_evidence_attention = AttentionAggregator(self.config.hidden_size)
        
        # RAG components (optional)
        if self.config.enable_rag:
            self.rag_retrieval_head = nn.Linear(self.config.hidden_size, self.config.rag_retrieval_dim)
            self.rag_generator = nn.Linear(self.config.hidden_size, self.tokenizer.vocab_size)
        
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
        
        self.logger.info(f"Initialized FactVerifier with {sum(p.numel() for p in self.parameters()):,} parameters")
    
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
        evidence_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Prepare concatenated claim-evidence input."""
        
        # Limit number of evidence pieces
        evidence_list = evidence_list[:self.config.max_evidence_pieces]
        
        if evidence_list:
            # Concatenate evidence pieces
            combined_evidence = self.config.evidence_separator.join(evidence_list)
            
            # Create claim-evidence pair
            combined_text = claim + self.config.claim_evidence_separator + combined_evidence
        else:
            combined_text = claim
        
        # Process with text processor
        inputs = self.text_processor.process_text(
            combined_text,
            max_length=self.config.max_sequence_length,
            truncation=True
        )
        
        return inputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fact verification.
        
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
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        outputs = {
            'logits': logits,
            'sequence_output': sequence_output,
            'pooled_output': pooled_output
        }
        
        # Include attention weights if requested
        if return_attn and roberta_outputs.attentions is not None:
            # Save subset of attention heads for interpretability
            attention_subset = []
            for layer_attention in roberta_outputs.attentions[-4:]:  # Last 4 layers
                # Average across heads for simplicity
                avg_attention = layer_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
                attention_subset.append(avg_attention)
            
            outputs['attention_weights'] = torch.stack(attention_subset, dim=1)  # [batch_size, 4, seq_len, seq_len]
        
        # RAG components (if enabled)
        if self.config.enable_rag:
            rag_embeddings = self.rag_retrieval_head(pooled_output)
            rag_logits = self.rag_generator(pooled_output)
            
            outputs.update({
                'rag_embeddings': rag_embeddings,
                'rag_logits': rag_logits
            })
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_loss(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute classification loss with optional class weighting."""
        
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
    
    def verify(
        self,
        claim: str,
        evidence: Union[str, List[str]],
        return_attention: bool = False,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Verify a claim against evidence.
        
        Args:
            claim: Claim text to verify
            evidence: Evidence text(s)
            return_attention: Whether to return attention weights
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with verification results
        """
        self.eval()
        
        # Prepare evidence list
        if isinstance(evidence, str):
            evidence_list = [evidence]
        else:
            evidence_list = evidence
        
        with torch.no_grad():
            # Prepare inputs
            inputs = self._prepare_claim_evidence_input(claim, evidence_list)
            
            # Move to device
            device = next(self.parameters()).device
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = self.forward(**inputs, return_attention=return_attention)
            
            # Get predictions
            logits = outputs['logits']
            predicted_class = torch.argmax(logits, dim=-1).item()
            predicted_label = self.config.class_names[predicted_class]
            
            result = {
                'label': predicted_label,
                'class_id': predicted_class,
                'claim': claim,
                'evidence': evidence_list
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
    
    def verify_batch(
        self,
        claims: List[str],
        evidence_lists: List[List[str]],
        return_attention: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple claims in batch.
        
        Args:
            claims: List of claim texts
            evidence_lists: List of evidence lists (one per claim)
            return_attention: Whether to return attention weights
            
        Returns:
            List of verification results
        """
        if len(claims) != len(evidence_lists):
            raise ValueError("Number of claims must match number of evidence lists")
        
        self.eval()
        
        results = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            batch_size = 8  # Adjust based on memory constraints
            
            for i in range(0, len(claims), batch_size):
                batch_claims = claims[i:i + batch_size]
                batch_evidence = evidence_lists[i:i + batch_size]
                
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
                    predicted_label = self.config.class_names[predicted_class]
                    
                    result = {
                        'label': predicted_label,
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
    
    def extract_important_tokens(
        self,
        claim: str,
        evidence: List[str],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract most important tokens based on attention weights.
        
        Args:
            claim: Claim text
            evidence: Evidence texts
            top_k: Number of top tokens to extract
            
        Returns:
            Dictionary with important tokens and their attention scores
        """
        result = self.verify(claim, evidence, return_attention=True)
        
        if 'attention_weights' not in result:
            return {'claim_tokens': [], 'evidence_tokens': []}
        
        # Get tokens
        inputs = self._prepare_claim_evidence_input(claim, evidence)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'])
        
        # Average attention across layers and positions
        attention_weights = result['attention_weights']  # [num_layers, seq_len, seq_len]
        
        # Focus on attention to [CLS] token (position 0)
        cls_attention = attention_weights[:, 0, :].mean(axis=0)  # [seq_len]
        
        # Get top-k tokens
        token_importance = [(tokens[i], cls_attention[i]) for i in range(len(tokens))]
        token_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Separate claim and evidence tokens (rough heuristic)
        separator_idx = -1
        for i, (token, _) in enumerate(token_importance):
            if '</s>' in token:
                separator_idx = i
                break
        
        if separator_idx > 0:
            claim_tokens = token_importance[:separator_idx][:top_k]
            evidence_tokens = token_importance[separator_idx:][:top_k]
        else:
            claim_tokens = token_importance[:top_k]
            evidence_tokens = []
        
        return {
            'claim_tokens': claim_tokens,
            'evidence_tokens': evidence_tokens
        }
    
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
        
        self.logger.info(f"FactVerifier saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[FactVerifierConfig] = None
    ) -> 'FactVerifier':
        """Load model from pretrained checkpoint."""
        
        model_path = Path(model_path)
        
        # Load configuration
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = FactVerifierConfig(**config_dict)
            else:
                config = FactVerifierConfig()
        
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
            'model_type': 'FactVerifier',
            'base_model': self.config.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'max_sequence_length': self.config.max_sequence_length,
            'evidence_aggregation': self.config.evidence_aggregation,
            'rag_enabled': self.config.enable_rag
        }


def main():
    """Example usage of FactVerifier."""
    
    # Initialize verifier
    config = FactVerifierConfig(
        max_sequence_length=256,
        return_attention_weights=True
    )
    
    verifier = FactVerifier(config)
    
    print("=== Fact Verification Example ===")
    print(f"Model info: {verifier.get_model_info()}")
    
    # Test cases
    test_cases = [
        {
            "claim": "COVID-19 vaccines are highly effective",
            "evidence": [
                "Clinical trials show COVID-19 vaccines have 90-95% efficacy",
                "Real-world data confirms vaccine effectiveness against severe illness"
            ]
        },
        {
            "claim": "The Earth is flat",
            "evidence": [
                "Satellite images show Earth is spherical",
                "Ships disappear over the horizon due to Earth's curvature"
            ]
        },
        {
            "claim": "Chocolate improves cognitive function",
            "evidence": [
                "Some studies suggest dark chocolate may have cognitive benefits",
                "Research on chocolate and brain function shows mixed results"
            ]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Claim: {case['claim']}")
        print("Evidence:")
        for j, ev in enumerate(case['evidence']):
            print(f"  {j+1}. {ev}")
        
        # Verify claim
        result = verifier.verify(case['claim'], case['evidence'])
        
        print(f"\nVerification Result:")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print("  Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"    {label}: {prob:.3f}")
        
        # Extract important tokens
        important_tokens = verifier.extract_important_tokens(case['claim'], case['evidence'])
        if important_tokens['claim_tokens']:
            print("  Important claim tokens:")
            for token, score in important_tokens['claim_tokens'][:5]:
                print(f"    {token}: {score:.3f}")
    
    # Test batch verification
    print("\n=== Batch Verification ===")
    claims = [case['claim'] for case in test_cases]
    evidence_lists = [case['evidence'] for case in test_cases]
    
    batch_results = verifier.verify_batch(claims, evidence_lists)
    
    for i, result in enumerate(batch_results):
        print(f"{i+1}. {result['claim'][:50]}... -> {result['label']} ({result['confidence']:.3f})")
    
    # Test model serialization
    print("\n=== Model Serialization Test ===")
    save_path = "test_fact_verifier"
    verifier.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    loaded_verifier = FactVerifier.from_pretrained(save_path)
    print("Model loaded successfully")
    
    # Test loaded model
    test_result = loaded_verifier.verify("Test claim", ["Test evidence"])
    print(f"Loaded model test: {test_result['label']}")


if __name__ == "__main__":
    main()
