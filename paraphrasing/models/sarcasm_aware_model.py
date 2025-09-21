#!/usr/bin/env python3
"""
Sarcasm-Aware Paraphrase Generation Model

Extends T5/BART models by conditioning on sarcasm embeddings from the sarcasm_detection module.
Enables context-aware paraphrasing that preserves or modifies sarcastic intent.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, T5Config,
    BartForConditionalGeneration, BartTokenizer, BartConfig,
    GenerationConfig
)
from dataclasses import dataclass, field
import json
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.utils.logging_utils import get_logger
from .t5_paraphraser import T5Paraphraser, T5ParaphraserConfig
from .bart_paraphraser import BARTParaphraser, BARTParaphraserConfig


@dataclass
class SarcasmAwareConfig:
    """Configuration for sarcasm-aware paraphraser."""
    
    # Base model configuration
    base_model_type: str = "t5"  # "t5" or "bart"
    base_model_name: str = "t5-base"
    
    # Sarcasm conditioning
    sarcasm_embedding_dim: int = 256
    sarcasm_fusion_type: str = "concatenate"  # "concatenate", "attention", "gate"
    sarcasm_detection_model_path: Optional[str] = None
    freeze_sarcasm_detector: bool = True
    
    # Training parameters
    max_input_length: int = 128
    max_target_length: int = 128
    learning_rate: float = 3e-5
    sarcasm_loss_weight: float = 0.1
    
    # Generation parameters
    num_beams: int = 4
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    
    # Sarcasm control
    enable_sarcasm_control: bool = True
    sarcasm_control_tokens: List[str] = field(default_factory=lambda: [
        "<sarcastic>", "<non_sarcastic>", "<preserve_sarcasm>"
    ])
    
    # Hardware optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


class SarcasmFusionLayer(nn.Module):
    """Layer for fusing sarcasm representations with text encodings."""
    
    def __init__(
        self,
        text_dim: int,
        sarcasm_dim: int,
        fusion_type: str = "concatenate",
        dropout: float = 0.1
    ):
        """
        Initialize sarcasm fusion layer.
        
        Args:
            text_dim: Text encoding dimension
            sarcasm_dim: Sarcasm embedding dimension
            fusion_type: Type of fusion ("concatenate", "attention", "gate")
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.text_dim = text_dim
        self.sarcasm_dim = sarcasm_dim
        
        if fusion_type == "concatenate":
            self.output_dim = text_dim + sarcasm_dim
            self.projection = nn.Linear(self.output_dim, text_dim)
        
        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.sarcasm_projection = nn.Linear(sarcasm_dim, text_dim)
            self.output_dim = text_dim
        
        elif fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(text_dim + sarcasm_dim, text_dim),
                nn.Sigmoid()
            )
            self.sarcasm_projection = nn.Linear(sarcasm_dim, text_dim)
            self.output_dim = text_dim
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        text_encodings: torch.Tensor,
        sarcasm_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse text encodings with sarcasm embeddings.
        
        Args:
            text_encodings: Text encoder outputs [batch, seq_len, text_dim]
            sarcasm_embeddings: Sarcasm embeddings [batch, sarcasm_dim]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Fused representations [batch, seq_len, output_dim]
        """
        batch_size, seq_len, _ = text_encodings.shape
        
        if self.fusion_type == "concatenate":
            # Expand sarcasm embeddings to sequence length
            sarcasm_expanded = sarcasm_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate along feature dimension
            fused = torch.cat([text_encodings, sarcasm_expanded], dim=-1)
            
            # Project back to original dimension
            output = self.projection(fused)
        
        elif self.fusion_type == "attention":
            # Project sarcasm embeddings to text dimension
            sarcasm_projected = self.sarcasm_projection(sarcasm_embeddings)
            sarcasm_expanded = sarcasm_projected.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Apply cross-attention
            output, _ = self.attention(
                query=text_encodings,
                key=sarcasm_expanded,
                value=sarcasm_expanded,
                key_padding_mask=None
            )
            
            # Residual connection
            output = output + text_encodings
        
        elif self.fusion_type == "gate":
            # Project sarcasm embeddings
            sarcasm_projected = self.sarcasm_projection(sarcasm_embeddings)
            sarcasm_expanded = sarcasm_projected.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Compute gate
            gate_input = torch.cat([text_encodings, sarcasm_expanded], dim=-1)
            gate_weights = self.gate(gate_input)
            
            # Apply gating
            output = gate_weights * text_encodings + (1 - gate_weights) * sarcasm_expanded
        
        # Apply normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class SarcasmAwareParaphraser(BaseMultimodalModel):
    """
    Sarcasm-aware paraphrase generation model.
    
    Extends T5/BART with sarcasm conditioning for context-aware paraphrasing
    that can preserve, modify, or neutralize sarcastic content.
    """
    
    def __init__(self, config: SarcasmAwareConfig):
        """
        Initialize sarcasm-aware paraphraser.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger("SarcasmAwareParaphraser")
        
        # Initialize base paraphraser
        self.base_paraphraser = self._create_base_model()
        
        # Load sarcasm detection model
        self.sarcasm_detector = self._load_sarcasm_detector()
        
        # Create sarcasm fusion layer
        self.sarcasm_fusion = SarcasmFusionLayer(
            text_dim=self.base_paraphraser.config.d_model,
            sarcasm_dim=config.sarcasm_embedding_dim,
            fusion_type=config.sarcasm_fusion_type
        )
        
        # Add sarcasm control tokens
        self._add_sarcasm_control_tokens()
        
        # Sarcasm classification head for auxiliary loss
        self.sarcasm_classifier = nn.Sequential(
            nn.Linear(self.base_paraphraser.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary sarcasm classification
        )
        
        self.logger.info(f"Initialized sarcasm-aware paraphraser with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _create_base_model(self) -> Union[T5Paraphraser, BARTParaphraser]:
        """Create base paraphraser model."""
        
        if self.config.base_model_type.lower() == "t5":
            base_config = T5ParaphraserConfig(
                model_name=self.config.base_model_name,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                mixed_precision=self.config.mixed_precision
            )
            return T5Paraphraser(base_config)
        
        elif self.config.base_model_type.lower() == "bart":
            base_config = BARTParaphraserConfig(
                model_name=self.config.base_model_name,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                mixed_precision=self.config.mixed_precision
            )
            return BARTParaphraser(base_config)
        
        else:
            raise ValueError(f"Unknown base model type: {self.config.base_model_type}")
    
    def _load_sarcasm_detector(self) -> Optional[nn.Module]:
        """Load pretrained sarcasm detection model."""
        
        if not self.config.sarcasm_detection_model_path:
            self.logger.warning("No sarcasm detection model path provided")
            return None
        
        try:
            # Import sarcasm detection model
            from sarcasm_detection.models import MultimodalSarcasmModel
            
            # Load pretrained model
            checkpoint = torch.load(self.config.sarcasm_detection_model_path, map_location='cpu')
            
            # Create model with loaded config
            sarcasm_config = checkpoint.get('config', {})
            sarcasm_model = MultimodalSarcasmModel(sarcasm_config)
            sarcasm_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Freeze if specified
            if self.config.freeze_sarcasm_detector:
                for param in sarcasm_model.parameters():
                    param.requires_grad = False
            
            self.logger.info("Loaded pretrained sarcasm detection model")
            return sarcasm_model
            
        except Exception as e:
            self.logger.warning(f"Failed to load sarcasm detection model: {e}")
            return None
    
    def _add_sarcasm_control_tokens(self):
        """Add sarcasm control tokens to tokenizer."""
        
        tokenizer = self.base_paraphraser.tokenizer
        
        special_tokens = {
            'additional_special_tokens': tokenizer.additional_special_tokens + self.config.sarcasm_control_tokens
        }
        
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            # Resize model embeddings
            self.base_paraphraser.model.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"Added {num_added} sarcasm control tokens")
    
    def extract_sarcasm_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract sarcasm features from input text.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            
        Returns:
            Sarcasm embeddings [batch, sarcasm_dim]
        """
        if self.sarcasm_detector is None:
            # Return zero embeddings if no sarcasm detector
            batch_size = input_ids.shape[0]
            return torch.zeros(batch_size, self.config.sarcasm_embedding_dim, device=input_ids.device)
        
        with torch.set_grad_enabled(not self.config.freeze_sarcasm_detector):
            # Create input for sarcasm detector
            sarcasm_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Get sarcasm embeddings
            sarcasm_outputs = self.sarcasm_detector(**sarcasm_inputs)
            
            # Extract embeddings (assuming pooled output)
            if 'pooled_output' in sarcasm_outputs:
                embeddings = sarcasm_outputs['pooled_output']
            elif 'hidden_states' in sarcasm_outputs:
                # Pool the hidden states
                hidden_states = sarcasm_outputs['hidden_states']
                embeddings = torch.mean(hidden_states, dim=1)
            else:
                # Fallback to mean pooling of last hidden state
                embeddings = torch.mean(sarcasm_outputs['logits'], dim=1)
            
            # Project to desired dimension
            if embeddings.shape[-1] != self.config.sarcasm_embedding_dim:
                if not hasattr(self, 'sarcasm_projection'):
                    self.sarcasm_projection = nn.Linear(
                        embeddings.shape[-1],
                        self.config.sarcasm_embedding_dim
                    ).to(embeddings.device)
                embeddings = self.sarcasm_projection(embeddings)
            
            return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sarcasm_labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sarcasm-aware paraphraser.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            labels: Target labels for paraphrasing
            sarcasm_labels: Sarcasm labels for auxiliary loss
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            
        Returns:
            Dictionary containing model outputs
        """
        # Extract sarcasm features
        sarcasm_embeddings = self.extract_sarcasm_features(input_ids, attention_mask)
        
        # Get base model encoder outputs
        encoder_outputs = self.base_paraphraser.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Fuse sarcasm information with encoder outputs
        fused_encodings = self.sarcasm_fusion(
            text_encodings=encoder_outputs.last_hidden_state,
            sarcasm_embeddings=sarcasm_embeddings,
            attention_mask=attention_mask
        )
        
        # Update encoder outputs with fused representations
        encoder_outputs.last_hidden_state = fused_encodings
        
        # Run decoder
        decoder_outputs = self.base_paraphraser.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_encodings,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        
        # Compute main paraphrasing loss
        paraphrase_loss = None
        if labels is not None:
            logits = self.base_paraphraser.model.lm_head(decoder_outputs.last_hidden_state)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            paraphrase_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Compute auxiliary sarcasm loss
        sarcasm_loss = None
        if sarcasm_labels is not None:
            # Pool encoder representations for classification
            pooled_output = torch.mean(fused_encodings, dim=1)
            sarcasm_logits = self.sarcasm_classifier(pooled_output)
            
            sarcasm_loss_fn = nn.CrossEntropyLoss()
            sarcasm_loss = sarcasm_loss_fn(sarcasm_logits, sarcasm_labels)
        
        # Combine losses
        total_loss = None
        if paraphrase_loss is not None:
            total_loss = paraphrase_loss
            if sarcasm_loss is not None:
                total_loss += self.config.sarcasm_loss_weight * sarcasm_loss
        
        return {
            'loss': total_loss,
            'paraphrase_loss': paraphrase_loss,
            'sarcasm_loss': sarcasm_loss,
            'logits': decoder_outputs.last_hidden_state,
            'hidden_states': fused_encodings,
            'sarcasm_embeddings': sarcasm_embeddings
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sarcasm_control: Optional[str] = None,
        generation_strategy: str = "beam_search",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate sarcasm-aware paraphrases.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            sarcasm_control: Sarcasm control directive
            generation_strategy: Generation strategy
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        self.eval()
        
        with torch.no_grad():
            # Add sarcasm control token if specified
            if sarcasm_control and sarcasm_control in self.config.sarcasm_control_tokens:
                control_token_id = self.base_paraphraser.tokenizer.encode(
                    sarcasm_control, add_special_tokens=False
                )[0]
                
                # Prepend control token to input
                control_ids = torch.full(
                    (input_ids.shape[0], 1),
                    control_token_id,
                    device=input_ids.device
                )
                input_ids = torch.cat([control_ids, input_ids], dim=1)
                
                # Extend attention mask
                control_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device
                )
                attention_mask = torch.cat([control_mask, attention_mask], dim=1)
            
            # Extract sarcasm features
            sarcasm_embeddings = self.extract_sarcasm_features(input_ids, attention_mask)
            
            # Get encoder outputs
            encoder_outputs = self.base_paraphraser.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Fuse with sarcasm information
            fused_encodings = self.sarcasm_fusion(
                text_encodings=encoder_outputs.last_hidden_state,
                sarcasm_embeddings=sarcasm_embeddings,
                attention_mask=attention_mask
            )
            
            # Update encoder outputs
            encoder_outputs.last_hidden_state = fused_encodings
            
            # Generate using base model with modified encoder outputs
            generated_ids = self.base_paraphraser.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **generation_kwargs
            )
            
            # Decode generated sequences
            generated_sequences = []
            for seq in generated_ids:
                decoded = self.base_paraphraser.tokenizer.decode(seq, skip_special_tokens=True)
                generated_sequences.append(decoded)
            
            return {
                'generated_sequences': generated_sequences,
                'generated_ids': generated_ids,
                'sarcasm_embeddings': sarcasm_embeddings,
                'sarcasm_control': sarcasm_control
            }
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        sarcasm_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            labels: Target labels
            sarcasm_labels: Sarcasm labels for auxiliary loss
            
        Returns:
            Training loss
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sarcasm_labels=sarcasm_labels,
            **kwargs
        )
        
        return outputs['loss']
    
    def prepare_inputs(
        self,
        source_texts: List[str],
        target_texts: Optional[List[str]] = None,
        sarcasm_labels: Optional[List[int]] = None,
        sarcasm_control: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training or generation.
        
        Args:
            source_texts: Source texts to paraphrase
            target_texts: Target paraphrases (for training)
            sarcasm_labels: Sarcasm labels for auxiliary loss
            sarcasm_control: Sarcasm control directive
            
        Returns:
            Dictionary of tokenized inputs
        """
        # Add sarcasm control prefix if specified
        if sarcasm_control and sarcasm_control in self.config.sarcasm_control_tokens:
            source_texts = [sarcasm_control + " " + text for text in source_texts]
        
        # Use base model's prepare_inputs
        inputs = self.base_paraphraser.prepare_inputs(source_texts, target_texts)
        
        # Add sarcasm labels if provided
        if sarcasm_labels:
            inputs['sarcasm_labels'] = torch.tensor(sarcasm_labels, dtype=torch.long)
        
        return inputs
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration.
        
        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save base model
        base_model_path = save_path / "base_model"
        self.base_paraphraser.save_pretrained(str(base_model_path))
        
        # Save sarcasm-specific components
        sarcasm_components = {
            'sarcasm_fusion': self.sarcasm_fusion.state_dict(),
            'sarcasm_classifier': self.sarcasm_classifier.state_dict()
        }
        
        if hasattr(self, 'sarcasm_projection'):
            sarcasm_components['sarcasm_projection'] = self.sarcasm_projection.state_dict()
        
        torch.save(sarcasm_components, save_path / "sarcasm_components.pt")
        
        # Save config
        config_path = save_path / "sarcasm_aware_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Sarcasm-aware model saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[SarcasmAwareConfig] = None
    ) -> 'SarcasmAwareParaphraser':
        """
        Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to saved model
            config: Optional configuration override
            
        Returns:
            Loaded SarcasmAwareParaphraser instance
        """
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "sarcasm_aware_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = SarcasmAwareConfig(**config_dict)
            else:
                config = SarcasmAwareConfig()
        
        # Create instance
        instance = cls(config)
        
        # Load base model
        base_model_path = model_path / "base_model"
        if instance.config.base_model_type.lower() == "t5":
            instance.base_paraphraser = T5Paraphraser.from_pretrained(str(base_model_path))
        else:
            instance.base_paraphraser = BARTParaphraser.from_pretrained(str(base_model_path))
        
        # Load sarcasm components
        sarcasm_components_path = model_path / "sarcasm_components.pt"
        if sarcasm_components_path.exists():
            components = torch.load(sarcasm_components_path, map_location='cpu')
            
            instance.sarcasm_fusion.load_state_dict(components['sarcasm_fusion'])
            instance.sarcasm_classifier.load_state_dict(components['sarcasm_classifier'])
            
            if 'sarcasm_projection' in components:
                # Recreate projection layer
                proj_state = components['sarcasm_projection']
                in_features = list(proj_state.values())[0].shape[1]
                instance.sarcasm_projection = nn.Linear(in_features, instance.config.sarcasm_embedding_dim)
                instance.sarcasm_projection.load_state_dict(proj_state)
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        base_info = self.base_paraphraser.get_model_info()
        
        return {
            **base_info,
            'model_type': 'SarcasmAwareParaphraser',
            'base_model_type': self.config.base_model_type,
            'sarcasm_fusion_type': self.config.sarcasm_fusion_type,
            'sarcasm_embedding_dim': self.config.sarcasm_embedding_dim,
            'has_sarcasm_detector': self.sarcasm_detector is not None,
            'sarcasm_control_tokens': self.config.sarcasm_control_tokens
        }


def main():
    """Example usage of sarcasm-aware paraphraser."""
    
    # Configuration
    config = SarcasmAwareConfig(
        base_model_type="t5",
        base_model_name="t5-small",
        sarcasm_embedding_dim=128,
        sarcasm_fusion_type="concatenate",
        max_input_length=64,
        max_target_length=64
    )
    
    # Create model
    model = SarcasmAwareParaphraser(config)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test text preparation
    source_texts = [
        "Oh great, another Monday morning meeting.",
        "I just love working overtime on weekends."
    ]
    target_texts = [
        "Another Monday morning meeting, how wonderful.",
        "Working overtime on weekends is really enjoyable."
    ]
    sarcasm_labels = [1, 1]  # Both are sarcastic
    
    # Prepare inputs
    inputs = model.prepare_inputs(source_texts, target_texts, sarcasm_labels)
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items() if torch.is_tensor(v)]}")
    
    # Test forward pass
    outputs = model(**inputs)
    print(f"Total loss: {outputs['loss']:.4f}")
    if outputs['paraphrase_loss'] is not None:
        print(f"Paraphrase loss: {outputs['paraphrase_loss']:.4f}")
    if outputs['sarcasm_loss'] is not None:
        print(f"Sarcasm loss: {outputs['sarcasm_loss']:.4f}")
    
    # Test generation with sarcasm control
    for control in [None, "<sarcastic>", "<non_sarcastic>"]:
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            sarcasm_control=control,
            max_length=64
        )
        
        print(f"\nGeneration with control '{control}':")
        for i, (source, generated_text) in enumerate(zip(source_texts, generated['generated_sequences'])):
            print(f"{i+1}. Source: {source}")
            print(f"   Generated: {generated_text}")


if __name__ == "__main__":
    main()
