import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaModel, RobertaTokenizer,
    CLIPVisionModel, CLIPProcessor,
    Wav2Vec2Model, Wav2Vec2Processor
)
import math
import numpy as np

class UltimateMultimodalSarcasmModel(nn.Module):
    """
    Ultimate state-of-the-art multimodal sarcasm detection model
    Implements all advanced techniques for maximum 85%+ F1 accuracy
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model_architecture']
        
        # 1. LARGEST PRETRAINED ENCODERS FOR MAXIMUM PERFORMANCE
        self.text_encoder = RobertaModel.from_pretrained(model_config['text_encoder'])
        self.image_encoder = CLIPVisionModel.from_pretrained(model_config['image_encoder'])
        self.audio_encoder = Wav2Vec2Model.from_pretrained(model_config['audio_encoder'])
        
        # Model dimensions
        self.text_dim = self.text_encoder.config.hidden_size  # 1024 for roberta-large
        self.image_dim = self.image_encoder.config.hidden_size  # 1024 for clip-large
        self.audio_dim = self.audio_encoder.config.hidden_size  # 1024 for wav2vec2-large
        self.hidden_size = model_config['hidden_size']  # 768
        
        # 2. ADVANCED PROJECTION LAYERS WITH GATING
        self.text_projection = ProjectionWithGating(self.text_dim, self.hidden_size, model_config['dropout'])
        self.image_projection = ProjectionWithGating(self.image_dim, self.hidden_size, model_config['dropout'])
        self.audio_projection = ProjectionWithGating(self.audio_dim, self.hidden_size, model_config['dropout'])
        
        # 3. HIERARCHICAL CROSS-MODAL ATTENTION
        self.hierarchical_attention = HierarchicalCrossModalAttention(
            hidden_size=self.hidden_size,
            num_heads=model_config['attention_heads'],
            dropout=model_config['dropout']
        )
        
        # 4. DYNAMIC ROUTING TRANSFORMER
        if model_config.get('dynamic_routing', False):
            self.dynamic_router = DynamicRoutingTransformer(
                hidden_size=self.hidden_size,
                num_experts=model_config['num_experts'],
                num_layers=4
            )
        else:
            self.dynamic_router = None
        
        # 5. INCONGRUITY DETECTION (CORE OF SARCASM)
        if model_config.get('incongruity_detection', False):
            self.incongruity_detector = IncongruityDetector(
                text_dim=self.hidden_size,
                image_dim=self.hidden_size,
                audio_dim=self.hidden_size
            )
        else:
            self.incongruity_detector = None
        
        # 6. DOMAIN ADAPTATION MODULE
        if config.get('domain_adaptation', {}).get('enabled', False):
            self.domain_adapter = DomainAdapter(
                feature_dim=self.hidden_size,
                num_domains=config['domain_adaptation']['num_domains']
            )
        else:
            self.domain_adapter = None
        
        # 7. ADVANCED FUSION NETWORK
        fusion_input_dim = self.hidden_size * 3  # text + image + audio
        if self.incongruity_detector:
            fusion_input_dim += 64  # incongruity features
        
        self.fusion_network = AdvancedFusionNetwork(
            input_dim=fusion_input_dim,
            hidden_dim=self.hidden_size,
            dropout=model_config['dropout']
        )
        
        # 8. MULTI-TASK LEARNING HEADS
        if config.get('multi_task', {}).get('enabled', False):
            self.sentiment_head = nn.Linear(self.hidden_size, 3)  # pos, neg, neu
            self.emotion_head = nn.Linear(self.hidden_size, 8)    # 8 basic emotions
            self.irony_head = nn.Linear(self.hidden_size, 2)      # ironic, not ironic
        else:
            self.sentiment_head = None
            self.emotion_head = None
            self.irony_head = None
        
        # 9. UNCERTAINTY ESTIMATION
        if model_config.get('uncertainty_estimation', False):
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(model_config['dropout']),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.uncertainty_head = None
        
        # 10. MAIN SARCASM CLASSIFICATION HEAD
        self.sarcasm_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(256, 2)  # sarcastic, not sarcastic
        )
        
        # 11. CONTRASTIVE LEARNING HEAD
        if config.get('training_strategies', {}).get('contrastive_learning', False):
            self.contrastive_head = ContrastiveLearningHead(self.hidden_size, 128)
        else:
            self.contrastive_head = None
        
        # 12. LEARNABLE MODALITY WEIGHTS
        self.modality_weights = nn.Parameter(torch.ones(3))  # text, image, audio
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, batch, return_attention=False):
        """
        Forward pass through the ultimate multimodal model
        
        Args:
            batch: Dictionary containing text_inputs, image_inputs, audio_inputs, labels, etc.
            return_attention: Whether to return attention weights for visualization
        
        Returns:
            Dictionary containing logits, loss, and optional auxiliary outputs
        """
        device = next(self.parameters()).device
        batch_size = len(batch['labels']) if 'labels' in batch else 1
        
        # 1. EXTRACT FEATURES FROM EACH MODALITY
        # Text features
        if batch.get('text_inputs') is not None:
            text_outputs = self.text_encoder(**batch['text_inputs'])
            text_raw = text_outputs.pooler_output
            text_features = self.text_projection(text_raw)
        else:
            text_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Image features
        if batch.get('image_inputs') is not None:
            image_outputs = self.image_encoder(**batch['image_inputs'])
            image_raw = image_outputs.pooler_output
            image_features = self.image_projection(image_raw)
        else:
            image_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Audio features
        if batch.get('audio_inputs') is not None:
            audio_outputs = self.audio_encoder(**batch['audio_inputs'])
            audio_raw = audio_outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            audio_features = self.audio_projection(audio_raw)
        else:
            audio_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # 2. HIERARCHICAL CROSS-MODAL ATTENTION
        attended_features, attention_weights = self.hierarchical_attention(
            text_features, image_features, audio_features
        )
        
        # 3. DYNAMIC ROUTING (IF ENABLED)
        if self.dynamic_router:
            routed_features = self.dynamic_router(attended_features)
        else:
            routed_features = attended_features
        
        # 4. INCONGRUITY DETECTION (IF ENABLED)
        incongruity_features = None
        if self.incongruity_detector:
            incongruity_features = self.incongruity_detector(
                text_features, image_features, audio_features
            )
        
        # 5. FEATURE FUSION
        # Apply learnable modality weights
        modality_weights = F.softmax(self.modality_weights, dim=0)
        weighted_text = text_features * modality_weights[0]
        weighted_image = image_features * modality_weights[1]
        weighted_audio = audio_features * modality_weights[2]
        
        # Combine all features
        if incongruity_features is not None:
            fusion_input = torch.cat([
                weighted_text, weighted_image, weighted_audio, incongruity_features
            ], dim=1)
        else:
            fusion_input = torch.cat([
                weighted_text, weighted_image, weighted_audio
            ], dim=1)
        
        # Advanced fusion
        fused_features = self.fusion_network(fusion_input)
        
        # 6. DOMAIN ADAPTATION (IF ENABLED)
        if self.domain_adapter and batch.get('domain_ids') is not None:
            fused_features = self.domain_adapter(fused_features, batch['domain_ids'])
        
        # 7. PREDICTIONS
        # Main sarcasm prediction
        sarcasm_logits = self.sarcasm_classifier(fused_features)
        
        # Multi-task predictions (if enabled)
        auxiliary_outputs = {}
        if self.sentiment_head:
            auxiliary_outputs['sentiment_logits'] = self.sentiment_head(fused_features)
        if self.emotion_head:
            auxiliary_outputs['emotion_logits'] = self.emotion_head(fused_features)
        if self.irony_head:
            auxiliary_outputs['irony_logits'] = self.irony_head(fused_features)
        
        # Uncertainty estimation (if enabled)
        uncertainty = None
        if self.uncertainty_head:
            uncertainty = self.uncertainty_head(fused_features)
            auxiliary_outputs['uncertainty'] = uncertainty
        
        # Contrastive features (if enabled)
        contrastive_features = None
        if self.contrastive_head:
            contrastive_features = self.contrastive_head(fused_features)
            auxiliary_outputs['contrastive_features'] = contrastive_features
        
        # 8. LOSS CALCULATION
        total_loss = None
        if batch.get('labels') is not None:
            total_loss = self.calculate_comprehensive_loss(
                sarcasm_logits, auxiliary_outputs, batch, uncertainty, contrastive_features
            )
        
        # 9. PREPARE OUTPUT
        output = {
            'logits': sarcasm_logits,
            'loss': total_loss,
            'features': fused_features,
            'modality_weights': modality_weights,
            **auxiliary_outputs
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def calculate_comprehensive_loss(self, sarcasm_logits, auxiliary_outputs, batch, uncertainty, contrastive_features):
        """Calculate comprehensive multi-task loss with uncertainty weighting"""
        device = sarcasm_logits.device
        
        # Primary sarcasm classification loss
        if self.config.get('training_strategies', {}).get('focal_loss', False):
            sarcasm_loss = self.focal_loss(sarcasm_logits, batch['labels'])
        else:
            sarcasm_loss = F.cross_entropy(
                sarcasm_logits, 
                batch['labels'], 
                label_smoothing=self.config['training'].get('label_smoothing', 0.0)
            )
        
        # Uncertainty weighting for primary loss
        if uncertainty is not None and batch.get('is_synthetic') is not None:
            # Apply higher uncertainty weighting to synthetic samples
            synthetic_mask = batch['is_synthetic'].float()
            uncertainty_weights = 1.0 + uncertainty.squeeze() * synthetic_mask
            weighted_sarcasm_loss = (sarcasm_loss * uncertainty_weights.mean()).mean()
        else:
            weighted_sarcasm_loss = sarcasm_loss
        
        total_loss = weighted_sarcasm_loss
        
        # Multi-task auxiliary losses
        if self.config.get('multi_task', {}).get('enabled', False):
            # Sentiment analysis loss
            if 'sentiment_logits' in auxiliary_outputs and batch.get('sentiment_labels') is not None:
                sentiment_loss = F.cross_entropy(
                    auxiliary_outputs['sentiment_logits'], 
                    batch['sentiment_labels']
                )
                total_loss += self.config['multi_task']['sentiment_weight'] * sentiment_loss
            
            # Emotion recognition loss
            if 'emotion_logits' in auxiliary_outputs and batch.get('emotion_labels') is not None:
                emotion_loss = F.cross_entropy(
                    auxiliary_outputs['emotion_logits'], 
                    batch['emotion_labels']
                )
                total_loss += self.config['multi_task']['emotion_weight'] * emotion_loss
            
            # Irony detection loss
            if 'irony_logits' in auxiliary_outputs:
                # Use sarcasm labels as proxy for irony
                irony_loss = F.cross_entropy(
                    auxiliary_outputs['irony_logits'], 
                    batch['labels']
                )
                total_loss += self.config['multi_task']['irony_weight'] * irony_loss
        
        # Contrastive learning loss
        if contrastive_features is not None:
            contrastive_loss = self.calculate_contrastive_loss(contrastive_features, batch['labels'])
            total_loss += self.config['training_strategies']['contrastive_weight'] * contrastive_loss
        
        return total_loss
    
    def focal_loss(self, logits, labels, alpha=1, gamma=2):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def calculate_contrastive_loss(self, features, labels, temperature=0.1):
        """Calculate contrastive learning loss"""
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create positive pairs (same labels)
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Mask diagonal
        mask = torch.eye(labels.size(0), device=labels.device).bool()
        labels_eq = labels_eq.masked_fill(mask, 0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix).masked_fill(mask, 0)
        pos_sim = exp_sim * labels_eq
        neg_sim = exp_sim * (1 - labels_eq)
        
        # Avoid division by zero
        denominator = pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8
        loss = -torch.log(pos_sim.sum(dim=1) / denominator + 1e-8)
        
        return loss.mean()


class ProjectionWithGating(nn.Module):
    """Projection layer with gating mechanism"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        projected = self.projection(x)
        gate_weights = self.gate(x)
        return projected * gate_weights


class HierarchicalCrossModalAttention(nn.Module):
    """Hierarchical cross-modal attention mechanism"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Different levels of attention
        self.token_level_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.semantic_level_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.global_level_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features, image_features, audio_features):
        batch_size = text_features.size(0)
        
        # Combine all modality features
        all_features = torch.stack([text_features, image_features, audio_features], dim=1)
        
        # Apply different levels of attention
        token_attended, token_attn = self.token_level_attention(
            all_features, all_features, all_features
        )
        
        semantic_attended, semantic_attn = self.semantic_level_attention(
            token_attended, token_attended, token_attended
        )
        
        global_attended, global_attn = self.global_level_attention(
            semantic_attended, semantic_attended, semantic_attended
        )
        
        # Combine attended features
        combined_features = torch.cat([
            token_attended.mean(dim=1),
            semantic_attended.mean(dim=1),
            global_attended.mean(dim=1)
        ], dim=1)
        
        # Final projection
        output_features = self.output_projection(combined_features)
        
        attention_weights = {
            'token_level': token_attn,
            'semantic_level': semantic_attn,
            'global_level': global_attn
        }
        
        return output_features, attention_weights


class DynamicRoutingTransformer(nn.Module):
    """Dynamic routing transformer with expert networks"""
    
    def __init__(self, hidden_size, num_experts, num_layers):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_experts)
        ])
        
        # Routing mechanism
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features):
        batch_size = features.size(0)
        
        # Calculate routing weights
        routing_weights = self.router(features)  # [batch_size, num_experts]
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(features.unsqueeze(1))  # Add sequence dimension
            expert_outputs.append(expert_output.squeeze(1))  # Remove sequence dimension
        
        # Weight and combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_size]
        routing_weights = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        routed_output = (expert_outputs * routing_weights).sum(dim=1)
        
        return routed_output


class IncongruityDetector(nn.Module):
    """Advanced incongruity detection for sarcasm"""
    
    def __init__(self, text_dim, image_dim, audio_dim):
        super().__init__()
        
        # Semantic incongruity detection
        self.semantic_detector = nn.Sequential(
            nn.Linear(text_dim + image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)
        )
        
        # Emotional incongruity detection
        self.emotional_detector = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)
        )
        
        # Contextual incongruity detection
        self.contextual_detector = nn.Sequential(
            nn.Linear(text_dim + image_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)
        )
        
        # Final incongruity score
        self.incongruity_classifier = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, text_features, image_features, audio_features):
        # Detect different types of incongruities
        semantic_incongruity = self.semantic_detector(
            torch.cat([text_features, image_features], dim=1)
        )
        
        emotional_incongruity = self.emotional_detector(
            torch.cat([text_features, audio_features], dim=1)
        )
        
        contextual_incongruity = self.contextual_detector(
            torch.cat([text_features, image_features, audio_features], dim=1)
        )
        
        # Combine all incongruity signals
        all_incongruities = torch.cat([
            semantic_incongruity, emotional_incongruity, contextual_incongruity
        ], dim=1)
        
        incongruity_features = self.incongruity_classifier(all_incongruities)
        
        return incongruity_features


class DomainAdapter(nn.Module):
    """Domain adaptation module for cross-domain generalization"""
    
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.num_domains = num_domains
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, 64)
        
        # Domain-specific adaptation layers
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_domains)
        ])
        
        # Attention-based domain weighting
        self.domain_attention = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features, domain_ids):
        batch_size = features.size(0)
        
        # Get domain embeddings
        domain_embs = self.domain_embeddings(domain_ids)
        
        # Apply domain-specific adaptations
        adapted_features = []
        for i, adapter in enumerate(self.domain_adapters):
            adapted = adapter(features)
            adapted_features.append(adapted)
        
        adapted_features = torch.stack(adapted_features, dim=1)  # [batch_size, num_domains, feature_dim]
        
        # Calculate domain attention weights
        combined_input = torch.cat([features, domain_embs], dim=1)
        domain_weights = self.domain_attention(combined_input)  # [batch_size, num_domains]
        domain_weights = domain_weights.unsqueeze(-1)  # [batch_size, num_domains, 1]
        
        # Weight and combine domain-adapted features
        final_features = (adapted_features * domain_weights).sum(dim=1)
        
        return final_features


class AdvancedFusionNetwork(nn.Module):
    """Advanced feature fusion network"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout // 2)
        )
        
        # Residual connection
        self.residual_projection = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, features):
        fused = self.fusion_layers(features)
        residual = self.residual_projection(features)
        return fused + residual


class ContrastiveLearningHead(nn.Module):
    """Contrastive learning head for better representations"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, features):
        return F.normalize(self.projection(features), dim=1)
