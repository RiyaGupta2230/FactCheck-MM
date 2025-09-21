#!/usr/bin/env python3
"""
Paraphrase Quality Scoring Model

A classifier/regression model that predicts paraphrase quality scores (0-1) based on
semantic similarity, fluency, and other quality metrics. Can be trained on synthetic
labels (BLEU/ROUGE) and human ratings, and serves as a reward function for RL training.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    RobertaModel, RobertaTokenizer, RobertaConfig,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass, field
import json
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pickle

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_model import BaseMultimodalModel
from shared.utils.logging_utils import get_logger
from shared.utils.metrics import calculate_bleu_score, calculate_rouge_score


@dataclass
class QualityScorerConfig:
    """Configuration for paraphrase quality scorer."""
    
    # Model architecture
    base_model_name: str = "roberta-large"
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    num_quality_features: int = 8  # Number of explicit quality features
    
    # Input processing
    max_length: int = 256
    input_format: str = "triplet"  # "triplet" (src, tgt, para) or "pair" (src, para)
    
    # Training parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Task configuration
    task_type: str = "regression"  # "regression" or "classification"
    num_quality_levels: int = 5  # For classification (1-5 quality levels)
    
    # Quality features
    use_explicit_features: bool = True
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'bleu': 0.2,
        'rouge_l': 0.2,
        'semantic_similarity': 0.3,
        'fluency': 0.2,
        'diversity': 0.1
    })
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_probability: float = 0.1
    
    # Hardware optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


class QualityFeatureExtractor:
    """Extract explicit quality features for paraphrase evaluation."""
    
    def __init__(self, config: QualityScorerConfig):
        """
        Initialize feature extractor.
        
        Args:
            config: Quality scorer configuration
        """
        self.config = config
        self.logger = get_logger("QualityFeatureExtractor")
        
        # Initialize sentence transformer for semantic similarity
        self.similarity_model = None
        self._setup_similarity_model()
        
        # Initialize fluency model (GPT-2 for perplexity)
        self.fluency_model = None
        self.fluency_tokenizer = None
        self._setup_fluency_model()
    
    def _setup_similarity_model(self):
        """Setup sentence transformer for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded sentence transformer for semantic similarity")
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _setup_fluency_model(self):
        """Setup fluency model for perplexity computation."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            self.fluency_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.fluency_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            if self.fluency_tokenizer.pad_token is None:
                self.fluency_tokenizer.pad_token = self.fluency_tokenizer.eos_token
            
            self.fluency_model.eval()
            self.logger.info("Loaded GPT-2 for fluency assessment")
            
        except Exception as e:
            self.logger.warning(f"Failed to load fluency model: {e}")
    
    def extract_bleu_features(self, source_texts: List[str], paraphrases: List[str]) -> np.ndarray:
        """Extract BLEU score features."""
        bleu_scores = []
        for source, paraphrase in zip(source_texts, paraphrases):
            try:
                bleu = calculate_bleu_score([source], paraphrase)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
        
        return np.array(bleu_scores)
    
    def extract_rouge_features(self, source_texts: List[str], paraphrases: List[str]) -> np.ndarray:
        """Extract ROUGE-L score features."""
        rouge_scores = []
        for source, paraphrase in zip(source_texts, paraphrases):
            try:
                rouge = calculate_rouge_score(source, paraphrase)
                rouge_scores.append(rouge['rouge-l']['f'])
            except:
                rouge_scores.append(0.0)
        
        return np.array(rouge_scores)
    
    def extract_semantic_similarity(self, source_texts: List[str], paraphrases: List[str]) -> np.ndarray:
        """Extract semantic similarity features."""
        if self.similarity_model is None:
            return np.zeros(len(source_texts))
        
        try:
            source_embeddings = self.similarity_model.encode(source_texts)
            paraphrase_embeddings = self.similarity_model.encode(paraphrases)
            
            # Compute cosine similarities
            similarities = []
            for src_emb, para_emb in zip(source_embeddings, paraphrase_embeddings):
                similarity = np.dot(src_emb, para_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(para_emb))
                similarities.append(max(0.0, similarity))  # Ensure non-negative
            
            return np.array(similarities)
            
        except Exception as e:
            self.logger.warning(f"Error computing semantic similarity: {e}")
            return np.zeros(len(source_texts))
    
    def extract_fluency_features(self, texts: List[str]) -> np.ndarray:
        """Extract fluency features based on perplexity."""
        if self.fluency_model is None or self.fluency_tokenizer is None:
            return np.ones(len(texts))  # Default to neutral fluency
        
        fluency_scores = []
        
        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize text
                    inputs = self.fluency_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    # Compute perplexity
                    outputs = self.fluency_model(**inputs, labels=inputs.input_ids)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    
                    # Convert to fluency score (lower perplexity = higher fluency)
                    fluency = 1.0 / (1.0 + perplexity / 100.0)
                    fluency_scores.append(fluency)
                    
                except:
                    fluency_scores.append(0.5)  # Default neutral fluency
        
        return np.array(fluency_scores)
    
    def extract_diversity_features(self, texts: List[str], n: int = 3) -> np.ndarray:
        """Extract diversity features using distinct-n metric."""
        diversity_scores = []
        
        for text in texts:
            tokens = text.lower().split()
            
            if len(tokens) < n:
                diversity_scores.append(0.0)
                continue
            
            # Generate n-grams
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngrams.add(ngram)
            
            # Calculate distinct-n ratio
            total_ngrams = len(tokens) - n + 1
            distinct_ratio = len(ngrams) / max(total_ngrams, 1)
            diversity_scores.append(distinct_ratio)
        
        return np.array(diversity_scores)
    
    def extract_length_features(self, source_texts: List[str], paraphrases: List[str]) -> np.ndarray:
        """Extract length-based features."""
        features = []
        
        for source, paraphrase in zip(source_texts, paraphrases):
            source_len = len(source.split())
            para_len = len(paraphrases.split())
            
            # Length ratio
            length_ratio = para_len / max(source_len, 1)
            
            # Length difference (normalized)
            length_diff = abs(para_len - source_len) / max(source_len, 1)
            
            features.append([length_ratio, length_diff])
        
        return np.array(features)
    
    def extract_all_features(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        target_texts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract all quality features.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            target_texts: Target reference texts (optional)
            
        Returns:
            Feature matrix [batch_size, num_features]
        """
        features_list = []
        
        # Basic quality metrics
        bleu_features = self.extract_bleu_features(source_texts, paraphrases)
        rouge_features = self.extract_rouge_features(source_texts, paraphrases)
        semantic_features = self.extract_semantic_similarity(source_texts, paraphrases)
        fluency_features = self.extract_fluency_features(paraphrases)
        diversity_features = self.extract_diversity_features(paraphrases)
        
        # Length-based features
        length_features = self.extract_length_features(source_texts, paraphrases)
        
        # Combine all features
        features_list = [
            bleu_features.reshape(-1, 1),
            rouge_features.reshape(-1, 1),
            semantic_features.reshape(-1, 1),
            fluency_features.reshape(-1, 1),
            diversity_features.reshape(-1, 1),
            length_features  # Already 2D
        ]
        
        # Include target-based features if available
        if target_texts:
            target_bleu = self.extract_bleu_features(target_texts, paraphrases)
            target_rouge = self.extract_rouge_features(target_texts, paraphrases)
            target_semantic = self.extract_semantic_similarity(target_texts, paraphrases)
            
            features_list.extend([
                target_bleu.reshape(-1, 1),
                target_rouge.reshape(-1, 1),
                target_semantic.reshape(-1, 1)
            ])
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=1)
        
        return all_features


class QualityScorer(BaseMultimodalModel):
    """
    Paraphrase quality scoring model.
    
    Predicts quality scores (0-1) for paraphrases based on contextual representations
    and explicit quality features. Can be used as a reward function for RL training.
    """
    
    def __init__(self, config: QualityScorerConfig):
        """
        Initialize quality scorer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.logger = get_logger("QualityScorer")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        
        # Feature extractor
        self.feature_extractor = QualityFeatureExtractor(config) if config.use_explicit_features else None
        
        # Quality prediction head
        encoder_dim = self.encoder.config.hidden_size
        feature_dim = config.num_quality_features if config.use_explicit_features else 0
        
        self.quality_head = nn.Sequential(
            nn.Linear(encoder_dim + feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(
                config.hidden_dim // 2,
                config.num_quality_levels if config.task_type == "classification" else 1
            )
        )
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        
        self.logger.info(f"Initialized quality scorer with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def encode_texts(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        target_texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode text inputs using transformer encoder.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            target_texts: Target reference texts (optional)
            
        Returns:
            Encoded representations [batch_size, hidden_dim]
        """
        # Format inputs based on configuration
        if self.config.input_format == "triplet" and target_texts:
            # Format: [CLS] source [SEP] target [SEP] paraphrase [SEP]
            input_texts = []
            for src, tgt, para in zip(source_texts, target_texts, paraphrases):
                formatted = f"{src} [SEP] {tgt} [SEP] {para}"
                input_texts.append(formatted)
        else:
            # Format: [CLS] source [SEP] paraphrase [SEP]
            input_texts = []
            for src, para in zip(source_texts, paraphrases):
                formatted = f"{src} [SEP] {para}"
                input_texts.append(formatted)
        
        # Tokenize inputs
        encoding = self.tokenizer(
            input_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode with transformer
        encoder_outputs = self.encoder(**encoding)
        
        # Pool the representations (use CLS token or mean pooling)
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled_output = encoder_outputs.pooler_output
        else:
            # Mean pooling with attention mask
            hidden_states = encoder_outputs.last_hidden_state
            attention_mask = encoding['attention_mask']
            
            pooled_output = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
            pooled_output = pooled_output / attention_mask.sum(dim=1, keepdim=True)
        
        return pooled_output
    
    def forward(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        target_texts: Optional[List[str]] = None,
        quality_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for quality scoring.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            target_texts: Target reference texts (optional)
            quality_labels: Ground truth quality labels
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode texts
        text_representations = self.encode_texts(source_texts, paraphrases, target_texts)
        
        # Extract explicit features if enabled
        if self.config.use_explicit_features and self.feature_extractor:
            explicit_features = self.feature_extractor.extract_all_features(
                source_texts, paraphrases, target_texts
            )
            explicit_features = torch.tensor(explicit_features, dtype=torch.float32, device=text_representations.device)
            
            # Concatenate text representations with explicit features
            combined_representations = torch.cat([text_representations, explicit_features], dim=1)
        else:
            combined_representations = text_representations
        
        # Predict quality scores
        quality_logits = self.quality_head(combined_representations)
        
        # Compute loss if labels provided
        loss = None
        if quality_labels is not None:
            if self.config.task_type == "regression":
                loss_fn = nn.MSELoss()
                quality_scores = torch.sigmoid(quality_logits.squeeze(-1))  # Ensure [0,1] range
                loss = loss_fn(quality_scores, quality_labels.float())
            else:  # classification
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(quality_logits, quality_labels.long())
        
        return {
            'loss': loss,
            'logits': quality_logits,
            'text_representations': text_representations,
            'combined_representations': combined_representations
        }
    
    def predict_quality(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        target_texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Predict quality scores for paraphrases.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            target_texts: Target reference texts (optional)
            
        Returns:
            Quality scores [batch_size]
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(source_texts, paraphrases, target_texts)
            logits = outputs['logits']
            
            if self.config.task_type == "regression":
                # Apply sigmoid to ensure [0,1] range
                scores = torch.sigmoid(logits.squeeze(-1))
            else:
                # Convert classification probabilities to scores
                probs = F.softmax(logits, dim=-1)
                # Weighted average based on quality levels
                level_weights = torch.arange(1, self.config.num_quality_levels + 1, device=probs.device).float()
                level_weights = level_weights / self.config.num_quality_levels  # Normalize to [0,1]
                scores = torch.sum(probs * level_weights, dim=-1)
            
            return scores
    
    def compute_loss(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        quality_labels: torch.Tensor,
        target_texts: Optional[List[str]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            quality_labels: Ground truth quality labels
            target_texts: Target reference texts (optional)
            
        Returns:
            Training loss
        """
        outputs = self.forward(
            source_texts=source_texts,
            paraphrases=paraphrases,
            target_texts=target_texts,
            quality_labels=quality_labels,
            **kwargs
        )
        
        return outputs['loss']
    
    def generate_synthetic_labels(
        self,
        source_texts: List[str],
        paraphrases: List[str],
        target_texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Generate synthetic quality labels based on automatic metrics.
        
        Args:
            source_texts: Source texts
            paraphrases: Generated paraphrases
            target_texts: Target reference texts (optional)
            
        Returns:
            Synthetic quality labels [batch_size]
        """
        if not self.feature_extractor:
            # Simple length-based heuristic if no feature extractor
            labels = []
            for src, para in zip(source_texts, paraphrases):
                src_len = len(src.split())
                para_len = len(para.split())
                
                # Prefer paraphrases with similar length and non-empty content
                if para_len == 0:
                    score = 0.0
                else:
                    length_penalty = abs(src_len - para_len) / max(src_len, 1)
                    score = max(0.0, 1.0 - length_penalty)
                
                labels.append(score)
            
            return torch.tensor(labels)
        
        # Extract features for quality computation
        features = self.feature_extractor.extract_all_features(
            source_texts, paraphrases, target_texts
        )
        
        # Compute weighted combination of quality metrics
        weights = list(self.config.feature_weights.values())
        weights = weights[:features.shape[1]]  # Match available features
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        # Weighted average of normalized features
        normalized_features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0) + 1e-8)
        quality_scores = np.dot(normalized_features, weights)
        
        # Ensure [0,1] range
        quality_scores = np.clip(quality_scores, 0.0, 1.0)
        
        return torch.tensor(quality_scores)
    
    def evaluate_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate quality predictions.
        
        Args:
            predictions: Predicted quality scores
            targets: Target quality scores
            
        Returns:
            Evaluation metrics
        """
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        metrics = {}
        
        if self.config.task_type == "regression":
            # Regression metrics
            metrics['mse'] = mean_squared_error(target_np, pred_np)
            metrics['mae'] = mean_absolute_error(target_np, pred_np)
            metrics['r2'] = r2_score(target_np, pred_np)
            
            # Correlation metrics
            pearson_r, pearson_p = pearsonr(pred_np, target_np)
            spearman_r, spearman_p = spearmanr(pred_np, target_np)
            
            metrics['pearson_r'] = pearson_r
            metrics['spearman_r'] = spearman_r
        
        else:
            # Classification metrics
            from sklearn.metrics import accuracy_score, f1_score
            
            pred_classes = torch.argmax(predictions, dim=-1).cpu().numpy()
            target_classes = targets.cpu().numpy()
            
            metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
            metrics['f1'] = f1_score(target_classes, pred_classes, average='weighted')
        
        return metrics
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration.
        
        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder and tokenizer
        self.encoder.save_pretrained(save_path / "encoder")
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # Save quality head
        torch.save(self.quality_head.state_dict(), save_path / "quality_head.pt")
        
        # Save feature extractor if used
        if self.feature_extractor:
            # Save configuration (models will be reloaded on load)
            feature_config = {
                'use_explicit_features': True,
                'feature_weights': self.config.feature_weights
            }
            with open(save_path / "feature_config.json", 'w') as f:
                json.dump(feature_config, f, indent=2)
        
        # Save config
        config_path = save_path / "quality_scorer_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Quality scorer saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[QualityScorerConfig] = None
    ) -> 'QualityScorer':
        """
        Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to saved model
            config: Optional configuration override
            
        Returns:
            Loaded QualityScorer instance
        """
        model_path = Path(model_path)
        
        # Load config
        if config is None:
            config_path = model_path / "quality_scorer_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = QualityScorerConfig(**config_dict)
            else:
                config = QualityScorerConfig()
        
        # Create instance
        instance = cls(config)
        
        # Load encoder and tokenizer
        instance.encoder = AutoModel.from_pretrained(model_path / "encoder")
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
        
        # Load quality head
        quality_head_path = model_path / "quality_head.pt"
        if quality_head_path.exists():
            instance.quality_head.load_state_dict(
                torch.load(quality_head_path, map_location='cpu')
            )
        
        # Setup feature extractor if used
        feature_config_path = model_path / "feature_config.json"
        if feature_config_path.exists():
            instance.feature_extractor = QualityFeatureExtractor(config)
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'QualityScorer',
            'base_model': self.config.base_model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'task_type': self.config.task_type,
            'input_format': self.config.input_format,
            'use_explicit_features': self.config.use_explicit_features,
            'num_quality_features': self.config.num_quality_features,
            'feature_weights': self.config.feature_weights
        }


def main():
    """Example usage of quality scorer."""
    
    # Configuration
    config = QualityScorerConfig(
        base_model_name="roberta-base",  # Use base model for testing
        max_length=128,
        task_type="regression",
        use_explicit_features=True
    )
    
    # Create model
    model = QualityScorer(config)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test data
    source_texts = [
        "The weather is beautiful today.",
        "I love programming in Python.",
        "This movie is really boring."
    ]
    
    paraphrases = [
        "Today's weather is wonderful.",
        "Python programming is enjoyable for me.",
        "The film is quite dull."
    ]
    
    target_texts = [
        "It's a lovely day weather-wise.",
        "I enjoy coding with Python.",
        "This film lacks excitement."
    ]
    
    # Generate synthetic labels
    synthetic_labels = model.generate_synthetic_labels(source_texts, paraphrases, target_texts)
    print(f"Synthetic labels: {synthetic_labels}")
    
    # Test quality prediction
    quality_scores = model.predict_quality(source_texts, paraphrases, target_texts)
    print(f"Quality scores: {quality_scores}")
    
    # Test training step
    loss = model.compute_loss(source_texts, paraphrases, synthetic_labels, target_texts)
    print(f"Training loss: {loss:.4f}")
    
    # Test explicit features if available
    if model.feature_extractor:
        features = model.feature_extractor.extract_all_features(source_texts, paraphrases, target_texts)
        print(f"Explicit features shape: {features.shape}")
        print(f"Sample features: {features[0]}")
    
    # Test evaluation metrics
    # Create dummy predictions for evaluation
    dummy_predictions = torch.rand_like(synthetic_labels)
    metrics = model.evaluate_predictions(dummy_predictions, synthetic_labels)
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
