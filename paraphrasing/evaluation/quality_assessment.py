#!/usr/bin/env python3
"""
Automated Quality Assessment for Paraphrase Evaluation

Integrates trained quality scorer models with rule-based fallback scoring to provide
comprehensive quality assessment for paraphrase candidates. Supports batch scoring,
threshold-based filtering, and RL reward signal generation.

Example Usage:
    >>> from paraphrasing.evaluation import QualityAssessor
    >>> 
    >>> # Initialize with trained quality scorer
    >>> assessor = QualityAssessor(quality_model_path="models/quality_scorer")
    >>> 
    >>> # Score paraphrase candidates
    >>> sources = ["The weather is nice", "I love coding"]
    >>> candidates = ["It's a beautiful day", "Programming is enjoyable"]
    >>> references = ["Today is lovely", "I enjoy programming"]
    >>> scores = assessor.score_batch(candidates, sources, references)
    >>> print(f"Quality scores: {scores}")
    >>> 
    >>> # Filter high-quality candidates
    >>> filtered = assessor.filter_by_quality(candidates, sources, min_score=0.7)
    >>> print(f"High-quality candidates: {len(filtered)}")
    >>>
    >>> # Generate RL rewards
    >>> rewards = assessor.create_rl_rewards(candidates, sources, references)
    >>> print(f"RL rewards: {rewards}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
import numpy as np
import pandas as pd
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger
from shared.utils.metrics import calculate_bleu_score, calculate_rouge_score

# Import quality scorer (with fallback)
try:
    from paraphrasing.models.quality_scorer import QualityScorer, QualityScorerConfig
    QUALITY_SCORER_AVAILABLE = True
except ImportError:
    QUALITY_SCORER_AVAILABLE = False

# Import evaluation metrics
try:
    from .generation_metrics import GenerationMetrics, MetricsConfig
    from .semantic_similarity import SemanticSimilarity, SimilarityConfig
    EVAL_METRICS_AVAILABLE = True
except ImportError:
    EVAL_METRICS_AVAILABLE = False

# Optional dependencies for advanced scoring
try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class QualityAssessmentConfig:
    """Configuration for quality assessment."""
    
    # Quality scorer configuration
    quality_model_path: Optional[str] = None
    use_quality_model: bool = True
    quality_model_batch_size: int = 32
    
    # Fallback scoring weights
    fallback_weights: Dict[str, float] = field(default_factory=lambda: {
        'bleu': 0.25,
        'rouge_l': 0.25,
        'bertscore': 0.3,
        'semantic_similarity': 0.15,
        'diversity_penalty': 0.05
    })
    
    # Rule-based scoring parameters
    min_length_ratio: float = 0.3  # Minimum length ratio (candidate/source)
    max_length_ratio: float = 3.0  # Maximum length ratio
    diversity_ngram: int = 2  # N-gram for diversity calculation
    repetition_penalty: float = 0.1  # Penalty for repetitive text
    
    # Semantic similarity configuration
    similarity_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.3  # Minimum semantic similarity
    
    # Quality thresholds
    min_quality_score: float = 0.0
    max_quality_score: float = 1.0
    default_quality_score: float = 0.5
    
    # Filtering and reward configuration
    filter_threshold: float = 0.6
    reward_scaling: float = 1.0
    reward_shift: float = 0.0  # Shift rewards (e.g., -0.5 to center around 0)
    
    # Export configuration
    save_detailed_scores: bool = True
    include_individual_metrics: bool = True


class QualityAssessor:
    """
    Automated quality assessment for paraphrase evaluation.
    
    Provides comprehensive quality scoring using trained models with rule-based
    fallbacks, batch processing, and integration with RL training pipelines.
    """
    
    def __init__(self, config: Optional[QualityAssessmentConfig] = None):
        """
        Initialize quality assessor.
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config or QualityAssessmentConfig()
        self.logger = get_logger("QualityAssessor")
        
        # Load quality scorer model
        self.quality_scorer = self._load_quality_scorer()
        
        # Initialize fallback evaluation tools
        self.generation_metrics = None
        self.semantic_similarity = None
        
        if EVAL_METRICS_AVAILABLE:
            metrics_config = MetricsConfig(
                compute_diversity=True,
                rouge_types=["rougeL"],
                bertscore_model="microsoft/deberta-base-mnli"
            )
            self.generation_metrics = GenerationMetrics(metrics_config)
            
            similarity_config = SimilarityConfig(
                model_name=self.config.similarity_model,
                similarity_metric="cosine"
            )
            self.semantic_similarity = SemanticSimilarity(similarity_config)
        
        # Initialize sentence transformer for fallback
        self.sentence_transformer = self._load_sentence_transformer()
        
        # Scoring statistics
        self.score_history = []
        
        self.logger.info("Initialized QualityAssessor")
        self.logger.info(f"Quality scorer available: {self.quality_scorer is not None}")
        self.logger.info(f"Fallback metrics available: {EVAL_METRICS_AVAILABLE}")
    
    def _load_quality_scorer(self) -> Optional[QualityScorer]:
        """Load trained quality scorer model."""
        
        if not self.config.use_quality_model or not self.config.quality_model_path:
            return None
        
        if not QUALITY_SCORER_AVAILABLE:
            self.logger.warning("Quality scorer module not available")
            return None
        
        try:
            model_path = Path(self.config.quality_model_path)
            if not model_path.exists():
                self.logger.warning(f"Quality scorer path not found: {model_path}")
                return None
            
            quality_scorer = QualityScorer.from_pretrained(str(model_path))
            quality_scorer.eval()  # Set to evaluation mode
            
            self.logger.info(f"Loaded quality scorer from: {model_path}")
            return quality_scorer
            
        except Exception as e:
            self.logger.error(f"Failed to load quality scorer: {e}")
            return None
    
    def _load_sentence_transformer(self) -> Optional[SentenceTransformer]:
        """Load sentence transformer for semantic similarity."""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        try:
            model = SentenceTransformer(self.config.similarity_model)
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            return None
    
    def score_batch(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """
        Score a batch of paraphrase candidates.
        
        Args:
            candidates: List of candidate paraphrases
            sources: List of source texts
            references: Optional list of reference paraphrases
            
        Returns:
            List of quality scores (0-1 range)
        """
        if len(candidates) != len(sources):
            raise ValueError("Candidates and sources must have same length")
        
        if references and len(references) != len(sources):
            raise ValueError("References and sources must have same length")
        
        # Use quality scorer if available
        if self.quality_scorer:
            try:
                scores = self._score_with_quality_model(candidates, sources, references)
                self.score_history.extend(scores)
                return scores
            except Exception as e:
                self.logger.warning(f"Quality scorer failed, using fallback: {e}")
        
        # Fallback to rule-based scoring
        scores = self._score_with_fallback_rules(candidates, sources, references)
        self.score_history.extend(scores)
        return scores
    
    def _score_with_quality_model(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """Score using trained quality model."""
        
        # Process in batches for memory efficiency
        batch_size = self.config.quality_model_batch_size
        all_scores = []
        
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            batch_sources = sources[i:i + batch_size]
            batch_references = references[i:i + batch_size] if references else None
            
            # Get quality scores
            with torch.no_grad():
                batch_scores = self.quality_scorer.predict_quality(
                    source_texts=batch_sources,
                    paraphrases=batch_candidates,
                    target_texts=batch_references
                )
            
            all_scores.extend(batch_scores.cpu().tolist())
        
        # Ensure scores are in valid range
        clipped_scores = [
            max(self.config.min_quality_score, 
                min(self.config.max_quality_score, score))
            for score in all_scores
        ]
        
        return clipped_scores
    
    def _score_with_fallback_rules(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """Score using rule-based fallback methods."""
        
        scores = []
        weights = self.config.fallback_weights
        
        for i, (candidate, source) in enumerate(zip(candidates, sources)):
            reference = references[i] if references else None
            
            individual_scores = {}
            
            # BLEU score
            try:
                target_for_bleu = reference if reference else source
                bleu = calculate_bleu_score([target_for_bleu], candidate)
                individual_scores['bleu'] = bleu
            except:
                individual_scores['bleu'] = 0.0
            
            # ROUGE-L score
            try:
                target_for_rouge = reference if reference else source
                rouge = calculate_rouge_score(target_for_rouge, candidate)
                individual_scores['rouge_l'] = rouge.get('rouge-l', {}).get('f', 0.0)
            except:
                individual_scores['rouge_l'] = 0.0
            
            # BERTScore
            individual_scores['bertscore'] = self._compute_bertscore_fallback(
                candidate, reference if reference else source
            )
            
            # Semantic similarity
            individual_scores['semantic_similarity'] = self._compute_semantic_similarity_fallback(
                candidate, source
            )
            
            # Diversity/length penalties
            diversity_penalty = self._compute_diversity_penalty(candidate)
            length_penalty = self._compute_length_penalty(candidate, source)
            repetition_penalty = self._compute_repetition_penalty(candidate)
            
            individual_scores['diversity_penalty'] = 1.0 - (
                diversity_penalty + length_penalty + repetition_penalty
            ) / 3.0
            
            # Weighted combination
            total_score = sum(
                weights.get(key, 0) * score
                for key, score in individual_scores.items()
            )
            
            # Normalize and clip
            normalized_score = max(0.0, min(1.0, total_score))
            scores.append(normalized_score)
        
        return scores
    
    def _compute_bertscore_fallback(self, candidate: str, reference: str) -> float:
        """Compute BERTScore with fallback."""
        
        if BERTSCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score.score([candidate], [reference], lang="en", verbose=False)
                return F1.item()
            except:
                pass
        
        # Simple word overlap fallback
        candidate_words = set(candidate.lower().split())
        reference_words = set(reference.lower().split())
        
        if not candidate_words or not reference_words:
            return 0.0
        
        intersection = len(candidate_words & reference_words)
        union = len(candidate_words | reference_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_semantic_similarity_fallback(self, candidate: str, source: str) -> float:
        """Compute semantic similarity with fallback."""
        
        if self.semantic_similarity:
            try:
                similarities = self.semantic_similarity.compute_similarity([source], [candidate])
                return similarities[0] if similarities else 0.0
            except:
                pass
        
        if self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode([source, candidate])
                similarity = F.cosine_similarity(
                    torch.tensor(embeddings[0]).unsqueeze(0),
                    torch.tensor(embeddings[1]).unsqueeze(0)
                ).item()
                return max(0.0, similarity)
            except:
                pass
        
        # Word overlap fallback
        return self._compute_bertscore_fallback(candidate, source)
    
    def _compute_diversity_penalty(self, text: str) -> float:
        """Compute penalty for lack of diversity."""
        
        words = text.lower().split()
        if len(words) < self.config.diversity_ngram:
            return 1.0  # Maximum penalty for very short texts
        
        # Compute distinct n-grams
        ngrams = []
        for i in range(len(words) - self.config.diversity_ngram + 1):
            ngram = tuple(words[i:i + self.config.diversity_ngram])
            ngrams.append(ngram)
        
        if not ngrams:
            return 1.0
        
        unique_ngrams = len(set(ngrams))
        diversity_ratio = unique_ngrams / len(ngrams)
        
        return 1.0 - diversity_ratio  # Convert to penalty
    
    def _compute_length_penalty(self, candidate: str, source: str) -> float:
        """Compute penalty for inappropriate length."""
        
        candidate_words = len(candidate.split())
        source_words = len(source.split())
        
        if source_words == 0:
            return 1.0 if candidate_words > 0 else 0.0
        
        length_ratio = candidate_words / source_words
        
        if length_ratio < self.config.min_length_ratio or length_ratio > self.config.max_length_ratio:
            # Penalty for being too short or too long
            if length_ratio < self.config.min_length_ratio:
                penalty = (self.config.min_length_ratio - length_ratio) / self.config.min_length_ratio
            else:
                penalty = (length_ratio - self.config.max_length_ratio) / self.config.max_length_ratio
            
            return min(1.0, penalty)
        
        return 0.0  # No penalty for appropriate length
    
    def _compute_repetition_penalty(self, text: str) -> float:
        """Compute penalty for repetitive text."""
        
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        # Count consecutive repetitions
        repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetitions += 1
        
        repetition_ratio = repetitions / (len(words) - 1)
        
        return repetition_ratio * self.config.repetition_penalty
    
    def score_single(
        self,
        candidate: str,
        source: str,
        reference: Optional[str] = None
    ) -> float:
        """Score a single paraphrase candidate."""
        
        scores = self.score_batch([candidate], [source], [reference] if reference else None)
        return scores[0]
    
    def filter_by_quality(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, List]:
        """
        Filter candidates by quality threshold.
        
        Args:
            candidates: List of candidate paraphrases
            sources: List of source texts
            references: Optional list of reference paraphrases
            min_score: Minimum quality score (uses config default if None)
            
        Returns:
            Dictionary with filtered candidates and their indices
        """
        threshold = min_score if min_score is not None else self.config.filter_threshold
        
        scores = self.score_batch(candidates, sources, references)
        
        filtered_data = {
            'candidates': [],
            'sources': [],
            'references': [],
            'scores': [],
            'original_indices': []
        }
        
        for i, (candidate, source, score) in enumerate(zip(candidates, sources, scores)):
            if score >= threshold:
                filtered_data['candidates'].append(candidate)
                filtered_data['sources'].append(source)
                filtered_data['scores'].append(score)
                filtered_data['original_indices'].append(i)
                
                if references:
                    filtered_data['references'].append(references[i])
        
        if not references:
            del filtered_data['references']
        
        self.logger.info(f"Filtered {len(filtered_data['candidates'])}/{len(candidates)} candidates (threshold: {threshold})")
        
        return filtered_data
    
    def create_rl_rewards(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """
        Create RL reward signals from quality scores.
        
        Args:
            candidates: List of candidate paraphrases
            sources: List of source texts
            references: Optional list of reference paraphrases
            
        Returns:
            List of reward values for RL training
        """
        # Get quality scores
        scores = self.score_batch(candidates, sources, references)
        
        # Transform scores to rewards
        rewards = []
        for score in scores:
            # Apply scaling and shifting
            reward = (score * self.config.reward_scaling) + self.config.reward_shift
            rewards.append(reward)
        
        return rewards
    
    def save_detailed_scores(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None,
        output_file: str = "quality_scores.csv",
        include_individual_metrics: bool = None
    ) -> str:
        """
        Save detailed quality scores to file.
        
        Args:
            candidates: List of candidate paraphrases
            sources: List of source texts  
            references: Optional list of reference paraphrases
            output_file: Output file path
            include_individual_metrics: Whether to include individual metric scores
            
        Returns:
            Path to saved file
        """
        include_individual = (
            include_individual_metrics if include_individual_metrics is not None 
            else self.config.include_individual_metrics
        )
        
        # Get overall scores
        overall_scores = self.score_batch(candidates, sources, references)
        
        # Prepare data
        data = {
            'id': [f"sample_{i:06d}" for i in range(len(candidates))],
            'source': sources,
            'candidate': candidates,
            'quality_score': overall_scores
        }
        
        if references:
            data['reference'] = references
        
        # Add individual metrics if requested and quality scorer not used
        if include_individual and not self.quality_scorer:
            individual_metrics = self._compute_individual_metrics_batch(
                candidates, sources, references
            )
            
            for metric_name in individual_metrics:
                data[f'metric_{metric_name}'] = individual_metrics[metric_name]
        
        # Add metadata
        data['timestamp'] = [datetime.now().isoformat()] * len(candidates)
        data['scorer_type'] = ['quality_model' if self.quality_scorer else 'rule_based'] * len(candidates)
        
        # Save to file
        df = pd.DataFrame(data)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        else:
            # Default to JSON
            df.to_json(output_path, orient='records', indent=2)
        
        self.logger.info(f"Saved detailed scores to: {output_path}")
        return str(output_path)
    
    def _compute_individual_metrics_batch(
        self,
        candidates: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Compute individual metrics for all samples."""
        
        metrics = {
            'bleu': [],
            'rouge_l': [],
            'bertscore': [],
            'semantic_similarity': [],
            'diversity_penalty': []
        }
        
        for i, (candidate, source) in enumerate(zip(candidates, sources)):
            reference = references[i] if references else None
            
            # BLEU
            try:
                target = reference if reference else source
                bleu = calculate_bleu_score([target], candidate)
                metrics['bleu'].append(bleu)
            except:
                metrics['bleu'].append(0.0)
            
            # ROUGE-L
            try:
                target = reference if reference else source
                rouge = calculate_rouge_score(target, candidate)
                metrics['rouge_l'].append(rouge.get('rouge-l', {}).get('f', 0.0))
            except:
                metrics['rouge_l'].append(0.0)
            
            # BERTScore
            target = reference if reference else source
            metrics['bertscore'].append(self._compute_bertscore_fallback(candidate, target))
            
            # Semantic similarity
            metrics['semantic_similarity'].append(
                self._compute_semantic_similarity_fallback(candidate, source)
            )
            
            # Diversity penalty
            metrics['diversity_penalty'].append(self._compute_diversity_penalty(candidate))
        
        return metrics
    
    def get_score_statistics(self) -> Dict[str, float]:
        """Get statistics on computed quality scores."""
        
        if not self.score_history:
            return {}
        
        scores = np.array(self.score_history)
        
        return {
            'count': len(scores),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'above_threshold': int(np.sum(scores >= self.config.filter_threshold)),
            'below_threshold': int(np.sum(scores < self.config.filter_threshold))
        }
    
    def clear_history(self):
        """Clear score history."""
        self.score_history.clear()


def main():
    """Example usage of QualityAssessor."""
    
    # Sample data
    sources = [
        "The weather is beautiful today.",
        "I love programming in Python.",
        "This movie is really interesting.",
        "Machine learning is fascinating."
    ]
    
    candidates = [
        "Today's weather is lovely.",
        "Python programming is enjoyable.",
        "The film is quite engaging.", 
        "AI algorithms are captivating."
    ]
    
    references = [
        "It's a lovely day weather-wise.",
        "I enjoy coding with Python.",
        "This film is engaging.",
        "ML concepts are intriguing."
    ]
    
    # Initialize quality assessor
    config = QualityAssessmentConfig(
        quality_model_path=None,  # Use fallback scoring
        filter_threshold=0.6,
        save_detailed_scores=True
    )
    
    assessor = QualityAssessor(config)
    
    print("=== Quality Assessment Example ===")
    print(f"Sources: {len(sources)}")
    print(f"Candidates: {len(candidates)}")
    print(f"References: {len(references)}")
    
    # Score batch
    print("\n=== Quality Scores ===")
    scores = assessor.score_batch(candidates, sources, references)
    
    for i, (src, cand, ref, score) in enumerate(zip(sources, candidates, references, scores)):
        print(f"{i+1}. Score: {score:.3f}")
        print(f"   Source: {src}")
        print(f"   Candidate: {cand}")
        print(f"   Reference: {ref}")
        print()
    
    # Filter by quality
    print("=== Quality Filtering ===")
    filtered = assessor.filter_by_quality(candidates, sources, references, min_score=0.5)
    
    print(f"Filtered candidates: {len(filtered['candidates'])}/{len(candidates)}")
    for i, (cand, score) in enumerate(zip(filtered['candidates'], filtered['scores'])):
        print(f"{i+1}. {cand} (score: {score:.3f})")
    
    # Create RL rewards
    print("\n=== RL Rewards ===")
    rewards = assessor.create_rl_rewards(candidates, sources, references)
    
    for i, reward in enumerate(rewards):
        print(f"Sample {i+1}: {reward:.3f}")
    
    # Get statistics
    print("\n=== Score Statistics ===")
    stats = assessor.get_score_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Save detailed scores
    print("\n=== Saving Detailed Scores ===")
    output_file = assessor.save_detailed_scores(
        candidates, sources, references, 
        "example_quality_scores.csv"
    )
    print(f"Detailed scores saved to: {output_file}")


if __name__ == "__main__":
    main()
