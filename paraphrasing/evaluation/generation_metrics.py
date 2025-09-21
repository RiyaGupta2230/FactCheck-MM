#!/usr/bin/env python3
"""
Generation Metrics for Paraphrase Evaluation

Comprehensive implementation of automatic evaluation metrics including BLEU, ROUGE,
METEOR, chrF, BERTScore, and diversity metrics with batching support and configurable
tokenization for robust paraphrase evaluation.

Example Usage:
    >>> from paraphrasing.evaluation import GenerationMetrics
    >>> 
    >>> # Initialize metrics computer
    >>> metrics = GenerationMetrics(smooth_bleu=True, use_bertscore=True)
    >>> 
    >>> # Update with samples
    >>> references = ["The weather is beautiful today", "I love programming"]
    >>> predictions = ["Today's weather is lovely", "Programming is enjoyable"]
    >>> metrics.update(references, predictions)
    >>> 
    >>> # Compute all metrics
    >>> scores = metrics.compute()
    >>> print(f"BLEU: {scores['bleu']:.3f}")
    >>> print(f"ROUGE-L: {scores['rouge_l']:.3f}")
    >>> print(f"BERTScore: {scores['bertscore_f1']:.3f}")
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict, Counter
import logging
import warnings
from dataclasses import dataclass, field

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Import evaluation libraries with fallbacks
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    warnings.warn("sacrebleu not available. Install with: pip install sacrebleu")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge-score not available. Install with: pip install rouge-score")

try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    warnings.warn("bert-score not available. Install with: pip install bert-score")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    warnings.warn("NLTK not available for METEOR. Install with: pip install nltk")


@dataclass
class MetricsConfig:
    """Configuration for generation metrics computation."""
    
    # BLEU configuration
    smooth_bleu: bool = True
    bleu_tokenizer: str = "13a"  # "13a", "intl", "zh", "ja-mecab"
    
    # ROUGE configuration
    rouge_types: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    rouge_use_stemmer: bool = True
    
    # BERTScore configuration
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    bertscore_lang: str = "en"
    bertscore_rescale: bool = True
    
    # METEOR configuration
    meteor_alpha: float = 0.9
    meteor_beta: float = 3.0
    meteor_gamma: float = 0.5
    
    # Diversity metrics
    compute_diversity: bool = True
    diversity_ngrams: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # General configuration
    lowercase: bool = False
    remove_punctuation: bool = False
    batch_size: int = 64  # For BERTScore batching


class GenerationMetrics:
    """
    Comprehensive generation metrics computer for paraphrase evaluation.
    
    Supports BLEU, ROUGE, METEOR, chrF, BERTScore, and diversity metrics
    with accumulation across batches and configurable options.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize generation metrics computer.
        
        Args:
            config: Metrics configuration. If None, uses default config.
        """
        self.config = config or MetricsConfig()
        self.logger = get_logger("GenerationMetrics")
        
        # Initialize metric computers
        self._setup_metric_computers()
        
        # Storage for accumulated predictions and references
        self.predictions = []
        self.references = []
        
        # Batch accumulation
        self.batch_predictions = []
        self.batch_references = []
        
        self.logger.info("Initialized GenerationMetrics with available metrics:")
        self.logger.info(f"  BLEU: {SACREBLEU_AVAILABLE}")
        self.logger.info(f"  ROUGE: {ROUGE_AVAILABLE}")
        self.logger.info(f"  METEOR: {METEOR_AVAILABLE}")
        self.logger.info(f"  BERTScore: {BERTSCORE_AVAILABLE}")
    
    def _setup_metric_computers(self):
        """Setup individual metric computers."""
        
        # ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.config.rouge_types,
                use_stemmer=self.config.rouge_use_stemmer
            )
        else:
            self.rouge_scorer = None
        
        # Download NLTK data for METEOR if needed
        if METEOR_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                self.logger.info("Downloading NLTK WordNet data for METEOR...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text based on configuration."""
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def compute_bleu(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute BLEU scores using sacrebleu."""
        
        if not SACREBLEU_AVAILABLE or not predictions:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        # Preprocess texts
        pred_processed = [self._preprocess_text(p) for p in predictions]
        ref_processed = [[self._preprocess_text(r)] for r in references]
        
        try:
            # Compute BLEU score
            bleu = sacrebleu.corpus_bleu(
                pred_processed,
                list(zip(*ref_processed)),
                smooth_method='exp' if self.config.smooth_bleu else 'none',
                tokenize=self.config.bleu_tokenizer
            )
            
            # Compute individual n-gram BLEU scores
            bleu_scores = {
                'bleu': bleu.score / 100.0,  # Convert to 0-1 range
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': bleu.score / 100.0
            }
            
            # Compute individual n-gram scores
            for n in [1, 2, 3]:
                try:
                    n_bleu = sacrebleu.corpus_bleu(
                        pred_processed,
                        list(zip(*ref_processed)),
                        smooth_method='exp' if self.config.smooth_bleu else 'none',
                        tokenize=self.config.bleu_tokenizer,
                        force=True,
                        max_ngram_order=n
                    )
                    bleu_scores[f'bleu_{n}'] = n_bleu.score / 100.0
                except:
                    continue
            
            return bleu_scores
            
        except Exception as e:
            self.logger.warning(f"BLEU computation failed: {e}")
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    def compute_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        
        if not ROUGE_AVAILABLE or not predictions:
            return {f'{rouge_type}_{metric}': 0.0 
                   for rouge_type in self.config.rouge_types 
                   for metric in ['precision', 'recall', 'fmeasure']}
        
        rouge_scores = defaultdict(list)
        
        for ref, pred in zip(references, predictions):
            # Preprocess texts
            ref_processed = self._preprocess_text(ref)
            pred_processed = self._preprocess_text(pred)
            
            try:
                scores = self.rouge_scorer.score(ref_processed, pred_processed)
                
                for rouge_type in self.config.rouge_types:
                    if rouge_type in scores:
                        rouge_scores[f'{rouge_type}_precision'].append(scores[rouge_type].precision)
                        rouge_scores[f'{rouge_type}_recall'].append(scores[rouge_type].recall)
                        rouge_scores[f'{rouge_type}_fmeasure'].append(scores[rouge_type].fmeasure)
                        
            except Exception as e:
                self.logger.warning(f"ROUGE computation failed for sample: {e}")
                # Add zero scores for failed samples
                for rouge_type in self.config.rouge_types:
                    rouge_scores[f'{rouge_type}_precision'].append(0.0)
                    rouge_scores[f'{rouge_type}_recall'].append(0.0)
                    rouge_scores[f'{rouge_type}_fmeasure'].append(0.0)
        
        # Average scores
        avg_rouge_scores = {}
        for key, values in rouge_scores.items():
            avg_rouge_scores[key] = np.mean(values) if values else 0.0
        
        # Add convenient aliases
        if 'rougeL_fmeasure' in avg_rouge_scores:
            avg_rouge_scores['rouge_l'] = avg_rouge_scores['rougeL_fmeasure']
        if 'rouge1_fmeasure' in avg_rouge_scores:
            avg_rouge_scores['rouge_1'] = avg_rouge_scores['rouge1_fmeasure']
        if 'rouge2_fmeasure' in avg_rouge_scores:
            avg_rouge_scores['rouge_2'] = avg_rouge_scores['rouge2_fmeasure']
        
        return avg_rouge_scores
    
    def compute_meteor(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute METEOR scores."""
        
        if not METEOR_AVAILABLE or not predictions:
            return {'meteor': 0.0}
        
        meteor_scores = []
        
        for ref, pred in zip(references, predictions):
            # Preprocess texts
            ref_processed = self._preprocess_text(ref)
            pred_processed = self._preprocess_text(pred)
            
            try:
                # Tokenize
                ref_tokens = ref_processed.split()
                pred_tokens = pred_processed.split()
                
                if ref_tokens and pred_tokens:
                    score = meteor_score(
                        [ref_tokens], pred_tokens,
                        alpha=self.config.meteor_alpha,
                        beta=self.config.meteor_beta,
                        gamma=self.config.meteor_gamma
                    )
                    meteor_scores.append(score)
                else:
                    meteor_scores.append(0.0)
                    
            except Exception as e:
                self.logger.warning(f"METEOR computation failed for sample: {e}")
                meteor_scores.append(0.0)
        
        return {'meteor': np.mean(meteor_scores) if meteor_scores else 0.0}
    
    def compute_chrf(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute chrF scores using sacrebleu."""
        
        if not SACREBLEU_AVAILABLE or not predictions:
            return {'chrf': 0.0, 'chrf_plus_plus': 0.0}
        
        # Preprocess texts
        pred_processed = [self._preprocess_text(p) for p in predictions]
        ref_processed = [[self._preprocess_text(r)] for r in references]
        
        try:
            # Compute chrF
            chrf = sacrebleu.corpus_chrf(
                pred_processed,
                list(zip(*ref_processed))
            )
            
            # Compute chrF++
            chrf_plus_plus = sacrebleu.corpus_chrf(
                pred_processed,
                list(zip(*ref_processed)),
                word_order=2
            )
            
            return {
                'chrf': chrf.score / 100.0,
                'chrf_plus_plus': chrf_plus_plus.score / 100.0
            }
            
        except Exception as e:
            self.logger.warning(f"chrF computation failed: {e}")
            return {'chrf': 0.0, 'chrf_plus_plus': 0.0}
    
    def compute_bertscore(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Compute BERTScore."""
        
        if not BERTSCORE_AVAILABLE or not predictions:
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        # Preprocess texts
        pred_processed = [self._preprocess_text(p) for p in predictions]
        ref_processed = [self._preprocess_text(r) for r in references]
        
        try:
            # Compute BERTScore in batches
            all_precision = []
            all_recall = []
            all_f1 = []
            
            batch_size = self.config.batch_size
            for i in range(0, len(predictions), batch_size):
                batch_pred = pred_processed[i:i + batch_size]
                batch_ref = ref_processed[i:i + batch_size]
                
                P, R, F1 = bert_score.score(
                    batch_pred,
                    batch_ref,
                    model_type=self.config.bertscore_model,
                    lang=self.config.bertscore_lang,
                    rescale_with_baseline=self.config.bertscore_rescale,
                    verbose=False
                )
                
                all_precision.extend(P.tolist())
                all_recall.extend(R.tolist())
                all_f1.extend(F1.tolist())
            
            return {
                'bertscore_precision': np.mean(all_precision),
                'bertscore_recall': np.mean(all_recall),
                'bertscore_f1': np.mean(all_f1)
            }
            
        except Exception as e:
            self.logger.warning(f"BERTScore computation failed: {e}")
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
    
    def compute_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute diversity metrics (distinct-n)."""
        
        if not predictions or not self.config.compute_diversity:
            return {f'distinct_{n}': 0.0 for n in self.config.diversity_ngrams}
        
        diversity_scores = {}
        
        for n in self.config.diversity_ngrams:
            all_ngrams = []
            total_ngrams = 0
            
            for pred in predictions:
                # Preprocess and tokenize
                pred_processed = self._preprocess_text(pred)
                tokens = pred_processed.split()
                
                if len(tokens) >= n:
                    # Extract n-grams
                    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                    all_ngrams.extend(ngrams)
                    total_ngrams += len(ngrams)
            
            # Calculate distinct-n ratio
            if total_ngrams > 0:
                unique_ngrams = len(set(all_ngrams))
                diversity_scores[f'distinct_{n}'] = unique_ngrams / total_ngrams
            else:
                diversity_scores[f'distinct_{n}'] = 0.0
        
        return diversity_scores
    
    def compute_repetition_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute repetition-based metrics."""
        
        if not predictions:
            return {'repetition_rate': 0.0, 'avg_repetitions_per_sample': 0.0}
        
        repetition_rates = []
        total_repetitions = []
        
        for pred in predictions:
            pred_processed = self._preprocess_text(pred)
            tokens = pred_processed.split()
            
            if len(tokens) > 1:
                # Count repeated consecutive tokens
                repetitions = 0
                for i in range(len(tokens) - 1):
                    if tokens[i] == tokens[i + 1]:
                        repetitions += 1
                
                repetition_rate = repetitions / (len(tokens) - 1)
                repetition_rates.append(repetition_rate)
                total_repetitions.append(repetitions)
            else:
                repetition_rates.append(0.0)
                total_repetitions.append(0)
        
        return {
            'repetition_rate': np.mean(repetition_rates),
            'avg_repetitions_per_sample': np.mean(total_repetitions)
        }
    
    def update(self, references: List[str], predictions: List[str]):
        """
        Update metrics with new batch of references and predictions.
        
        Args:
            references: List of reference texts
            predictions: List of predicted texts
        """
        if len(references) != len(predictions):
            raise ValueError(f"Mismatch in lengths: {len(references)} references vs {len(predictions)} predictions")
        
        self.batch_references.extend(references)
        self.batch_predictions.extend(predictions)
        
        # Also store in main lists
        self.references.extend(references)
        self.predictions.extend(predictions)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all available metrics on accumulated data.
        
        Returns:
            Dictionary of metric scores
        """
        if not self.batch_predictions:
            return {}
        
        all_metrics = {}
        
        # Compute each metric
        try:
            bleu_scores = self.compute_bleu(self.batch_references, self.batch_predictions)
            all_metrics.update(bleu_scores)
        except Exception as e:
            self.logger.warning(f"BLEU computation failed: {e}")
        
        try:
            rouge_scores = self.compute_rouge(self.batch_references, self.batch_predictions)
            all_metrics.update(rouge_scores)
        except Exception as e:
            self.logger.warning(f"ROUGE computation failed: {e}")
        
        try:
            meteor_scores = self.compute_meteor(self.batch_references, self.batch_predictions)
            all_metrics.update(meteor_scores)
        except Exception as e:
            self.logger.warning(f"METEOR computation failed: {e}")
        
        try:
            chrf_scores = self.compute_chrf(self.batch_references, self.batch_predictions)
            all_metrics.update(chrf_scores)
        except Exception as e:
            self.logger.warning(f"chrF computation failed: {e}")
        
        try:
            bertscore_scores = self.compute_bertscore(self.batch_references, self.batch_predictions)
            all_metrics.update(bertscore_scores)
        except Exception as e:
            self.logger.warning(f"BERTScore computation failed: {e}")
        
        try:
            diversity_scores = self.compute_diversity_metrics(self.batch_predictions)
            all_metrics.update(diversity_scores)
        except Exception as e:
            self.logger.warning(f"Diversity computation failed: {e}")
        
        try:
            repetition_scores = self.compute_repetition_metrics(self.batch_predictions)
            all_metrics.update(repetition_scores)
        except Exception as e:
            self.logger.warning(f"Repetition computation failed: {e}")
        
        return all_metrics
    
    def reset(self):
        """Reset all accumulated data."""
        self.predictions.clear()
        self.references.clear()
        self.batch_predictions.clear()
        self.batch_references.clear()
    
    def get_sample_count(self) -> int:
        """Get number of accumulated samples."""
        return len(self.batch_predictions)
    
    def compute_pairwise_metrics(
        self,
        references: List[str],
        predictions: List[str]
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each reference-prediction pair individually.
        
        Args:
            references: List of reference texts
            predictions: List of predicted texts
            
        Returns:
            List of metric dictionaries, one per sample
        """
        if len(references) != len(predictions):
            raise ValueError("References and predictions must have same length")
        
        pairwise_metrics = []
        
        for ref, pred in zip(references, predictions):
            sample_metrics = {}
            
            # Compute metrics for this sample
            try:
                bleu = self.compute_bleu([ref], [pred])
                sample_metrics.update(bleu)
            except:
                sample_metrics.update({'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0})
            
            try:
                rouge = self.compute_rouge([ref], [pred])
                sample_metrics.update(rouge)
            except:
                pass
            
            try:
                meteor = self.compute_meteor([ref], [pred])
                sample_metrics.update(meteor)
            except:
                sample_metrics.update({'meteor': 0.0})
            
            try:
                chrf = self.compute_chrf([ref], [pred])
                sample_metrics.update(chrf)
            except:
                sample_metrics.update({'chrf': 0.0, 'chrf_plus_plus': 0.0})
            
            pairwise_metrics.append(sample_metrics)
        
        return pairwise_metrics


def main():
    """Example usage of GenerationMetrics."""
    
    # Create sample data
    references = [
        "The weather is beautiful today.",
        "I love programming in Python.",
        "This movie is really interesting.",
        "Machine learning is fascinating."
    ]
    
    predictions = [
        "Today's weather is lovely.",
        "Python programming is enjoyable.",
        "The film is quite engaging.", 
        "ML algorithms are captivating."
    ]
    
    # Initialize metrics with custom config
    config = MetricsConfig(
        smooth_bleu=True,
        compute_diversity=True,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    
    metrics = GenerationMetrics(config)
    
    print("=== Generation Metrics Example ===")
    print(f"References: {len(references)}")
    print(f"Predictions: {len(predictions)}")
    
    # Update metrics
    metrics.update(references, predictions)
    
    # Compute all metrics
    scores = metrics.compute()
    
    print("\n=== Results ===")
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score:.4f}")
    
    # Test pairwise computation
    print("\n=== Pairwise Metrics (First Sample) ===")
    pairwise = metrics.compute_pairwise_metrics(references[:1], predictions[:1])
    for metric_name, score in pairwise[0].items():
        print(f"{metric_name}: {score:.4f}")
    
    # Reset and test incremental updates
    print("\n=== Incremental Updates ===")
    metrics.reset()
    
    for i in range(0, len(references), 2):
        batch_refs = references[i:i+2]
        batch_preds = predictions[i:i+2]
        metrics.update(batch_refs, batch_preds)
        print(f"After batch {i//2 + 1}: {metrics.get_sample_count()} samples")
    
    final_scores = metrics.compute()
    print(f"\nFinal BLEU: {final_scores.get('bleu', 0):.4f}")
    print(f"Final ROUGE-L: {final_scores.get('rouge_l', 0):.4f}")


if __name__ == "__main__":
    main()
