# sarcasm_detection/evaluation/error_analysis.py
"""
Comprehensive Error Analysis for Sarcasm Detection
Detailed misclassification reports, failure case analysis, and error pattern detection.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import re
from textstat import flesch_reading_ease
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

from shared.utils import get_logger
from shared.datasets import MultimodalCollator
from ..models import TextSarcasmModel, MultimodalSarcasmModel, EnsembleSarcasmModel
from .evaluator import SarcasmEvaluator


@dataclass
class ErrorAnalysisConfig:
    """Configuration for error analysis."""
    
    # Analysis scope
    analyze_text_features: bool = True
    analyze_linguistic_patterns: bool = True
    analyze_confidence_errors: bool = True
    analyze_dataset_specific_errors: bool = True
    
    # Error categorization
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    min_error_frequency: int = 5
    max_examples_per_category: int = 50
    
    # Text analysis features
    text_features: List[str] = field(default_factory=lambda: [
        'length', 'readability', 'sentiment', 'punctuation',
        'capitalization', 'question_marks', 'exclamation_marks'
    ])
    
    # Pattern detection
    detect_irony_patterns: bool = True
    detect_syntactic_patterns: bool = True
    detect_semantic_patterns: bool = True
    
    # Output settings
    generate_detailed_reports: bool = True
    save_error_examples: bool = True
    create_visualizations: bool = True


class ErrorAnalyzer:
    """Comprehensive error analyzer for sarcasm detection models."""
    
    def __init__(
        self,
        model: Union[TextSarcasmModel, MultimodalSarcasmModel, EnsembleSarcasmModel],
        datasets: Dict[str, Any],
        config: Union[ErrorAnalysisConfig, Dict[str, Any]] = None
    ):
        """
        Initialize error analyzer.
        
        Args:
            model: Model to analyze
            datasets: Datasets for analysis
            config: Error analysis configuration
        """
        if isinstance(config, dict):
            config = ErrorAnalysisConfig(**config)
        elif config is None:
            config = ErrorAnalysisConfig()
        
        self.model = model
        self.datasets = datasets
        self.config = config
        
        self.logger = get_logger("ErrorAnalyzer")
        
        # Initialize evaluator
        self.evaluator = SarcasmEvaluator(model)
        
        # Initialize text analysis tools
        self._setup_text_analysis_tools()
        
        self.logger.info(f"Initialized error analyzer for {len(datasets)} datasets")
    
    def _setup_text_analysis_tools(self):
        """Setup text analysis tools."""
        
        # Irony detection patterns
        self.irony_patterns = [
            (r'\boh\s+(sure|great|wonderful|perfect|nice)\b', 'sarcastic_interjection'),
            (r'\byeah\s+(right|sure)\b', 'skeptical_agreement'),
            (r'\bas\s+if\b', 'dismissive_phrase'),
            (r'\bi\s+bet\b', 'ironic_agreement'),
            (r'\btotally\b', 'ironic_emphasis'),
            (r'\babsolutely\s+(not|never)\b', 'negated_emphasis'),
            (r'\bcouldn\'t\s+agree\s+more\b', 'excessive_agreement'),
            (r'\bsure\s+thing\b', 'dismissive_agreement'),
            (r'\bwow\b.*\bimpressive\b', 'mock_admiration'),
            (r'\bbrilliant\b.*\bidea\b', 'sarcastic_praise')
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), label) 
                                for pattern, label in self.irony_patterns]
        
        # Syntactic patterns
        self.syntactic_patterns = [
            (r'\.{3,}', 'ellipsis'),
            (r'[!]{2,}', 'multiple_exclamation'),
            (r'[?]{2,}', 'multiple_question'),
            (r'[A-Z]{3,}', 'all_caps_words'),
            (r'\b\w*([a-z])\1{2,}\w*\b', 'repeated_letters')
        ]
        
        self.compiled_syntactic = [(re.compile(pattern), label) 
                                 for pattern, label in self.syntactic_patterns]
    
    def run_comprehensive_error_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive error analysis across all datasets.
        
        Returns:
            Complete error analysis results
        """
        self.logger.info("Starting comprehensive error analysis")
        
        results = {
            'configuration': self.config.__dict__,
            'dataset_errors': {},
            'aggregated_analysis': {},
            'error_patterns': {},
            'recommendations': []
        }
        
        # Analyze errors for each dataset
        all_errors = []
        for dataset_name, dataset in self.datasets.items():
            self.logger.info(f"Analyzing errors for dataset: {dataset_name}")
            
            dataset_errors = self._analyze_dataset_errors(dataset, dataset_name)
            results['dataset_errors'][dataset_name] = dataset_errors
            
            # Collect all errors for aggregated analysis
            if 'error_samples' in dataset_errors:
                for error in dataset_errors['error_samples']:
                    error['dataset'] = dataset_name
                    all_errors.append(error)
        
        # Aggregated analysis across all datasets
        if all_errors:
            results['aggregated_analysis'] = self._perform_aggregated_analysis(all_errors)
            results['error_patterns'] = self._detect_error_patterns(all_errors)
            results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_dataset_errors(
        self,
        dataset,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Analyze errors for a specific dataset."""
        
        # Get predictions and collect errors
        evaluation_results = self.evaluator.evaluate_dataset(dataset, dataset_name)
        
        if 'predictions' not in evaluation_results:
            self.logger.warning(f"No predictions available for {dataset_name}")
            return {}
        
        predictions_data = evaluation_results['predictions']
        
        # Identify error samples
        error_samples = []
        correct_samples = []
        
        for i, (pred, prob, label, sample_id) in enumerate(zip(
            predictions_data['predictions'],
            predictions_data['probabilities'],
            predictions_data['labels'],
            predictions_data['sample_ids']
        )):
            # Get original sample
            try:
                sample = dataset[i % len(dataset)]  # Handle potential index mismatch
            except:
                continue
            
            is_error = (pred != label)
            confidence = max(prob)
            
            sample_analysis = {
                'sample_id': sample_id,
                'predicted': pred,
                'actual': label,
                'confidence': confidence,
                'probabilities': prob,
                'is_error': is_error,
                'text': sample.get('text', ''),
                'dataset': dataset_name
            }
            
            # Add text features if available
            if sample_analysis['text']:
                sample_analysis['text_features'] = self._extract_text_features(sample_analysis['text'])
                sample_analysis['linguistic_patterns'] = self._detect_linguistic_patterns(sample_analysis['text'])
            
            if is_error:
                error_samples.append(sample_analysis)
            else:
                correct_samples.append(sample_analysis)
        
        # Analyze error patterns
        error_analysis = {
            'total_samples': len(predictions_data['predictions']),
            'total_errors': len(error_samples),
            'error_rate': len(error_samples) / len(predictions_data['predictions']) if predictions_data['predictions'] else 0,
            'error_samples': error_samples[:self.config.max_examples_per_category],
            'error_breakdown': self._categorize_errors(error_samples),
            'confidence_analysis': self._analyze_confidence_errors(error_samples),
            'text_feature_analysis': self._analyze_text_features(error_samples, correct_samples)
        }
        
        return error_analysis
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive text features."""
        
        features = {}
        
        if not text or not text.strip():
            return features
        
        # Basic length features
        features['char_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Readability
        try:
            features['flesch_score'] = flesch_reading_ease(text)
        except:
            features['flesch_score'] = 0
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        features['comma_count'] = text.count(',')
        
        # Capitalization features
        features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Special characters
        features['quotation_marks'] = text.count('"') + text.count("'")
        features['parentheses'] = text.count('(') + text.count(')')
        features['brackets'] = text.count('[') + text.count(']')
        
        return features
    
    def _detect_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Detect linguistic patterns in text."""
        
        patterns = {
            'irony_patterns': [],
            'syntactic_patterns': [],
            'semantic_indicators': []
        }
        
        if not text:
            return patterns
        
        # Detect irony patterns
        for pattern, label in self.compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                patterns['irony_patterns'].append({
                    'pattern': label,
                    'matches': len(matches),
                    'examples': matches[:3]  # First 3 matches
                })
        
        # Detect syntactic patterns
        for pattern, label in self.compiled_syntactic:
            matches = pattern.findall(text)
            if matches:
                patterns['syntactic_patterns'].append({
                    'pattern': label,
                    'matches': len(matches),
                    'examples': matches[:3]
                })
        
        # Semantic indicators
        text_lower = text.lower()
        
        # Contrast indicators
        contrast_words = ['but', 'however', 'although', 'though', 'yet', 'nevertheless']
        patterns['contrast_indicators'] = sum(1 for word in contrast_words if word in text_lower)
        
        # Intensifiers
        intensifiers = ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally']
        patterns['intensifiers'] = sum(1 for word in intensifiers if word in text_lower)
        
        # Hedging
        hedges = ['maybe', 'perhaps', 'possibly', 'probably', 'seems', 'appears']
        patterns['hedging'] = sum(1 for word in hedges if word in text_lower)
        
        return patterns
    
    def _categorize_errors(self, error_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize errors by type and characteristics."""
        
        categorization = {
            'by_confidence': defaultdict(int),
            'by_error_type': defaultdict(int),
            'by_text_length': defaultdict(int),
            'by_sentiment': defaultdict(int)
        }
        
        for error in error_samples:
            confidence = error.get('confidence', 0)
            actual = error.get('actual', 0)
            predicted = error.get('predicted', 0)
            
            # Confidence-based categorization
            if confidence < 0.5:
                categorization['by_confidence']['low_confidence'] += 1
            elif confidence < 0.7:
                categorization['by_confidence']['medium_confidence'] += 1
            else:
                categorization['by_confidence']['high_confidence'] += 1
            
            # Error type
            if actual == 1 and predicted == 0:
                categorization['by_error_type']['false_negative'] += 1
            elif actual == 0 and predicted == 1:
                categorization['by_error_type']['false_positive'] += 1
            
            # Text length
            if 'text_features' in error:
                word_count = error['text_features'].get('word_count', 0)
                if word_count < 5:
                    categorization['by_text_length']['very_short'] += 1
                elif word_count < 15:
                    categorization['by_text_length']['short'] += 1
                elif word_count < 30:
                    categorization['by_text_length']['medium'] += 1
                else:
                    categorization['by_text_length']['long'] += 1
                
                # Sentiment
                sentiment = error['text_features'].get('sentiment_polarity', 0)
                if sentiment < -0.1:
                    categorization['by_sentiment']['negative'] += 1
                elif sentiment > 0.1:
                    categorization['by_sentiment']['positive'] += 1
                else:
                    categorization['by_sentiment']['neutral'] += 1
        
        # Convert to regular dict for JSON serialization
        return {k: dict(v) for k, v in categorization.items()}
    
    def _analyze_confidence_errors(self, error_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence-related error patterns."""
        
        confidence_analysis = {
            'high_confidence_errors': [],
            'low_confidence_errors': [],
            'confidence_statistics': {}
        }
        
        confidences = [error.get('confidence', 0) for error in error_samples]
        
        if confidences:
            confidence_analysis['confidence_statistics'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
            
            # High confidence errors (model is wrong but very confident)
            high_conf_errors = [error for error in error_samples if error.get('confidence', 0) > 0.8]
            confidence_analysis['high_confidence_errors'] = high_conf_errors[:20]  # Top 20
            
            # Low confidence errors (model is uncertain and wrong)
            low_conf_errors = [error for error in error_samples if error.get('confidence', 0) < 0.6]
            confidence_analysis['low_confidence_errors'] = low_conf_errors[:20]
        
        return confidence_analysis
    
    def _analyze_text_features(
        self,
        error_samples: List[Dict[str, Any]],
        correct_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze text features that correlate with errors."""
        
        if not self.config.analyze_text_features:
            return {}
        
        feature_analysis = {}
        
        # Extract features for errors and correct predictions
        error_features = []
        correct_features = []
        
        for error in error_samples:
            if 'text_features' in error:
                error_features.append(error['text_features'])
        
        for correct in correct_samples[:len(error_features)]:  # Match sample sizes
            if 'text_features' in correct:
                correct_features.append(correct['text_features'])
        
        if not error_features or not correct_features:
            return feature_analysis
        
        # Compare feature distributions
        for feature in self.config.text_features:
            if feature in ['length', 'readability', 'sentiment']:
                # Map feature names to actual feature keys
                feature_key_map = {
                    'length': 'word_count',
                    'readability': 'flesch_score',
                    'sentiment': 'sentiment_polarity'
                }
                actual_key = feature_key_map.get(feature, feature)
                
                error_values = [f.get(actual_key, 0) for f in error_features if actual_key in f]
                correct_values = [f.get(actual_key, 0) for f in correct_features if actual_key in f]
                
                if error_values and correct_values:
                    feature_analysis[feature] = {
                        'error_mean': float(np.mean(error_values)),
                        'correct_mean': float(np.mean(correct_values)),
                        'error_std': float(np.std(error_values)),
                        'correct_std': float(np.std(correct_values)),
                        'difference': float(np.mean(error_values) - np.mean(correct_values))
                    }
        
        return feature_analysis
    
    def _perform_aggregated_analysis(self, all_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform aggregated analysis across all errors."""
        
        aggregated = {
            'total_errors': len(all_errors),
            'error_by_dataset': Counter(error['dataset'] for error in all_errors),
            'most_common_patterns': {},
            'challenging_samples': []
        }
        
        # Find most common irony patterns across all errors
        all_patterns = []
        for error in all_errors:
            if 'linguistic_patterns' in error:
                for pattern_info in error['linguistic_patterns'].get('irony_patterns', []):
                    all_patterns.append(pattern_info['pattern'])
        
        aggregated['most_common_patterns'] = dict(Counter(all_patterns).most_common(10))
        
        # Find most challenging samples (high confidence errors)
        challenging = sorted(
            [error for error in all_errors if error.get('confidence', 0) > 0.8],
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )
        aggregated['challenging_samples'] = challenging[:20]
        
        return aggregated
    
    def _detect_error_patterns(self, all_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect systematic error patterns."""
        
        patterns = {
            'linguistic_patterns': defaultdict(int),
            'length_patterns': defaultdict(list),
            'sentiment_patterns': defaultdict(list),
            'dataset_patterns': defaultdict(lambda: defaultdict(int))
        }
        
        for error in all_errors:
            dataset = error.get('dataset', 'unknown')
            error_type = 'false_positive' if error.get('predicted') == 1 else 'false_negative'
            
            # Dataset-specific patterns
            patterns['dataset_patterns'][dataset][error_type] += 1
            
            if 'linguistic_patterns' in error:
                # Linguistic patterns
                for pattern_info in error['linguistic_patterns'].get('irony_patterns', []):
                    patterns['linguistic_patterns'][pattern_info['pattern']] += 1
            
            if 'text_features' in error:
                features = error['text_features']
                
                # Length patterns
                word_count = features.get('word_count', 0)
                patterns['length_patterns'][error_type].append(word_count)
                
                # Sentiment patterns
                sentiment = features.get('sentiment_polarity', 0)
                patterns['sentiment_patterns'][error_type].append(sentiment)
        
        # Convert to regular dict and compute statistics
        result_patterns = {}
        
        result_patterns['linguistic_patterns'] = dict(patterns['linguistic_patterns'])
        result_patterns['dataset_patterns'] = {k: dict(v) for k, v in patterns['dataset_patterns'].items()}
        
        # Length and sentiment statistics
        for error_type in ['false_positive', 'false_negative']:
            if patterns['length_patterns'][error_type]:
                result_patterns[f'{error_type}_length_stats'] = {
                    'mean': float(np.mean(patterns['length_patterns'][error_type])),
                    'std': float(np.std(patterns['length_patterns'][error_type]))
                }
            
            if patterns['sentiment_patterns'][error_type]:
                result_patterns[f'{error_type}_sentiment_stats'] = {
                    'mean': float(np.mean(patterns['sentiment_patterns'][error_type])),
                    'std': float(np.std(patterns['sentiment_patterns'][error_type]))
                }
        
        return result_patterns
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on error analysis."""
        
        recommendations = []
        
        # Check aggregated analysis
        if 'aggregated_analysis' in results:
            total_errors = results['aggregated_analysis'].get('total_errors', 0)
            
            if total_errors > 100:
                recommendations.append(
                    "High error count detected. Consider additional training data or model architecture improvements."
                )
        
        # Check error patterns
        if 'error_patterns' in results:
            patterns = results['error_patterns']
            
            # Linguistic pattern recommendations
            common_patterns = patterns.get('linguistic_patterns', {})
            if common_patterns:
                most_common = max(common_patterns.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Most common error pattern: '{most_common[0]}' ({most_common[1]} occurrences). "
                    f"Consider adding training examples with this pattern."
                )
            
            # Dataset-specific recommendations
            dataset_patterns = patterns.get('dataset_patterns', {})
            for dataset, error_types in dataset_patterns.items():
                if error_types.get('false_positive', 0) > error_types.get('false_negative', 0) * 2:
                    recommendations.append(
                        f"Dataset '{dataset}' shows high false positive rate. "
                        f"Consider adjusting decision threshold or adding negative examples."
                    )
                elif error_types.get('false_negative', 0) > error_types.get('false_positive', 0) * 2:
                    recommendations.append(
                        f"Dataset '{dataset}' shows high false negative rate. "
                        f"Consider improving sarcasm detection sensitivity or adding positive examples."
                    )
        
        # Check individual dataset errors
        for dataset_name, dataset_errors in results.get('dataset_errors', {}).items():
            error_rate = dataset_errors.get('error_rate', 0)
            
            if error_rate > 0.3:
                recommendations.append(
                    f"High error rate ({error_rate:.2%}) on dataset '{dataset_name}'. "
                    f"Consider dataset-specific fine-tuning or data augmentation."
                )
            
            # Confidence analysis recommendations
            confidence_analysis = dataset_errors.get('confidence_analysis', {})
            high_conf_errors = len(confidence_analysis.get('high_confidence_errors', []))
            
            if high_conf_errors > 10:
                recommendations.append(
                    f"Dataset '{dataset_name}' has {high_conf_errors} high-confidence errors. "
                    f"These represent systematic model blind spots that need attention."
                )
        
        return recommendations
    
    def save_error_analysis(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        experiment_name: str = "error_analysis"
    ):
        """Save comprehensive error analysis results."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save error examples
        if self.config.save_error_examples:
            examples_data = []
            
            for dataset_name, dataset_errors in results.get('dataset_errors', {}).items():
                for error in dataset_errors.get('error_samples', []):
                    examples_data.append({
                        'dataset': dataset_name,
                        'sample_id': error.get('sample_id', ''),
                        'text': error.get('text', ''),
                        'actual': error.get('actual', 0),
                        'predicted': error.get('predicted', 0),
                        'confidence': error.get('confidence', 0),
                        'error_type': 'false_positive' if error.get('predicted') == 1 else 'false_negative'
                    })
            
            if examples_data:
                examples_df = pd.DataFrame(examples_data)
                examples_file = output_dir / f"{experiment_name}_error_examples.csv"
                examples_df.to_csv(examples_file, index=False)
        
        # Generate summary report
        self._generate_error_report(results, output_dir, experiment_name)
        
        self.logger.info(f"Saved error analysis results to {output_dir}")
    
    def _generate_error_report(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        experiment_name: str
    ):
        """Generate human-readable error analysis report."""
        
        report_file = output_dir / f"{experiment_name}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Error Analysis Report: {experiment_name}\n\n")
            
            # Summary
            if 'aggregated_analysis' in results:
                agg = results['aggregated_analysis']
                f.write("## Summary\n\n")
                f.write(f"- Total errors analyzed: {agg.get('total_errors', 0)}\n")
                f.write(f"- Datasets analyzed: {len(agg.get('error_by_dataset', {}))}\n\n")
                
                # Error distribution by dataset
                f.write("### Error Distribution by Dataset\n\n")
                for dataset, count in agg.get('error_by_dataset', {}).items():
                    f.write(f"- {dataset}: {count} errors\n")
                f.write("\n")
            
            # Most common patterns
            if 'error_patterns' in results:
                patterns = results['error_patterns']
                
                if 'linguistic_patterns' in patterns:
                    f.write("### Most Common Error Patterns\n\n")
                    for pattern, count in sorted(patterns['linguistic_patterns'].items(), 
                                               key=lambda x: x[1], reverse=True)[:10]:
                        f.write(f"- {pattern}: {count} occurrences\n")
                    f.write("\n")
            
            # Recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n\n")
            
            # Dataset-specific analysis
            f.write("## Dataset-Specific Analysis\n\n")
            for dataset_name, dataset_errors in results.get('dataset_errors', {}).items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- Error rate: {dataset_errors.get('error_rate', 0):.2%}\n")
                f.write(f"- Total errors: {dataset_errors.get('total_errors', 0)}\n")
                
                # Error breakdown
                breakdown = dataset_errors.get('error_breakdown', {})
                if 'by_error_type' in breakdown:
                    f.write(f"- False positives: {breakdown['by_error_type'].get('false_positive', 0)}\n")
                    f.write(f"- False negatives: {breakdown['by_error_type'].get('false_negative', 0)}\n")
                
                f.write("\n")


class MisclassificationAnalyzer:
    """Specialized analyzer for misclassification patterns."""
    
    def __init__(self, error_analyzer: ErrorAnalyzer):
        """Initialize with an error analyzer."""
        self.error_analyzer = error_analyzer
        self.logger = get_logger("MisclassificationAnalyzer")
    
    def analyze_misclassification_patterns(
        self,
        error_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze specific misclassification patterns."""
        
        self.logger.info("Analyzing misclassification patterns")
        
        patterns = {
            'false_positive_patterns': self._analyze_false_positives(error_results),
            'false_negative_patterns': self._analyze_false_negatives(error_results),
            'confidence_patterns': self._analyze_confidence_patterns(error_results),
            'linguistic_confusion': self._analyze_linguistic_confusion(error_results)
        }
        
        return patterns
    
    def _analyze_false_positives(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze false positive patterns (non-sarcastic predicted as sarcastic)."""
        
        false_positives = []
        
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                if error.get('actual') == 0 and error.get('predicted') == 1:
                    false_positives.append(error)
        
        analysis = {
            'count': len(false_positives),
            'common_features': self._extract_common_features(false_positives),
            'examples': false_positives[:10]  # Top 10 examples
        }
        
        return analysis
    
    def _analyze_false_negatives(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze false negative patterns (sarcastic predicted as non-sarcastic)."""
        
        false_negatives = []
        
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                if error.get('actual') == 1 and error.get('predicted') == 0:
                    false_negatives.append(error)
        
        analysis = {
            'count': len(false_negatives),
            'common_features': self._extract_common_features(false_negatives),
            'examples': false_negatives[:10]
        }
        
        return analysis
    
    def _analyze_confidence_patterns(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence-related misclassification patterns."""
        
        high_conf_errors = []
        low_conf_errors = []
        
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                confidence = error.get('confidence', 0)
                if confidence > 0.8:
                    high_conf_errors.append(error)
                elif confidence < 0.6:
                    low_conf_errors.append(error)
        
        return {
            'high_confidence_errors': {
                'count': len(high_conf_errors),
                'patterns': self._extract_common_features(high_conf_errors),
                'examples': high_conf_errors[:5]
            },
            'low_confidence_errors': {
                'count': len(low_conf_errors),
                'patterns': self._extract_common_features(low_conf_errors),
                'examples': low_conf_errors[:5]
            }
        }
    
    def _analyze_linguistic_confusion(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze linguistic features that cause confusion."""
        
        all_errors = []
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            all_errors.extend(dataset_errors.get('error_samples', []))
        
        confusion_analysis = {
            'ambiguous_phrases': self._find_ambiguous_phrases(all_errors),
            'conflicting_signals': self._find_conflicting_signals(all_errors),
            'subtle_sarcasm': self._find_subtle_sarcasm(all_errors)
        }
        
        return confusion_analysis
    
    def _extract_common_features(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common features from a set of errors."""
        
        if not errors:
            return {}
        
        features = {}
        
        # Text length analysis
        lengths = [error.get('text_features', {}).get('word_count', 0) for error in errors]
        if lengths:
            features['length_stats'] = {
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'std': float(np.std(lengths))
            }
        
        # Sentiment analysis
        sentiments = [error.get('text_features', {}).get('sentiment_polarity', 0) for error in errors]
        if sentiments:
            features['sentiment_stats'] = {
                'mean': float(np.mean(sentiments)),
                'std': float(np.std(sentiments))
            }
        
        # Common linguistic patterns
        pattern_counts = defaultdict(int)
        for error in errors:
            patterns = error.get('linguistic_patterns', {})
            for pattern_info in patterns.get('irony_patterns', []):
                pattern_counts[pattern_info['pattern']] += 1
        
        features['common_patterns'] = dict(Counter(pattern_counts).most_common(5))
        
        return features
    
    def _find_ambiguous_phrases(self, errors: List[Dict[str, Any]]) -> List[str]:
        """Find phrases that are ambiguous for sarcasm detection."""
        
        # This is a simplified implementation
        # In practice, you might use more sophisticated NLP techniques
        
        ambiguous_phrases = []
        
        for error in errors:
            text = error.get('text', '').lower()
            confidence = error.get('confidence', 0)
            
            # Look for phrases in borderline confidence errors
            if 0.4 < confidence < 0.6:
                # Extract potential ambiguous phrases (2-4 word sequences)
                words = text.split()
                for i in range(len(words) - 1):
                    for j in range(i + 2, min(i + 5, len(words) + 1)):
                        phrase = ' '.join(words[i:j])
                        if len(phrase) > 5:  # Ignore very short phrases
                            ambiguous_phrases.append(phrase)
        
        # Return most common ambiguous phrases
        return [phrase for phrase, count in Counter(ambiguous_phrases).most_common(10)]
    
    def _find_conflicting_signals(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find examples with conflicting sarcasm signals."""
        
        conflicting = []
        
        for error in errors:
            text_features = error.get('text_features', {})
            linguistic_patterns = error.get('linguistic_patterns', {})
            
            # Check for conflicting signals
            sentiment = text_features.get('sentiment_polarity', 0)
            irony_patterns = len(linguistic_patterns.get('irony_patterns', []))
            
            # Positive sentiment but irony patterns (or vice versa)
            if (sentiment > 0.1 and irony_patterns > 0) or (sentiment < -0.1 and irony_patterns == 0):
                conflicting.append({
                    'text': error.get('text', ''),
                    'sentiment': sentiment,
                    'irony_patterns': irony_patterns,
                    'actual': error.get('actual'),
                    'predicted': error.get('predicted')
                })
        
        return conflicting[:10]  # Return top 10
    
    def _find_subtle_sarcasm(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find examples of subtle sarcasm that the model missed."""
        
        subtle_sarcasm = []
        
        for error in errors:
            # Look for false negatives with few obvious sarcasm markers
            if error.get('actual') == 1 and error.get('predicted') == 0:
                linguistic_patterns = error.get('linguistic_patterns', {})
                text_features = error.get('text_features', {})
                
                # Few explicit markers but still sarcastic
                irony_count = len(linguistic_patterns.get('irony_patterns', []))
                punctuation_score = (
                    text_features.get('exclamation_count', 0) + 
                    text_features.get('ellipsis_count', 0)
                )
                
                if irony_count == 0 and punctuation_score < 2:
                    subtle_sarcasm.append({
                        'text': error.get('text', ''),
                        'confidence': error.get('confidence', 0),
                        'features': text_features
                    })
        
        return subtle_sarcasm[:10]


class FailureCaseAnalyzer:
    """Analyzer for systematic failure cases."""
    
    def __init__(self, error_analyzer: ErrorAnalyzer):
        """Initialize with an error analyzer."""
        self.error_analyzer = error_analyzer
        self.logger = get_logger("FailureCaseAnalyzer")
    
    def identify_failure_modes(
        self,
        error_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify systematic failure modes."""
        
        self.logger.info("Identifying systematic failure modes")
        
        failure_modes = {
            'consistent_failures': self._find_consistent_failures(error_results),
            'context_dependence': self._analyze_context_dependence(error_results),
            'domain_transfer': self._analyze_domain_transfer(error_results),
            'length_sensitivity': self._analyze_length_sensitivity(error_results)
        }
        
        return failure_modes
    
    def _find_consistent_failures(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find patterns that consistently cause failures."""
        
        # Aggregate errors across datasets
        all_errors = []
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            all_errors.extend(dataset_errors.get('error_samples', []))
        
        # Group by similar patterns
        pattern_errors = defaultdict(list)
        
        for error in all_errors:
            patterns = error.get('linguistic_patterns', {})
            for pattern_info in patterns.get('irony_patterns', []):
                pattern_errors[pattern_info['pattern']].append(error)
        
        # Find patterns with high error rates
        consistent_failures = {}
        for pattern, errors in pattern_errors.items():
            if len(errors) >= 5:  # Minimum threshold
                error_rate = len(errors) / (len(errors) + 10)  # Simplified calculation
                consistent_failures[pattern] = {
                    'error_count': len(errors),
                    'estimated_error_rate': error_rate,
                    'examples': errors[:3]
                }
        
        return consistent_failures
    
    def _analyze_context_dependence(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context-dependent failures."""
        
        context_analysis = {
            'short_context_errors': [],
            'long_context_errors': [],
            'conversation_errors': []
        }
        
        # This would require additional context information
        # For now, we'll analyze based on text length as a proxy
        
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                text_features = error.get('text_features', {})
                word_count = text_features.get('word_count', 0)
                
                if word_count < 5:
                    context_analysis['short_context_errors'].append(error)
                elif word_count > 30:
                    context_analysis['long_context_errors'].append(error)
        
        return {k: v[:10] for k, v in context_analysis.items()}  # Limit examples
    
    def _analyze_domain_transfer(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain transfer failures."""
        
        domain_analysis = {}
        
        # Compare error rates across datasets (as proxy for domains)
        for dataset_name, dataset_errors in error_results.get('dataset_errors', {}).items():
            error_rate = dataset_errors.get('error_rate', 0)
            total_errors = dataset_errors.get('total_errors', 0)
            
            domain_analysis[dataset_name] = {
                'error_rate': error_rate,
                'total_errors': total_errors,
                'relative_difficulty': 'high' if error_rate > 0.3 else 'medium' if error_rate > 0.15 else 'low'
            }
        
        return domain_analysis
    
    def _analyze_length_sensitivity(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity to text length."""
        
        length_buckets = {
            'very_short': [],  # 1-3 words
            'short': [],       # 4-10 words
            'medium': [],      # 11-25 words
            'long': []         # 26+ words
        }
        
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                text_features = error.get('text_features', {})
                word_count = text_features.get('word_count', 0)
                
                if word_count <= 3:
                    length_buckets['very_short'].append(error)
                elif word_count <= 10:
                    length_buckets['short'].append(error)
                elif word_count <= 25:
                    length_buckets['medium'].append(error)
                else:
                    length_buckets['long'].append(error)
        
        # Calculate error rates by length
        length_analysis = {}
        for bucket, errors in length_buckets.items():
            if errors:
                length_analysis[bucket] = {
                    'error_count': len(errors),
                    'examples': errors[:5]
                }
        
        return length_analysis


class ErrorPatternDetector:
    """Detector for recurring error patterns."""
    
    def __init__(self):
        """Initialize error pattern detector."""
        self.logger = get_logger("ErrorPatternDetector")
    
    def detect_recurring_patterns(
        self,
        error_results: Dict[str, Any],
        min_frequency: int = 3
    ) -> Dict[str, Any]:
        """Detect recurring error patterns across datasets."""
        
        self.logger.info("Detecting recurring error patterns")
        
        patterns = {
            'text_patterns': self._detect_text_patterns(error_results, min_frequency),
            'feature_patterns': self._detect_feature_patterns(error_results, min_frequency),
            'temporal_patterns': self._detect_temporal_patterns(error_results),
            'cross_dataset_patterns': self._detect_cross_dataset_patterns(error_results)
        }
        
        return patterns
    
    def _detect_text_patterns(
        self,
        error_results: Dict[str, Any],
        min_frequency: int
    ) -> Dict[str, Any]:
        """Detect recurring text patterns in errors."""
        
        # Extract n-grams from error texts
        error_texts = []
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                text = error.get('text', '').lower()
                if text:
                    error_texts.append(text)
        
        # Find common n-grams
        from collections import Counter
        
        # Bigrams
        bigrams = []
        for text in error_texts:
            words = text.split()
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        trigrams = []
        for text in error_texts:
            words = text.split()
            for i in range(len(words) - 2):
                trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return {
            'common_bigrams': dict(Counter(bigrams).most_common(20)),
            'common_trigrams': dict(Counter(trigrams).most_common(20))
        }
    
    def _detect_feature_patterns(
        self,
        error_results: Dict[str, Any],
        min_frequency: int
    ) -> Dict[str, Any]:
        """Detect patterns in text features."""
        
        feature_patterns = {}
        
        # Collect all features
        all_features = []
        for dataset_errors in error_results.get('dataset_errors', {}).values():
            for error in dataset_errors.get('error_samples', []):
                if 'text_features' in error:
                    all_features.append(error['text_features'])
        
        if not all_features:
            return feature_patterns
        
        # Analyze feature distributions
        for feature_name in ['word_count', 'sentiment_polarity', 'flesch_score']:
            values = [f.get(feature_name, 0) for f in all_features if feature_name in f]
            if values:
                feature_patterns[feature_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'quartiles': [float(q) for q in np.percentile(values, [25, 50, 75])]
                }
        
        return feature_patterns
    
    def _detect_temporal_patterns(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect temporal patterns (if timestamp data is available)."""
        
        # Placeholder for temporal analysis
        # This would require timestamp information in the error data
        
        return {
            'note': 'Temporal analysis requires timestamp data in error samples'
        }
    
    def _detect_cross_dataset_patterns(self, error_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns that occur across multiple datasets."""
        
        # Find patterns that appear in multiple datasets
        cross_patterns = {}
        
        # Collect patterns by dataset
        dataset_patterns = {}
        for dataset_name, dataset_errors in error_results.get('dataset_errors', {}).items():
            patterns = set()
            for error in dataset_errors.get('error_samples', []):
                linguistic_patterns = error.get('linguistic_patterns', {})
                for pattern_info in linguistic_patterns.get('irony_patterns', []):
                    patterns.add(pattern_info['pattern'])
            dataset_patterns[dataset_name] = patterns
        
        # Find patterns common to multiple datasets
        all_patterns = set()
        for patterns in dataset_patterns.values():
            all_patterns.update(patterns)
        
        for pattern in all_patterns:
            datasets_with_pattern = [
                dataset for dataset, patterns in dataset_patterns.items()
                if pattern in patterns
            ]
            if len(datasets_with_pattern) >= 2:  # Appears in at least 2 datasets
                cross_patterns[pattern] = datasets_with_pattern
        
        return cross_patterns
