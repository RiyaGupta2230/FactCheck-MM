#!/usr/bin/env python3
"""
Human Evaluation Tools for Paraphrase Assessment

Provides tools for preparing human evaluation datasets, collecting annotations,
computing inter-annotator agreement, and integrating human ratings into the
quality training pipeline for improved paraphrase evaluation.

Example Usage:
    >>> from paraphrasing.evaluation import HumanEvalHooks
    >>> 
    >>> # Prepare evaluation file
    >>> samples = [
    ...     {"source": "The weather is nice", "candidate": "It's a beautiful day",
    ...      "reference": "Today's weather is lovely", "sarcasm_flag": False}
    >>> ]
    >>> HumanEvalHooks.create_human_eval_file(samples, "eval.csv")
    >>> 
    >>> # Process annotations
    >>> evaluator = HumanEvalHooks()
    >>> stats = evaluator.compute_annotation_stats("annotations.csv")
    >>> print(f"Average rating: {stats['mean_rating']:.2f}")
    >>>
    >>> # Compute inter-annotator agreement
    >>> agreement = evaluator.compute_inter_annotator_agreement("annotations.csv")
    >>> print(f"Fleiss' kappa: {agreement['fleiss_kappa']:.3f}")

Integration with Quality Trainer:
    The human ratings collected through this module can be integrated into the
    quality scorer training pipeline by:
    
    1. Collecting annotations using create_human_eval_file() and annotation tools
    2. Processing ratings with compute_annotation_stats() 
    3. Converting to quality_trainer format using prepare_quality_training_data()
    4. Training quality scorer with human ratings as supervision signal
    
    See quality_trainer.py documentation for detailed integration steps.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
from dataclasses import dataclass, field
import csv

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional imports for web interface
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# Statistical analysis imports
try:
    from scipy.stats import kendalltau, spearmanr, pearsonr
    from sklearn.metrics import cohen_kappa_score
    import krippendorff
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False


@dataclass
class HumanEvalConfig:
    """Configuration for human evaluation."""
    
    # Evaluation setup
    rating_scale: Tuple[int, int] = (1, 5)  # (min, max) rating scale
    include_reference: bool = True
    include_metadata: bool = True
    
    # Annotation fields
    required_fields: List[str] = field(default_factory=lambda: [
        'id', 'source', 'candidate', 'rating'
    ])
    optional_fields: List[str] = field(default_factory=lambda: [
        'reference', 'sarcasm_flag', 'dataset_name', 'comments'
    ])
    
    # Quality criteria for instructions
    quality_dimensions: List[str] = field(default_factory=lambda: [
        'semantic_similarity', 'fluency', 'naturalness', 'adequacy'
    ])
    
    # Inter-annotator agreement
    min_annotators: int = 2
    agreement_method: str = "fleiss_kappa"  # "fleiss_kappa", "krippendorff", "pearson"
    
    # Export configuration
    output_format: str = "csv"  # "csv", "jsonl", "json"
    include_timestamp: bool = True


class HumanEvalHooks:
    """
    Human evaluation utilities for paraphrase assessment.
    
    Provides tools for creating evaluation files, collecting annotations,
    analyzing inter-annotator agreement, and preparing data for quality training.
    """
    
    def __init__(self, config: Optional[HumanEvalConfig] = None):
        """
        Initialize human evaluation tools.
        
        Args:
            config: Human evaluation configuration
        """
        self.config = config or HumanEvalConfig()
        self.logger = get_logger("HumanEvalHooks")
        
        self.logger.info("Initialized HumanEvalHooks")
        if STREAMLIT_AVAILABLE:
            self.logger.info("Streamlit available for web interface")
        if GRADIO_AVAILABLE:
            self.logger.info("Gradio available for web interface")
        if STATS_AVAILABLE:
            self.logger.info("Statistical analysis libraries available")
    
    @staticmethod
    def create_human_eval_file(
        samples: List[Dict[str, Any]],
        out_path: Path,
        config: Optional[HumanEvalConfig] = None,
        annotator_id: Optional[str] = None
    ) -> Path:
        """
        Create human evaluation file from paraphrase samples.
        
        Args:
            samples: List of evaluation samples with fields:
                - source: Original source text
                - candidate: Generated paraphrase candidate  
                - reference: Reference paraphrase (optional)
                - sarcasm_flag: Whether text contains sarcasm (optional)
                - dataset_name: Source dataset name (optional)
            out_path: Output file path
            config: Evaluation configuration
            annotator_id: Optional annotator identifier
            
        Returns:
            Path to created evaluation file
        """
        config = config or HumanEvalConfig()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger = get_logger("create_human_eval_file")
        
        # Prepare evaluation data
        eval_data = []
        
        for i, sample in enumerate(samples):
            eval_item = {
                'id': f"sample_{i:06d}",
                'source': sample.get('source', ''),
                'candidate': sample.get('candidate', ''),
                'rating': '',  # To be filled by annotator
            }
            
            # Add optional fields
            if config.include_reference and 'reference' in sample:
                eval_item['reference'] = sample['reference']
            
            if config.include_metadata:
                eval_item['sarcasm_flag'] = sample.get('sarcasm_flag', False)
                eval_item['dataset_name'] = sample.get('dataset_name', 'unknown')
            
            # Add annotator information
            if annotator_id:
                eval_item['annotator_id'] = annotator_id
            
            if config.include_timestamp:
                eval_item['created_at'] = datetime.now().isoformat()
            
            # Add comments field for additional feedback
            eval_item['comments'] = ''
            
            eval_data.append(eval_item)
        
        # Save in specified format
        if config.output_format == "csv":
            df = pd.DataFrame(eval_data)
            df.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            
        elif config.output_format == "jsonl":
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in eval_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        elif config.output_format == "json":
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported output format: {config.output_format}")
        
        logger.info(f"Created evaluation file with {len(eval_data)} samples: {out_path}")
        
        # Create instruction file
        instruction_path = out_path.parent / f"{out_path.stem}_instructions.txt"
        HumanEvalHooks._create_instruction_file(instruction_path, config)
        
        return out_path
    
    @staticmethod
    def _create_instruction_file(
        instruction_path: Path,
        config: HumanEvalConfig
    ):
        """Create instruction file for human annotators."""
        
        min_rating, max_rating = config.rating_scale
        
        instructions = f"""
PARAPHRASE EVALUATION INSTRUCTIONS
=================================

Thank you for participating in this paraphrase evaluation task. Please rate the quality 
of paraphrase candidates on a scale from {min_rating} to {max_rating}.

RATING SCALE:
{max_rating} - Excellent: Perfect paraphrase with same meaning, natural and fluent
{max_rating-1} - Good: High-quality paraphrase with minor issues
{(max_rating+min_rating)//2} - Adequate: Acceptable paraphrase with some meaning preserved
{min_rating+1} - Poor: Low-quality paraphrase with significant issues  
{min_rating} - Bad: Completely incorrect or unrelated to source

EVALUATION CRITERIA:
"""
        
        criteria_descriptions = {
            'semantic_similarity': "Does the candidate preserve the meaning of the source?",
            'fluency': "Is the candidate grammatically correct and natural-sounding?",
            'naturalness': "Does the candidate sound like something a human would say?",
            'adequacy': "Is the candidate an appropriate paraphrase overall?"
        }
        
        for criterion in config.quality_dimensions:
            if criterion in criteria_descriptions:
                instructions += f"- {criterion.title()}: {criteria_descriptions[criterion]}\n"
        
        instructions += f"""
INSTRUCTIONS:
1. Read the source text carefully
2. Evaluate how well the candidate paraphrases the source
3. Consider the reference paraphrase as a quality example (if provided)
4. Assign a rating from {min_rating} to {max_rating} based on overall quality
5. Add optional comments explaining your reasoning

EXAMPLES:
Source: "The weather is beautiful today."
Candidate: "Today's weather is lovely." → Rating: {max_rating} (excellent paraphrase)
Candidate: "Weather good today." → Rating: {(max_rating+min_rating)//2} (adequate but not fluent)
Candidate: "I like pizza." → Rating: {min_rating} (completely unrelated)

Please be consistent in your ratings and take your time to carefully evaluate each example.
"""
        
        with open(instruction_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
    
    def load_annotations(self, file_path: str) -> pd.DataFrame:
        """
        Load human annotations from file.
        
        Args:
            file_path: Path to annotations file
            
        Returns:
            DataFrame with annotation data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                df = pd.DataFrame(data)
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Loaded {len(df)} annotations from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {e}")
            raise
    
    def compute_annotation_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Compute basic statistics on human annotations.
        
        Args:
            file_path: Path to annotations file
            
        Returns:
            Dictionary of annotation statistics
        """
        df = self.load_annotations(file_path)
        
        # Convert ratings to numeric, handling any non-numeric values
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Remove rows with missing ratings
        valid_df = df.dropna(subset=['rating_numeric'])
        
        if len(valid_df) == 0:
            return {'error': 'No valid ratings found'}
        
        ratings = valid_df['rating_numeric'].values
        
        stats = {
            'total_annotations': len(df),
            'valid_annotations': len(valid_df),
            'missing_ratings': len(df) - len(valid_df),
            
            'mean_rating': float(np.mean(ratings)),
            'median_rating': float(np.median(ratings)),
            'std_rating': float(np.std(ratings)),
            'min_rating': float(np.min(ratings)),
            'max_rating': float(np.max(ratings)),
            
            'rating_distribution': dict(valid_df['rating_numeric'].value_counts().sort_index()),
        }
        
        # Add annotator statistics if available
        if 'annotator_id' in df.columns:
            annotator_stats = {}
            for annotator in df['annotator_id'].unique():
                annotator_data = valid_df[valid_df['annotator_id'] == annotator]
                if len(annotator_data) > 0:
                    annotator_stats[annotator] = {
                        'count': len(annotator_data),
                        'mean_rating': float(np.mean(annotator_data['rating_numeric'])),
                        'std_rating': float(np.std(annotator_data['rating_numeric']))
                    }
            
            stats['annotator_stats'] = annotator_stats
            stats['num_annotators'] = len(annotator_stats)
        
        return stats
    
    def compute_inter_annotator_agreement(self, file_path: str) -> Dict[str, float]:
        """
        Compute inter-annotator agreement metrics.
        
        Args:
            file_path: Path to annotations file with multiple annotators
            
        Returns:
            Dictionary of agreement metrics
        """
        if not STATS_AVAILABLE:
            self.logger.warning("Statistical libraries not available for agreement computation")
            return {'error': 'Statistical libraries not available'}
        
        df = self.load_annotations(file_path)
        
        if 'annotator_id' not in df.columns:
            return {'error': 'annotator_id column required for agreement computation'}
        
        # Convert ratings to numeric
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating_numeric'])
        
        # Pivot to get annotator ratings per sample
        pivot_df = df.pivot(index='id', columns='annotator_id', values='rating_numeric')
        
        # Remove samples with missing annotations
        complete_samples = pivot_df.dropna()
        
        if len(complete_samples) < 2:
            return {'error': 'Insufficient overlapping annotations for agreement computation'}
        
        agreement_metrics = {}
        
        # Get annotator pairs for pairwise agreement
        annotators = list(complete_samples.columns)
        
        if len(annotators) < 2:
            return {'error': 'At least 2 annotators required'}
        
        # Pairwise correlations
        correlations = []
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                ann1_ratings = complete_samples[annotators[i]].values
                ann2_ratings = complete_samples[annotators[j]].values
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(ann1_ratings, ann2_ratings)
                correlations.append(pearson_r)
                
                # Store specific pair if only 2 annotators
                if len(annotators) == 2:
                    agreement_metrics['pearson_r'] = pearson_r
                    agreement_metrics['pearson_p'] = pearson_p
                    
                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(ann1_ratings, ann2_ratings)
                    agreement_metrics['spearman_r'] = spearman_r
                    agreement_metrics['spearman_p'] = spearman_p
                    
                    # Kendall's tau
                    kendall_tau, kendall_p = kendalltau(ann1_ratings, ann2_ratings)
                    agreement_metrics['kendall_tau'] = kendall_tau
                    agreement_metrics['kendall_p'] = kendall_p
        
        # Average correlation across all pairs
        if correlations:
            agreement_metrics['mean_pairwise_correlation'] = np.mean(correlations)
            agreement_metrics['std_pairwise_correlation'] = np.std(correlations)
        
        # Fleiss' kappa (for multiple annotators)
        if len(annotators) >= 2:
            try:
                # Convert to format expected by agreement libraries
                ratings_matrix = complete_samples.values
                
                # For Krippendorff's alpha (if available)
                if hasattr(krippendorff, 'alpha'):
                    krippendorff_alpha = krippendorff.alpha(
                        reliability_data=ratings_matrix.T,
                        level_of_measurement='interval'
                    )
                    agreement_metrics['krippendorff_alpha'] = krippendorff_alpha
                
                # Simplified Fleiss' kappa approximation
                # (Note: This is a simplified version; for production use, 
                #  consider using dedicated agreement libraries)
                agreement_metrics['fleiss_kappa_approx'] = self._compute_fleiss_kappa_approx(ratings_matrix)
                
            except Exception as e:
                self.logger.warning(f"Advanced agreement computation failed: {e}")
        
        agreement_metrics['num_samples'] = len(complete_samples)
        agreement_metrics['num_annotators'] = len(annotators)
        
        return agreement_metrics
    
    def _compute_fleiss_kappa_approx(self, ratings_matrix: np.ndarray) -> float:
        """
        Compute approximation of Fleiss' kappa.
        
        Note: This is a simplified implementation. For production use,
        consider using specialized libraries like `krippendorff` or `statsmodels`.
        """
        n_subjects, n_raters = ratings_matrix.shape
        
        # Get unique categories
        categories = np.unique(ratings_matrix)
        n_categories = len(categories)
        
        # Create category counts matrix
        counts = np.zeros((n_subjects, n_categories))
        for i, category in enumerate(categories):
            counts[:, i] = np.sum(ratings_matrix == category, axis=1)
        
        # Compute proportions
        p_j = np.sum(counts, axis=0) / (n_subjects * n_raters)
        
        # Compute P_e (expected agreement)
        P_e = np.sum(p_j ** 2)
        
        # Compute P_o (observed agreement)  
        P_o = 0.0
        for i in range(n_subjects):
            for j in range(n_categories):
                P_o += counts[i, j] * (counts[i, j] - 1)
        P_o = P_o / (n_subjects * n_raters * (n_raters - 1))
        
        # Fleiss' kappa
        if P_e == 1.0:
            return 1.0
        else:
            return (P_o - P_e) / (1 - P_e)
    
    def prepare_quality_training_data(
        self,
        annotation_file: str,
        output_file: str,
        min_rating: Optional[float] = None
    ) -> str:
        """
        Prepare human annotations for quality scorer training.
        
        Args:
            annotation_file: Path to human annotations
            output_file: Path to output training data file
            min_rating: Minimum rating threshold for filtering
            
        Returns:
            Path to prepared training data file
        """
        df = self.load_annotations(annotation_file)
        
        # Convert ratings to numeric and normalize to [0, 1]
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating_numeric'])
        
        # Normalize ratings to [0, 1] scale
        min_scale, max_scale = self.config.rating_scale
        df['quality_score'] = (df['rating_numeric'] - min_scale) / (max_scale - min_scale)
        
        # Filter by minimum rating if specified
        if min_rating is not None:
            df = df[df['quality_score'] >= min_rating]
        
        # Prepare training format
        training_data = []
        for _, row in df.iterrows():
            training_sample = {
                'source_text': row['source'],
                'paraphrase_text': row['candidate'],
                'quality_score': row['quality_score']
            }
            
            # Add reference if available
            if 'reference' in row and pd.notna(row['reference']):
                training_sample['target_text'] = row['reference']
            
            training_data.append(training_sample)
        
        # Save training data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Prepared {len(training_data)} quality training samples: {output_path}")
        
        return str(output_path)
    
    def create_streamlit_annotation_app(self, eval_file: str, output_file: str) -> Optional[str]:
        """
        Create a Streamlit app for web-based annotation.
        
        Args:
            eval_file: Path to evaluation samples file
            output_file: Path to save annotations
            
        Returns:
            Path to Streamlit app file or None if unavailable
        """
        if not STREAMLIT_AVAILABLE:
            self.logger.warning("Streamlit not available")
            return None
        
        app_code = f'''
import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.title("Paraphrase Quality Annotation")

# Load evaluation data
@st.cache_data
def load_eval_data():
    return pd.read_csv("{eval_file}")

df = load_eval_data()

# Session state for tracking progress
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if 'annotations' not in st.session_state:
    st.session_state.annotations = []

# Progress tracking
st.progress((st.session_state.current_index + 1) / len(df))
st.write(f"Sample {{st.session_state.current_index + 1}} of {{len(df)}}")

# Current sample
if st.session_state.current_index < len(df):
    sample = df.iloc[st.session_state.current_index]
    
    st.subheader("Source Text")
    st.write(sample['source'])
    
    if 'reference' in sample:
        st.subheader("Reference Paraphrase")
        st.write(sample.get('reference', 'N/A'))
    
    st.subheader("Candidate Paraphrase")
    st.write(sample['candidate'])
    
    # Rating input
    rating = st.selectbox(
        "Quality Rating (1=Poor, 5=Excellent)",
        options=[1, 2, 3, 4, 5],
        index=2
    )
    
    # Comments
    comments = st.text_area("Comments (optional)")
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.experimental_rerun()
    
    with col2:
        if st.button("Save & Next"):
            # Save annotation
            annotation = {{
                'id': sample['id'],
                'rating': rating,
                'comments': comments,
                'annotator_id': 'streamlit_user'
            }}
            st.session_state.annotations.append(annotation)
            
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
            else:
                st.success("All samples annotated!")
            st.experimental_rerun()
    
    with col3:
        if st.button("Export Annotations"):
            output_df = pd.DataFrame(st.session_state.annotations)
            output_df.to_csv("{output_file}", index=False)
            st.success(f"Annotations exported to {{output_file}}")

else:
    st.success("Annotation complete!")
    if st.session_state.annotations:
        output_df = pd.DataFrame(st.session_state.annotations)
        output_df.to_csv("{output_file}", index=False)
        st.write(f"Final annotations saved to {{output_file}}")
'''
        
        app_file = Path(output_file).parent / "streamlit_annotation_app.py"
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(app_code)
        
        self.logger.info(f"Created Streamlit annotation app: {app_file}")
        self.logger.info(f"Run with: streamlit run {app_file}")
        
        return str(app_file)


def main():
    """CLI interface for human evaluation tools."""
    
    parser = argparse.ArgumentParser(description="Human evaluation tools for paraphrasing")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create evaluation file
    create_parser = subparsers.add_parser('create', help='Create human evaluation file')
    create_parser.add_argument('--samples', type=str, required=True, help='Input samples file (JSON)')
    create_parser.add_argument('--output', type=str, required=True, help='Output evaluation file')
    create_parser.add_argument('--format', type=str, default='csv', choices=['csv', 'jsonl', 'json'],
                              help='Output format')
    create_parser.add_argument('--annotator-id', type=str, help='Annotator identifier')
    
    # Analyze annotations
    analyze_parser = subparsers.add_parser('analyze', help='Analyze annotation statistics')
    analyze_parser.add_argument('--file', type=str, required=True, help='Annotations file')
    analyze_parser.add_argument('--agreement', action='store_true', help='Compute inter-annotator agreement')
    
    # Prepare for training
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for quality training')
    prepare_parser.add_argument('--annotations', type=str, required=True, help='Annotations file')
    prepare_parser.add_argument('--output', type=str, required=True, help='Output training data file')
    prepare_parser.add_argument('--min-rating', type=float, help='Minimum rating threshold')
    
    # Create web app
    webapp_parser = subparsers.add_parser('webapp', help='Create Streamlit annotation app')
    webapp_parser.add_argument('--eval-file', type=str, required=True, help='Evaluation file')
    webapp_parser.add_argument('--output', type=str, required=True, help='Output annotations file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize evaluator
    evaluator = HumanEvalHooks()
    
    if args.command == 'create':
        # Load sample data
        with open(args.samples, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # Create config
        config = HumanEvalConfig(output_format=args.format)
        
        # Create evaluation file
        output_path = HumanEvalHooks.create_human_eval_file(
            samples, Path(args.output), config, args.annotator_id
        )
        print(f"Created evaluation file: {output_path}")
    
    elif args.command == 'analyze':
        # Compute statistics
        stats = evaluator.compute_annotation_stats(args.file)
        
        print("=== Annotation Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # Compute agreement if requested
        if args.agreement:
            print("\n=== Inter-Annotator Agreement ===")
            agreement = evaluator.compute_inter_annotator_agreement(args.file)
            
            for key, value in agreement.items():
                print(f"{key}: {value}")
    
    elif args.command == 'prepare':
        # Prepare training data
        output_file = evaluator.prepare_quality_training_data(
            args.annotations, args.output, args.min_rating
        )
        print(f"Prepared quality training data: {output_file}")
    
    elif args.command == 'webapp':
        # Create Streamlit app
        app_file = evaluator.create_streamlit_annotation_app(args.eval_file, args.output)
        if app_file:
            print(f"Created Streamlit app: {app_file}")
            print(f"Run with: streamlit run {app_file}")
        else:
            print("Streamlit not available")


if __name__ == "__main__":
    main()
