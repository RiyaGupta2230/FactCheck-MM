"""
Metrics Computation for FactCheck-MM
Task-specific metrics for sarcasm detection, paraphrasing, and fact verification.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

from .logging_utils import get_logger

try:
    from sacrebleu import corpus_bleu, sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


class BaseMetrics:
    """Base class for metrics computation."""
    
    def __init__(self, task_name: str):
        """
        Initialize metrics computer.
        
        Args:
            task_name: Name of the task
        """
        self.task_name = task_name
        self.logger = get_logger(f"Metrics_{task_name}")
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            self.logger.warning("Failed to download NLTK data")
    
    def compute_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List],
        labels: Union[torch.Tensor, np.ndarray, List],
        mode: str = "eval"
    ) -> Dict[str, float]:
        """
        Compute metrics for predictions.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            mode: Computation mode ('train', 'eval', 'test')
            
        Returns:
            Dictionary of computed metrics
        """
        raise NotImplementedError("Subclasses must implement compute_metrics")


class ClassificationMetrics(BaseMetrics):
    """Metrics for classification tasks (sarcasm detection, stance classification)."""
    
    def __init__(self, task_name: str, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize classification metrics.
        
        Args:
            task_name: Task name
            num_classes: Number of classes
            class_names: Names of classes
        """
        super().__init__(task_name)
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    def compute_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List],
        labels: Union[torch.Tensor, np.ndarray, List],
        mode: str = "eval"
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predictions (logits or class indices)
            labels: True labels
            mode: Computation mode
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                # Convert logits to class predictions
                predictions = torch.argmax(predictions, dim=-1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=range(self.num_classes)
        )
        
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        metrics = {
            f"{mode}_accuracy": float(accuracy),
            f"{mode}_precision": float(precision),
            f"{mode}_recall": float(recall),
            f"{mode}_f1": float(f1),
            f"{mode}_precision_macro": float(precision_macro),
            f"{mode}_recall_macro": float(recall_macro),
            f"{mode}_f1_macro": float(f1_macro),
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f"{mode}_precision_{class_name}"] = float(precision_per_class[i])
                metrics[f"{mode}_recall_{class_name}"] = float(recall_per_class[i])
                metrics[f"{mode}_f1_{class_name}"] = float(f1_per_class[i])
        
        # ROC AUC for binary classification
        if self.num_classes == 2:
            try:
                auc = roc_auc_score(labels, predictions)
                metrics[f"{mode}_auc"] = float(auc)
            except ValueError:
                pass  # AUC not computable (single class in batch)
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predictions
            labels: True labels
            
        Returns:
            Confusion matrix
        """
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=-1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        return confusion_matrix(labels, predictions)
    
    def get_classification_report(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            predictions: Predictions
            labels: True labels
            
        Returns:
            Classification report string
        """
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=-1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        return classification_report(
            labels, predictions, 
            target_names=self.class_names,
            digits=4
        )


class GenerationMetrics(BaseMetrics):
    """Metrics for text generation tasks (paraphrasing)."""
    
    def __init__(self, task_name: str):
        """Initialize generation metrics."""
        super().__init__(task_name)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def compute_bleu_score(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: Generated texts
            references: Reference texts (list of lists)
            
        Returns:
            BLEU scores
        """
        bleu_scores = {}
        
        if SACREBLEU_AVAILABLE:
            # Corpus-level BLEU
            try:
                bleu = corpus_bleu(predictions, references)
                bleu_scores['bleu'] = bleu.score
                bleu_scores['bleu_1'] = bleu.precisions[0]
                bleu_scores['bleu_2'] = bleu.precisions[1]
                bleu_scores['bleu_3'] = bleu.precisions[2]
                bleu_scores['bleu_4'] = bleu.precisions[3]
            except Exception as e:
                self.logger.debug(f"BLEU computation failed: {e}")
        else:
            # Fallback to sentence-level BLEU
            try:
                from nltk.translate.bleu_score import sentence_bleu
                bleu_scores_list = []
                for pred, refs in zip(predictions, references):
                    score = sentence_bleu(refs, pred.split())
                    bleu_scores_list.append(score)
                bleu_scores['bleu'] = np.mean(bleu_scores_list)
            except Exception as e:
                self.logger.debug(f"NLTK BLEU computation failed: {e}")
                bleu_scores['bleu'] = 0.0
        
        return bleu_scores
    
    def compute_rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            ROUGE scores
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Average scores
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def compute_bert_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            BERTScore metrics
        """
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            self.logger.debug(f"BERTScore computation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }
    
    def compute_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute semantic similarity using sentence transformers.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            Semantic similarity scores
        """
        if self.sentence_model is None:
            return {'semantic_similarity': 0.0}
        
        try:
            # Encode texts
            pred_embeddings = self.sentence_model.encode(predictions)
            ref_embeddings = self.sentence_model.encode(references)
            
            # Compute cosine similarities
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = np.dot(pred_emb, ref_emb) / (
                    np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
                )
                similarities.append(similarity)
            
            return {
                'semantic_similarity': np.mean(similarities),
                'semantic_similarity_std': np.std(similarities)
            }
        except Exception as e:
            self.logger.debug(f"Semantic similarity computation failed: {e}")
            return {'semantic_similarity': 0.0}
    
    def compute_metrics(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        mode: str = "eval"
    ) -> Dict[str, float]:
        """
        Compute generation metrics.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            mode: Computation mode
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Ensure references is list of lists for BLEU
        if isinstance(references[0], str):
            bleu_references = [[ref] for ref in references]
            rouge_references = references
        else:
            bleu_references = references
            rouge_references = [ref[0] for ref in references]  # Use first reference for ROUGE
        
        # BLEU scores
        bleu_scores = self.compute_bleu_score(predictions, bleu_references)
        for key, value in bleu_scores.items():
            metrics[f"{mode}_{key}"] = float(value)
        
        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, rouge_references)
        for key, value in rouge_scores.items():
            metrics[f"{mode}_{key}"] = float(value)
        
        # BERTScore
        bert_scores = self.compute_bert_score(predictions, rouge_references)
        for key, value in bert_scores.items():
            metrics[f"{mode}_{key}"] = float(value)
        
        # Semantic similarity
        semantic_scores = self.compute_semantic_similarity(predictions, rouge_references)
        for key, value in semantic_scores.items():
            metrics[f"{mode}_{key}"] = float(value)
        
        # Length statistics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in rouge_references]
        
        metrics[f"{mode}_pred_length_mean"] = float(np.mean(pred_lengths))
        metrics[f"{mode}_pred_length_std"] = float(np.std(pred_lengths))
        metrics[f"{mode}_ref_length_mean"] = float(np.mean(ref_lengths))
        metrics[f"{mode}_length_ratio"] = float(np.mean(pred_lengths) / np.mean(ref_lengths))
        
        return metrics


class FactVerificationMetrics(BaseMetrics):
    """Metrics for fact verification tasks."""
    
    def __init__(self, task_name: str):
        """Initialize fact verification metrics."""
        super().__init__(task_name)
        
        # Standard FEVER classes
        self.fever_classes = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        self.classification_metrics = ClassificationMetrics(
            task_name, 
            num_classes=3, 
            class_names=self.fever_classes
        )
    
    def compute_fever_score(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute FEVER score (strict accuracy considering evidence).
        
        Args:
            predictions: List of predictions with 'label' and 'evidence' keys
            gold_labels: List of gold labels with same structure
            
        Returns:
            FEVER score metrics
        """
        correct_label = 0
        correct_evidence = 0
        correct_both = 0
        total = len(predictions)
        
        for pred, gold in zip(predictions, gold_labels):
            pred_label = pred.get('label', 'NOT ENOUGH INFO')
            gold_label = gold.get('label', 'NOT ENOUGH INFO')
            pred_evidence = set(pred.get('evidence', []))
            gold_evidence = set(gold.get('evidence', []))
            
            # Label accuracy
            if pred_label == gold_label:
                correct_label += 1
                
                # Evidence accuracy (only for SUPPORTS/REFUTES)
                if gold_label in ['SUPPORTS', 'REFUTES']:
                    if len(pred_evidence.intersection(gold_evidence)) > 0:
                        correct_evidence += 1
                        correct_both += 1
                else:
                    # For NOT ENOUGH INFO, no evidence is needed
                    correct_both += 1
        
        return {
            'fever_score': correct_both / total if total > 0 else 0.0,
            'label_accuracy': correct_label / total if total > 0 else 0.0,
            'evidence_precision': correct_evidence / max(sum(1 for pred in predictions 
                                                            if pred.get('label') in ['SUPPORTS', 'REFUTES']), 1),
            'evidence_recall': correct_evidence / max(sum(1 for gold in gold_labels 
                                                         if gold.get('label') in ['SUPPORTS', 'REFUTES']), 1)
        }
    
    def compute_retrieval_metrics(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics (Precision@k, Recall@k, MRR).
        
        Args:
            retrieved_docs: List of retrieved document lists for each query
            relevant_docs: List of relevant document lists for each query
            k_values: K values for Precision@k and Recall@k
            
        Returns:
            Retrieval metrics
        """
        metrics = {}
        
        # Precision@k and Recall@k
        for k in k_values:
            precisions_at_k = []
            recalls_at_k = []
            
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                retrieved_k = retrieved[:k]
                relevant_set = set(relevant)
                retrieved_set = set(retrieved_k)
                
                # Precision@k
                if len(retrieved_k) > 0:
                    precision = len(retrieved_set.intersection(relevant_set)) / len(retrieved_k)
                else:
                    precision = 0.0
                precisions_at_k.append(precision)
                
                # Recall@k
                if len(relevant) > 0:
                    recall = len(retrieved_set.intersection(relevant_set)) / len(relevant)
                else:
                    recall = 0.0
                recalls_at_k.append(recall)
            
            metrics[f'precision_at_{k}'] = np.mean(precisions_at_k)
            metrics[f'recall_at_{k}'] = np.mean(recalls_at_k)
        
        # Mean Reciprocal Rank (MRR)
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            relevant_set = set(relevant)
            rr = 0.0
            
            for i, doc in enumerate(retrieved):
                if doc in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            
            reciprocal_ranks.append(rr)
        
        metrics['mrr'] = np.mean(reciprocal_ranks)
        
        return metrics
    
    def compute_metrics(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        labels: Union[torch.Tensor, List[Dict[str, Any]]],
        mode: str = "eval"
    ) -> Dict[str, float]:
        """
        Compute fact verification metrics.
        
        Args:
            predictions: Predictions (logits or structured predictions)
            labels: Labels (class indices or structured labels)
            mode: Computation mode
            
        Returns:
            Dictionary of metrics
        """
        if isinstance(predictions, torch.Tensor):
            # Simple classification case
            return self.classification_metrics.compute_metrics(
                predictions, labels, mode
            )
        
        # Structured prediction case
        metrics = {}
        
        # Extract class predictions for classification metrics
        pred_classes = []
        true_classes = []
        
        for pred, label in zip(predictions, labels):
            pred_label = pred.get('label', 'NOT ENOUGH INFO')
            true_label = label.get('label', 'NOT ENOUGH INFO')
            
            pred_classes.append(self.fever_classes.index(pred_label))
            true_classes.append(self.fever_classes.index(true_label))
        
        # Classification metrics
        class_metrics = self.classification_metrics.compute_metrics(
            pred_classes, true_classes, mode
        )
        metrics.update(class_metrics)
        
        # FEVER score
        fever_metrics = self.compute_fever_score(predictions, labels)
        for key, value in fever_metrics.items():
            metrics[f"{mode}_{key}"] = float(value)
        
        return metrics


class MetricsComputer:
    """Main metrics computer that delegates to task-specific metrics."""
    
    def __init__(self, task_name: str, **kwargs):
        """
        Initialize metrics computer for specific task.
        
        Args:
            task_name: Task name
            **kwargs: Task-specific arguments
        """
        self.task_name = task_name
        self.logger = get_logger("MetricsComputer")
        
        # Initialize appropriate metrics computer
        if task_name == "sarcasm_detection":
            self.metrics = ClassificationMetrics(task_name, num_classes=2, 
                                                class_names=['non_sarcastic', 'sarcastic'])
        elif task_name == "paraphrasing":
            self.metrics = GenerationMetrics(task_name)
        elif task_name == "fact_verification":
            self.metrics = FactVerificationMetrics(task_name)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        self.logger.info(f"Initialized metrics computer for {task_name}")
    
    def compute_metrics(self, *args, **kwargs) -> Dict[str, float]:
        """Compute metrics using task-specific implementation."""
        return self.metrics.compute_metrics(*args, **kwargs)
    
    def get_primary_metric(self) -> str:
        """Get primary metric name for this task."""
        if self.task_name == "sarcasm_detection":
            return "eval_f1"
        elif self.task_name == "paraphrasing":
            return "eval_bleu"
        elif self.task_name == "fact_verification":
            return "eval_fever_score"
        else:
            return "eval_accuracy"
