#!/usr/bin/env python3

"""
Unified Fact Verification Dataset Loader
Combines FEVER and LIAR datasets into a unified fact-checking dataset with
consistent label schema, balanced sampling, and curriculum learning capabilities
for comprehensive fact verification model training.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .fever_loader import FeverDataset, FeverDatasetConfig
from .liar_loader import LiarDataset, LiarDatasetConfig
from shared.preprocessing.text_processor import TextProcessor, TextProcessorConfig
from shared.datasets.data_loaders import ChunkedDataLoader
from shared.utils.logging_utils import get_logger

@dataclass
class UnifiedFactDatasetConfig:
    """Configuration for unified fact verification dataset."""
    
    use_fever: bool = True
    use_liar: bool = True
    fever_only: bool = False
    liar_only: bool = False
    
    balance_datasets: bool = True
    fever_sample_ratio: float = 0.7
    liar_sample_ratio: float = 0.3
    balance_labels: bool = True
    max_samples_per_dataset: Optional[int] = None
    
    unified_label_mapping: Dict[str, int] = field(default_factory=lambda: {
        'SUPPORTS': 0,
        'REFUTES': 1,
        'NOT_ENOUGH_INFO': 2
    })
    
    liar_to_unified: Dict[str, str] = field(default_factory=lambda: {
        'true': 'SUPPORTS',
        'mostly-true': 'SUPPORTS',
        'half-true': 'NOT_ENOUGH_INFO',
        'mostly-false': 'REFUTES',
        'false': 'REFUTES',
        'pants-fire': 'REFUTES'
    })
    
    curriculum_learning: bool = False
    curriculum_stage: float = 1.0
    curriculum_stages: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0])
    
    max_text_length: int = 256
    standardize_text_format: bool = True
    include_source_info: bool = True
    
    shuffle_datasets: bool = True
    seed: int = 42
    cache_unified_samples: bool = True

class UnifiedFactDataset(Dataset):
    """
    Unified fact verification dataset combining FEVER and LIAR.
    Provides consistent three-class labeling (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
    with flexible sampling strategies and curriculum learning support.
    """
    
    def __init__(
        self,
        split: str = "train",
        config: Optional[UnifiedFactDatasetConfig] = None,
        fever_config: Optional[FeverDatasetConfig] = None,
        liar_config: Optional[LiarDatasetConfig] = None,
        text_processor: Optional[TextProcessor] = None
    ):
        """
        Initialize unified fact verification dataset.
        
        Args:
            split: Data split ('train', 'test', 'valid')
            config: Unified dataset configuration
            fever_config: FEVER dataset configuration
            liar_config: LIAR dataset configuration
            text_processor: Text preprocessing pipeline
        """
        self.split = split
        self.config = config or UnifiedFactDatasetConfig()
        self.logger = get_logger("UnifiedFactDataset")
        
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if text_processor is None:
            processor_config = TextProcessorConfig(
                model_name="roberta-base",
                max_length=self.config.max_text_length,
                lowercase=False,
                remove_special_chars=False
            )
            self.text_processor = TextProcessor(processor_config)
        else:
            self.text_processor = text_processor
        
        self.fever_dataset = None
        self.liar_dataset = None
        
        self.unified_samples = []
        self.dataset_indices = []
        
        self._load_constituent_datasets(fever_config, liar_config)
        self._create_unified_dataset()
        
        self.logger.info(
            f"Created unified {split} dataset with {len(self.unified_samples)} samples "
            f"(FEVER: {self._count_fever_samples()}, LIAR: {self._count_liar_samples()})"
        )
    
    def _load_constituent_datasets(
        self,
        fever_config: Optional[FeverDatasetConfig],
        liar_config: Optional[LiarDatasetConfig]
    ):
        """Load individual FEVER and LIAR datasets."""
        if self.config.use_fever and not self.config.liar_only:
            try:
                if fever_config is None:
                    fever_config = FeverDatasetConfig(
                        max_claim_length=self.config.max_text_length // 2,
                        max_evidence_length=self.config.max_text_length // 2,
                        filter_no_evidence=False
                    )
                self.fever_dataset = FeverDataset(
                    self.split,
                    config=fever_config,
                    text_processor=self.text_processor
                )
                self.logger.info(f"Loaded FEVER dataset: {len(self.fever_dataset)} samples")
            except Exception as e:
                self.logger.error(f"Failed to load FEVER dataset: {e}")
                raise
        
        if self.config.use_liar and not self.config.fever_only:
            try:
                if liar_config is None:
                    liar_config = LiarDatasetConfig(
                        max_statement_length=self.config.max_text_length,
                        binary_labels=False,
                        include_context_info=True
                    )
                self.liar_dataset = LiarDataset(
                    self.split,
                    config=liar_config,
                    text_processor=self.text_processor
                )
                self.logger.info(f"Loaded LIAR dataset: {len(self.liar_dataset)} samples")
            except Exception as e:
                self.logger.error(f"Failed to load LIAR dataset: {e}")
                raise
    
    def _create_unified_dataset(self):
        """Create unified dataset by combining and balancing constituent datasets."""
        fever_samples = self._extract_fever_samples()
        liar_samples = self._extract_liar_samples()
        
        if self.config.curriculum_learning:
            fever_samples, liar_samples = self._apply_curriculum_filtering(fever_samples, liar_samples)
        
        if self.config.balance_datasets:
            fever_samples, liar_samples = self._balance_datasets(fever_samples, liar_samples)
        
        all_samples = fever_samples + liar_samples
        dataset_indices = (['fever'] * len(fever_samples) +
                          ['liar'] * len(liar_samples))
        
        if self.config.shuffle_datasets:
            combined = list(zip(all_samples, dataset_indices))
            random.shuffle(combined)
            all_samples, dataset_indices = zip(*combined)
            all_samples = list(all_samples)
            dataset_indices = list(dataset_indices)
        
        self.unified_samples = all_samples
        self.dataset_indices = dataset_indices
        
        self.logger.info(
            f"Created unified dataset: {len(fever_samples)} FEVER + "
            f"{len(liar_samples)} LIAR = {len(self.unified_samples)} total"
        )
    
    def _extract_fever_samples(self) -> List[Dict[str, Any]]:
        """Extract and convert FEVER samples to unified format."""
        if not self.fever_dataset:
            return []
        
        fever_samples = []
        for i in range(len(self.fever_dataset)):
            try:
                original_sample = self.fever_dataset[i]
                
                unified_sample = {
                    'claim': original_sample['claim_text'],
                    'evidence': original_sample.get('evidence_text', []),
                    'original_label': original_sample['label_text'],
                    'unified_label': original_sample['label_text'],
                    'label_id': original_sample['label'].item(),
                    'source_dataset': 'fever',
                    'original_id': original_sample['sample_id']
                }
                
                if self.config.standardize_text_format:
                    unified_sample.update(self._standardize_text_inputs(unified_sample))
                
                fever_samples.append(unified_sample)
            except Exception as e:
                self.logger.warning(f"Error converting FEVER sample {i}: {e}")
                continue
        
        return fever_samples
    
    def _extract_liar_samples(self) -> List[Dict[str, Any]]:
        """Extract and convert LIAR samples to unified format."""
        if not self.liar_dataset:
            return []
        
        liar_samples = []
        for i in range(len(self.liar_dataset)):
            try:
                original_sample = self.liar_dataset[i]
                original_label = original_sample['original_label']
                
                unified_label = self.config.liar_to_unified.get(original_label, 'NOT_ENOUGH_INFO')
                unified_label_id = self.config.unified_label_mapping.get(unified_label, 2)
                
                unified_sample = {
                    'claim': original_sample['statement_text'],
                    'evidence': [],
                    'original_label': original_label,
                    'unified_label': unified_label,
                    'label_id': unified_label_id,
                    'source_dataset': 'liar',
                    'original_id': original_sample['sample_id']
                }
                
                if 'speaker' in original_sample:
                    unified_sample['speaker'] = original_sample['speaker']
                    unified_sample['party_affiliation'] = original_sample.get('party_affiliation', '')
                
                if self.config.standardize_text_format:
                    unified_sample.update(self._standardize_text_inputs(unified_sample))
                
                liar_samples.append(unified_sample)
            except Exception as e:
                self.logger.warning(f"Error converting LIAR sample {i}: {e}")
                continue
        
        return liar_samples
    
    def _standardize_text_inputs(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Standardize text inputs for unified processing."""
        claim = sample['claim']
        evidence = sample.get('evidence', [])
        
        if evidence and len(evidence) > 0:
            if isinstance(evidence, list):
                evidence_text = " [SEP] ".join(evidence)
            else:
                evidence_text = str(evidence)
            
            inputs = self.text_processor.process_text_pair(
                claim, evidence_text,
                max_length_a=self.config.max_text_length // 2,
                max_length_b=self.config.max_text_length // 2,
                truncation="longest_first"
            )
        else:
            inputs = self.text_processor.process_text(
                claim,
                max_length=self.config.max_text_length,
                truncation=True
            )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    
    def _apply_curriculum_filtering(
        self,
        fever_samples: List[Dict[str, Any]],
        liar_samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply curriculum learning filtering based on stage."""
        stage = self.config.curriculum_stage
        
        if stage <= 0.0:
            return fever_samples, []
        elif stage >= 1.0:
            return fever_samples, liar_samples
        else:
            liar_count = int(len(liar_samples) * stage)
            return fever_samples, liar_samples[:liar_count]
    
    def _balance_datasets(
        self,
        fever_samples: List[Dict[str, Any]],
        liar_samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Balance samples between datasets according to configured ratios."""
        if not fever_samples or not liar_samples:
            return fever_samples, liar_samples
        
        total_available = len(fever_samples) + len(liar_samples)
        
        if self.config.max_samples_per_dataset:
            total_target = min(total_available, self.config.max_samples_per_dataset * 2)
        else:
            total_target = total_available
        
        fever_target = int(total_target * self.config.fever_sample_ratio)
        liar_target = int(total_target * self.config.liar_sample_ratio)
        
        if len(fever_samples) > fever_target:
            fever_samples = random.sample(fever_samples, fever_target)
        if len(liar_samples) > liar_target:
            liar_samples = random.sample(liar_samples, liar_target)
        
        self.logger.info(
            f"Balanced datasets: {len(fever_samples)} FEVER, {len(liar_samples)} LIAR"
        )
        
        if self.config.balance_labels:
            fever_samples = self._balance_labels_within_dataset(fever_samples)
            liar_samples = self._balance_labels_within_dataset(liar_samples)
        
        return fever_samples, liar_samples
    
    def _balance_labels_within_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance labels within a single dataset."""
        label_groups = {}
        for sample in samples:
            label_id = sample['label_id']
            if label_id not in label_groups:
                label_groups[label_id] = []
            label_groups[label_id].append(sample)
        
        if len(label_groups) <= 1:
            return samples
        
        min_size = min(len(group) for group in label_groups.values())
        
        balanced_samples = []
        for group in label_groups.values():
            balanced_samples.extend(random.sample(group, min(len(group), min_size)))
        
        return balanced_samples
    
    def _count_fever_samples(self) -> int:
        """Count FEVER samples in unified dataset."""
        return sum(1 for idx in self.dataset_indices if idx == 'fever')
    
    def _count_liar_samples(self) -> int:
        """Count LIAR samples in unified dataset."""
        return sum(1 for idx in self.dataset_indices if idx == 'liar')
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.unified_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the unified dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with unified model inputs
        """
        if idx >= len(self.unified_samples):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.unified_samples)}")
        
        sample = self.unified_samples[idx]
        source_dataset = self.dataset_indices[idx]
        
        result = {
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'label': torch.tensor(sample['label_id'], dtype=torch.long)
        }
        
        result['sample_id'] = sample['original_id']
        result['claim'] = sample['claim']
        result['evidence'] = sample.get('evidence', [])
        result['unified_label'] = sample['unified_label']
        result['original_label'] = sample['original_label']
        result['source_dataset'] = source_dataset
        
        if source_dataset == 'liar' and 'speaker' in sample:
            result['speaker'] = sample['speaker']
            result['party_affiliation'] = sample.get('party_affiliation', '')
        
        return result
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the unified dataset."""
        stats = {
            'total_samples': len(self.unified_samples),
            'fever_samples': self._count_fever_samples(),
            'liar_samples': self._count_liar_samples(),
            'unified_label_distribution': {},
            'source_distribution': {
                'fever': self._count_fever_samples() / len(self.unified_samples) * 100,
                'liar': self._count_liar_samples() / len(self.unified_samples) * 100
            }
        }
        
        for sample in self.unified_samples:
            label = sample['unified_label']
            stats['unified_label_distribution'][label] = stats['unified_label_distribution'].get(label, 0) + 1
        
        if self.unified_samples:
            claim_lengths = [len(s['claim'].split()) for s in self.unified_samples]
            stats['avg_claim_length'] = np.mean(claim_lengths)
            stats['max_claim_length'] = np.max(claim_lengths)
            stats['min_claim_length'] = np.min(claim_lengths)
            
            evidence_counts = [
                len(s.get('evidence', [])) for s in self.unified_samples
            ]
            stats['avg_evidence_sentences'] = np.mean(evidence_counts)
            stats['samples_with_evidence'] = sum(1 for c in evidence_counts if c > 0)
            stats['evidence_coverage'] = stats['samples_with_evidence'] / len(self.unified_samples) * 100
        
        return stats
    
    def get_label_mapping_info(self) -> Dict[str, Any]:
        """Get information about label mappings used."""
        return {
            'unified_labels': list(self.config.unified_label_mapping.keys()),
            'unified_mapping': self.config.unified_label_mapping,
            'liar_to_unified': self.config.liar_to_unified
        }
    
    def create_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        use_chunked: bool = False
    ) -> Union[torch.utils.data.DataLoader, ChunkedDataLoader]:
        """
        Create a DataLoader for the unified dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            use_chunked: Whether to use chunked loading
            
        Returns:
            DataLoader instance
        """
        if use_chunked:
            return ChunkedDataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                chunk_size=1000
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_fn
            )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        collated = {}
        
        for key in ['input_ids', 'attention_mask', 'label']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        metadata_keys = [
            'sample_id', 'claim', 'evidence', 'unified_label',
            'original_label', 'source_dataset', 'speaker', 'party_affiliation'
        ]
        for key in metadata_keys:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated

def main():
    """Example usage of UnifiedFactDataset."""
    configs = [
        ("Balanced Mix", UnifiedFactDatasetConfig(
            use_fever=True, use_liar=True, balance_datasets=True
        )),
        ("FEVER Only", UnifiedFactDatasetConfig(
            fever_only=True
        )),
        ("LIAR Only", UnifiedFactDatasetConfig(
            liar_only=True
        )),
        ("Curriculum Stage 0.5", UnifiedFactDatasetConfig(
            use_fever=True, use_liar=True, curriculum_learning=True, curriculum_stage=0.5
        ))
    ]
    
    for config_name, config in configs:
        try:
            print(f"\n=== {config_name} ===")
            
            unified_data = UnifiedFactDataset('train', config=config)
            print(f"Dataset size: {len(unified_data)}")
            
            stats = unified_data.get_dataset_statistics()
            print("\nDataset Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
            
            if len(unified_data) > 0:
                sample = unified_data[0]
                print(f"\nSample:")
                print(f"Claim: {sample['claim'][:100]}...")
                print(f"Unified Label: {sample['unified_label']} ({sample['label'].item()})")
                print(f"Source: {sample['source_dataset']}")
                print(f"Input shape: {sample['input_ids'].shape}")
                
                if sample['evidence']:
                    print(f"Evidence: {str(sample['evidence'])[:100]}...")
            
            if config_name == "Balanced Mix":
                mapping_info = unified_data.get_label_mapping_info()
                print(f"\nLabel Mapping:")
                for k, v in mapping_info['liar_to_unified'].items():
                    print(f"  LIAR '{k}' -> '{v}'")
        
        except Exception as e:
            print(f"Error with {config_name}: {e}")
    
    print("\nNote: Ensure both FEVER and LIAR data files are available:")
    print("  data/FEVER/fever_train.jsonl")
    print("  data/LIAR/train_formatted.csv")

if __name__ == "__main__":
    main()
