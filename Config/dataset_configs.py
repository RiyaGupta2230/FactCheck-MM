"""
Dataset Configuration for FactCheck-MM
Paths and settings for all 10 datasets used in the pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class DatasetConfig:
    """Base configuration for a single dataset."""
    name: str
    path: Path
    task: str
    modalities: List[str]
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    format: str = "json"  # json, csv, jsonl, tsv
    encoding: str = "utf-8"
    preprocessing_required: bool = True

@dataclass
class DatasetConfigs:
    """Complete dataset configuration for all tasks."""
    
    # Root data directory
    data_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    # Sarcasm Detection Datasets
    sarcasm_datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "mustard": DatasetConfig(
            name="mustard",
            path=Path("mustard_repo"),
            task="sarcasm_detection",
            modalities=["text", "audio", "video", "context"],
            train_file="data/sarcasm_data.json",
            format="json"
        ),
        "mmsd2": DatasetConfig(
            name="mmsd2",
            path=Path("mmsd2"),
            task="sarcasm_detection",
            modalities=["text", "image"],
            train_file="text_json_final/train.json",
            val_file="text_json_final/valid.json",
            test_file="text_json_final/test.json",
            format="json"
        ),
        "sarcnet": DatasetConfig(
            name="sarcnet",
            path=Path("sarcnet/SarcNet Image-Text"),
            task="sarcasm_detection",
            modalities=["text", "image"],
            train_file="SarcNetTrain.csv",
            val_file="SarcNetVal.csv",
            test_file="SarcNetTest.csv",
            format="csv"
        ),
        "sarc": DatasetConfig(
            name="sarc",
            path=Path("sarc"),
            task="sarcasm_detection",
            modalities=["text", "context"],
            train_file="train-balanced-sarcasm.csv",
            format="csv"
        ),
        "sarcasm_headlines": DatasetConfig(
            name="sarcasm_headlines",
            path=Path("Sarcasm Headlines"),
            task="sarcasm_detection",
            modalities=["text"],
            train_file="Sarcasm_Headlines_Dataset.json",
            format="json"
        )
    })
    
    # Paraphrasing Datasets
    paraphrasing_datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "paranmt": DatasetConfig(
            name="paranmt",
            path=Path("paranmt"),
            task="paraphrasing",
            modalities=["text"],
            train_file="para-nmt-5m-processed.txt",
            format="txt"
        ),
        "mrpc": DatasetConfig(
            name="mrpc",
            path=Path("MRPC"),
            task="paraphrasing",
            modalities=["text"],
            train_file="train.tsv",
            val_file="dev.tsv",
            test_file="test.tsv",
            format="tsv"
        ),
        "quora": DatasetConfig(
            name="quora",
            path=Path("quora"),
            task="paraphrasing",
            modalities=["text"],
            train_file="train.csv",
            test_file="test.csv",
            format="csv"
        )
    })
    
    # Fact Verification Datasets
    fact_verification_datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "fever": DatasetConfig(
            name="fever",
            path=Path("FEVER"),
            task="fact_verification",
            modalities=["text"],
            train_file="fever_train.jsonl",
            test_file="fever_test.jsonl",
            format="jsonl"
        ),
        "liar": DatasetConfig(
            name="liar",
            path=Path("LIAR"),
            task="fact_verification",
            modalities=["text", "metadata"],
            train_file="train_formatted.csv",
            val_file="valid.tsv",
            test_file="test.tsv",
            format="mixed"  # CSV for train, TSV for val/test
        )
    })
    
    # Dataset processing settings
    processing_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_text_length": 512,
        "image_size": 224,
        "audio_sample_rate": 16000,
        "video_fps": 30,
        "cache_processed_data": True,
        "preprocessing_batch_size": 1000,
        "num_preprocessing_workers": 4
    })
    
    # Data splitting configuration
    split_config: Dict[str, Any] = field(default_factory=lambda: {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 42,
        "stratify": True,
        "min_samples_per_class": 10
    })
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._validate_paths()
        self._setup_absolute_paths()
    
    def _validate_paths(self):
        """Validate that dataset paths exist."""
        missing_paths = []
        all_datasets = {**self.sarcasm_datasets, **self.paraphrasing_datasets, **self.fact_verification_datasets}
        
        for dataset_name, config in all_datasets.items():
            full_path = self.data_root / config.path
            if not full_path.exists():
                missing_paths.append(f"{dataset_name}: {full_path}")
        
        if missing_paths:
            print("⚠️ Warning: Missing dataset paths detected:")
            for path in missing_paths:
                print(f"   - {path}")
            print("Run the dataset download script to fetch missing datasets.")
    
    def _setup_absolute_paths(self):
        """Convert relative paths to absolute paths."""
        for dataset_dict in [self.sarcasm_datasets, self.paraphrasing_datasets, self.fact_verification_datasets]:
            for config in dataset_dict.values():
                config.path = self.data_root / config.path
    
    def get_datasets_by_task(self, task: str) -> Dict[str, DatasetConfig]:
        """Get all datasets for a specific task."""
        task_datasets = {
            "sarcasm_detection": self.sarcasm_datasets,
            "paraphrasing": self.paraphrasing_datasets,
            "fact_verification": self.fact_verification_datasets
        }
        
        if task not in task_datasets:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_datasets.keys())}")
        
        return task_datasets[task]
    
    def get_multimodal_datasets(self) -> Dict[str, DatasetConfig]:
        """Get all datasets that contain multiple modalities."""
        multimodal_datasets = {}
        all_datasets = {**self.sarcasm_datasets, **self.paraphrasing_datasets, **self.fact_verification_datasets}
        
        for name, config in all_datasets.items():
            if len(config.modalities) > 1:
                multimodal_datasets[name] = config
        
        return multimodal_datasets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {
            "total_datasets": 0,
            "by_task": {},
            "by_modality": {"text": 0, "image": 0, "audio": 0, "video": 0},
            "multimodal_count": 0
        }
        
        all_datasets = {**self.sarcasm_datasets, **self.paraphrasing_datasets, **self.fact_verification_datasets}
        info["total_datasets"] = len(all_datasets)
        
        # Count by task
        for task in ["sarcasm_detection", "paraphrasing", "fact_verification"]:
            info["by_task"][task] = len(self.get_datasets_by_task(task))
        
        # Count by modality
        for config in all_datasets.values():
            if len(config.modalities) > 1:
                info["multimodal_count"] += 1
            for modality in config.modalities:
                if modality in info["by_modality"]:
                    info["by_modality"][modality] += 1
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "data_root": str(self.data_root),
            "sarcasm_datasets": {k: v.__dict__ for k, v in self.sarcasm_datasets.items()},
            "paraphrasing_datasets": {k: v.__dict__ for k, v in self.paraphrasing_datasets.items()},
            "fact_verification_datasets": {k: v.__dict__ for k, v in self.fact_verification_datasets.items()},
            "processing_config": self.processing_config,
            "split_config": self.split_config
        }
