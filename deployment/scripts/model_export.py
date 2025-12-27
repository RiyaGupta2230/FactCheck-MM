#!/usr/bin/env python3
"""
Model Export Utility for FactCheck-MM Production Deployment

Exports trained model checkpoints to deployment-ready formats with metadata
generation for production inference. Supports TorchScript compilation and
safe PyTorch format exports.

Usage:
    # Export all models
    python deployment/scripts/model_export.py --export-all
    
    # Export specific model
    python deployment/scripts/model_export.py --task sarcasm_detection --checkpoint checkpoints/sarcasm/
    
    # Export with TorchScript compilation
    python deployment/scripts/model_export.py --export-all --torchscript

Example:
    >>> from deployment.scripts.model_export import ModelExporter
    >>> 
    >>> exporter = ModelExporter(output_dir="deployment/models")
    >>> exporter.export_model("sarcasm_detection", "checkpoints/sarcasm/best.pt")
    >>> exporter.export_all_models()
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime
import shutil
import traceback

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from shared.utils.checkpoint_manager import CheckpointManager
from config.model_config import ModelConfig


class ModelExporter:
    """
    Exports trained models to production-ready formats.
    
    Handles model loading, optimization, compilation, and metadata generation
    for deployment in production environments.
    """
    
    def __init__(
        self,
        output_dir: str = "deployment/models",
        device: str = "cpu",
        logger: Optional[Any] = None
    ):
        """
        Initialize model exporter.
        
        Args:
            output_dir: Directory to save exported models
            device: Target device for export (cpu or cuda)
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("ModelExporter")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        
        self.logger.info(f"ModelExporter initialized (device: {self.device})")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Model registry with default checkpoint paths
        self.model_registry = {
            "sarcasm_detection": {
                "checkpoint_path": "checkpoints/multimodal_sarcasm/best_model/",
                "config_path": "config/model_configs/sarcasm_detection.yaml",
                "task": "multimodal_sarcasm_detection",
                "datasets": ["SARC", "MMSD2", "MUStARD"]
            },
            "paraphrasing_t5": {
                "checkpoint_path": "checkpoints/t5_paraphraser/",
                "config_path": "config/model_configs/paraphrasing.yaml",
                "task": "paraphrase_generation",
                "datasets": ["ParaNMT-5M", "MRPC", "Quora"]
            },
            "paraphrasing_bart": {
                "checkpoint_path": "checkpoints/bart_paraphraser/",
                "config_path": "config/model_configs/paraphrasing.yaml",
                "task": "paraphrase_generation",
                "datasets": ["ParaNMT-5M", "MRPC"]
            },
            "fact_verification": {
                "checkpoint_path": "checkpoints/fact_verification/general_model/",
                "config_path": "config/model_configs/fact_verification.yaml",
                "task": "fact_checking",
                "datasets": ["FEVER", "LIAR"]
            }
        }
    
    def export_model(
        self,
        task_name: str,
        checkpoint_path: Optional[str] = None,
        export_torchscript: bool = False,
        export_onnx: bool = False,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Export a single model to deployment format.
        
        Args:
            task_name: Name of the task (must be in model_registry)
            checkpoint_path: Optional custom checkpoint path
            export_torchscript: Whether to compile to TorchScript
            export_onnx: Whether to export to ONNX format
            optimize: Whether to optimize model for inference
        
        Returns:
            Dictionary with export results
        """
        self.logger.info(f"Starting export for task: {task_name}")
        
        # Get model configuration
        if task_name not in self.model_registry:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.model_registry.keys())}")
        
        model_info = self.model_registry[task_name]
        checkpoint_path = checkpoint_path or model_info["checkpoint_path"]
        checkpoint_path = Path(checkpoint_path)
        
        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load model based on task type
            model = self._load_model(task_name, checkpoint_path)
            
            # Move to target device
            model = model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            # Optimize for inference
            if optimize:
                model = self._optimize_model(model)
            
            # Create task-specific output directory
            task_output_dir = self.output_dir / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            export_results = {
                "task_name": task_name,
                "checkpoint_path": str(checkpoint_path),
                "output_dir": str(task_output_dir),
                "device": str(self.device),
                "exports": []
            }
            
            # Export standard PyTorch format
            pytorch_path = self._export_pytorch(model, task_output_dir, task_name)
            export_results["exports"].append({
                "format": "pytorch",
                "path": str(pytorch_path)
            })
            
            # Export TorchScript if requested
            if export_torchscript:
                try:
                    torchscript_path = self._export_torchscript(model, task_output_dir, task_name)
                    export_results["exports"].append({
                        "format": "torchscript",
                        "path": str(torchscript_path)
                    })
                except Exception as e:
                    self.logger.warning(f"TorchScript export failed: {e}")
            
            # Export ONNX if requested
            if export_onnx:
                try:
                    onnx_path = self._export_onnx(model, task_output_dir, task_name)
                    export_results["exports"].append({
                        "format": "onnx",
                        "path": str(onnx_path)
                    })
                except Exception as e:
                    self.logger.warning(f"ONNX export failed: {e}")
            
            # Generate metadata
            metadata = self._generate_metadata(task_name, model, model_info)
            metadata_path = task_output_dir / "metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            export_results["metadata_path"] = str(metadata_path)
            
            # Copy configuration files
            self._copy_config_files(model_info, task_output_dir)
            
            self.logger.info(f"Export completed successfully for {task_name}")
            self.logger.info(f"Output directory: {task_output_dir}")
            
            return export_results
            
        except Exception as e:
            self.logger.error(f"Export failed for {task_name}: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_model(self, task_name: str, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint based on task type."""
        
        self.logger.info(f"Loading {task_name} model...")
        
        try:
            if task_name == "sarcasm_detection":
                from sarcasm_detection.models import MultimodalSarcasmDetector
                model = MultimodalSarcasmDetector.from_pretrained(str(checkpoint_path))
                
            elif task_name.startswith("paraphrasing_t5"):
                from paraphrasing.models import T5Paraphraser
                model = T5Paraphraser.from_pretrained(str(checkpoint_path))
                
            elif task_name.startswith("paraphrasing_bart"):
                from paraphrasing.models import BARTParaphraser
                model = BARTParaphraser.from_pretrained(str(checkpoint_path))
                
            elif task_name == "fact_verification":
                from fact_verification.models import FactCheckPipeline
                model = FactCheckPipeline.from_pretrained(str(checkpoint_path))
                
            else:
                raise ValueError(f"Unknown task type: {task_name}")
            
            self.logger.info(f"Model loaded successfully: {type(model).__name__}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference."""
        
        self.logger.info("Optimizing model for inference...")
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply inference optimizations
        model.eval()
        
        # Fuse operations if possible (for CNN-based components)
        if hasattr(torch.quantization, 'fuse_modules') and hasattr(model, 'fuse_model'):
            try:
                model.fuse_model()
                self.logger.info("Model layers fused successfully")
            except:
                self.logger.debug("Model fusion not applicable")
        
        return model
    
    def _export_pytorch(self, model: nn.Module, output_dir: Path, name: str) -> Path:
        """Export model in standard PyTorch format."""
        
        self.logger.info("Exporting PyTorch model...")
        
        # Save model state dict
        output_path = output_dir / f"{name}_model.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': type(model).__name__,
            'export_date': datetime.now().isoformat(),
            'device': str(self.device)
        }, output_path)
        
        self.logger.info(f"PyTorch model saved to: {output_path}")
        return output_path
    
    def _export_torchscript(self, model: nn.Module, output_dir: Path, name: str) -> Path:
        """Export model to TorchScript format."""
        
        self.logger.info("Exporting TorchScript model...")
        
        # Trace or script the model
        try:
            # Try tracing first
            example_input = self._get_example_input(model)
            traced_model = torch.jit.trace(model, example_input)
            
        except Exception as e:
            self.logger.warning(f"Tracing failed, attempting scripting: {e}")
            # Fall back to scripting
            traced_model = torch.jit.script(model)
        
        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save
        output_path = output_dir / f"{name}_torchscript.pt"
        traced_model.save(str(output_path))
        
        self.logger.info(f"TorchScript model saved to: {output_path}")
        return output_path
    
    def _export_onnx(self, model: nn.Module, output_dir: Path, name: str) -> Path:
        """Export model to ONNX format."""
        
        self.logger.info("Exporting ONNX model...")
        
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("ONNX export requires onnx and onnxruntime packages")
        
        # Get example input
        example_input = self._get_example_input(model)
        
        # Export to ONNX
        output_path = output_dir / f"{name}_model.onnx"
        
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        self.logger.info(f"ONNX model saved to: {output_path}")
        return output_path
    
    def _get_example_input(self, model: nn.Module) -> torch.Tensor:
        """Generate example input for model tracing."""
        
        # This is a simplified version - in production, you'd need
        # task-specific example inputs
        batch_size = 1
        seq_length = 128
        
        # Default to text input (most common)
        return torch.randint(0, 1000, (batch_size, seq_length)).to(self.device)
    
    def _generate_metadata(
        self,
        task_name: str,
        model: nn.Module,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata JSON for exported model."""
        
        self.logger.info("Generating model metadata...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metadata = {
            "model_name": task_name,
            "model_class": type(model).__name__,
            "task": model_info["task"],
            "training_datasets": model_info["datasets"],
            "export_date": datetime.now().isoformat(),
            "export_device": str(self.device),
            "model_info": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "parameter_size_mb": (total_params * 4) / (1024 * 1024)  # Assume float32
            },
            "device_compatibility": {
                "cpu": True,
                "cuda": torch.cuda.is_available(),
                "mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            },
            "deployment_info": {
                "framework": "PyTorch",
                "pytorch_version": torch.__version__,
                "python_version": sys.version.split()[0],
                "recommended_batch_size": 16
            }
        }
        
        # Try to load evaluation metrics if available
        try:
            eval_results_path = Path(model_info["checkpoint_path"]) / "eval_results.json"
            if eval_results_path.exists():
                with open(eval_results_path, 'r') as f:
                    metadata["evaluation_metrics"] = json.load(f)
        except:
            self.logger.debug("No evaluation metrics found")
        
        return metadata
    
    def _copy_config_files(self, model_info: Dict[str, Any], output_dir: Path):
        """Copy configuration files to export directory."""
        
        config_path = Path(model_info.get("config_path", ""))
        
        if config_path.exists():
            dest_path = output_dir / "config.yaml"
            shutil.copy2(config_path, dest_path)
            self.logger.info(f"Configuration copied to: {dest_path}")
    
    def export_all_models(
        self,
        export_torchscript: bool = False,
        export_onnx: bool = False,
        skip_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Export all registered models.
        
        Args:
            export_torchscript: Whether to compile to TorchScript
            export_onnx: Whether to export to ONNX format
            skip_missing: Whether to skip missing checkpoints or raise error
        
        Returns:
            Dictionary with export results for all models
        """
        self.logger.info("Starting batch export for all models...")
        
        results = {
            "export_date": datetime.now().isoformat(),
            "total_models": len(self.model_registry),
            "successful_exports": 0,
            "failed_exports": 0,
            "models": {}
        }
        
        for task_name in self.model_registry.keys():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Exporting: {task_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                export_result = self.export_model(
                    task_name=task_name,
                    export_torchscript=export_torchscript,
                    export_onnx=export_onnx
                )
                
                results["models"][task_name] = {
                    "status": "success",
                    "result": export_result
                }
                results["successful_exports"] += 1
                
            except FileNotFoundError as e:
                if skip_missing:
                    self.logger.warning(f"Skipping {task_name}: {e}")
                    results["models"][task_name] = {
                        "status": "skipped",
                        "reason": str(e)
                    }
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"Failed to export {task_name}: {e}")
                results["models"][task_name] = {
                    "status": "failed",
                    "error": str(e)
                }
                results["failed_exports"] += 1
        
        # Save summary
        summary_path = self.output_dir / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Batch Export Summary")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Models: {results['total_models']}")
        self.logger.info(f"Successful: {results['successful_exports']}")
        self.logger.info(f"Failed: {results['failed_exports']}")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return results


def main():
    """Main entry point for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Export FactCheck-MM models for production deployment"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["sarcasm_detection", "paraphrasing_t5", "paraphrasing_bart", "fact_verification"],
        help="Specific task to export"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Custom checkpoint path"
    )
    
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all registered models"
    )
    
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Export to TorchScript format"
    )
    
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export to ONNX format"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment/models",
        help="Output directory for exported models"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target device for export"
    )
    
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip missing checkpoints instead of failing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create exporter
    exporter = ModelExporter(
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Execute export
    if args.export_all:
        results = exporter.export_all_models(
            export_torchscript=args.torchscript,
            export_onnx=args.onnx,
            skip_missing=args.skip_missing
        )
        print("\nExport completed!")
        print(f"Successful: {results['successful_exports']}/{results['total_models']}")
        
    elif args.task:
        result = exporter.export_model(
            task_name=args.task,
            checkpoint_path=args.checkpoint,
            export_torchscript=args.torchscript,
            export_onnx=args.onnx
        )
        print("\nExport completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        
    else:
        parser.print_help()
        print("\nError: Must specify either --task or --export-all")
        sys.exit(1)


if __name__ == "__main__":
    main()
