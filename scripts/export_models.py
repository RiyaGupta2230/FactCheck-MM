#!/usr/bin/env python3
"""
Model Export Script for FactCheck-MM
Export trained models to TorchScript, ONNX, and other production formats.
"""

import sys
import os
import argparse
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.logging_utils import get_logger


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    # Export formats
    export_torchscript: bool = True
    export_onnx: bool = True
    export_tflite: bool = False
    export_coreml: bool = False
    
    # Optimization
    optimize_for_inference: bool = True
    quantize_model: bool = False
    
    # Input specifications
    max_sequence_length: int = 512
    batch_size: int = 1
    
    # Metadata
    version: str = "1.0.0"
    description: str = ""
    author: str = "FactCheck-MM"


class ModelExporter:
    """Handles export of FactCheck-MM models to various formats."""
    
    def __init__(self, export_config: ExportConfig):
        """Initialize model exporter."""
        self.config = export_config
        self.logger = get_logger("ModelExporter")
        
        # Check available export libraries
        self.available_formats = self._check_export_dependencies()
        self.logger.info(f"Available export formats: {list(self.available_formats.keys())}")
    
    def _check_export_dependencies(self) -> Dict[str, bool]:
        """Check which export formats are available."""
        formats = {
            'torchscript': True,  # Always available with PyTorch
            'onnx': False,
            'tflite': False,
            'coreml': False
        }
        
        # Check ONNX
        try:
            import onnx
            import onnxruntime
            formats['onnx'] = True
            self.logger.info("✅ ONNX export available")
        except ImportError:
            self.logger.warning("⚠️ ONNX not available - install with: pip install onnx onnxruntime")
        
        # Check Core ML
        try:
            import coremltools
            formats['coreml'] = True
            self.logger.info("✅ Core ML export available")
        except ImportError:
            self.logger.warning("⚠️ Core ML not available - install with: pip install coremltools")
        
        return formats
    
    def load_model_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint."""
        self.logger.info(f"📥 Loading model from: {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine model type
        metadata = checkpoint.get('metadata', {})
        model_type = metadata.get('model_type', 'unknown')
        model_config = checkpoint.get('config', {})
        
        self.logger.info(f"🧠 Model type: {model_type}")
        
        # Load appropriate model class
        model = None
        
        if model_type in ['sarcasm', 'text_sarcasm', 'multimodal_sarcasm']:
            if 'multimodal' in model_type:
                from sarcasm_detection.models import MultimodalSarcasmModel
                model = MultimodalSarcasmModel(model_config)
            else:
                from sarcasm_detection.models import RobertaSarcasmModel
                model = RobertaSarcasmModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model is None:
            raise ValueError(f"Could not load model of type: {model_type}")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        result = {
            'model': model,
            'config': model_config,
            'metadata': metadata,
            'checkpoint': checkpoint
        }
        
        self.logger.info("✅ Model loaded successfully")
        return result
    
    def create_sample_inputs(self, model_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create sample inputs for export."""
        config = model_info['config']
        metadata = model_info['metadata']
        model_type = metadata.get('model_type', 'unknown')
        
        batch_size = self.config.batch_size
        seq_len = self.config.max_sequence_length
        
        sample_inputs = {}
        
        if 'sarcasm' in model_type:
            # Text inputs (always present)
            sample_inputs['input_ids'] = torch.randint(0, 30522, (batch_size, seq_len))
            sample_inputs['attention_mask'] = torch.ones(batch_size, seq_len)
            
            # Multimodal inputs (if multimodal model)
            if 'multimodal' in model_type:
                modalities = config.get('modalities', ['text'])
                
                if 'audio' in modalities:
                    audio_dim = config.get('audio_hidden_dim', 768)
                    sample_inputs['audio_features'] = torch.randn(batch_size, 100, audio_dim)
                
                if 'image' in modalities:
                    image_dim = config.get('image_hidden_dim', 768)
                    sample_inputs['image_features'] = torch.randn(batch_size, 196, image_dim)
                
                if 'video' in modalities:
                    video_dim = config.get('video_hidden_dim', 768)
                    sample_inputs['video_features'] = torch.randn(batch_size, 16, video_dim)
        
        self.logger.info(f"📊 Created sample inputs: {list(sample_inputs.keys())}")
        return sample_inputs
    
    def export_to_torchscript(
        self, 
        model: torch.nn.Module, 
        sample_inputs: Dict[str, torch.Tensor],
        output_path: Path
    ) -> bool:
        """Export model to TorchScript."""
        try:
            self.logger.info("🔄 Exporting to TorchScript...")
            
            model.eval()
            
            with torch.no_grad():
                # Use scripting for complex models
                traced_model = torch.jit.script(model)
                
                if self.config.optimize_for_inference:
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                
                # Save model
                traced_model.save(str(output_path))
            
            # Verify export
            loaded_model = torch.jit.load(str(output_path))
            
            self.logger.info(f"✅ TorchScript export successful: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TorchScript export failed: {e}")
            return False
    
    def export_to_onnx(
        self, 
        model: torch.nn.Module, 
        sample_inputs: Dict[str, torch.Tensor],
        output_path: Path
    ) -> bool:
        """Export model to ONNX."""
        if not self.available_formats['onnx']:
            self.logger.error("❌ ONNX not available")
            return False
        
        try:
            import onnx
            import onnxruntime
            
            self.logger.info("🔄 Exporting to ONNX...")
            
            model.eval()
            
            # For sarcasm models, typically use text inputs
            input_names = ['input_ids', 'attention_mask']
            input_tuple = (sample_inputs['input_ids'], sample_inputs['attention_mask'])
            
            dynamic_axes = {
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            torch.onnx.export(
                model,
                input_tuple,
                str(output_path),
                input_names=input_names,
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=11,
                do_constant_folding=True
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"✅ ONNX export successful: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ONNX export failed: {e}")
            return False
    
    def create_model_metadata(
        self, 
        model_info: Dict[str, Any], 
        export_paths: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Create comprehensive model metadata."""
        metadata = {
            'model_info': {
                'name': model_info['metadata'].get('model_type', 'unknown'),
                'version': self.config.version,
                'description': self.config.description,
                'author': self.config.author,
                'framework': 'PyTorch'
            },
            'model_config': model_info['config'],
            'export_config': {
                'max_sequence_length': self.config.max_sequence_length,
                'batch_size': self.config.batch_size,
                'optimized_for_inference': self.config.optimize_for_inference
            },
            'exported_formats': {},
            'usage_instructions': {
                'torchscript': {
                    'load': "model = torch.jit.load('model.pt')",
                    'inference': "output = model(input_ids, attention_mask)"
                },
                'onnx': {
                    'load': "session = onnxruntime.InferenceSession('model.onnx')",
                    'inference': "output = session.run(None, {'input_ids': ids, 'attention_mask': mask})"
                }
            }
        }
        
        # Add file information for each exported format
        for format_name, path in export_paths.items():
            if path.exists():
                metadata['exported_formats'][format_name] = {
                    'filename': path.name,
                    'size_mb': path.stat().st_size / (1024 * 1024),
                    'path': str(path)
                }
        
        return metadata
    
    def export_model(
        self, 
        checkpoint_path: str, 
        output_dir: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export model to all specified formats."""
        
        # Load model
        model_info = self.load_model_checkpoint(checkpoint_path)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model name
        if model_name is None:
            model_type = model_info['metadata'].get('model_type', 'model')
            model_name = f"{model_type}_v{self.config.version}"
        
        self.logger.info(f"📦 Exporting model: {model_name}")
        
        # Create sample inputs
        sample_inputs = self.create_sample_inputs(model_info)
        
        # Export to different formats
        export_results = {}
        export_paths = {}
        
        # TorchScript
        if self.config.export_torchscript:
            torchscript_path = output_dir / f"{model_name}.pt"
            success = self.export_to_torchscript(
                model_info['model'], sample_inputs, torchscript_path
            )
            export_results['torchscript'] = success
            if success:
                export_paths['torchscript'] = torchscript_path
        
        # ONNX
        if self.config.export_onnx and self.available_formats['onnx']:
            onnx_path = output_dir / f"{model_name}.onnx"
            success = self.export_to_onnx(
                model_info['model'], sample_inputs, onnx_path
            )
            export_results['onnx'] = success
            if success:
                export_paths['onnx'] = onnx_path
        
        # Create and save metadata
        metadata = self.create_model_metadata(model_info, export_paths)
        metadata_path = output_dir / f"{model_name}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Summary
        successful_exports = sum(export_results.values())
        total_exports = len([x for x in [
            self.config.export_torchscript,
            self.config.export_onnx and self.available_formats['onnx']
        ] if x])
        
        self.logger.info(f"✅ Export completed: {successful_exports}/{total_exports} formats successful")
        
        return {
            'model_name': model_name,
            'export_results': export_results,
            'export_paths': export_paths,
            'metadata_path': metadata_path,
            'output_directory': output_dir
        }


def main():
    """Main entry point for model export."""
    parser = argparse.ArgumentParser(description="FactCheck-MM Model Export")
    
    # Input/Output
    parser.add_argument('checkpoint_path', help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--model-name', help='Custom model name')
    
    # Export formats
    parser.add_argument('--torchscript', action='store_true', default=True,
                       help='Export to TorchScript')
    parser.add_argument('--onnx', action='store_true', default=True,
                       help='Export to ONNX')
    
    # Model specifications
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Export batch size')
    
    # Metadata
    parser.add_argument('--version', default='1.0.0',
                       help='Model version')
    parser.add_argument('--description', help='Model description')
    
    args = parser.parse_args()
    
    # Create export config
    export_config = ExportConfig(
        export_torchscript=args.torchscript,
        export_onnx=args.onnx,
        max_sequence_length=args.max_length,
        batch_size=args.batch_size,
        version=args.version,
        description=args.description or f"FactCheck-MM model exported from {args.checkpoint_path}"
    )
    
    try:
        # Initialize exporter
        exporter = ModelExporter(export_config)
        
        # Export model
        results = exporter.export_model(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        # Print results
        print(f"\n🎉 Model export completed!")
        print(f"📂 Output directory: {results['output_directory']}")
        print(f"📋 Metadata: {results['metadata_path']}")
        
        print("\n📦 Exported formats:")
        for format_name, success in results['export_results'].items():
            status = "✅" if success else "❌"
            print(f"   {status} {format_name}")
        
    except Exception as e:
        print(f"💥 Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
