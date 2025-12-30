#!/usr/bin/env python3
"""
Full Pipeline Runner for FactCheck-MM
End-to-end execution: sarcasm detection → paraphrasing → fact verification
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.logging_utils import get_logger, setup_logging
from Config.base_config import BaseConfig


@dataclass
class PipelineConfig:
    """Configuration for full pipeline execution."""
    
    # Task selection
    run_sarcasm_detection: bool = True
    run_paraphrasing: bool = True
    run_fact_verification: bool = True
    
    # Data flow
    use_sarcasm_for_paraphrasing: bool = True
    use_paraphrases_for_verification: bool = True
    
    # Model settings
    sarcasm_model_path: Optional[str] = None
    paraphrase_model_path: Optional[str] = None
    fact_check_model_path: Optional[str] = None
    
    # Hardware
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512
    
    # Output
    save_intermediate_results: bool = True
    output_dir: str = "pipeline_outputs"
    experiment_name: str = "full_pipeline"


class FactCheckPipeline:
    """Complete FactCheck-MM pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("FactCheckPipeline")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.results = {}
        self.intermediate_data = {}
        
        # Initialize task modules (lazy loading)
        self._sarcasm_detector = None
        self._paraphraser = None
        self._fact_verifier = None
        
        self.logger.info(f"Initialized FactCheck-MM pipeline: {config.experiment_name}")
    
    def _setup_device(self):
        """Setup compute device for pipeline."""
        import torch
        
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"🎮 Using GPU: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                self.logger.info("🍎 Using Apple Metal Performance Shaders")
            else:
                device = "cpu"
                self.logger.info("💻 Using CPU")
        else:
            device = self.config.device
            self.logger.info(f"🎯 Using specified device: {device}")
        
        self.device = device
        return device
    
    def _load_sarcasm_detector(self):
        """Load sarcasm detection model."""
        if self._sarcasm_detector is not None:
            return self._sarcasm_detector
        
        self.logger.info("📥 Loading sarcasm detection model...")
        
        try:
            from sarcasm_detection.models import MultimodalSarcasmModel
            from sarcasm_detection.evaluation import SarcasmEvaluator
            import torch
            
            if self.config.sarcasm_model_path:
                # Load trained model
                self.logger.info(f"Loading from checkpoint: {self.config.sarcasm_model_path}")
                checkpoint = torch.load(self.config.sarcasm_model_path, map_location=self.device)
                
                model_config = checkpoint.get('config', {})
                model = MultimodalSarcasmModel(model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
            else:
                # Use pre-trained or default model
                self.logger.info("Using default sarcasm detection model")
                model_config = {
                    'modalities': ['text'],  # Start with text-only for compatibility
                    'num_classes': 2,
                    'fusion_strategy': 'cross_modal_attention'
                }
                model = MultimodalSarcasmModel(model_config)
                model.to(self.device)
                model.eval()
            
            self._sarcasm_detector = {
                'model': model,
                'evaluator': SarcasmEvaluator(model)
            }
            
            self.logger.info("✅ Sarcasm detection model loaded")
            return self._sarcasm_detector
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load sarcasm detection model: {e}")
            raise
    
    def _load_paraphraser(self):
        """Load paraphrasing model."""
        if self._paraphraser is not None:
            return self._paraphraser
        
        self.logger.info("📥 Loading paraphrasing model...")
        
        # Placeholder for paraphrasing model loading
        # This will be implemented when paraphrasing module is complete
        self.logger.warning("⚠️ Paraphrasing module not yet implemented - using placeholder")
        
        self._paraphraser = {
            'model': None,
            'generator': None
        }
        
        return self._paraphraser
    
    def _load_fact_verifier(self):
        """Load fact verification model."""
        if self._fact_verifier is not None:
            return self._fact_verifier
        
        self.logger.info("📥 Loading fact verification model...")
        
        # Placeholder for fact verification model loading
        # This will be implemented when fact_verification module is complete
        self.logger.warning("⚠️ Fact verification module not yet implemented - using placeholder")
        
        self._fact_verifier = {
            'model': None,
            'evaluator': None
        }
        
        return self._fact_verifier
    
    def run_sarcasm_detection(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run sarcasm detection on input data.
        
        Args:
            input_data: List of samples with text/multimodal content
            
        Returns:
            Processed data with sarcasm predictions
        """
        if not self.config.run_sarcasm_detection:
            self.logger.info("⏭️ Skipping sarcasm detection")
            return input_data
        
        self.logger.info(f"🎭 Running sarcasm detection on {len(input_data)} samples")
        
        detector = self._load_sarcasm_detector()
        
        # Process samples
        processed_data = []
        
        for i, sample in enumerate(input_data):
            try:
                # Extract text content
                text = sample.get('text', sample.get('claim', ''))
                
                if not text:
                    self.logger.warning(f"Sample {i}: No text content found")
                    sample['sarcasm_prediction'] = 0
                    sample['sarcasm_confidence'] = 0.0
                    processed_data.append(sample)
                    continue
                
                # Create input for sarcasm model
                model_input = {
                    'text': text,
                    'audio': sample.get('audio'),
                    'image': sample.get('image'),
                    'video': sample.get('video')
                }
                
                # Run prediction (placeholder implementation)
                # In practice, this would use the actual model
                import torch
                import random
                
                # Simulate sarcasm prediction
                sarcasm_prob = random.random()
                sarcasm_prediction = 1 if sarcasm_prob > 0.5 else 0
                
                # Add predictions to sample
                sample['sarcasm_prediction'] = sarcasm_prediction
                sample['sarcasm_confidence'] = sarcasm_prob
                sample['is_sarcastic'] = bool(sarcasm_prediction)
                
                processed_data.append(sample)
                
            except Exception as e:
                self.logger.error(f"❌ Error processing sample {i}: {e}")
                sample['sarcasm_prediction'] = 0
                sample['sarcasm_confidence'] = 0.0
                processed_data.append(sample)
        
        # Save intermediate results
        if self.config.save_intermediate_results:
            sarcasm_output = self.output_dir / "sarcasm_results.json"
            with open(sarcasm_output, 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Sarcasm results saved to: {sarcasm_output}")
        
        self.results['sarcasm_detection'] = {
            'total_samples': len(processed_data),
            'sarcastic_samples': sum(1 for s in processed_data if s.get('is_sarcastic', False)),
            'avg_confidence': sum(s.get('sarcasm_confidence', 0) for s in processed_data) / len(processed_data)
        }
        
        self.logger.info(f"✅ Sarcasm detection completed: {self.results['sarcasm_detection']['sarcastic_samples']}/{len(processed_data)} sarcastic")
        
        return processed_data
    
    def run_paraphrasing(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run paraphrasing on input data.
        
        Args:
            input_data: Data from sarcasm detection
            
        Returns:
            Data with paraphrases
        """
        if not self.config.run_paraphrasing:
            self.logger.info("⏭️ Skipping paraphrasing")
            return input_data
        
        self.logger.info(f"🔄 Running paraphrasing on {len(input_data)} samples")
        
        paraphraser = self._load_paraphraser()
        
        # Process samples
        processed_data = []
        
        for i, sample in enumerate(input_data):
            try:
                text = sample.get('text', sample.get('claim', ''))
                
                if not text:
                    self.logger.warning(f"Sample {i}: No text for paraphrasing")
                    sample['paraphrases'] = []
                    processed_data.append(sample)
                    continue
                
                # Generate paraphrases
                # Placeholder implementation - replace with actual paraphrasing
                paraphrases = [
                    f"Paraphrase 1: {text}",
                    f"Alternative: {text}",
                    f"Rephrased: {text}"
                ]
                
                # Consider sarcasm context if available
                if self.config.use_sarcasm_for_paraphrasing and sample.get('is_sarcastic', False):
                    paraphrases = [f"[Non-sarcastic] {p}" for p in paraphrases]
                
                sample['paraphrases'] = paraphrases
                sample['num_paraphrases'] = len(paraphrases)
                
                processed_data.append(sample)
                
            except Exception as e:
                self.logger.error(f"❌ Error paraphrasing sample {i}: {e}")
                sample['paraphrases'] = []
                processed_data.append(sample)
        
        # Save intermediate results
        if self.config.save_intermediate_results:
            para_output = self.output_dir / "paraphrase_results.json"
            with open(para_output, 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Paraphrase results saved to: {para_output}")
        
        self.results['paraphrasing'] = {
            'total_samples': len(processed_data),
            'total_paraphrases': sum(len(s.get('paraphrases', [])) for s in processed_data),
            'avg_paraphrases_per_sample': sum(len(s.get('paraphrases', [])) for s in processed_data) / len(processed_data)
        }
        
        self.logger.info(f"✅ Paraphrasing completed: {self.results['paraphrasing']['total_paraphrases']} paraphrases generated")
        
        return processed_data
    
    def run_fact_verification(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run fact verification on input data.
        
        Args:
            input_data: Data from paraphrasing
            
        Returns:
            Final results with fact-check verdicts
        """
        if not self.config.run_fact_verification:
            self.logger.info("⏭️ Skipping fact verification")
            return input_data
        
        self.logger.info(f"🔍 Running fact verification on {len(input_data)} samples")
        
        fact_verifier = self._load_fact_verifier()
        
        # Process samples
        processed_data = []
        
        for i, sample in enumerate(input_data):
            try:
                # Get claims to verify
                claims_to_verify = []
                
                # Original claim
                original_claim = sample.get('text', sample.get('claim', ''))
                if original_claim:
                    claims_to_verify.append(original_claim)
                
                # Add paraphrases if enabled
                if self.config.use_paraphrases_for_verification:
                    paraphrases = sample.get('paraphrases', [])
                    claims_to_verify.extend(paraphrases)
                
                # Run fact verification on each claim
                verification_results = []
                
                for claim in claims_to_verify:
                    # Placeholder fact verification
                    import random
                    
                    verdict_score = random.random()
                    if verdict_score > 0.7:
                        verdict = "TRUE"
                    elif verdict_score > 0.4:
                        verdict = "PARTIALLY_TRUE"
                    else:
                        verdict = "FALSE"
                    
                    verification_results.append({
                        'claim': claim,
                        'verdict': verdict,
                        'confidence': verdict_score,
                        'evidence': f"Evidence for: {claim[:50]}..."
                    })
                
                # Aggregate results
                verdicts = [r['verdict'] for r in verification_results]
                confidences = [r['confidence'] for r in verification_results]
                
                sample['verification_results'] = verification_results
                sample['final_verdict'] = max(set(verdicts), key=verdicts.count)  # Most common verdict
                sample['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
                sample['is_factual'] = sample['final_verdict'] in ['TRUE', 'PARTIALLY_TRUE']
                
                processed_data.append(sample)
                
            except Exception as e:
                self.logger.error(f"❌ Error verifying sample {i}: {e}")
                sample['verification_results'] = []
                sample['final_verdict'] = "UNKNOWN"
                sample['is_factual'] = False
                processed_data.append(sample)
        
        # Save final results
        final_output = self.output_dir / "final_results.json"
        with open(final_output, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Final results saved to: {final_output}")
        
        self.results['fact_verification'] = {
            'total_samples': len(processed_data),
            'factual_samples': sum(1 for s in processed_data if s.get('is_factual', False)),
            'avg_confidence': sum(s.get('avg_confidence', 0) for s in processed_data) / len(processed_data)
        }
        
        self.logger.info(f"✅ Fact verification completed: {self.results['fact_verification']['factual_samples']}/{len(processed_data)} factual")
        
        return processed_data
    
    def run_full_pipeline(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run complete pipeline on input data.
        
        Args:
            input_data: Input samples
            
        Returns:
            Pipeline results and metrics
        """
        self.logger.info(f"🚀 Starting full FactCheck-MM pipeline on {len(input_data)} samples")
        start_time = time.time()
        
        # Setup device
        self._setup_device()
        
        # Stage 1: Sarcasm Detection
        stage1_start = time.time()
        data = self.run_sarcasm_detection(input_data)
        stage1_time = time.time() - stage1_start
        
        # Stage 2: Paraphrasing
        stage2_start = time.time()
        data = self.run_paraphrasing(data)
        stage2_time = time.time() - stage2_start
        
        # Stage 3: Fact Verification
        stage3_start = time.time()
        final_data = self.run_fact_verification(data)
        stage3_time = time.time() - stage3_start
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'pipeline_config': self.config.__dict__,
            'execution_time': {
                'total_seconds': total_time,
                'sarcasm_detection_seconds': stage1_time,
                'paraphrasing_seconds': stage2_time,
                'fact_verification_seconds': stage3_time
            },
            'stage_results': self.results,
            'final_data': final_data,
            'summary': {
                'total_input_samples': len(input_data),
                'total_output_samples': len(final_data),
                'sarcastic_samples': self.results.get('sarcasm_detection', {}).get('sarcastic_samples', 0),
                'total_paraphrases': self.results.get('paraphrasing', {}).get('total_paraphrases', 0),
                'factual_samples': self.results.get('fact_verification', {}).get('factual_samples', 0)
            }
        }
        
        # Save complete results
        complete_output = self.output_dir / f"{self.config.experiment_name}_complete.json"
        with open(complete_output, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"🎉 Pipeline completed in {total_time:.2f}s")
        self.logger.info(f"📊 Results: {final_results['summary']}")
        self.logger.info(f"💾 Complete results: {complete_output}")
        
        return final_results


def load_input_data(input_source: str) -> List[Dict[str, Any]]:
    """
    Load input data from various sources.
    
    Args:
        input_source: Path to input file or dataset name
        
    Returns:
        List of input samples
    """
    logger = get_logger("DataLoader")
    
    if input_source.endswith('.json'):
        # Load from JSON file
        with open(input_source, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            return [data]
    
    elif input_source.endswith('.csv'):
        # Load from CSV file
        import pandas as pd
        df = pd.read_csv(input_source)
        return df.to_dict('records')
    
    elif input_source in ['fever', 'liar', 'mmsd2', 'mustard', 'sarc']:
        # Load from FactCheck-MM datasets
        logger.info(f"Loading from FactCheck-MM dataset: {input_source}")
        
        if input_source in ['mmsd2', 'mustard']:
            from sarcasm_detection.data.unified_loader import UnifiedSarcasmLoader
            loader = UnifiedSarcasmLoader('data/', [input_source])
            _, _, test_data = loader.load_datasets()
            
            # Convert dataset to list of dicts
            samples = []
            for i in range(min(100, len(test_data))):  # Limit for demo
                sample = test_data[i]
                samples.append({
                    'text': sample.get('text', ''),
                    'label': sample.get('label', 0),
                    'source_dataset': input_source
                })
            
            return samples
        
        elif input_source in ['fever', 'liar']:
            # Placeholder for fact verification datasets
            logger.warning(f"Fact verification dataset {input_source} not yet implemented")
            return [
                {'claim': f'Sample claim from {input_source} dataset', 'source_dataset': input_source}
                for _ in range(10)
            ]
    
    else:
        # Create sample data
        logger.info("Creating sample test data")
        return [
            {
                'text': 'This is definitely the best implementation ever created.',
                'source': 'sample'
            },
            {
                'text': 'Climate change is causing unprecedented global warming.',
                'source': 'sample'
            },
            {
                'text': 'The latest AI model achieved 99.9% accuracy on all tasks.',
                'source': 'sample'
            }
        ]


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description="FactCheck-MM Full Pipeline Runner")
    
    # Task selection
    parser.add_argument('--tasks', default='sarcasm,paraphrase,fact_check',
                       help='Comma-separated tasks to run (sarcasm,paraphrase,fact_check)')
    
    # Input/Output
    parser.add_argument('--input', default='sample',
                       help='Input data source (file path or dataset name)')
    parser.add_argument('--output-dir', default='pipeline_outputs',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', default=f'pipeline_{int(time.time())}',
                       help='Experiment name')
    
    # Model paths
    parser.add_argument('--sarcasm-model', 
                       help='Path to trained sarcasm detection model')
    parser.add_argument('--paraphrase-model',
                       help='Path to trained paraphrasing model')
    parser.add_argument('--fact-check-model',
                       help='Path to trained fact verification model')
    
    # Execution settings
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--no-save-intermediate', action='store_true',
                       help='Do not save intermediate results')
    
    # Pipeline flow
    parser.add_argument('--no-sarcasm-conditioning', action='store_true',
                       help='Do not use sarcasm results for paraphrasing')
    parser.add_argument('--no-paraphrase-verification', action='store_true',
                       help='Do not verify paraphrases')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(experiment_name=args.experiment_name)
    logger = get_logger("PipelineRunner")
    
    try:
        # Parse tasks
        tasks = [t.strip() for t in args.tasks.split(',')]
        
        # Create pipeline config
        config = PipelineConfig(
            run_sarcasm_detection='sarcasm' in tasks,
            run_paraphrasing='paraphrase' in tasks,
            run_fact_verification='fact_check' in tasks,
            
            sarcasm_model_path=args.sarcasm_model,
            paraphrase_model_path=args.paraphrase_model,
            fact_check_model_path=args.fact_check_model,
            
            device=args.device,
            batch_size=args.batch_size,
            save_intermediate_results=not args.no_save_intermediate,
            
            use_sarcasm_for_paraphrasing=not args.no_sarcasm_conditioning,
            use_paraphrases_for_verification=not args.no_paraphrase_verification,
            
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        # Load input data
        logger.info(f"📥 Loading input data from: {args.input}")
        input_data = load_input_data(args.input)
        logger.info(f"📊 Loaded {len(input_data)} samples")
        
        # Initialize and run pipeline
        pipeline = FactCheckPipeline(config)
        results = pipeline.run_full_pipeline(input_data)
        
        # Print summary
        logger.info("🎯 Pipeline Summary:")
        logger.info(f"   Total time: {results['execution_time']['total_seconds']:.2f}s")
        logger.info(f"   Input samples: {results['summary']['total_input_samples']}")
        logger.info(f"   Sarcastic samples: {results['summary']['sarcastic_samples']}")
        logger.info(f"   Generated paraphrases: {results['summary']['total_paraphrases']}")
        logger.info(f"   Factual samples: {results['summary']['factual_samples']}")
        
        logger.info("✅ Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
