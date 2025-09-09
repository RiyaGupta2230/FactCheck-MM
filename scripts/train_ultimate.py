import torch
import yaml
import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
import json
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging for the main training script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ultimate_training_main.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load and validate configuration"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['training', 'paths', 'model_architecture', 'datasets']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config

def discover_training_chunks(config):
    """Discover available training chunks"""
    chunks_dir = Path(config['paths']['chunks_dir'])
    
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    # Look for training and test chunks
    train_chunks = sorted(chunks_dir.glob("ultimate_train_chunk_*.csv"))
    test_chunks = sorted(chunks_dir.glob("ultimate_test_chunk_*.csv"))
    
    # If no train/test split, look for regular chunks
    if not train_chunks:
        all_chunks = sorted(chunks_dir.glob("ultimate_chunk_*.csv"))
        return all_chunks, []
    
    return train_chunks, test_chunks

def train_single_chunk(trainer, chunk_name, train_chunk_path, test_chunk_path=None):
    """Train model on a single chunk"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎯 Starting training for chunk: {chunk_name}")
    
    # Import dataset loader
    from src.milestone1.enhanced_dataset import create_ultimate_dataloader
    
    # Create data loaders
    train_dataloader = create_ultimate_dataloader(
        train_chunk_path, 
        trainer.config, 
        split='train',
        batch_size=trainer.config['training']['batch_size'],
        shuffle=True
    )
    
    val_dataloader = None
    if test_chunk_path and test_chunk_path.exists():
        val_dataloader = create_ultimate_dataloader(
            test_chunk_path,
            trainer.config,
            split='test',
            batch_size=trainer.config['training']['batch_size'],
            shuffle=False
        )
    
    # Train the chunk
    start_time = time.time()
    results = trainer.train_ultimate_chunk(chunk_name, train_dataloader, val_dataloader)
    end_time = time.time()
    
    # Add timing information
    results['training_time_minutes'] = (end_time - start_time) / 60
    results['training_time_hours'] = results['training_time_minutes'] / 60
    
    logger.info(f"✅ Chunk {chunk_name} completed in {results['training_time_minutes']:.1f} minutes")
    logger.info(f"🎯 Best F1: {results['best_f1']:.4f}, Best Accuracy: {results['best_accuracy']:.4f}")
    
    return results

def save_training_summary(all_results, config, total_time):
    """Save comprehensive training summary"""
    logger = logging.getLogger(__name__)
    
    # Calculate overall statistics
    summary = {
        'training_summary': {
            'start_time': datetime.now().isoformat(),
            'total_training_time_hours': total_time / 3600,
            'total_chunks_trained': len(all_results),
            'config_used': config
        },
        'performance_statistics': {
            'best_f1_score': max(result['best_f1'] for result in all_results) if all_results else 0.0,
            'average_f1_score': sum(result['best_f1'] for result in all_results) / len(all_results) if all_results else 0.0,
            'best_accuracy': max(result['best_accuracy'] for result in all_results) if all_results else 0.0,
            'average_accuracy': sum(result['best_accuracy'] for result in all_results) / len(all_results) if all_results else 0.0
        },
        'chunk_results': all_results,
        'hardware_info': {
            'device_used': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'mixed_precision': config['hardware']['mixed_precision']
        }
    }
    
    # Save summary
    summary_path = Path(config['paths']['logs_dir']) / 'ultimate_training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📋 Training summary saved to: {summary_path}")
    
    # Display final statistics
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("🎉 ULTIMATE TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Total chunks trained: {len(all_results)}")
        logger.info(f"⏱️ Total training time: {total_time/3600:.1f} hours")
        logger.info(f"🏆 Best F1 score: {summary['performance_statistics']['best_f1_score']:.4f}")
        logger.info(f"📈 Average F1 score: {summary['performance_statistics']['average_f1_score']:.4f}")
        logger.info(f"🏆 Best accuracy: {summary['performance_statistics']['best_accuracy']:.4f}")
        logger.info(f"📈 Average accuracy: {summary['performance_statistics']['average_accuracy']:.4f}")
        
        # Performance assessment
        best_f1 = summary['performance_statistics']['best_f1_score']
        if best_f1 >= 0.85:
            logger.info("🥇 OUTSTANDING PERFORMANCE! Ready for deployment!")
        elif best_f1 >= 0.80:
            logger.info("🥈 EXCELLENT PERFORMANCE! Very competitive results!")
        elif best_f1 >= 0.75:
            logger.info("🥉 GOOD PERFORMANCE! Solid results!")
        else:
            logger.info("📈 MODERATE PERFORMANCE - Consider ensemble building!")
        
        logger.info("="*80)
    
    return summary

def main():
    """Main training function"""
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Ultimate Multimodal Sarcasm Detection Training')
    parser.add_argument('--config', default='config/milestone1_ultimate_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--chunk', type=str, help='Train specific chunk only (e.g., ultimate_train_chunk_001)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--dry_run', action='store_true', help='Perform dry run without actual training')
    parser.add_argument('--build_ensemble', action='store_true', help='Build ensemble after training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 ULTIMATE MULTIMODAL SARCASM DETECTION TRAINING")
    logger.info("🎯 Target: 85%+ F1 Score")
    logger.info("🔥 RTX 2050 Optimized")
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"✅ Configuration loaded from: {args.config}")
        
        # Discover training chunks
        train_chunks, test_chunks = discover_training_chunks(config)
        logger.info(f"📦 Found {len(train_chunks)} training chunks")
        if test_chunks:
            logger.info(f"📦 Found {len(test_chunks)} test chunks")
        
        if not train_chunks:
            logger.error("❌ No training chunks found! Please run data preparation first.")
            return 1
        
        # Initialize trainer
        from src.milestone1.advanced_trainer import UltimateRTXTrainer
        trainer = UltimateRTXTrainer(args.config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Dry run check
        if args.dry_run:
            logger.info("🧪 Dry run mode - validating setup without training")
            logger.info("✅ Setup validation successful!")
            return 0
        
        # Training loop
        all_results = []
        total_start_time = time.time()
        
        if args.chunk:
            # Train specific chunk
            chunk_name = args.chunk
            train_chunk_path = None
            test_chunk_path = None
            
            # Find the specified chunk
            for train_chunk in train_chunks:
                if chunk_name in train_chunk.name:
                    train_chunk_path = train_chunk
                    # Find corresponding test chunk
                    test_chunk_name = train_chunk.name.replace('train_', 'test_')
                    test_chunk_path = train_chunk.parent / test_chunk_name
                    break
            
            if not train_chunk_path:
                logger.error(f"❌ Chunk '{chunk_name}' not found!")
                return 1
            
            # Train single chunk
            result = train_single_chunk(trainer, chunk_name, train_chunk_path, test_chunk_path)
            all_results.append(result)
            
        else:
            # Train all chunks
            for i, train_chunk in enumerate(train_chunks):
                chunk_name = train_chunk.stem
                
                # Find corresponding test chunk
                test_chunk_path = None
                if test_chunks:
                    test_chunk_name = train_chunk.name.replace('train_', 'test_')
                    potential_test_chunk = train_chunk.parent / test_chunk_name
                    if potential_test_chunk.exists():
                        test_chunk_path = potential_test_chunk
                
                logger.info(f"\n{'='*60}")
                logger.info(f"📦 Processing chunk {i+1}/{len(train_chunks)}: {chunk_name}")
                logger.info(f"{'='*60}")
                
                # Train chunk
                result = train_single_chunk(trainer, chunk_name, train_chunk, test_chunk_path)
                all_results.append(result)
                
                # Save intermediate results
                intermediate_summary = {
                    'completed_chunks': i + 1,
                    'total_chunks': len(train_chunks),
                    'results_so_far': all_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                intermediate_path = Path(config['paths']['logs_dir']) / 'intermediate_results.json'
                intermediate_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(intermediate_path, 'w') as f:
                    json.dump(intermediate_summary, f, indent=2, default=str)
        
        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        
        # Save final training summary
        summary = save_training_summary(all_results, config, total_training_time)
        
        # Build ensemble if requested
        if args.build_ensemble:
            logger.info("\n🔧 Building ultimate ensemble...")
            try:
                from scripts.build_ultimate_ensemble import UltimateEnsembleBuilder
                ensemble_builder = UltimateEnsembleBuilder(args.config)
                ensemble_metrics = ensemble_builder.build_complete_ensemble()
                
                logger.info(f"🎉 Ensemble built successfully!")
                logger.info(f"🎯 Ensemble F1: {ensemble_metrics['f1']:.4f}")
                logger.info(f"🎯 Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Ensemble building failed: {e}")
        
        logger.info("\n🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
