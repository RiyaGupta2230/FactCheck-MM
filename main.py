#!/usr/bin/env python3
"""
FactCheck-MM: Multimodal Fact-Checking Pipeline
Main CLI entrypoint for training, evaluation, and inference.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from Config import get_config
from shared.utils import setup_logging, get_logger
from shared import MultimodalEncoder


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    
    parser = argparse.ArgumentParser(
        description="FactCheck-MM: Multimodal Fact-Checking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--config", 
        type=str,
        default="default",
        help="Configuration profile name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, -vvv)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        "task",
        choices=["sarcasm_detection", "paraphrasing", "fact_verification", "multitask"],
        help="Task to train"
    )
    train_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to use (default: all available for task)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint path"
    )
    train_parser.add_argument(
        "--chunked",
        action="store_true",
        help="Enable chunked training for low-memory devices"
    )
    train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    eval_parser.add_argument(
        "task",
        choices=["sarcasm_detection", "paraphrasing", "fact_verification", "all"],
        help="Task to evaluate"
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to evaluate on"
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation results"
    )
    eval_parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save detailed predictions"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on new data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument(
        "task",
        choices=["sarcasm_detection", "paraphrasing", "fact_verification"],
        help="Task for prediction"
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        help="Input file or text string"
    )
    predict_parser.add_argument(
        "--input-type",
        choices=["text", "file", "batch"],
        default="text",
        help="Type of input"
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )
    predict_parser.add_argument(
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Output format"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system and dataset information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    info_parser.add_argument(
        "--datasets",
        action="store_true",
        help="Show dataset information"
    )
    info_parser.add_argument(
        "--system",
        action="store_true",
        help="Show system information"  
    )
    info_parser.add_argument(
        "--models",
        action="store_true",
        help="Show model information"
    )
    
    return parser


def setup_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup the environment and load configuration."""
    
    # Setup logging
    log_level = "ERROR" if args.quiet else "INFO"
    if args.verbose >= 1:
        log_level = "DEBUG"
    
    setup_logging(level=log_level)
    logger = get_logger("main")
    
    # Load configuration
    config = get_config(args.config)
    
    # Override device if specified
    if args.device != "auto":
        config["base"].device = args.device
    
    logger.info(f"🚀 Starting FactCheck-MM")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {config['base'].device}")
    
    return config


def cmd_train(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Execute training command."""
    logger = get_logger("train")
    
    # Import task-specific trainers
    if args.task == "sarcasm_detection":
        from sarcasm_detection.training import train_multimodal
        trainer = train_multimodal.MultimodalSarcasmTrainer(config)
    elif args.task == "paraphrasing":
        from paraphrasing.training import train_generation
        trainer = train_generation.ParaphraseTrainer(config)
    elif args.task == "fact_verification":
        from fact_verification.training import train_end_to_end
        trainer = train_end_to_end.FactVerificationTrainer(config)
    elif args.task == "multitask":
        from experiments.multitask_learning import joint_trainer
        trainer = joint_trainer.MultiTaskTrainer(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Override config with CLI arguments
    training_config = config["training"].get_config_for_task(args.task)
    
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.optimizer.learning_rate = args.learning_rate
    if args.chunked:
        config["training"].chunked_training.enabled = True
    if args.no_wandb:
        config["base"].use_wandb = False
    
    logger.info(f"Training {args.task} model")
    logger.info(f"Datasets: {args.datasets or 'all available'}")
    logger.info(f"Epochs: {training_config.num_epochs}")
    logger.info(f"Batch size: {training_config.batch_size}")
    
    # Start training
    try:
        trainer.train(
            datasets=args.datasets,
            resume_from_checkpoint=args.resume
        )
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def cmd_eval(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Execute evaluation command."""
    logger = get_logger("eval")
    
    # Import task-specific evaluators
    if args.task == "sarcasm_detection":
        from sarcasm_detection.evaluation import evaluator
        eval_class = evaluator.SarcasmEvaluator
    elif args.task == "paraphrasing":
        from paraphrasing.evaluation import generation_metrics
        eval_class = generation_metrics.ParaphraseEvaluator
    elif args.task == "fact_verification":
        from fact_verification.evaluation import pipeline_evaluation
        eval_class = pipeline_evaluation.FactVerificationEvaluator
    elif args.task == "all":
        # Run evaluation for all tasks
        logger.info("Evaluating all tasks")
        for task in ["sarcasm_detection", "paraphrasing", "fact_verification"]:
            args.task = task
            cmd_eval(args, config)
        return
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    logger.info(f"Evaluating {args.task} model")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Initialize evaluator
    evaluator_instance = eval_class(config)
    
    try:
        results = evaluator_instance.evaluate(
            checkpoint_path=args.checkpoint,
            datasets=args.datasets,
            output_dir=args.output_dir,
            save_predictions=args.save_predictions
        )
        
        logger.info("✅ Evaluation completed successfully")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


def cmd_predict(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Execute prediction command."""
    logger = get_logger("predict")
    
    # Import task-specific predictors
    if args.task == "sarcasm_detection":
        from sarcasm_detection.models import multimodal_sarcasm
        model_class = multimodal_sarcasm.MultimodalSarcasmModel
    elif args.task == "paraphrasing":
        from paraphrasing.models import t5_paraphraser
        model_class = t5_paraphraser.T5ParaphraserModel
    elif args.task == "fact_verification":
        from fact_verification.models import end_to_end_model
        model_class = end_to_end_model.FactVerificationModel
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    logger.info(f"Running inference for {args.task}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Input: {args.input}")
    
    try:
        # Load model
        model = model_class.from_checkpoint(args.checkpoint)
        model.eval()
        
        # Process input based on type
        if args.input_type == "text":
            results = model.predict_text(args.input)
        elif args.input_type == "file":
            results = model.predict_file(args.input)
        elif args.input_type == "batch":
            results = model.predict_batch(args.input)
        else:
            raise ValueError(f"Unknown input type: {args.input_type}")
        
        # Save or print results
        if args.output:
            model.save_predictions(results, args.output, format=args.format)
            logger.info(f"Results saved to {args.output}")
        else:
            print(results)
        
        logger.info("✅ Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise


def cmd_info(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Show system and dataset information."""
    logger = get_logger("info")
    
    if args.system or not (args.datasets or args.models):
        print("\n🖥️  System Information:")
        print(f"   Device: {config['base'].device}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        device_info = config['base'].get_device_info()
        for key, value in device_info.items():
            print(f"   {key}: {value}")
    
    if args.datasets or not (args.system or args.models):
        print("\n📊 Dataset Information:")
        dataset_info = config["datasets"].get_dataset_info()
        print(f"   Total datasets: {dataset_info['total_datasets']}")
        print(f"   Multimodal datasets: {dataset_info['multimodal_count']}")
        print("   By task:")
        for task, count in dataset_info["by_task"].items():
            print(f"     {task}: {count}")
        print("   By modality:")
        for modality, count in dataset_info["by_modality"].items():
            print(f"     {modality}: {count}")
    
    if args.models or not (args.system or args.datasets):
        print("\n🤖 Model Information:")
        model_configs = config["models"]
        print(f"   Sarcasm Detection: {model_configs.sarcasm_detection.text_model_name}")
        print(f"   Paraphrasing: {model_configs.paraphrasing.base_model_name}")
        print(f"   Fact Verification: {model_configs.fact_verification.verifier_model_name}")
        print(f"   Shared Encoder Hidden Size: {model_configs.shared_encoder['hidden_dim']}")
        print(f"   Multi-task Learning: {model_configs.multitask_learning['enabled']}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Setup environment and load config
        config = setup_environment(args)
        
        # Execute command
        if args.command == "train":
            cmd_train(args, config)
        elif args.command == "eval":
            cmd_eval(args, config)
        elif args.command == "predict":
            cmd_predict(args, config)
        elif args.command == "info":
            cmd_info(args, config)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
