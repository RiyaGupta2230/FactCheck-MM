#!/usr/bin/env python3

import subprocess
import sys
import yaml
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run chunked training pipeline")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip dataset chunking")
    parser.add_argument("--skip-training", action="store_true", help="Skip chunk training")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble building")
    
    args = parser.parse_args()
    
    print("🚀 FACTCHECK-MM CHUNKED TRAINING PIPELINE")
    print("=" * 60)
    
    success_count = 0
    
    # Step 1: Create chunks
    if not args.skip_chunking:
        cmd = [sys.executable, "-m", "src.milestone3.chunk_datasets"]
        if run_command(cmd, "📊 STEP 1: CREATING DATASET CHUNKS"):
            success_count += 1
    else:
        print("⏭️  STEP 1 SKIPPED: Dataset chunking")
        success_count += 1
    
    # Step 2: Train chunks
    if not args.skip_training:
        cmd = [sys.executable, "scripts/train_milestone3.py", "--chunked"]
        if run_command(cmd, "🏋️  STEP 2: TRAINING CHUNK MODELS"):
            success_count += 1
    else:
        print("⏭️  STEP 2 SKIPPED: Chunk training")
        success_count += 1
    
    # Step 3: Build ensemble
    if not args.skip_ensemble:
        cmd = [sys.executable, "-m", "src.milestone3.ensemble_builder"]
        if run_command(cmd, "🤖 STEP 3: BUILDING ENSEMBLE"):
            success_count += 1
    else:
        print("⏭️  STEP 3 SKIPPED: Ensemble building")
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/3 steps")
    
    if success_count == 3:
        print("✅ All steps completed successfully!")
        print("📁 Final model: models/final_ensemble_model.pt")
        print("📁 Chunk models: models/chunk_models/")
        print("📊 Results: training_results.json")

if __name__ == "__main__":
    main()
