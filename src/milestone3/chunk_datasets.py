import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path
import yaml
import sys

class DatasetChunker:
    def __init__(self, config_path="config/training_config.yaml"):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunk_size = self.config['chunking']['chunk_size']
        self.chunks_dir = Path(self.config['paths']['chunks_dir'])
        self.chunks_dir.mkdir(exist_ok=True)
        
        print(f"Initialized DatasetChunker with chunk size: {self.chunk_size}")
        print(f"Output directory: {self.chunks_dir}")
    
    def load_datasets(self):
        """Load all datasets for chunking"""
        datasets = {}
        
        # Load SARC dataset
        sarc_path = Path(self.config['datasets']['sarc']['train'])
        if sarc_path.exists():
            print(f"Loading SARC dataset: {sarc_path}")
            sarc_df = pd.read_csv(sarc_path)
            
            # Standardize column names for SARC
            if 'comment' in sarc_df.columns:
                sarc_df = sarc_df.rename(columns={'comment': 'text'})
            elif 'text' not in sarc_df.columns:
                text_cols = [col for col in sarc_df.columns 
                        if any(word in col.lower() for word in ['text', 'comment', 'content', 'sentence'])]
                if text_cols:
                    sarc_df = sarc_df.rename(columns={text_cols[0]: 'text'})
            
            # Ensure label column exists
            if 'label' not in sarc_df.columns:
                label_cols = [col for col in sarc_df.columns 
                            if any(word in col.lower() for word in ['label', 'sarcasm', 'target', 'class'])]
                if label_cols:
                    sarc_df = sarc_df.rename(columns={label_cols[0]: 'label'})
            
            datasets['sarc'] = sarc_df
            print(f"Loaded SARC: {len(sarc_df)} samples")
        
        # Load News Headlines (FIXED VERSION)
        headlines_path = Path(self.config['datasets']['headlines']['data'])
        if headlines_path.exists():
            print(f"Loading Headlines dataset: {headlines_path}")
            try:
                import json
                headlines_data = []
                
                with open(headlines_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    f.seek(0)
                    
                    if first_line.startswith('['):
                        # Standard JSON array
                        headlines_data = json.load(f)
                    else:
                        # JSONL format (one JSON per line)
                        for line in f:
                            line = line.strip()
                            if line:
                                headlines_data.append(json.loads(line))
                
                headlines_df = pd.DataFrame(headlines_data)
                headlines_df = headlines_df.rename(columns={'headline': 'text', 'is_sarcastic': 'label'})
                datasets['headlines'] = headlines_df
                print(f"Loaded Headlines: {len(headlines_df)} samples")
                
            except Exception as e:
                print(f"⚠️  Error loading headlines: {e}")
                print("Skipping headlines dataset...")
        
        # Load LIAR dataset (keep existing code)
            
        liar_path = Path(self.config['datasets']['liar']['train'])
        if liar_path.exists():
            print(f"Loading LIAR dataset: {liar_path}")
            try:
                # Load TSV without headers (LIAR format)
                liar_df = pd.read_csv(liar_path, sep='\t', header=None, quoting=3)
                
                # Assign correct column names (standard LIAR format)
                liar_df.columns = ['id', 'label', 'text', 'subject', 'speaker', 'job', 'state', 'party',
                                'barely_true', 'false_ct', 'half_true', 'mostly_true', 'pants_fire', 'context']
                
                # Keep only text and label columns
                liar_df = liar_df[['text', 'label']].copy()
                
                # Convert 6-class labels to binary
                label_mapping = {
                    'true': 1, 'mostly-true': 1, 'half-true': 1,
                    'barely-true': 0, 'false': 0, 'pants-fire': 0
                }
                liar_df['label'] = liar_df['label'].astype(str).str.lower().map(label_mapping)
                liar_df = liar_df.dropna()  # Remove any unmapped labels
                
                datasets['liar'] = liar_df
                print(f"Loaded LIAR: {len(liar_df)} samples")
                
            except Exception as e:
                print(f"⚠️  Error loading LIAR: {e}")
                print("Skipping LIAR dataset...")

        return datasets

    
    def create_chunks(self):
        """Create balanced chunks from all datasets"""
        datasets = self.load_datasets()
        
        if not datasets:
            print("No datasets found! Please check your data paths in config.")
            return {}
        
        chunk_info = {}
        total_chunks_created = 0
        
        for dataset_name, df in datasets.items():
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            print(f"Original samples: {len(df)}")
            
            # Ensure required columns exist
            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"❌ Missing required columns in {dataset_name}. Available: {list(df.columns)}")
                print("Required: 'text' and 'label' columns")
                continue
            
            # Clean data
            original_len = len(df)
            df = df.dropna(subset=['text', 'label'])
            df = df[df['text'].astype(str).str.len() > 0]  # Remove empty texts
            
            print(f"After cleaning: {len(df)} samples (removed {original_len - len(df)})")
            
            if len(df) == 0:
                print(f"❌ No valid data remaining in {dataset_name}")
                continue
            
            # Create chunks
            chunks = self._chunk_dataframe(df, dataset_name)
            chunk_info[dataset_name] = {
                'total_samples': len(df),
                'num_chunks': len(chunks),
                'chunk_paths': chunks,
                'samples_per_chunk': self.chunk_size
            }
            
            total_chunks_created += len(chunks)
            print(f"✅ Created {len(chunks)} chunks for {dataset_name}")
        
        # Save chunk information
        chunk_info_path = self.chunks_dir / 'chunk_info.json'
        with open(chunk_info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"CHUNKING SUMMARY")
        print(f"{'='*60}")
        print(f"Total datasets processed: {len(chunk_info)}")
        print(f"Total chunks created: {total_chunks_created}")
        print(f"Chunk info saved to: {chunk_info_path}")
        
        return chunk_info
    
    def _chunk_dataframe(self, df, dataset_name):
        """Split dataframe into chunks and save"""
        chunks = []
        num_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        # Shuffle data for better distribution
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(df_shuffled))
            
            chunk_df = df_shuffled.iloc[start_idx:end_idx].copy()
            chunk_path = self.chunks_dir / f"{dataset_name}_chunk_{i+1:03d}.csv"
            
            # Ensure we have both classes in each chunk (if possible)
            if 'label' in chunk_df.columns and len(chunk_df['label'].unique()) < 2 and len(df_shuffled) > self.chunk_size:
                print(f"⚠️  Chunk {i+1} has only one class, attempting rebalance...")
                # Simple rebalancing by mixing with next chunk
                if i < num_chunks - 1:
                    next_start = end_idx
                    next_end = min(next_start + self.chunk_size, len(df_shuffled))
                    if next_end > next_start:
                        next_chunk = df_shuffled.iloc[next_start:next_start + min(100, next_end - next_start)]
                        chunk_df = pd.concat([chunk_df[:-50], next_chunk[:50]], ignore_index=True)
            
            # Save chunk
            chunk_df.to_csv(chunk_path, index=False)
            chunks.append(str(chunk_path))
            
            # Print chunk statistics
            if 'label' in chunk_df.columns:
                label_counts = chunk_df['label'].value_counts()
                print(f"  Chunk {i+1:03d}: {len(chunk_df)} samples, labels: {dict(label_counts)}")
            else:
                print(f"  Chunk {i+1:03d}: {len(chunk_df)} samples")
        
        return chunks
    
    def validate_chunks(self):
        """Validate created chunks"""
        chunk_info_path = self.chunks_dir / 'chunk_info.json'
        if not chunk_info_path.exists():
            print("No chunk info found. Run create_chunks() first.")
            return False
        
        with open(chunk_info_path, 'r') as f:
            chunk_info = json.load(f)
        
        print(f"\n{'='*50}")
        print(f"CHUNK VALIDATION")
        print(f"{'='*50}")
        
        valid_chunks = 0
        total_chunks = 0
        
        for dataset_name, info in chunk_info.items():
            print(f"\nValidating {dataset_name} chunks:")
            dataset_valid = 0
            
            for chunk_path in info['chunk_paths']:
                total_chunks += 1
                chunk_file = Path(chunk_path)
                
                if chunk_file.exists():
                    try:
                        df = pd.read_csv(chunk_file)
                        if len(df) > 0 and 'text' in df.columns and 'label' in df.columns:
                            valid_chunks += 1
                            dataset_valid += 1
                        else:
                            print(f"  ❌ Invalid chunk: {chunk_file.name}")
                    except Exception as e:
                        print(f"  ❌ Error reading {chunk_file.name}: {e}")
                else:
                    print(f"  ❌ Missing chunk: {chunk_file.name}")
            
            print(f"  {dataset_name}: {dataset_valid}/{len(info['chunk_paths'])} chunks valid")
        
        print(f"\nValidation Summary: {valid_chunks}/{total_chunks} chunks valid")
        return valid_chunks == total_chunks

def main():
    parser = argparse.ArgumentParser(description="Create dataset chunks for training")
    parser.add_argument("--config", default="config/training_config.yaml", help="Config file path")
    parser.add_argument("--validate", action="store_true", help="Validate existing chunks")
    args = parser.parse_args()
    
    try:
        chunker = DatasetChunker(args.config)
        
        if args.validate:
            chunker.validate_chunks()
        else:
            chunk_info = chunker.create_chunks()
            
            if chunk_info:
                print(f"\n🎉 Chunking completed successfully!")
                print(f"📁 Chunks saved to: {Path('data/chunks').absolute()}")
                print(f"📊 Ready for training with: python scripts/train_milestone3.py")
            else:
                print(f"\n❌ Chunking failed. Please check your configuration and data files.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
