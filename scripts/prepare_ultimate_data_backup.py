import pandas as pd
import json
import logging
from pathlib import Path
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_your_exact_data():
    """Data preparation for YOUR exact file structure"""
    logger.info("🚀 PREPARING DATA FROM YOUR EXACT STRUCTURE")
    
    all_samples = []
    
    # 1. Load MUStARD JSONL files (NOT JSON!)
    mustard_files = [
        'data/multimodal/mustard_bert/bert-output-context.jsonl',
        'data/multimodal/mustard_bert/bert-output.jsonl'
    ]
    
    for file_path in mustard_files:
        if Path(file_path).exists():
            try:
                logger.info(f"📋 Loading JSONL: {file_path}")
                
                # Read JSONL (one JSON per line)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                logger.info(f"✅ Found {len(lines)} lines in {file_path}")
                
                for i, line in enumerate(lines[:10000]):  # Limit for performance
                    try:
                        item = json.loads(line.strip())
                        
                        text = str(item.get('utterance', item.get('text', '')))
                        label = int(item.get('sarcasm', 0))
                        context = str(item.get('context', ''))
                        
                        # Enhanced text with context
                        if context and len(context.split()) <= 20:
                            enhanced_text = f"{context} [SEP] {text}"
                        else:
                            enhanced_text = text
                        
                        if enhanced_text and len(enhanced_text.split()) >= 3:
                            all_samples.append({
                                'text': enhanced_text,
                                'label': label,
                                'image_path': '',
                                'audio_path': '',
                                'domain': 'tv_dialogue',
                                'language': 'en',
                                'quality_score': min(len(enhanced_text.split()), 50),
                                'source': 'mustard',
                                'context_length': len(enhanced_text),
                                'has_context': bool(context)
                            })
                    except json.JSONDecodeError:
                        continue
                        
                logger.info(f"✅ Loaded {len(all_samples)} samples from MUStARD")
                        
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
    
    # 2. Load root multimodal JSON files
    root_json_files = [
        'data/multimodal/train.json',
        'data/multimodal/val.json'
    ]
    
    for file_path in root_json_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"✅ Loaded {file_path}: {len(data)} samples")
                
                for item in data:
                    if isinstance(item, dict):
                        text = str(item.get('text', item.get('utterance', item.get('sentence', ''))))
                        label = item.get('label', item.get('sarcasm', 0))
                        
                        if isinstance(label, str):
                            label = 1 if label.lower() in ['1', 'true', 'sarcastic'] else 0
                        
                        if text and len(text.split()) >= 3:
                            all_samples.append({
                                'text': text,
                                'label': int(label),
                                'image_path': item.get('image_path', ''),
                                'audio_path': item.get('audio_path', ''),
                                'domain': 'general',
                                'language': 'en',
                                'quality_score': min(len(text.split()), 50),
                                'source': 'multimodal_root',
                                'context_length': len(text)
                            })
                            
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
    
    # 3. Load Text Sarcasm CSV files
    text_files = [
        'data/text/sarc/train-balanced-sarcasm.csv',
        'data/text/sarc/test-balanced.csv'
    ]
    
    for file_path in text_files:
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                logger.info(f"✅ Loaded {file_path}: {len(df)} samples")
                logger.info(f"   Columns: {list(df.columns)}")
                
                for _, row in df.iterrows():
                    # Try multiple column names
                    text = str(row.get('text', row.get('sentence', row.get('comment', row.get('tweet', '')))))
                    label = row.get('label', row.get('sarcasm', row.get('class', 0)))
                    
                    if isinstance(label, str):
                        label = 1 if label.lower() in ['1', 'true', 'sarcastic'] else 0
                    
                    if text and len(text.split()) >= 3:
                        all_samples.append({
                            'text': text,
                            'label': int(label),
                            'image_path': '',
                            'audio_path': '',
                            'domain': 'general',
                            'language': 'en',
                            'quality_score': min(len(text.split()), 50),
                            'source': 'text_sarc',
                            'context_length': len(text)
                        })
                        
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
    
    # 4. Load Headlines Dataset
    headlines_file = 'data/text/Sarcasm_Headlines_Dataset.json'
    if Path(headlines_file).exists():
        try:
            with open(headlines_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded {headlines_file}: {len(data)} samples")
            
            for item in data[:5000]:  # Limit for balance
                if isinstance(item, dict):
                    text = str(item.get('headline', item.get('text', '')))
                    label = item.get('is_sarcastic', 0)
                    
                    if text and len(text.split()) >= 3:
                        all_samples.append({
                            'text': text,
                            'label': int(label),
                            'image_path': '',
                            'audio_path': '',
                            'domain': 'news',
                            'language': 'en',
                            'quality_score': min(len(text.split()), 50),
                            'source': 'headlines',
                            'context_length': len(text)
                        })
                        
        except Exception as e:
            logger.error(f"❌ Error loading {headlines_file}: {e}")
    
    logger.info(f"🎯 Total samples collected: {len(all_samples)}")
    
    if not all_samples:
        logger.error("❌ No samples found!")
        return
    
    # 5. Smart balancing
    sarcastic = [s for s in all_samples if s['label'] == 1]
    non_sarcastic = [s for s in all_samples if s['label'] == 0]
    
    target_sarcastic = min(len(sarcastic), int(len(non_sarcastic) * 0.67))
    target_non_sarcastic = min(len(non_sarcastic), int(target_sarcastic * 1.5))
    
    # Prioritize high-quality samples
    sarcastic.sort(key=lambda x: x['quality_score'], reverse=True)
    non_sarcastic.sort(key=lambda x: x['quality_score'], reverse=True)
    
    balanced_samples = (sarcastic[:target_sarcastic] + 
                       non_sarcastic[:target_non_sarcastic])
    
    random.shuffle(balanced_samples)
    logger.info(f"✅ Balanced dataset: {target_non_sarcastic} non-sarcastic, {target_sarcastic} sarcastic")
    
    # 6. Create chunks
    chunks_dir = Path("data/chunks/milestone1_ultimate")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # 80/20 split
    split_point = int(0.8 * len(balanced_samples))
    train_samples = balanced_samples[:split_point]
    test_samples = balanced_samples[split_point:]
    
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Create training chunks (800 samples each)
    chunk_size = 800
    train_chunks = []
    
    for i in range(0, len(train_samples), chunk_size):
        chunk = train_samples[i:i + chunk_size]
        chunk_df = pd.DataFrame(chunk)
        
        chunk_file = chunks_dir / f"ultimate_train_chunk_{i//chunk_size + 1:03d}.csv"
        chunk_df.to_csv(chunk_file, index=False, encoding='utf-8')
        train_chunks.append(chunk_file)
        
        # Log chunk stats
        label_dist = chunk_df['label'].value_counts()
        logger.info(f"Created {chunk_file.name}: {len(chunk)} samples "
                   f"({label_dist.get(0,0)} non-sarcastic, {label_dist.get(1,0)} sarcastic)")
    
    # Create test chunks
    test_chunks = []
    for i in range(0, len(test_samples), chunk_size):
        chunk = test_samples[i:i + chunk_size]
        chunk_df = pd.DataFrame(chunk)
        
        chunk_file = chunks_dir / f"ultimate_test_chunk_{i//chunk_size + 1:03d}.csv"
        chunk_df.to_csv(chunk_file, index=False, encoding='utf-8')
        test_chunks.append(chunk_file)
        
        logger.info(f"Created {chunk_file.name}: {len(chunk)} samples")
    
    # Success summary
    logger.info("🎉 DATA PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info(f"✅ Training chunks: {len(train_chunks)}")
    logger.info(f"✅ Test chunks: {len(test_chunks)}")
    logger.info(f"✅ Total samples: {len(balanced_samples)}")
    logger.info("🚀 Ready for ultimate training!")

if __name__ == "__main__":
    prepare_your_exact_data()
