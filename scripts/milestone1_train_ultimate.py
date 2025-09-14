"""
Ultimate training script for milestone1 sarcasm detection
Optimized for maximum accuracy with your prepared chunks
"""
import os
import yaml
import torch
import pandas as pd
import numpy as np
import glob
import logging
from PIL import Image
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
     AutoImageProcessor, AutoProcessor,
    RobertaForSequenceClassification, 
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW 
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
from milestone1_feature_engineering import prepare_features_for_training
from src.utils.milestone1_model_utils import MultimodalSarcasmModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateSarcasmDataset(Dataset):
    """Multimodal dataset with optional image/audio and modality mask"""
    def __init__(self, df, feature_cols, tokenizer, max_length,
                 use_image=False, vision_processor=None,
                 use_audio=False, audio_processor=None, audio_sr=16000,
                 modality_dropout_prob=0.0):
        self.df = df.reset_index(drop=True)
        self.features = df[feature_cols].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_image = use_image
        self.vision_processor = vision_processor
        self.use_audio = use_audio
        self.audio_processor = audio_processor
        self.audio_sr = audio_sr
        self.modality_dropout_prob = modality_dropout_prob

        self.has_image_col = 'image_path' in df.columns
        self.has_audio_col = 'audio_path' in df.columns

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])

        # text
        enc = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        # modality mask [text, image?, audio?]
        mask_list = [5]  # text present

        # Image branch
        if self.use_image and self.has_image_col and pd.notna(row.get('image_path', None)):
            try:
                img = Image.open(row['image_path']).convert('RGB')
                proc = self.vision_processor(images=img, return_tensors='pt')
                image_pixels = proc['pixel_values'].squeeze(0)  # [3, H, W]
                mask_list.append(1)
            except Exception:
                mask_list.append(0)
        elif self.use_image:
            mask_list.append(0)

        # Audio branch
        if self.use_audio and self.has_audio_col and pd.notna(row.get('audio_path', None)):
            try:
                wav, sr = torchaudio.load(row['audio_path'])  # [C, T]
                wav = wav.mean(dim=0)  # mono
                if sr != self.audio_sr:
                    wav = torchaudio.functional.resample(wav, sr, self.audio_sr)
                proc = self.audio_processor(wav.numpy(), sampling_rate=self.audio_sr, return_tensors='pt', padding='max_length', max_length=self.audio_sr * 5, truncation=True)
                audio_values = proc['input_values'].squeeze(0)     # [T_fixed]
                audio_attn = proc.get('attention_mask', None)
                if audio_attn is not None:
                    audio_attn = audio_attn.squeeze(0)
                mask_list.append(1)
            except Exception:
                mask_list.append(0)
        elif self.use_audio:
            mask_list.append(0)

        # Modality dropout (robust to missing at inference)
        if self.modality_dropout_prob > 0:
            import random
            # never drop text, but maybe drop image/audio when present
            for i in range(1, len(mask_list)):
                if mask_list[i] == 1 and random.random() < self.modality_dropout_prob:
                    mask_list[i] = 0
                    if i == 1 and self.use_image:
                        image_pixels = None
                    if (i == 2 and self.use_audio) or (i == 1 and self.use_audio and not self.use_image):
                        audio_values, audio_attn = None, None

        modality_mask = torch.tensor(mask_list, dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_pixels': image_pixels if image_pixels is not None else torch.tensor(0),
            'audio_values': audio_values if audio_values is not None else torch.tensor(0),
            'audio_attn': audio_attn if audio_attn is not None else torch.tensor(0),
            'modality_mask': modality_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class UltimateTrainer:
    """Ultimate training class for maximum accuracy"""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() or torch.backends.mps.is_available() 
            else 'cpu'
        )
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging for training"""
        log_dir = os.path.join(self.config['paths']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'ultimate_training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train_ultimate_model(self):
        """Train the ultimate sarcasm detection model"""
        
        logger.info("🚀 Starting Ultimate Sarcasm Detection Training")
        logger.info(f"🎯 Target F1: {self.config['project']['target_f1']}")
        logger.info(f"📱 Device: {self.device}")
        
        # Initialize tokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained(self.config['model']['name'])
        use_image = self.config['modalities']['use_image']
        use_audio = self.config['modalities']['use_audio']

        vision_processor = None
        audio_processor = None
        if use_image:
            vision_processor = AutoImageProcessor.from_pretrained(self.config['vision_encoder']['model_name'])
        if use_audio:
            audio_processor = AutoProcessor.from_pretrained(self.config['audio_encoder']['model_name'])
            sample_rate = int(self.config['audio_encoder']['sample_rate'])

        
        # Get all training chunks
        chunks_dir = self.config['data']['chunks_dir']
        train_pattern = self.config['data']['train_pattern']
        train_files = sorted(glob.glob(os.path.join(chunks_dir, train_pattern)))
        
        logger.info(f"📁 Found {len(train_files)} training chunks")
        
        # Training results storage
        training_results = []
        best_f1 = 0
        
        # Initialize model (will be reused across chunks)
        model = MultimodalSarcasmModel(
            text_model_name=self.config['model']['name'],
            num_labels=2,
            dropout=self.config['model']['dropout'],
            use_image=use_image,
            vision_name=self.config['vision_encoder']['model_name'] if use_image else None,
            vision_trainable=bool(self.config['vision_encoder']['trainable']) if use_image else False,
            use_audio=use_audio,
            audio_name=self.config['audio_encoder']['model_name'] if use_audio else None,
            audio_trainable=bool(self.config['audio_encoder']['trainable']) if use_audio else False,
            fusion_type=self.config['modalities']['fusion_type']
        )
        model.to(self.device)

        
        # Process each chunk
        for chunk_idx, chunk_file in enumerate(train_files):
            logger.info(f"\n🔄 Processing chunk {chunk_idx + 1}/{len(train_files)}")
            logger.info(f"📄 File: {os.path.basename(chunk_file)}")
            
            # Load and prepare chunk data
            df = pd.read_csv(chunk_file)
            bad = df['label'].isna() | (df['label'] < 0) | (df['label'] >= model.num_labels)
            if bad.any():
                n_bad = int(bad.sum())
                logger.error(f"Found {n_bad} invalid labels in {os.path.basename(chunk_file)}. Dropping them.")
                df = df.loc[~bad].reset_index(drop=True)
            logger.info(f"📊 Loaded {len(df)} samples")
            
            # Extract enhanced features
            df_enhanced, feature_columns = prepare_features_for_training(df)
            
            # Create dataset
            dataset = UltimateSarcasmDataset(
                df_enhanced, feature_columns, tokenizer, self.config['model']['max_length'],
                use_image=use_image, vision_processor=vision_processor,
                use_audio=use_audio, audio_processor=audio_processor, audio_sr=sample_rate if use_audio else 16000,
                modality_dropout_prob=float(self.config['modalities']['modality_dropout_prob'])
            )

            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['hardware']['dataloader_num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
            # Setup optimizer and scheduler for this chunk
            lr = float(self.config['training']['learning_rate'])
            wd = float(self.config['training']['weight_decay'])

            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=wd,
                eps=1e-8,           # stability on Apple Silicon
                betas=(0.9, 0.999)
            )
            
            total_steps = len(dataloader) * self.config['training']['epochs_per_chunk']
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['training']['warmup_steps'],
                num_training_steps=total_steps
            )
            
            # Train on this chunk
            chunk_results = self.train_on_chunk(
                model, dataloader, optimizer, scheduler, chunk_idx + 1
            )
            
            training_results.append({
                'chunk_idx': chunk_idx + 1,
                'chunk_file': chunk_file,
                'samples': len(df),
                'results': chunk_results
            })
            
            # Save best model
            final_f1 = chunk_results[-1]['f1']
            if final_f1 > best_f1:
                best_f1 = final_f1
                self.save_model(model, tokenizer, chunk_idx + 1, final_f1)
                logger.info(f"💾 New best model saved! F1: {final_f1:.4f}")
        
        # Save training history
        self.save_training_history(training_results)
        
        logger.info("\n🎉 Ultimate training completed!")
        logger.info(f"🏆 Best F1 Score: {best_f1:.4f}")
        
        return training_results
    
    def train_on_chunk(self, model, dataloader, optimizer, scheduler, chunk_num):
        """Train model on a single chunk"""
        
        model.train()
        epoch_results = []
        
        for epoch in range(self.config['training']['epochs_per_chunk']):
            total_loss = 0
            all_predictions = []
            all_labels = []
            
            pbar = tqdm(
                dataloader, 
                desc=f"Chunk {chunk_num}, Epoch {epoch + 1}"
            )
            
            for batch in pbar:
                optimizer.zero_grad()
                
                # Move to device
                batch = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                # ── Label validation (ADD THIS BLOCK) ─────────────────────────
                labels = batch['labels']
                # Ensure correct dtype for CE (long / int64)
                if labels.dtype not in (torch.int64, torch.long):
                    batch['labels'] = labels.long()
                    labels = batch['labels']

                # Ensure labels are finite and inside [0, num_labels)
                num_labels = int(model.num_labels)  # e.g., 2
                lbl_min = labels.min().item()
                lbl_max = labels.max().item()
                if (not torch.isfinite(labels.float()).all()) or (lbl_min < 0) or (lbl_max >= num_labels):
                    logger.error(
                        f"Invalid labels: min={lbl_min}, max={lbl_max}, expected in [0,{num_labels-1}]. Skipping batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Forward pass
                # Assemble optional tensors (use 0 tensors as placeholders)
                img = batch['image_pixels']
                img = img.to(self.device) if isinstance(img, torch.Tensor) and img.ndim > 0 else None
                aud = batch['audio_values']
                aud = aud.to(self.device) if isinstance(aud, torch.Tensor) and aud.ndim > 0 else None
                aud_attn = batch['audio_attn']
                aud_attn = aud_attn.to(self.device) if isinstance(aud_attn, torch.Tensor) else None

                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    image_pixels=img,
                    audio_values=aud,
                    audio_attn=aud_attn,
                    modality_mask=batch['modality_mask'],
                    labels=batch['labels']
                )

                
                loss = outputs.loss
                if not torch.isfinite(loss):
                    logger.warning("⚠️ Non-finite loss encountered, skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Collect metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Calculate epoch metrics
            epoch_loss = total_loss / len(dataloader)
            epoch_accuracy = accuracy_score(all_labels, all_predictions)
            epoch_f1 = f1_score(all_labels, all_predictions)
            
            epoch_result = {
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_accuracy,
                'f1': epoch_f1
            }
            
            epoch_results.append(epoch_result)
            
            logger.info(
                f"Chunk {chunk_num}, Epoch {epoch + 1}: "
                f"Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, F1: {epoch_f1:.4f}"
            )
        
        return epoch_results
    
    def save_model(self, model, tokenizer, chunk_idx, f1_score):
        """Save the model"""
        
        model_dir = self.config['paths']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        # Save with chunk and score info
        model_name = f"best_model_chunk_{chunk_idx}_f1_{f1_score:.4f}"
        model_path = os.path.join(model_dir, model_name)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Also save as latest best
        latest_path = os.path.join(model_dir, "latest_best")
        model.save_pretrained(latest_path)
        tokenizer.save_pretrained(latest_path)
    
    def save_training_history(self, results):
        """Save training history"""
        
        output_dir = self.config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        history_file = os.path.join(output_dir, 'ultimate_training_history.json')
        
        with open(history_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Training history saved: {history_file}")

def main():
    """Main training function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python milestone1_train_ultimate.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    trainer = UltimateTrainer(config_path)
    results = trainer.train_ultimate_model()
    
    print("✅ Ultimate training completed successfully!")
    print(f"📊 Processed {len(results)} chunks")
    
if __name__ == "__main__":
    main()
