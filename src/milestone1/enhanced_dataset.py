import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import cv2
from transformers import RobertaTokenizer, CLIPProcessor, Wav2Vec2Processor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UltimateMultimodalDataset(Dataset):
    """
    Ultimate dataset loader for multimodal sarcasm detection
    Handles enhanced data format from ultimate data preparation
    """
    
    def __init__(self, chunk_path, config, split='train', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load enhanced chunk data
        self.data = pd.read_csv(chunk_path)
        
        # Validate enhanced columns
        required_cols = ['text', 'label', 'domain', 'language', 'dataset', 
                        'has_audio', 'has_video', 'has_image', 'is_synthetic']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Enhanced data missing columns: {missing_cols}")
        
        # Initialize processors for different modalities
        self.tokenizer = RobertaTokenizer.from_pretrained(config['model_architecture']['text_encoder'])
        self.clip_processor = CLIPProcessor.from_pretrained(config['model_architecture']['image_encoder'])
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(config['model_architecture']['audio_encoder'])
        
        # Create mappings for categorical data
        self.domain_to_id = {
            'tv_shows': 0, 'social_media': 1, 'ted_talks': 2, 
            'memes': 3, 'news': 4, 'general': 5
        }
        self.lang_to_id = {'en': 0, 'zh': 1, 'es': 2, 'other': 3}
        self.dataset_to_id = {
            'mustard': 0, 'sarcnet': 1, 'mmsd2': 2, 'ur_funny': 3, 
            'spanish_sarcasm': 4, 'synthetic': 5
        }
        
        # Data statistics
        print(f"✅ Enhanced dataset loaded: {len(self.data)} samples")
        print(f"📊 Label distribution: {self.data['label'].value_counts().to_dict()}")
        print(f"🌍 Language distribution: {self.data['language'].value_counts().to_dict()}")
        print(f"🏠 Domain distribution: {self.data['domain'].value_counts().to_dict()}")
        print(f"🔬 Synthetic ratio: {self.data['is_synthetic'].mean():.2%}")
        
        # Multimodal statistics
        multimodal_stats = {
            'text_only': (~self.data['has_audio'] & ~self.data['has_video'] & ~self.data['has_image']).sum(),
            'text_image': (self.data['has_image'] & ~self.data['has_audio'] & ~self.data['has_video']).sum(),
            'text_audio': (self.data['has_audio'] & ~self.data['has_image'] & ~self.data['has_video']).sum(),
            'text_video': (self.data['has_video'] & ~self.data['has_image'] & ~self.data['has_audio']).sum(),
            'multimodal': (self.data['has_audio'] | self.data['has_video'] | self.data['has_image']).sum()
        }
        print(f"📱 Multimodal distribution: {multimodal_stats}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Process text with enhanced tokenization
        text_inputs = self.tokenizer(
            str(row['text']),
            max_length=self.config['training']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image if available
        image_inputs = None
        if row['has_image'] and pd.notna(row['image_path']):
            image_inputs = self.load_and_process_image(row['image_path'])
        
        # Process audio if available
        audio_inputs = None
        if row['has_audio'] and pd.notna(row['audio_path']):
            audio_inputs = self.load_and_process_audio(row['audio_path'])
        
        # Process video if available (extract frame or use video features)
        video_inputs = None
        if row['has_video'] and pd.notna(row['video_path']):
            video_inputs = self.load_and_process_video(row['video_path'])
        
        # Create comprehensive batch item
        item = {
            # Core inputs
            'text_inputs': {k: v.squeeze() for k, v in text_inputs.items()},
            'image_inputs': image_inputs,
            'audio_inputs': audio_inputs,
            'video_inputs': video_inputs,
            
            # Labels and IDs
            'labels': torch.tensor(int(row['label']), dtype=torch.long),
            'domain_ids': torch.tensor(self.domain_to_id.get(row['domain'], 5), dtype=torch.long),
            'language_ids': torch.tensor(self.lang_to_id.get(row['language'], 3), dtype=torch.long),
            'dataset_ids': torch.tensor(self.get_dataset_id(row['dataset']), dtype=torch.long),
            
            # Metadata
            'is_synthetic': torch.tensor(bool(row['is_synthetic']), dtype=torch.bool),
            'weight': torch.tensor(float(row.get('weight', 1.0)), dtype=torch.float),
            'has_multimodal': torch.tensor(
                bool(row['has_audio'] or row['has_video'] or row['has_image']), 
                dtype=torch.bool
            ),
            
            # Context information (for analysis)
            'context': str(row.get('context', '')),
            'speaker': str(row.get('speaker', '')),
            'dataset_source': row['dataset'],
            'sample_id': idx
        }
        
        # Generate auxiliary labels for multi-task learning if enabled
        if self.config.get('multi_task', {}).get('enabled', False):
            item.update(self.generate_auxiliary_labels(row))
        
        return item
    
    def load_and_process_image(self, image_path):
        """Load and process image with error handling"""
        try:
            if not Path(image_path).exists():
                return None
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)
            
            # Process with CLIP
            inputs = self.clip_processor(images=image, return_tensors='pt')
            return {k: v.squeeze() for k, v in inputs.items()}
            
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            return None
    
    def load_and_process_audio(self, audio_path):
        """Load and process audio with error handling"""
        try:
            if not Path(audio_path).exists():
                return None
            
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=16000, duration=30)  # Limit to 30 seconds
            
            # Ensure minimum length
            if len(audio) < 1600:  # 0.1 seconds at 16kHz
                return None
            
            # Process with Wav2Vec2
            inputs = self.wav2vec_processor(
                audio, sampling_rate=16000, return_tensors='pt'
            )
            return {k: v.squeeze() for k, v in inputs.items()}
            
        except Exception as e:
            print(f"⚠️ Error loading audio {audio_path}: {e}")
            return None
    
    def load_and_process_video(self, video_path):
        """Load and process video by extracting representative frame"""
        try:
            if not Path(video_path).exists():
                return None
            
            # Extract middle frame from video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            
            # Extract middle frame
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            
            # Process as image
            inputs = self.clip_processor(images=image, return_tensors='pt')
            return {k: v.squeeze() for k, v in inputs.items()}
            
        except Exception as e:
            print(f"⚠️ Error loading video {video_path}: {e}")
            return None
    
    def get_dataset_id(self, dataset_name):
        """Map dataset name to ID, handling augmented/synthetic variants"""
        # Extract base dataset name
        base_name = dataset_name.split('_')[0] if '_' in dataset_name else dataset_name
        
        if 'synthetic' in dataset_name:
            return self.dataset_to_id.get('synthetic', 5)
        else:
            return self.dataset_to_id.get(base_name, 0)
    
    def generate_auxiliary_labels(self, row):
        """Generate auxiliary labels for multi-task learning"""
        auxiliary_labels = {}
        
        # Sentiment labels (simplified heuristic)
        if self.config['multi_task'].get('sentiment_analysis', False):
            # Use sarcasm as proxy: sarcastic -> negative, non-sarcastic -> neutral
            sentiment_label = 1 if row['label'] == 1 else 2  # 0=pos, 1=neg, 2=neu
            auxiliary_labels['sentiment_labels'] = torch.tensor(sentiment_label, dtype=torch.long)
        
        # Emotion labels (simplified heuristic)
        if self.config['multi_task'].get('emotion_recognition', False):
            # Random emotion for demonstration (in practice, use proper emotion labels)
            emotion_label = np.random.randint(0, 8)  # 8 basic emotions
            auxiliary_labels['emotion_labels'] = torch.tensor(emotion_label, dtype=torch.long)
        
        # Irony labels (use sarcasm as proxy)
        if self.config['multi_task'].get('irony_detection', False):
            auxiliary_labels['irony_labels'] = torch.tensor(int(row['label']), dtype=torch.long)
        
        return auxiliary_labels


def create_ultimate_dataloader(chunk_path, config, split='train', batch_size=None, shuffle=None):
    """Create optimized dataloader for ultimate multimodal training"""
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataset
    dataset = UltimateMultimodalDataset(chunk_path, config, split=split)
    
    # Create dataloader with optimal settings for RTX 2050
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['hardware'].get('dataloader_num_workers', 4),
        pin_memory=config['hardware'].get('pin_memory', True),
        persistent_workers=config['hardware'].get('persistent_workers', True),
        drop_last=(split == 'train'),  # Drop last batch for training
        collate_fn=ultimate_collate_fn
    )
    
    return dataloader


def ultimate_collate_fn(batch):
    """
    Ultimate collate function to handle multimodal batch creation
    Properly handles None values and variable-length sequences
    """
    # Initialize batch dictionary
    collated_batch = {}
    
    # Text inputs (always present)
    text_inputs = {}
    for key in batch[0]['text_inputs'].keys():
        text_inputs[key] = torch.stack([item['text_inputs'][key] for item in batch])
    collated_batch['text_inputs'] = text_inputs
    
    # Image inputs (handle None values)
    image_items = [item['image_inputs'] for item in batch if item['image_inputs'] is not None]
    if image_items:
        image_inputs = {}
        for key in image_items[0].keys():
            image_inputs[key] = torch.stack([item[key] for item in image_items])
        collated_batch['image_inputs'] = image_inputs
    else:
        collated_batch['image_inputs'] = None
    
    # Audio inputs (handle None values)
    audio_items = [item['audio_inputs'] for item in batch if item['audio_inputs'] is not None]
    if audio_items:
        audio_inputs = {}
        for key in audio_items[0].keys():
            # Handle variable length audio sequences
            audio_tensors = [item[key] for item in audio_items]
            if key == 'input_values':
                # Pad audio sequences to same length
                max_length = max(tensor.shape[-1] for tensor in audio_tensors)
                padded_tensors = []
                for tensor in audio_tensors:
                    if len(tensor.shape) == 1:
                        tensor = tensor.unsqueeze(0)
                    padding_length = max_length - tensor.shape[-1]
                    if padding_length > 0:
                        padding = torch.zeros(tensor.shape[0], padding_length)
                        tensor = torch.cat([tensor, padding], dim=-1)
                    padded_tensors.append(tensor)
                audio_inputs[key] = torch.stack(padded_tensors)
            else:
                audio_inputs[key] = torch.stack(audio_tensors)
        collated_batch['audio_inputs'] = audio_inputs
    else:
        collated_batch['audio_inputs'] = None
    
    # Video inputs (handle None values) - treated same as images
    video_items = [item.get('video_inputs') for item in batch if item.get('video_inputs') is not None]
    if video_items:
        video_inputs = {}
        for key in video_items[0].keys():
            video_inputs[key] = torch.stack([item[key] for item in video_items])
        collated_batch['video_inputs'] = video_inputs
    else:
        collated_batch['video_inputs'] = None
    
    # Standard tensor fields
    tensor_fields = ['labels', 'domain_ids', 'language_ids', 'dataset_ids', 
                    'is_synthetic', 'weight', 'has_multimodal']
    
    for field in tensor_fields:
        if field in batch[0]:
            collated_batch[field] = torch.stack([item[field] for item in batch])
    
    # Auxiliary labels for multi-task learning
    auxiliary_fields = ['sentiment_labels', 'emotion_labels', 'irony_labels']
    for field in auxiliary_fields:
        if field in batch[0]:
            collated_batch[field] = torch.stack([item[field] for item in batch])
    
    # String fields (keep as lists)
    string_fields = ['context', 'speaker', 'dataset_source']
    for field in string_fields:
        if field in batch[0]:
            collated_batch[field] = [item[field] for item in batch]
    
    # Sample IDs
    if 'sample_id' in batch[0]:
        collated_batch['sample_ids'] = torch.tensor([item['sample_id'] for item in batch])
    
    return collated_batch


# Utility functions for dataset analysis
def analyze_dataset_distribution(chunk_path):
    """Analyze and print dataset distribution"""
    data = pd.read_csv(chunk_path)
    
    print(f"📊 Dataset Analysis for {Path(chunk_path).name}")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total samples: {len(data)}")
    print(f"Label distribution: {data['label'].value_counts().to_dict()}")
    print(f"Language distribution: {data['language'].value_counts().to_dict()}")
    print(f"Domain distribution: {data['domain'].value_counts().to_dict()}")
    
    # Multimodal statistics
    print(f"\nMultimodal Statistics:")
    print(f"Has audio: {data['has_audio'].sum()} ({data['has_audio'].mean():.1%})")
    print(f"Has video: {data['has_video'].sum()} ({data['has_video'].mean():.1%})")
    print(f"Has image: {data['has_image'].sum()} ({data['has_image'].mean():.1%})")
    print(f"Text only: {(~data['has_audio'] & ~data['has_video'] & ~data['has_image']).sum()}")
    
    # Quality statistics
    print(f"\nQuality Statistics:")
    print(f"Synthetic samples: {data['is_synthetic'].sum()} ({data['is_synthetic'].mean():.1%})")
    print(f"Average text length: {data['text_length'].mean():.1f} characters")
    print(f"Average word count: {data['word_count'].mean():.1f} words")
    
    return data

if __name__ == "__main__":
    # Example usage
    config_path = 'config/milestone1_ultimate_config.yaml'
    chunk_path = 'data/chunks/milestone1_ultimate/ultimate_train_chunk_001.csv'
    
    # Analyze dataset
    analyze_dataset_distribution(chunk_path)
    
    # Test dataloader
    print("\n🔧 Testing ultimate dataloader...")
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    dataloader = create_ultimate_dataloader(chunk_path, config, split='train', batch_size=2)
    
    for batch in dataloader:
        print("✅ Batch loaded successfully!")
        print(f"   Text inputs shape: {batch['text_inputs']['input_ids'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Has image inputs: {batch['image_inputs'] is not None}")
        print(f"   Has audio inputs: {batch['audio_inputs'] is not None}")
        break
