class UltimateMultimodalDataset(Dataset):
    """Enhanced dataset loader for ultimate performance"""
    
    def __init__(self, chunk_path, config, split='train'):
        self.config = config
        self.split = split
        
        # Load enhanced chunk data
        self.data = pd.read_csv(chunk_path)
        
        # Validate enhanced columns
        required_cols = ['text', 'label', 'domain', 'language', 'has_audio', 'has_video', 'has_image']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Enhanced data missing columns: {missing_cols}")
        
        # Initialize processors
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large')
        
        # Domain and language mappings
        self.domain_to_id = {'tv_shows': 0, 'social_media': 1, 'ted_talks': 2, 'memes': 3, 'news': 4, 'general': 5}
        self.lang_to_id = {'en': 0, 'zh': 1, 'es': 2, 'other': 3}
        
        print(f"✅ Enhanced dataset loaded: {len(self.data)} samples")
        print(f"📊 Domains: {self.data['domain'].value_counts().to_dict()}")
        print(f"🌍 Languages: {self.data['language'].value_counts().to_dict()}")
    
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
        
        # Process multimodal inputs based on availability
        image_inputs = None
        if row['has_image'] and pd.notna(row['image_path']):
            try:
                image = Image.open(row['image_path']).convert('RGB')
                image_inputs = self.clip_processor(images=image, return_tensors='pt')
            except Exception as e:
                pass  # Handle missing images gracefully
        
        audio_inputs = None
        if row['has_audio'] and pd.notna(row['audio_path']):
            try:
                audio, sr = librosa.load(row['audio_path'], sr=16000, duration=10)
                audio_inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors='pt')
            except Exception as e:
                pass  # Handle missing audio gracefully
        
        return {
            'text_inputs': {k: v.squeeze() for k, v in text_inputs.items()},
            'image_inputs': {k: v.squeeze() for k, v in image_inputs.items()} if image_inputs else None,
            'audio_inputs': {k: v.squeeze() for k, v in audio_inputs.items()} if audio_inputs else None,
            'labels': torch.tensor(int(row['label']), dtype=torch.long),
            'domain_ids': torch.tensor(self.domain_to_id.get(row['domain'], 5), dtype=torch.long),
            'language_ids': torch.tensor(self.lang_to_id.get(row['language'], 3), dtype=torch.long),
            'is_synthetic': torch.tensor(row.get('is_synthetic', False), dtype=torch.bool),
            'context': str(row.get('context', '')),
            'speaker': str(row.get('speaker', '')),
            'dataset_source': row['dataset']
        }
