import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoded = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FactVerificationModel(nn.Module):
    """Your original model architecture for fact verification"""
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def evaluate_model(model_path, tokenizer, test_data):
    """Evaluate a single model and return real metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Loading checkpoint from {model_path.name}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with correct architecture
    model_name = checkpoint.get("model_config", {}).get("model_name", "bert-base-uncased")
    model = FactVerificationModel(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = SimpleDataset(
        test_data['text'].tolist(), 
        test_data['label'].tolist(), 
        tokenizer
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    print(f"  Evaluating on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(inputs, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(loader)
    
    return {
        'avg_loss': float(avg_loss),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'num_samples': len(all_labels)
    }

def main():
    print("🔍 Evaluating your trained Milestone 3 models...")
    
    # Paths - ADJUST THESE TO YOUR ACTUAL PATHS
    project_root = Path(__file__).parent
    models_dir = project_root / 'models' / 'chunk_models'
    
    # Test data path - use your LIAR test set or validation set
    test_data_paths = [
        project_root / 'data' / 'fact_verification' / 'test.tsv',
        project_root / 'data' / 'fact_verification' / 'valid.tsv',
        project_root / 'data' / 'chunks' / 'liar_chunk_001.csv'  # fallback
    ]
    
    # Find available test data
    test_data_path = None
    for path in test_data_paths:
        if path.exists():
            test_data_path = path
            break
    
    if test_data_path is None:
        print("❌ No test data found! Please check these paths:")
        for path in test_data_paths:
            print(f"   {path}")
        return
    
    print(f"📊 Using test data: {test_data_path}")
    
    # Load test data
    if test_data_path.suffix == '.tsv':
        # LIAR format: tab-separated, may need column mapping
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, quoting=3)
        if len(test_data.columns) >= 3:
            # LIAR format: [id, label, text, ...]
            test_data = pd.DataFrame({
                'text': test_data.iloc[:, 2],  # text column
                'label': test_data.iloc[:, 1].map({
                    'true': 1, 'mostly-true': 1, 'half-true': 1,
                    'barely-true': 0, 'false': 0, 'pants-fire': 0
                })
            })
    else:
        # CSV format
        test_data = pd.read_csv(test_data_path)
    
    # Clean data
    test_data = test_data.dropna(subset=['text', 'label'])
    test_data = test_data[test_data['text'].astype(str).str.len() > 0]
    print(f"📈 Loaded {len(test_data)} test samples")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Evaluate all models
    results = {}
    model_files = sorted(models_dir.glob("*.pt"))
    
    print(f"\n🎯 Found {len(model_files)} trained models to evaluate:")
    for i, model_file in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Evaluating {model_file.name}")
        
        try:
            result = evaluate_model(model_file, tokenizer, test_data)
            
            # Clean model name for results
            model_key = model_file.stem.replace('model_', '')
            results[model_key] = result
            
            print(f"  ✅ Results: Loss={result['avg_loss']:.4f}, "
                  f"Acc={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed to evaluate {model_file.name}: {e}")
            continue
    
    # Save results
    results_path = project_root / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {results_path}")
    print(f"📊 Successfully evaluated {len(results)}/{len(model_files)} models")
    
    # Show best models
    if results:
        print(f"\n🏆 Top 5 models by F1 score:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        for i, (model_name, metrics) in enumerate(sorted_models[:5], 1):
            print(f"  {i}. {model_name:<25} F1: {metrics['f1_score']:.4f} "
                  f"Acc: {metrics['accuracy']:.4f} Loss: {metrics['avg_loss']:.4f}")

if __name__ == "__main__":
    main()
