import torch
import torch.nn.functional as F
from ..utils.preprocessing import TextPreprocessor, MultimodalPreprocessor

class SarcasmInference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.preprocessor = TextPreprocessor()
        
    def predict_text(self, text, return_probabilities=False):
        """Predict sarcasm for text input"""
        # Preprocess text
        encoding = self.preprocessor.tokenize(text, return_tensors='pt')
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'text_encoder'):  # Multimodal model in text-only mode
                outputs = self.model(input_ids, attention_mask)
            else:  # Text-only model
                outputs = self.model(input_ids, attention_mask)
            
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        result = {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'is_sarcastic': bool(prediction.item())
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'non_sarcastic': probabilities[0][0].item(),
                'sarcastic': probabilities[0][1].item()
            }
            
        return result
    
    def predict_multimodal(self, text, visual_features=None, audio_features=None):
        """Predict sarcasm using multimodal inputs"""
        if not hasattr(self.model, 'text_encoder'):
            raise ValueError("Model does not support multimodal inputs")
        
        # Preprocess inputs
        text_encoding = self.preprocessor.tokenize(text, return_tensors='pt')
        
        with torch.no_grad():
            text_ids = text_encoding['input_ids'].to(self.device)
            text_mask = text_encoding['attention_mask'].to(self.device)
            
            # Process optional features
            visual_feat = None
            if visual_features is not None:
                visual_feat = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
            audio_feat = None
            if audio_features is not None:
                audio_feat = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.model(text_ids, text_mask, visual_feat, audio_feat)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'is_sarcastic': bool(prediction.item()),
            'probabilities': {
                'non_sarcastic': probabilities[0][0].item(),
                'sarcastic': probabilities[0][1].item()
            }
        }
    
    def batch_predict(self, texts, batch_size=32):
        """Predict sarcasm for batch of texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict_text(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
        return results
