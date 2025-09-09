import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, T5Tokenizer

class ParaphraseInference:
    def __init__(self, model, device='cpu', model_type='detection'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.model_type = model_type
        
        if model_type == 'detection':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:  # generation
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    def detect_paraphrase(self, sentence1, sentence2, return_probabilities=False):
        """Detect if two sentences are paraphrases"""
        if self.model_type != 'detection':
            raise ValueError("Model is not configured for paraphrase detection")
        
        # Tokenize sentence pair
        encoding = self.tokenizer(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        result = {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'is_paraphrase': bool(prediction.item()),
            'similarity_score': probabilities[0][1].item()  # Paraphrase probability
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'not_paraphrase': probabilities[0][0].item(),
                'is_paraphrase': probabilities[0][1].item()
            }
            
        return result
    
    def generate_paraphrase(self, text):
        """Generate paraphrase of input text"""
        if self.model_type != 'generation':
            raise ValueError("Model is not configured for paraphrase generation")
        
        return self.model.generate_paraphrase(text)
    
    def sarcasm_to_literal(self, sarcastic_text):
        """Convert sarcastic text to literal meaning"""
        if self.model_type != 'generation':
            raise ValueError("Model is not configured for paraphrase generation")
        
        return self.model.sarcasm_to_literal(sarcastic_text)
    
    def batch_detect(self, sentence_pairs, batch_size=32):
        """Batch paraphrase detection"""
        results = []
        
        for i in range(0, len(sentence_pairs), batch_size):
            batch_pairs = sentence_pairs[i:i+batch_size]
            batch_results = []
            
            for pair in batch_pairs:
                result = self.detect_paraphrase(pair[0], pair[1])
                batch_results.append(result)
            
            results.extend(batch_results)
            
        return results
