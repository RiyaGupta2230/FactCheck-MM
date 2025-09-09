import torch
import torch.nn as nn
from transformers import (
    RobertaModel, RobertaTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertModel, BertTokenizer
)
import torch.nn.functional as F

class RoBERTaParaphraseDetector(nn.Module):
    """RoBERTa-based paraphrase detection model"""
    def __init__(self, model_name='roberta-base', num_classes=2, dropout=0.3):
        super(RoBERTaParaphraseDetector, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced classification head for paraphrase detection
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get RoBERTa outputs for sentence pair
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        
        return logits

class SiameseRoBERTa(nn.Module):
    """Siamese RoBERTa network for paraphrase detection"""
    def __init__(self, model_name='roberta-base', num_classes=2, dropout=0.3):
        super(SiameseRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Similarity computation layers
        self.similarity_head = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size * 4, 512),  # [CLS1, CLS2, |CLS1-CLS2|, CLS1*CLS2]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Encode both sentences separately
        outputs_1 = self.roberta(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.roberta(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
        cls_1 = outputs_1.pooler_output
        cls_2 = outputs_2.pooler_output
        
        # Create feature vector: [cls1, cls2, |cls1-cls2|, cls1*cls2]
        diff = torch.abs(cls_1 - cls_2)
        mult = cls_1 * cls_2
        combined = torch.cat([cls_1, cls_2, diff, mult], dim=1)
        
        logits = self.similarity_head(self.dropout(combined))
        return logits

class ParaphraseGenerator(nn.Module):
    """T5-based paraphrase generator"""
    def __init__(self, model_name='t5-small'):
        super(ParaphraseGenerator, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def generate_paraphrase(self, text, max_length=128, num_beams=4):
        """Generate paraphrase of input text"""
        # T5 instruction format
        input_text = f"paraphrase: {text}"
        
        input_ids = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
            
        paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrase
    
    def sarcasm_to_literal(self, sarcastic_text, max_length=128):
        """Convert sarcastic text to literal meaning"""
        # Special instruction for literal conversion
        input_text = f"convert sarcasm to literal meaning: {sarcastic_text}"
        
        input_ids = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.6
            )
            
        literal_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return literal_text
