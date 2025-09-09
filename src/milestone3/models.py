import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    DebertaV2Model, DebertaV2Tokenizer,
    RobertaModel, RobertaTokenizer
)
import torch.nn.functional as F
import numpy as np

class DeBERTaFactVerifier(nn.Module):
    """State-of-the-art DeBERTa-v3 based fact verification model"""
    def __init__(self, model_name='microsoft/deberta-v3-base', num_classes=3, dropout=0.3):
        super(DeBERTaFactVerifier, self).__init__()
        
        # Use the best available model: DeBERTa-v3
        self.model_name = model_name
        self.deberta = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Enhanced classification head with attention
        hidden_size = self.deberta.config.hidden_size
        self.attention_pooling = nn.MultiheadAttention(hidden_size, num_heads=12, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )
        
        # Class weights for imbalanced data
        self.class_weights = torch.tensor([1.0, 1.2, 0.8])  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Enhanced pooling using attention
        sequence_output = outputs.last_hidden_state
        
        # Apply attention pooling over sequence
        sequence_output_t = sequence_output.transpose(0, 1)  # [seq_len, batch, hidden]
        attended_output, _ = self.attention_pooling(
            sequence_output_t, sequence_output_t, sequence_output_t
        )
        
        # Use CLS token with attention enhancement
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        attended_cls = attended_output[0]  # First token (CLS) from attention
        
        # Combine CLS and attention representations
        combined_output = cls_output + attended_cls
        
        # Classification
        logits = self.classifier(combined_output)
        return logits

class FactVerifierWithEvidence(nn.Module):
    """Fact verifier that incorporates evidence retrieval"""
    def __init__(self, model_name='microsoft/deberta-v3-base', num_classes=3):
        super(FactVerifierWithEvidence, self).__init__()
        
        self.fact_verifier = DeBERTaFactVerifier(model_name, num_classes)
        self.evidence_retriever = EvidenceRetriever()
        
    def forward(self, claim, evidence_texts=None, input_ids=None, attention_mask=None):
        if input_ids is not None:
            return self.fact_verifier(input_ids, attention_mask)
        
        # Retrieve evidence if not provided
        if evidence_texts is None:
            evidence_texts = self.evidence_retriever.retrieve(claim)
        
        # Combine claim with evidence
        combined_text = f"Claim: {claim} Evidence: {' '.join(evidence_texts)}"
        
        # Tokenize combined text
        inputs = self.fact_verifier.tokenizer(
            combined_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return self.fact_verifier(inputs['input_ids'], inputs['attention_mask'])

class EvidenceRetriever(nn.Module):
    """Simple evidence retrieval system (can be enhanced with vector search)"""
    def __init__(self):
        super(EvidenceRetriever, self).__init__()
        
        # Knowledge base (in practice, this would be a large corpus)
        self.knowledge_base = [
            "Climate change is caused by greenhouse gas emissions from human activities.",
            "COVID-19 vaccines have been proven safe and effective in clinical trials.",
            "The Earth orbits around the Sun in approximately 365.25 days.",
            "Water boils at 100 degrees Celsius at sea level pressure.",
            "Artificial intelligence involves machines performing tasks that typically require human intelligence.",
        ]
        
    def retrieve(self, claim, top_k=3):
        """Simple keyword-based retrieval (enhance with embedding similarity)"""
        claim_words = set(claim.lower().split())
        
        scored_evidence = []
        for evidence in self.knowledge_base:
            evidence_words = set(evidence.lower().split())
            similarity = len(claim_words.intersection(evidence_words)) / len(claim_words.union(evidence_words))
            scored_evidence.append((evidence, similarity))
        
        # Return top-k most similar evidence
        sorted_evidence = sorted(scored_evidence, key=lambda x: x[1], reverse=True)
        return [evidence for evidence, _ in sorted_evidence[:top_k]]

class RobustFactVerifier(nn.Module):
    """Enhanced fact verifier with multiple verification strategies"""
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super(RobustFactVerifier, self).__init__()
        
        # Primary fact verifier
        self.primary_verifier = DeBERTaFactVerifier(model_name)
        
        # Secondary verifier for confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty quantification
        self.enable_mc_dropout = True
        
    def forward(self, input_ids, attention_mask, return_uncertainty=False):
        logits = self.primary_verifier(input_ids, attention_mask)
        
        if return_uncertainty and self.enable_mc_dropout:
            # Monte Carlo Dropout for uncertainty estimation
            uncertainties = []
            self.train()  # Enable dropout during inference
            
            for _ in range(10):  # MC samples
                with torch.no_grad():
                    mc_logits = self.primary_verifier(input_ids, attention_mask)
                    uncertainties.append(F.softmax(mc_logits, dim=1))
            
            # Calculate uncertainty as variance across MC samples
            uncertainties = torch.stack(uncertainties)
            mean_pred = torch.mean(uncertainties, dim=0)
            uncertainty = torch.var(uncertainties, dim=0).mean(dim=1)
            
            self.eval()  # Return to eval mode
            return logits, uncertainty
        
        return logits
    
    def predict_with_confidence(self, claim, threshold=0.7):
        """Predict with confidence thresholding"""
        inputs = self.primary_verifier.tokenizer(
            claim,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            logits, uncertainty = self.forward(
                inputs['input_ids'], 
                inputs['attention_mask'], 
                return_uncertainty=True
            )
            
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            prediction = torch.argmax(probabilities, dim=1)
            
            labels = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
            
            # If confidence is low, default to NOT_ENOUGH_INFO
            if confidence < threshold:
                prediction = torch.tensor([2])  # NOT_ENOUGH_INFO
                
            return {
                'prediction': labels[prediction.item()],
                'confidence': confidence.item(),
                'uncertainty': uncertainty.item() if uncertainty is not None else 0.0,
                'probabilities': {
                    'SUPPORTS': probabilities[0][0].item(),
                    'REFUTES': probabilities[0][1].item(), 
                    'NOT_ENOUGH_INFO': probabilities[0][2].item()
                }
            }

class RoBERTaFactVerifier(nn.Module):
    """RoBERTa-based fact verification model (stable alternative to DeBERTa)"""
    def __init__(self, model_name='roberta-base', num_classes=3, dropout=0.3):
        super(RoBERTaFactVerifier, self).__init__()
        
        self.model_name = model_name
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Enhanced classification head with attention
        hidden_size = self.roberta.config.hidden_size
        self.attention_pooling = nn.MultiheadAttention(hidden_size, num_heads=12, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Enhanced pooling using attention
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Apply attention pooling
        sequence_output_t = sequence_output.transpose(0, 1)
        attended_output, _ = self.attention_pooling(
            sequence_output_t, sequence_output_t, sequence_output_t
        )
        
        # Combine representations
        attended_cls = attended_output[0]
        combined_output = pooled_output + attended_cls
        
        # Classification
        logits = self.classifier(combined_output)
        return logits
