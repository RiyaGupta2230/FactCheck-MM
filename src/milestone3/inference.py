import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

class FactVerificationInference:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        
    def verify_claim(self, claim, evidence=None, return_probabilities=False):
        """Verify a factual claim"""
        
        # Combine claim with evidence if provided
        if evidence:
            text = f"Claim: {claim} Evidence: {evidence}"
        else:
            text = claim
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if hasattr(self.model, 'predict_with_confidence'):
                # Use robust prediction if available
                result = self.model.predict_with_confidence(text)
                return result
            else:
                # Standard prediction
                logits = self.model(**inputs)
                probabilities = F.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
        
        labels = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        
        result = {
            'verdict': labels[prediction.item()],
            'confidence': confidence.item(),
            'claim': claim,
            'evidence': evidence or "No evidence provided"
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'SUPPORTS': probabilities[0][0].item(),
                'REFUTES': probabilities[0][1].item(), 
                'NOT_ENOUGH_INFO': probabilities[0][2].item()
            }
            
        return result
    
    def batch_verify(self, claims, batch_size=16):
        """Batch fact verification"""
        results = []
        
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i:i+batch_size]
            batch_results = []
            
            for claim in batch_claims:
                result = self.verify_claim(claim)
                batch_results.append(result)
            
            results.extend(batch_results)
            
        return results
    
    def explain_verdict(self, claim, verdict_info):
        """Generate explanation for the verdict"""
        explanations = {
            'SUPPORTS': f"The claim '{claim}' is supported by available evidence.",
            'REFUTES': f"The claim '{claim}' is contradicted by available evidence.",
            'NOT_ENOUGH_INFO': f"There is insufficient evidence to verify the claim '{claim}'."
        }
        
        base_explanation = explanations.get(verdict_info['verdict'], "Unable to determine.")
        confidence_text = f" (Confidence: {verdict_info['confidence']:.1%})"
        
        return base_explanation + confidence_text
