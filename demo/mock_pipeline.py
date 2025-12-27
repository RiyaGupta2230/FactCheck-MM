"""
Mock Pipeline Functions for FactCheck-MM Demo

These functions simulate ML model outputs using heuristics.
Replace with real model calls for production use.

Author: FactCheck-MM Team
"""

import random
from typing import Dict, Optional, Any, List


def detect_sarcasm(
    text: Optional[str] = None,
    image: Optional[Any] = None,
    audio: Optional[Any] = None,
    video: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Mock sarcasm detection across modalities.
    
    Real implementation path:
        from sarcasm_detection.predict import detect_sarcasm
    
    Args:
        text: Input text claim
        image: Image file object
        audio: Audio file object
        video: Video file object
        
    Returns:
        dict: {'label': 'sarcastic'|'not_sarcastic', 'score': float}
    """
    # Simple keyword-based heuristic
    sarcastic_keywords = [
        'oh great', 'yeah right', 'sure thing', 'totally',
        'brilliant', 'fantastic', 'wonderful', 'amazing',
        'thanks', 'love it'
    ]
    
    is_sarcastic = False
    confidence = 0.65
    
    if text:
        text_lower = text.lower()
        
        # Check for sarcasm markers
        for keyword in sarcastic_keywords:
            if keyword in text_lower:
                is_sarcastic = True
                confidence = min(0.95, confidence + 0.15)
        
        # Exclamation marks and ellipsis often indicate sarcasm
        if '!' in text or '...' in text:
            confidence = min(0.98, confidence + 0.10)
            
        # Contradictory statements
        if ('not' in text_lower or "n't" in text_lower) and any(kw in text_lower for kw in sarcastic_keywords):
            is_sarcastic = True
            confidence = 0.88
    
    # Multimodal signals (mocked)
    if image:
        confidence = min(0.99, confidence + 0.08)  # Visual cues detected
    if audio:
        confidence = min(0.99, confidence + 0.12)  # Prosody analyzed
    if video:
        confidence = min(0.99, confidence + 0.15)  # Temporal context
    
    return {
        'label': 'sarcastic' if is_sarcastic else 'not_sarcastic',
        'score': confidence if is_sarcastic else 1 - confidence,
        'modalities_used': {
            'text': text is not None,
            'image': image is not None,
            'audio': audio is not None,
            'video': video is not None
        }
    }


def paraphrase_text(text: str) -> str:
    # Lowercase for matching
    t = text.lower()

    # Common sarcasm indicators
    sarcasm_phrases = [
        "oh great", "yeah right", "amazing", "fantastic", 
        "just perfect", "wonderful", "fixed my social life",
        "thanks electricity board"
    ]

    # Remove exaggeration phrases
    for phrase in sarcasm_phrases:
        if phrase in t:
            t = t.replace(phrase, "")

    # Specific rewrite rules for known examples
    if "power cut" in t or "powercut" in t:
        return "There was a long power cut which caused inconvenience."

    if "delhi" in t and "fresh" in t:
        return "The air quality in Delhi was poor."

    if "scientists have discovered" in t:
        return "People often believe misinformation online when it appears scientific."

    # Fallback: simple cleaned sentence
    return t.strip().capitalize()



def verify_claim(claim_text: str) -> Dict[str, Any]:
    """
    Mock fact verification against knowledge bases.
    
    Real implementation path:
        from fact_verification.verify import verify_claim
    
    Args:
        claim_text: Literal claim to verify
        
    Returns:
        dict: {
            'verdict': 'SUPPORTS'|'REFUTES'|'NOT_ENOUGH_INFO',
            'confidence': float,
            'evidence': List[Dict],
            'explanation': str
        }
    """
    # Simple keyword-based verdict assignment
    claim_lower = claim_text.lower()
    
    # Known false claims (for demo)
    false_keywords = [
        'free electricity', 'ban all vehicles', 'banned all',
        'shut down', 'closed permanently', 'never happened'
    ]
    
    # Known true facts (for demo)
    true_keywords = [
        'metro', 'vaccination', 'government announced',
        'minister', 'prime minister'
    ]
    
    # Uncertain claims
    uncertain_keywords = [
        'might', 'possibly', 'reportedly', 'allegedly',
        'rumored', 'sources say'
    ]
    
    # Determine verdict
    if any(kw in claim_lower for kw in false_keywords):
        verdict = 'REFUTES'
        confidence = random.uniform(0.78, 0.94)
        explanation = "The claim contradicts verified information from reliable sources."
    elif any(kw in claim_lower for kw in uncertain_keywords):
        verdict = 'NOT_ENOUGH_INFO'
        confidence = random.uniform(0.55, 0.72)
        explanation = "Insufficient evidence found to confirm or refute the claim."
    elif any(kw in claim_lower for kw in true_keywords):
        verdict = 'SUPPORTS'
        confidence = random.uniform(0.81, 0.96)
        explanation = "The claim is corroborated by multiple reliable sources."
    else:
        # Default to uncertain
        verdict = 'NOT_ENOUGH_INFO'
        confidence = random.uniform(0.50, 0.68)
        explanation = "Limited information available to make a definitive judgment."
    
    # Generate mock evidence
    evidence = generate_mock_evidence(claim_text, verdict)
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'evidence': evidence,
        'explanation': explanation,
        'knowledge_bases_queried': ['Wikipedia', 'Wikidata', 'DBpedia', 'News APIs']
    }


def generate_mock_evidence(claim: str, verdict: str) -> List[Dict[str, Any]]:
    """Generate realistic-looking mock evidence snippets."""
    
    evidence_templates = {
        'SUPPORTS': [
            {
                'source': 'BBC News',
                'snippet': f'According to official reports, {claim[:50]}... has been confirmed by government sources.',
                'url': 'https://bbc.com/news/example',
                'relevance': random.uniform(0.85, 0.97)
            },
            {
                'source': 'Reuters',
                'snippet': f'Independent verification shows that the information regarding {claim[:40]}... is accurate.',
                'url': 'https://reuters.com/article/example',
                'relevance': random.uniform(0.78, 0.92)
            }
        ],
        'REFUTES': [
            {
                'source': 'FactCheck.org',
                'snippet': f'Our investigation found no evidence supporting the claim that {claim[:50]}... This appears to be misinformation.',
                'url': 'https://factcheck.org/example',
                'relevance': random.uniform(0.88, 0.96)
            },
            {
                'source': 'Snopes',
                'snippet': f'The statement about {claim[:40]}... has been debunked by multiple sources.',
                'url': 'https://snopes.com/fact-check/example',
                'relevance': random.uniform(0.82, 0.94)
            }
        ],
        'NOT_ENOUGH_INFO': [
            {
                'source': 'Wikipedia',
                'snippet': f'Related information about {claim[:40]}... is sparse and requires further verification.',
                'url': 'https://wikipedia.org/wiki/Example',
                'relevance': random.uniform(0.55, 0.73)
            },
            {
                'source': 'Academic Database',
                'snippet': 'No peer-reviewed studies or official statements directly address this claim.',
                'url': 'https://scholar.google.com/example',
                'relevance': random.uniform(0.48, 0.68)
            }
        ]
    }
    
    return evidence_templates.get(verdict, evidence_templates['NOT_ENOUGH_INFO'])[:2]


def process_multimodal_input(
    text: Optional[str] = None,
    image: Optional[Any] = None,
    audio: Optional[Any] = None,
    video: Optional[Any] = None
) -> Dict[str, Any]:
    """
    End-to-end pipeline processing.
    
    Args:
        text, image, audio, video: Input modalities
        
    Returns:
        dict: Complete pipeline results
    """
    # Step 1: Sarcasm detection
    sarcasm_result = detect_sarcasm(text, image, audio, video)
    
    # Step 2: Paraphrasing (if sarcastic)
    if sarcasm_result['label'] == 'sarcastic' and text:
        paraphrased = paraphrase_text(text)
    else:
        paraphrased = text if text else "[Claim extracted from media]"
    
    # Step 3: Fact verification
    verification_result = verify_claim(paraphrased)
    
    return {
        'sarcasm': sarcasm_result,
        'paraphrase': paraphrased,
        'verification': verification_result
    }


# ========== COMMAND LINE TEST ==========
if __name__ == "__main__":
    print("=" * 60)
    print("FactCheck-MM Mock Pipeline - Command Line Test")
    print("=" * 60)
    
    # Test case 1: Sarcastic text
    test1 = "Oh great, the government announced free electricity for everyone. That'll definitely happen."
    print("\n[TEST 1] Sarcastic claim:")
    print(f"Input: {test1}")
    result1 = process_multimodal_input(text=test1)
    print(f"Sarcasm: {result1['sarcasm']['label']} ({result1['sarcasm']['score']:.2%})")
    print(f"Paraphrase: {result1['paraphrase']}")
    print(f"Verdict: {result1['verification']['verdict']} ({result1['verification']['confidence']:.2%})")
    
    # Test case 2: Straightforward claim
    test2 = "The Prime Minister inaugurated the new metro line in Chennai."
    print("\n[TEST 2] Straightforward claim:")
    print(f"Input: {test2}")
    result2 = process_multimodal_input(text=test2)
    print(f"Sarcasm: {result2['sarcasm']['label']} ({result2['sarcasm']['score']:.2%})")
    print(f"Paraphrase: {result2['paraphrase']}")
    print(f"Verdict: {result2['verification']['verdict']} ({result2['verification']['confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("✅ Mock pipeline test completed successfully!")
    print("=" * 60)
