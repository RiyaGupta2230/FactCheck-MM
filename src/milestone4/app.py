from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import os
import sys


# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.milestone1.models import TextOnlySarcasmDetector
from src.milestone1.inference import SarcasmInference
from src.utils.config import config
from src.utils.helpers import get_device
from src.milestone2.models import RoBERTaParaphraseDetector, ParaphraseGenerator
from src.milestone2.inference import ParaphraseInference
from src.milestone3.models import DeBERTaFactVerifier
from src.milestone3.inference import FactVerificationInference

app = Flask(__name__, template_folder='../../ui/templates', static_folder='../../ui/static')
CORS(app)

# Global variables for models
sarcasm_model = None
sarcasm_inference = None
device = None
paraphrase_model = None
paraphrase_inference = None
paraphrase_generator = None
fact_model = None
fact_inference = None

def load_models():
    """Load all trained models"""
    global sarcasm_model, sarcasm_inference, paraphrase_model, paraphrase_inference, paraphrase_generator, fact_model, fact_inference, device
    
    device = get_device()
    print(f"Loading models on device: {device}")
    
    # Load CLIP sarcasm detection model (NEW)
    try:
        clip_model_path = 'checkpoints/clip_sarcasm/best_model.pt'
        if os.path.exists(clip_model_path):
            from src.milestone1.clip_models import CLIPSarcasmDetector
            sarcasm_model = CLIPSarcasmDetector()
            checkpoint = torch.load(clip_model_path, map_location=device, weights_only=False)
            sarcasm_model.load_state_dict(checkpoint['model_state_dict'])
            sarcasm_inference = SarcasmInference(sarcasm_model, device)
            print("CLIP Sarcasm model loaded successfully")
            return
    except Exception as e:
        print(f"CLIP model loading failed: {e}")

    # Load paraphrase detection model (NEW)
    try:
        paraphrase_model_path = 'checkpoints/paraphrase_detection/best_model.pt'
        if os.path.exists(paraphrase_model_path):
            paraphrase_model = RoBERTaParaphraseDetector()
            checkpoint = torch.load(paraphrase_model_path, map_location=device, weights_only=False)
            paraphrase_model.load_state_dict(checkpoint['model_state_dict'])
            paraphrase_inference = ParaphraseInference(paraphrase_model, device, model_type='detection')
            print("Paraphrase detection model loaded successfully")
        else:
            print(f"Paraphrase model not found at {paraphrase_model_path}")
    except Exception as e:
        print(f"Error loading paraphrase model: {e}")
    
    # Load fact verification model (NEW)
    try:
        fact_model_path = 'checkpoints/fact_verification/best_model.pt'
        if os.path.exists(fact_model_path):
            fact_model = DeBERTaFactVerifier()
            checkpoint = torch.load(fact_model_path, map_location=device, weights_only=False)
            fact_model.load_state_dict(checkpoint['model_state_dict'])
            fact_inference = FactVerificationInference(fact_model, device)
            print("Fact verification model loaded successfully")
        else:
            print(f"Fact verification model not found at {fact_model_path}")
    except Exception as e:
        print(f"Error loading fact verification model: {e}")

    # Fallback to original model (EXISTING CODE)
    try:
        model_path = os.path.join(config.get('checkpoints.multimodal_sarcasm'), 'best_model.pt')
        if os.path.exists(model_path):
            sarcasm_model = TextOnlySarcasmDetector()
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            sarcasm_model.load_state_dict(checkpoint['model_state_dict'])
            sarcasm_inference = SarcasmInference(sarcasm_model, device)
            print("Original Sarcasm model loaded successfully")
        else:
            print(f"No models found")
    except Exception as e:
        print(f"Error loading models: {e}")


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'sarcasm': sarcasm_model is not None,
            'paraphrase': paraphrase_model is not None,
            'fact_check': fact_model is not None  # Updated!
        }
    })

@app.route('/api/sarcasm-detection', methods=['POST'])
def detect_sarcasm():
    """Sarcasm detection endpoint"""
    try:
        if not sarcasm_inference:
            return jsonify({'error': 'Sarcasm model not loaded'}), 500
        
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction
        result = sarcasm_inference.predict_text(text, return_probabilities=True)
        
        # Format response
        response = {
            'is_sarcastic': result['is_sarcastic'],
            'confidence': round(result['confidence'], 3),
            'probabilities': {
                'sarcastic': round(result['probabilities']['sarcastic'], 3),
                'non_sarcastic': round(result['probabilities']['non_sarcastic'], 3)
            },
            'explanation': 'Detected based on language patterns and context' if result['is_sarcastic'] else 'No sarcastic indicators found'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/paraphrase-check', methods=['POST'])
def check_paraphrase():
    """Paraphrase detection endpoint"""
    try:
        data = request.json
        sentence1 = data.get('sentence1', '').strip()
        sentence2 = data.get('sentence2', '').strip()
        
        if not sentence1 or not sentence2:
            return jsonify({'error': 'Both sentences are required'}), 400
        
        if not paraphrase_inference:
            # Placeholder response when model is not trained
            import random
            is_paraphrase = random.choice([True, False])
            similarity = random.uniform(0.3, 0.9)

            response = {
                        'is_paraphrase': is_paraphrase,
                        'similarity_score': round(similarity, 3),
                        'confidence': round(similarity, 3),
                        'explanation': '⚠️ Using placeholder predictions. Train Milestone 2 to get real results.',
                        'status': 'Model not trained yet'
                    }
            return jsonify(response)
        
        # Get prediction from trained model
        result = paraphrase_inference.detect_paraphrase(sentence1, sentence2, return_probabilities=True)
        
        # Format response
        response = {
            'is_paraphrase': result['is_paraphrase'],
            'similarity_score': round(result['similarity_score'], 3),
            'confidence': round(result['confidence'], 3),
            'probabilities': {
                'is_paraphrase': round(result['probabilities']['is_paraphrase'], 3),
                'not_paraphrase': round(result['probabilities']['not_paraphrase'], 3)
            },
            'explanation': 'Sentences are paraphrases' if result['is_paraphrase'] else 'Sentences are not paraphrases',
            'status': 'Prediction from trained model'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Paraphrase detection failed: {str(e)}'}), 500
    

@app.route('/api/paraphrase-generate', methods=['POST'])
def generate_paraphrase():
    """Generate paraphrase of input text"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not paraphrase_generator:
            return jsonify({
                'paraphrase': text,
                'status': 'Paraphrase generator not available - returning original text'
            })
        
        paraphrase = paraphrase_generator.generate_paraphrase(text)
        
        return jsonify({
            'original_text': text,
            'paraphrase': paraphrase,
            'status': 'Generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Paraphrase generation failed: {str(e)}'}), 500


# Update the fact-check endpoint
@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    """Fact checking endpoint"""
    try:
        data = request.json
        claim = data.get('claim', '').strip()
        evidence = data.get('evidence', '').strip() or None
        
        if not claim:
            return jsonify({'error': 'No claim provided'}), 400
        
        if not fact_inference:
            # Enhanced placeholder response
            import random
            verdicts = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
            verdict = random.choice(verdicts)
            confidence = random.uniform(0.4, 0.8)
            response = {
                    'verdict': verdict,
                    'confidence': round(confidence, 3),
                    'claim': claim,
                    'evidence': evidence or "No evidence provided",
                    'explanation': f'⚠️ Using placeholder predictions. Train Milestone 3 to get real results.',
                    'status': 'Model not trained yet',
                    'probabilities': {
                        'SUPPORTS': round(random.uniform(0.2, 0.6), 3),
                        'REFUTES': round(random.uniform(0.2, 0.6), 3),
                        'NOT_ENOUGH_INFO': round(random.uniform(0.2, 0.6), 3)
                    }
                }
            return jsonify(response)
        # Get prediction from trained model
        result = fact_inference.verify_claim(claim, evidence, return_probabilities=True)
        
        # Add explanation
        result['explanation'] = fact_inference.explain_verdict(claim, result)
        result['status'] = 'Prediction from trained DeBERTa model'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Fact verification failed: {str(e)}'}), 500

if __name__ == '__main__':
    load_models()
    app.run(
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 5000),
        debug=config.get('api.debug', True)
    )