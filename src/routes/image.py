import os
import requests
import base64
from flask import Blueprint, jsonify, request
from io import BytesIO

image_bp = Blueprint('image', __name__)
print("✅ image routes loaded")

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/stable-diffusion-v1-5"
HF_TOKEN = os.environ.get('HUGGING_FACE_TOKEN', '')

@image_bp.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        # Get prompt from request
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        
        # Check if HF token is available
        if not HF_TOKEN:
            return jsonify({'error': 'Hugging Face token not configured'}), 500
        
        # Prepare headers for Hugging Face API
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
        
        # Make request to Hugging Face API
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            # Convert image bytes to base64
            image_bytes = response.content
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f"data:image/png;base64,{image_base64}",
                'prompt': prompt
            })
        elif response.status_code == 503:
            return jsonify({
                'error': 'Model is currently loading. Please try again in a few moments.',
                'retry_after': 20
            }), 503
        else:
            error_msg = f"Hugging Face API error: {response.status_code}"
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_msg += f" - {error_detail['error']}"
            except:
                pass
            
            return jsonify({'error': error_msg}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 408
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@image_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'AI Image Generator API',
        'hf_token_configured': bool(HF_TOKEN)
    })

