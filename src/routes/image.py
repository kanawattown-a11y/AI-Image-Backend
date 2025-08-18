# src/routes/image.py
import os
import requests
import base64
from flask import Blueprint, jsonify, request

image_bp = Blueprint('image', __name__)
print("✅ image routes loaded")

# --- اضبط هنا موديل Hugging Face الذي تريده ---
# مثال موثوق ومشهور:
HF_API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
# أو بدل السطر أعلاه بأي موديل inference آخر متوفر، مثل:
# HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
# --------------------------------------------------------------------
HF_TOKEN = os.environ.get('HUGGING_FACE_TOKEN', '').strip()

# Helpers
def _bad_request(msg, code=400):
    return jsonify({'success': False, 'error': msg}), code

def _call_hf_api(prompt, params=None, timeout=120):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
    }
    if params:
        payload["parameters"] = params

    return requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)

@image_bp.route('/generate-image', methods=['POST'])
def generate_image():
    # Read JSON safely
    data = request.get_json(silent=True)
    if not data:
        return _bad_request("Request body must be valid JSON with a 'prompt' field.", 415)

    prompt = data.get('prompt')
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return _bad_request("Prompt is required and must be a non-empty string.", 400)

    prompt = prompt.strip()

    # Optional: simple prompt length guard
    if len(prompt) > 1000:
        return _bad_request("Prompt is too long (max 1000 characters).", 400)

    if not HF_TOKEN:
        return _bad_request("Hugging Face token not configured. Set HUGGING_FACE_TOKEN in environment.", 500)

    # Optional parameters (you can adjust or expose from frontend later)
    hf_params = {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    }

    try:
        resp = _call_hf_api(prompt, params=hf_params, timeout=120)

        # Debug logs (helpful while developing; you can remove or lower verbosity in production)
        print("HF status:", resp.status_code)
        try:
            # print start of body for debugging (avoid printing tokens!)
            print("HF resp headers:", resp.headers.get('content-type'))
            print("HF resp text preview:", resp.text[:1000])
        except Exception:
            pass

        # If HF returned JSON (errors or model-specific JSON responses)
        content_type = resp.headers.get('content-type', '')

        if resp.status_code == 200:
            # If response is JSON, return it directly (some endpoints return JSON with useful info)
            if 'application/json' in content_type:
                try:
                    resp_json = resp.json()
                    # If JSON contains an 'error' field, forward it as error
                    if isinstance(resp_json, dict) and resp_json.get('error'):
                        return jsonify({'success': False, 'error': resp_json.get('error')}), 500
                    # Otherwise return the JSON result to client for further handling
                    return jsonify({'success': True, 'result': resp_json}), 200
                except ValueError:
                    # fallback: treat as bytes if JSON parse fails
                    pass

            # Otherwise assume binary image bytes (e.g., image/png)
            image_bytes = resp.content
            if not image_bytes:
                return _bad_request("Hugging Face returned empty content.", 500)

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:image/png;base64,{image_b64}"
            return jsonify({'success': True, 'image': data_url, 'prompt': prompt}), 200

        elif resp.status_code == 503:
            # Model is loading / busy
            return jsonify({'success': False, 'error': 'Model is loading. Please try again in a few moments.'}), 503
        else:
            # Try to extract error detail
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            msg = f"Hugging Face API error: {resp.status_code}"
            return jsonify({'success': False, 'error': msg, 'detail': err}), resp.status_code

    except requests.exceptions.Timeout:
        return _bad_request('Request to Hugging Face timed out. Try again later.', 408)
    except requests.exceptions.RequestException as e:
        return _bad_request(f'Network error when calling Hugging Face: {str(e)}', 502)
    except Exception as e:
        # unexpected
        print("Internal server error in generate_image:", str(e))
        return _bad_request(f'Internal server error: {str(e)}', 500)


@image_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'AI Image Generator API',
        'hf_token_configured': bool(HF_TOKEN),
        'hf_model_url': HF_API_URL
    })
