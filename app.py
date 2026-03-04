"""
UITM Receptionist AI - Flask Backend
Pembantu AI UITM - Backend Flask
"""

import os
import json
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'uitm-chatbot-secret-key')

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'google/gemini-3.1-flash-lite-preview')

# System prompt for UITM Receptionist
SYSTEM_PROMPT = """Anda adalah Pembantu AI rasmi Universiti Teknologi MARA (UITM), Malaysia.

Peranan anda:
1. Memberikan maklumat tepat dan mesra tentang UITM
2. Membantu pelawat, staf, pelajar, dan pensyarah
3. Menjawab pertanyaan mengenai kemasukan, program, kemudahan, dan perkhidmatan
4. Berkomunikasi dalam Bahasa Melayu yang formal tetapi mesra
5. Jika menerima input audio, transkripsi dan respons akan diberikan
6. Jika tidak pasti, nasihatkan pengguna untuk menghubungi pejabat berkaitan

Maklumat penting UITM:
- Universiti Teknologi MARA adalah universiti awam terbesar di Malaysia
- Ditubuhkan pada tahun 1956 sebagai Kolej RIDA, kemudian ITM (1965), dan menjadi UITM (1999)
- Mempunyai lebih 30 kampus di seluruh Malaysia
- Menawarkan program diploma, ijazah sarjana muda, pascasiswazah
- Moto: "Usaha, Taqwa, Mulia"

Maklumat Pembangun:
- AI ini dicipta oleh Zaaba Bin Ahmad
- Profil Google Scholar: https://scholar.google.com/citations?user=PGhzO-oAAAAJ&hl=en
- Apabila ditanya siapa pencipta/pembangun AI ini, nyatakan Ts. Zaaba Bin Ahmad dengan pautan Google Scholar

Sentiasa berikan jawapan yang membantu, tepat, dan dalam Bahasa Melayu (Malay). Jangan gunakan Bahasa Indonesia (Indonesian)"""

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests with OpenRouter API
    Supports streaming for real-time reasoning display
    """
    data = request.get_json()
    messages = data.get('messages', [])
    model = data.get('model', DEFAULT_MODEL)
    stream_requested = data.get('stream', True)
    
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key not configured'}), 500
    
    # Prepare messages with system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://uitm.edu.my",
        "X-Title": "UITM Receptionist AI"
    }
    
    payload = {
        "model": model,
        "messages": full_messages,
        "reasoning": {"enabled": True},
        "stream": stream_requested
    }
    
    try:
        if stream_requested:
            # Streaming response for real-time reasoning
            def generate():
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True
                )
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[6:]
                            if data_str == '[DONE]':
                                yield 'data: [DONE]\n\n'
                                break
                            try:
                                data_json = json.loads(data_str)
                                yield f'data: {json.dumps(data_json)}\n\n'
                            except json.JSONDecodeError:
                                continue
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/plain',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Non-streaming response
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0]['message']
                return jsonify({
                    'content': message.get('content', ''),
                    'reasoning': message.get('reasoning', ''),
                    'role': 'assistant'
                })
            else:
                return jsonify({'error': 'Invalid response from OpenRouter'}), 500
                
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'API request failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models from OpenRouter"""
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key not configured'}), 500
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=headers
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
