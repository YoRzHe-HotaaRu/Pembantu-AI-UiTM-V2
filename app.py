"""
UITM Receptionist AI - Flask Backend with RAG System
Pembantu AI UITM - Backend Flask dengan Sistem RAG
"""

import os
import json
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from dotenv import load_dotenv

# Import RAG system
from rag import RAGManager
from rag.image_handler import ImageHandler

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'uitm-chatbot-secret-key')

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'google/gemini-3.1-flash-lite-preview')

# RAG Configuration
ENABLE_RAG = os.getenv('ENABLE_RAG', 'true').lower() == 'true'
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '5'))
RAG_USE_ADVANCED = os.getenv('RAG_USE_ADVANCED', 'false').lower() == 'true'

# Initialize RAG System
rag_manager = None
image_handler = None

if ENABLE_RAG:
    try:
        print("\n" + "="*60)
        print("Initializing RAG System...")
        print("="*60)
        
        # Use lightweight mode by default (no heavy embeddings)
        rag_manager = RAGManager(use_advanced=RAG_USE_ADVANCED)
        rag_manager.initialize()
        
        # Initialize image handler (lightweight)
        try:
            image_handler = ImageHandler()
            image_handler.load_images()
        except Exception as e:
            print(f"Note: Image handler not initialized: {e}")
        
        print("\n✓ RAG System ready!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n⚠ Warning: Could not initialize RAG system: {e}")
        print("Continuing without RAG functionality...\n")
        import traceback
        traceback.print_exc()
        rag_manager = None

# Base system prompt for UITM Receptionist
BASE_SYSTEM_PROMPT = """Anda adalah Pembantu AI rasmi Universiti Teknologi MARA (UITM), Malaysia.

Peranan anda:
1. Memberikan maklumat tepat dan mesra tentang UITM
2. Membantu pelawat, staf, pelajar, dan pensyarah
3. Menjawab pertanyaan mengenai kemasukan, program, kemudahan, dan perkhidmatan
4. Berkomunikasi dalam Bahasa Melayu yang formal tetapi mesra
5. Jika menerima input audio, transkripsi dan respons akan diberikan
6. Jika tidak pasti, nasihatkan pengguna untuk menghubungi pejabat berkaitan

Garis Panduan Respons:
- JANGAN mulakan setiap respons dengan ucapan seperti "Selamat sejahtera!", "Waalaikumussalam", 
  "Assalamualaikum", atau memperkenalkan diri berulang kali
- Terus jawab soalan pengguna tanpa perkenalan yang berulang
- Hanya perkenalkan diri jika pengguna bertanya "Siapa awak?" atau soalan serupa
- Untuk sambungan perbualan, teruskan dari topik sebelumnya
- Elakkan penggunaan ucapan agama seperti "Waalaikumussalam" atau "Assalamualaikum"

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

RAG_INSTRUCTIONS = """

[ARAHAN RAG]
Anda mempunyai akses kepada pangkalan pengetahuan UITM. Gunakan maklumat yang diberikan dalam [KONTEKS] di bawah untuk menjawab soalan pengguna.

Garis panduan penggunaan konteks:
1. Gunakan maklumat daripada [KONTEKS] untuk menjawab soalan dengan tepat
2. Jika maklumat tidak mencukupi dalam konteks, gunakan pengetahuan umum anda tentang UITM
3. Jika anda tidak pasti, nyatakan dengan jujur dan cadangkan pengguna menghubungi pejabat berkaitan
4. Rujuk sumber dokumen jika menggunakan maklumat spesifik daripada konteks
5. Pastikan jawapan adalah dalam Bahasa Melayu yang betul

[KONTEKS]
{retrieved_context}
[/KONTEKS]
"""


def get_last_user_query(messages):
    """Extract the last user message from the conversation"""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            
            # Handle multimodal messages (e.g., voice messages)
            # Content can be a list of objects with 'text' and 'input_audio'
            if isinstance(content, list):
                # Extract text parts from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif item.get('type') == 'input_text':
                            text_parts.append(item.get('text', ''))
                # Join all text parts for RAG search
                return ' '.join(text_parts)
            
            # Regular text message
            return content
    return ''


def build_system_prompt(retrieved_context=''):
    """Build the complete system prompt with optional RAG context"""
    if retrieved_context and ENABLE_RAG:
        return BASE_SYSTEM_PROMPT + RAG_INSTRUCTIONS.format(retrieved_context=retrieved_context)
    return BASE_SYSTEM_PROMPT


@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests with OpenRouter API
    Supports streaming for real-time reasoning display
    Integrates with RAG system for context retrieval
    """
    data = request.get_json()
    messages = data.get('messages', [])
    model = data.get('model', DEFAULT_MODEL)
    stream_requested = data.get('stream', True)
    
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key not configured'}), 500
    
    # Get the user's last query for RAG
    user_query = get_last_user_query(messages)
    
    # Retrieve context using RAG if enabled
    retrieved_context = ''
    sources = []
    rag_used = False
    
    # Skip RAG for voice/multimodal messages
    # We can't search the knowledge base until we know the transcribed content
    last_message = messages[-1] if messages else None
    is_voice_message = False
    if last_message and last_message.get('role') == 'user':
        content = last_message.get('content', '')
        if isinstance(content, list):
            # Check if this is a voice message (contains input_audio)
            is_voice_message = any(
                isinstance(item, dict) and item.get('type') == 'input_audio'
                for item in content
            )
    
    if ENABLE_RAG and rag_manager and user_query and not is_voice_message:
        try:
            rag_result = rag_manager.query(
                query_text=user_query,
                top_k=RAG_TOP_K,
                format_context=True
            )
            retrieved_context = rag_result.get('context', '')
            sources = rag_result.get('sources', [])
            rag_used = len(sources) > 0  # RAG was used if we found sources
        except Exception as e:
            print(f"RAG query error: {e}")
    
    # Build system prompt with retrieved context
    system_prompt = build_system_prompt(retrieved_context)
    
    # Prepare messages with system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
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
                                # Send RAG metadata if available
                                rag_metadata = {}
                                if rag_used:
                                    rag_metadata['rag_used'] = True
                                if sources:
                                    rag_metadata['sources'] = sources
                                if rag_metadata:
                                    yield f'data: {json.dumps({"rag_metadata": rag_metadata})}\n\n'
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
                result = {
                    'content': message.get('content', ''),
                    'reasoning': message.get('reasoning', ''),
                    'role': 'assistant'
                }
                
                # Include RAG metadata if available
                if rag_used:
                    result['rag_used'] = True
                if sources:
                    result['sources'] = sources
                
                return jsonify(result)
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


# RAG API Endpoints

@app.route('/api/knowledge/search', methods=['GET'])
def knowledge_search():
    """Search the knowledge base"""
    if not rag_manager:
        return jsonify({'error': 'RAG system not initialized'}), 503
    
    query = request.args.get('q', '')
    top_k = request.args.get('top_k', RAG_TOP_K, type=int)
    category = request.args.get('category', None)
    
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    try:
        result = rag_manager.query(
            query_text=query,
            top_k=top_k,
            category_filter=category,
            format_context=False
        )
        
        # Format chunks for response
        chunks_data = []
        for chunk in result.get('chunks', []):
            chunks_data.append({
                'id': chunk.id,
                'content': chunk.content[:500] + '...' if len(chunk.content) > 500 else chunk.content,
                'doc_title': chunk.doc_title,
                'category': chunk.category,
                'relevance': round(chunk.combined_score, 3)
            })
        
        return jsonify({
            'query': query,
            'results': chunks_data,
            'total': len(chunks_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/categories', methods=['GET'])
def knowledge_categories():
    """Get list of available categories"""
    if not rag_manager:
        return jsonify({'error': 'RAG system not initialized'}), 503
    
    try:
        categories = rag_manager.get_categories()
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/stats', methods=['GET'])
def knowledge_stats():
    """Get statistics about the knowledge base"""
    if not rag_manager:
        return jsonify({'error': 'RAG system not initialized'}), 503
    
    try:
        stats = rag_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/reload', methods=['POST'])
def knowledge_reload():
    """Reload and reindex the knowledge base"""
    if not rag_manager:
        return jsonify({'error': 'RAG system not initialized'}), 503
    
    try:
        rag_manager.reload()
        return jsonify({
            'status': 'success',
            'message': 'Knowledge base reloaded successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/search', methods=['GET'])
def images_search():
    """Search for images in the knowledge base"""
    if not image_handler:
        return jsonify({'error': 'Image handler not initialized'}), 503
    
    query = request.args.get('q', '')
    limit = request.args.get('limit', 5, type=int)
    
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    try:
        images = image_handler.search_images(query, limit)
        return jsonify({
            'query': query,
            'images': [
                {
                    'id': img.id,
                    'filename': img.filename,
                    'description': img.description,
                    'category': img.category,
                    'url': img.url_path
                }
                for img in images
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/kb_assets/<path:filename>')
def serve_kb_assets(filename):
    """Serve knowledge base assets (images)"""
    return send_from_directory('knowledge_base/assets', filename)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
