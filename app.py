"""
UiTM Receptionist AI - Flask Backend with RAG System
Pembantu AI UiTM - Backend Flask dengan Sistem RAG
"""

import os
import json
import base64
import requests
import asyncio
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from dotenv import load_dotenv

# Import RAG system
from rag import RAGManager
from rag.image_handler import ImageHandler

# Import VTS system
from vts import VTSConnector, LipSyncAnalyzer, ExpressionMapper, AudioConverter
from vts import get_idle_animator, get_gesture_controller, get_player
from vts import get_parallel_analyzer

# Import optimized TTS
try:
    from tts_optimized import OptimizedMinimaxTTS, get_tts_instance, TTSChunk
    TTS_OPTIMIZED_AVAILABLE = True
except ImportError:
    TTS_OPTIMIZED_AVAILABLE = False
    print("[TTS] Optimized TTS module not available, using fallback")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'uitm-chatbot-secret-key')

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'google/gemini-3.1-flash-lite-preview')

# Minimax TTS Configuration
MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY')
MINIMAX_TTS_MODEL = os.getenv('MINIMAX_TTS_MODEL', 'speech-2.8-turbo')
MINIMAX_TTS_VOICE = os.getenv('MINIMAX_TTS_VOICE', 'Malay_male_1_v1')
MINIMAX_TTS_LANGUAGE = os.getenv('MINIMAX_TTS_LANGUAGE', 'Malay')

# RAG Configuration
ENABLE_RAG = os.getenv('ENABLE_RAG', 'true').lower() == 'true'
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '5'))
RAG_USE_ADVANCED = os.getenv('RAG_USE_ADVANCED', 'false').lower() == 'true'

# VTube Studio Configuration
VTS_ENABLED = os.getenv('VTS_ENABLED', 'false').lower() == 'true'
VTS_HOST = os.getenv('VTS_HOST', 'localhost')
VTS_PORT = int(os.getenv('VTS_PORT', '8001'))
VTS_AUTO_RECONNECT = os.getenv('VTS_AUTO_RECONNECT', 'true').lower() == 'true'
VTS_RECONNECT_INTERVAL = float(os.getenv('VTS_RECONNECT_INTERVAL', '5.0'))

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

# Initialize VTS System
vts_connector = None
vts_lip_sync = None
vts_expression_mapper = None
vts_audio_converter = None
vts_loop = None  # Persistent event loop for VTS
vts_idle_animator = None  # Idle animation controller
vts_gesture_controller = None  # Gesture controller for talking

def run_in_vts_loop(coro):
    """
    Run a coroutine in the VTS event loop.
    This ensures all VTS operations use the same event loop.
    """
    global vts_loop
    if vts_loop is None or not vts_loop.is_running():
        print("[VTS] Error: VTS loop not running")
        return None
    print(f"[VTS] Running coroutine in VTS loop: {coro}")
    future = asyncio.run_coroutine_threadsafe(coro, vts_loop)
    result = future.result(timeout=30)  # 30 second timeout
    print(f"[VTS] Coroutine completed")
    return result

def vts_loop_thread():
    """Run the VTS event loop in a background thread."""
    global vts_loop
    vts_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(vts_loop)
    vts_loop.run_forever()

if VTS_ENABLED:
    try:
        print("\n" + "="*60)
        print("Initializing VTube Studio Integration...")
        print("="*60)
        
        # Start the VTS event loop in a background thread
        vts_thread = threading.Thread(target=vts_loop_thread, daemon=True)
        vts_thread.start()
        print("[VTS] Started background event loop thread")
        
        # Wait for loop to be ready
        import time
        while vts_loop is None:
            time.sleep(0.1)
        
        # Initialize VTS components
        vts_connector = VTSConnector(
            host=VTS_HOST,
            port=VTS_PORT,
            auto_reconnect=VTS_AUTO_RECONNECT,
            reconnect_interval=VTS_RECONNECT_INTERVAL
        )
        vts_lip_sync = LipSyncAnalyzer()
        vts_expression_mapper = ExpressionMapper()
        vts_audio_converter = AudioConverter()
        
        # Initialize liveliness controllers
        vts_idle_animator = get_idle_animator(vts_connector)
        vts_gesture_controller = get_gesture_controller(vts_connector)
        
        print("\n✓ VTS Integration ready! (Will connect on first use)")
        print("✓ Idle animations and gesture control ready!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n⚠ Warning: Could not initialize VTS system: {e}")
        print("Continuing without VTS functionality...\n")
        import traceback
        traceback.print_exc()
        vts_connector = None

# Base system prompt for UiTM Receptionist
BASE_SYSTEM_PROMPT = """Anda adalah Pembantu AI rasmi Universiti Teknologi MARA (UiTM), Malaysia.

Peranan anda:
1. Memberikan maklumat tepat dan mesra tentang UiTM
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

Maklumat penting UiTM:
- Universiti Teknologi MARA adalah universiti awam terbesar di Malaysia
- Ditubuhkan pada tahun 1956 sebagai Kolej RIDA, kemudian ITM (1965), dan menjadi UiTM (1999)
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
Anda mempunyai akses kepada pangkalan pengetahuan UiTM. Gunakan maklumat yang diberikan dalam [KONTEKS] di bawah untuk menjawab soalan pengguna.

Garis panduan penggunaan konteks:
1. Gunakan maklumat daripada [KONTEKS] untuk menjawab soalan dengan tepat
2. Jika maklumat tidak mencukupi dalam konteks, gunakan pengetahuan umum anda tentang UiTM
3. Jika anda tidak pasti, nyatakan dengan jujur dan cadangkan pengguna menghubungi pejabat berkaitan
4. Rujuk sumber dokumen jika menggunakan maklumat spesifik daripada konteks
5. Pastikan jawapan adalah dalam Bahasa Melayu yang betul

[KONTEKS]
{retrieved_context}
[/KONTEKS]
"""

# Creator information configuration
CREATOR_INFO = {
    "name": "Ts. Zaaba Bin Ahmad",
    "title": "Pencipta AI Receptionist UiTM",
    "image_path": "/static/assets/Zaaba-Ahmad.webp",
    "google_scholar": "https://scholar.google.com/citations?user=PGhzO-oAAAAJ&hl=en",
    "description": "AI ini dicipta oleh **Ts. Zaaba Bin Ahmad**. Beliau adalah seorang pensyarah dan penyelidik di UiTM yang mengkhusus dalam pembangunan sistem pintar dan aplikasi berasaskan AI.",
    "triggers": [
        # Malay triggers
        "siapa pencipta", "siapa pembangun", "siapa yang buat", "siapa yang cipta",
        "siapa yang bina", "siapa owner", "siapa tuan", "pencipta ai",
        "pembangun ai", "siapa zaaba", "ts zaaba", "zaaba ahmad",
        # English triggers
        "who created you", "who is your creator", "who built you",
        "who made you", "who developed you", "your creator",
        "who is zaaba", "ts zaaba", "zaaba ahmad"
    ]
}


def detect_creator_question(query):
    """Detect if user is asking about the creator"""
    if not query:
        return False
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in CREATOR_INFO["triggers"])


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
    
    # Get the user's last query
    user_query = get_last_user_query(messages)
    
    # Check if user is asking about the creator
    if detect_creator_question(user_query):
        # Return structured response with creator info and image
        creator_response = {
            'type': 'creator_info',
            'content': CREATOR_INFO["description"],
            'image': {
                'url': CREATOR_INFO["image_path"],
                'alt': f"{CREATOR_INFO['name']} - {CREATOR_INFO['title']}",
                'title': CREATOR_INFO['name']
            },
            'links': [
                {
                    'text': 'Profil Google Scholar',
                    'url': CREATOR_INFO["google_scholar"],
                    'icon': 'external-link'
                }
            ],
            'role': 'assistant'
        }
        
        if stream_requested:
            # Return streaming format for creator info
            def generate_creator_response():
                # Send the response as a single chunk
                yield f'data: {json.dumps({"choices": [{"delta": {"content": json.dumps(creator_response)}}]})}\n\n'
                yield 'data: [DONE]\n\n'
            
            return Response(
                stream_with_context(generate_creator_response()),
                mimetype='text/plain',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            return jsonify(creator_response)
    
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
        "X-Title": "UiTM Receptionist AI"
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


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech using Minimax TTS API
    Returns audio file in MP3 format with Malay male voice
    Optionally includes lip sync data if VTS is enabled
    """
    if not MINIMAX_API_KEY:
        return jsonify({'error': 'Minimax API key not configured'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    include_lip_sync = data.get('include_lip_sync', False) and VTS_ENABLED
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Limit text length to prevent API errors (Minimax limit: 10,000 characters)
    if len(text) > 10000:
        text = text[:9997] + '...'
    
    try:
        from minimax_tts import MinimaxTTS, MinimaxTTSError
        
        tts = MinimaxTTS(
            api_key=MINIMAX_API_KEY,
            model=MINIMAX_TTS_MODEL,
            voice_id=MINIMAX_TTS_VOICE,
            language_boost=MINIMAX_TTS_LANGUAGE
        )
        
        audio_bytes = tts.generate_audio(text)
        
        # If lip sync is not requested or VTS is disabled, return audio only
        if not include_lip_sync:
            return Response(
                audio_bytes,
                mimetype='audio/mpeg',
                headers={
                    'Content-Disposition': 'inline; filename="speech.mp3"',
                    'Cache-Control': 'no-cache'
                }
            )
        
        # Generate lip sync data
        lip_sync_data = []
        duration = 0.0
        expression = None
        
        try:
            # Convert MP3 to WAV for analysis
            if vts_audio_converter and vts_audio_converter.is_available:
                print(f"[TTS] Converting MP3 to WAV for lip sync analysis...")
                wav_bytes = vts_audio_converter.convert_mp3_to_wav(audio_bytes)
                
                if wav_bytes:
                    # Analyze WAV for lip sync
                    print(f"[TTS] Analyzing WAV for lip sync ({len(wav_bytes)} bytes)...")
                    lip_sync_data = vts_lip_sync.analyze_wav_bytes(wav_bytes)
                    print(f"[TTS] Generated {len(lip_sync_data)} lip sync frames")
                    
                    # Get audio duration
                    duration = vts_audio_converter.get_audio_duration(audio_bytes, 'mp3') or 0.0
                    
                    # Extract emotion from text
                    if vts_expression_mapper:
                        expression = vts_expression_mapper.extract_emotion(text)
                else:
                    print(f"[TTS] WAV conversion returned empty")
            else:
                print(f"[TTS] Audio converter not available (is_available={vts_audio_converter.is_available if vts_audio_converter else 'None'})")
        except Exception as e:
            print(f"[TTS] Lip sync generation error: {e}")
            import traceback
            traceback.print_exc()
        
        # Return JSON with audio and lip sync data
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return jsonify({
            'audio': audio_base64,
            'audio_format': 'mp3',
            'lip_sync': lip_sync_data,
            'duration': duration,
            'expression': expression,
            'vts_enabled': VTS_ENABLED
        })
        
    except MinimaxTTSError as e:
        print(f"Minimax TTS error: {e.message}")
        return jsonify({'error': f'TTS error: {e.message}'}), 500
    except Exception as e:
        print(f"TTS unexpected error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/tts-optimized', methods=['POST'])
def text_to_speech_optimized():
    """
    Optimized TTS endpoint with streaming, caching, and parallel processing.
    Returns audio with lip sync data using optimized pipeline.
    """
    if not MINIMAX_API_KEY:
        return jsonify({'error': 'Minimax API key not configured'}), 500
    
    if not TTS_OPTIMIZED_AVAILABLE:
        return jsonify({'error': 'Optimized TTS not available'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    include_lip_sync = data.get('include_lip_sync', False) and VTS_ENABLED
    use_streaming = data.get('streaming', True)
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Limit text length
    if len(text) > 10000:
        text = text[:9997] + '...'
    
    try:
        import time
        start_time = time.time()
        
        # Get optimized TTS instance
        tts = get_tts_instance(
            api_key=MINIMAX_API_KEY,
            model=MINIMAX_TTS_MODEL,
            voice_id=MINIMAX_TTS_VOICE,
            language_boost=MINIMAX_TTS_LANGUAGE
        )
        
        all_audio = b""
        all_lip_sync = []
        chunk_index = 0
        
        # Process TTS with streaming
        async def process_tts():
            nonlocal all_audio, all_lip_sync, chunk_index
            
            async for chunk in tts.generate_audio_streaming(text):
                all_audio += chunk.audio_bytes
                
                # Process lip sync for this chunk in parallel
                if include_lip_sync and vts_audio_converter and vts_audio_converter.is_available:
                    try:
                        # Convert chunk to WAV
                        wav_bytes = vts_audio_converter.convert_mp3_to_wav(chunk.audio_bytes)
                        if wav_bytes:
                            # Use parallel analyzer
                            parallel_analyzer = get_parallel_analyzer()
                            lip_sync = await parallel_analyzer.analyze_wav_bytes_parallel(wav_bytes)
                            
                            # Adjust timestamps for chunk position
                            offset = chunk_index * (len(lip_sync) / 30.0) if lip_sync else 0
                            adjusted_lip_sync = [(t + offset, v) for t, v in lip_sync]
                            all_lip_sync.extend(adjusted_lip_sync)
                    except Exception as e:
                        print(f"[TTS-Optimized] Lip sync error for chunk: {e}")
                
                chunk_index += 1
        
        # Run async processing
        asyncio.run(process_tts())
        
        # Get audio duration
        duration = 0.0
        if vts_audio_converter:
            duration = vts_audio_converter.get_audio_duration(all_audio, 'mp3') or 0.0
        
        # Get expression
        expression = None
        if vts_expression_mapper:
            expression = vts_expression_mapper.extract_emotion(text)
        
        elapsed = time.time() - start_time
        print(f"[TTS-Optimized] Generated in {elapsed:.2f}s, {len(all_audio)} bytes, {len(all_lip_sync)} lip sync frames")
        
        # Return response
        audio_base64 = base64.b64encode(all_audio).decode('utf-8')
        
        return jsonify({
            'audio': audio_base64,
            'audio_format': 'mp3',
            'lip_sync': all_lip_sync,
            'duration': duration,
            'expression': expression,
            'vts_enabled': VTS_ENABLED,
            'processing_time': elapsed,
            'optimized': True
        })
        
    except Exception as e:
        print(f"[TTS-Optimized] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/vts/status', methods=['GET'])
def vts_status():
    """
    Get VTube Studio connection status
    """
    if not VTS_ENABLED:
        return jsonify({
            'enabled': False,
            'connected': False,
            'message': 'VTS integration is disabled'
        })
    
    connected = vts_connector is not None and vts_connector.is_connected
    
    return jsonify({
        'enabled': True,
        'connected': connected,
        'message': 'Connected' if connected else 'Not connected'
    })


@app.route('/vts/connect', methods=['POST'])
def vts_connect():
    """
    Connect to VTube Studio and start idle animations
    """
    global vts_idle_animator
    
    if not VTS_ENABLED:
        return jsonify({'error': 'VTS integration is disabled'}), 400
    
    if not vts_connector:
        return jsonify({'error': 'VTS connector not initialized'}), 500
    
    try:
        # Run async connect in the persistent VTS loop
        success = run_in_vts_loop(vts_connector.connect())
        
        if success:
            # Start idle animations
            if vts_idle_animator:
                try:
                    run_in_vts_loop(vts_idle_animator.start())
                    print("[VTS] Idle animations started")
                except Exception as e:
                    print(f"[VTS] Could not start idle animations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Connected to VTube Studio'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to connect to VTube Studio'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/vts/disconnect', methods=['POST'])
def vts_disconnect():
    """
    Disconnect from VTube Studio and stop idle animations
    """
    global vts_idle_animator
    
    if not VTS_ENABLED:
        return jsonify({'error': 'VTS integration is disabled'}), 400
    
    if not vts_connector:
        return jsonify({'error': 'VTS connector not initialized'}), 500
    
    try:
        # Stop idle animations first
        if vts_idle_animator:
            try:
                run_in_vts_loop(vts_idle_animator.stop())
                print("[VTS] Idle animations stopped")
            except Exception as e:
                print(f"[VTS] Error stopping idle animations: {e}")
        
        run_in_vts_loop(vts_connector.disconnect())
        
        return jsonify({
            'success': True,
            'message': 'Disconnected from VTube Studio'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/vts/set_mouth', methods=['POST'])
def vts_set_mouth():
    """
    Set mouth open value in VTube Studio
    Called by frontend during TTS playback for lip sync
    """
    if not VTS_ENABLED:
        return jsonify({'error': 'VTS integration is disabled'}), 400
    
    if not vts_connector or not vts_connector.is_connected:
        return jsonify({'error': 'VTS not connected'}), 400
    
    data = request.get_json()
    mouth_value = data.get('value', 0.0)
    
    try:
        # Set the mouth parameter
        params = vts_lip_sync.get_mouth_parameters(mouth_value)
        success = run_in_vts_loop(vts_connector.set_parameters(params))
        success = loop.run_until_complete(vts_connector.set_parameters(params))
        loop.close()
        
        return jsonify({'success': success})
        
    except Exception as e:
        print(f"[VTS] Error setting mouth: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/vts/play_lip_sync', methods=['POST'])
def vts_play_lip_sync():
    """
    Play lip sync data directly on VTS with gesture control
    Frontend sends the lip sync data and backend plays it in real-time
    Includes idle animation pause/resume and talking gestures
    """
    global vts_idle_animator, vts_gesture_controller
    
    if not VTS_ENABLED:
        return jsonify({'error': 'VTS integration is disabled'}), 400
    
    if not vts_connector or not vts_connector.is_connected:
        return jsonify({'error': 'VTS not connected'}), 400
    
    data = request.get_json()
    lip_sync_data = data.get('lip_sync', [])
    text = data.get('text', '')  # Optional text for emotion detection
    
    if not lip_sync_data:
        return jsonify({'success': True, 'message': 'No lip sync data'})
    
    try:
        from vts.lip_sync import LipSyncPlayer
        
        player = LipSyncPlayer(vts_lip_sync)
        
        # Set up liveliness controllers
        if vts_idle_animator or vts_gesture_controller:
            player.set_liveliness_controllers(vts_idle_animator, vts_gesture_controller)
        
        # Play lip sync with text for emotion-based gestures
        run_in_vts_loop(player.play_lip_sync(vts_connector, lip_sync_data, text=text))
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"[VTS] Error playing lip sync: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
