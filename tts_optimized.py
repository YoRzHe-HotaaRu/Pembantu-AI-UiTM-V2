"""
Optimized TTS Service with Streaming, Persistent WebSocket, and Parallel Processing
Handles Text-to-Speech synthesis using Minimax WebSocket API with performance optimizations.
"""

import asyncio
import websockets
import json
import ssl
import hashlib
import os
import re
import time
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class TTSChunk:
    """Represents a chunk of TTS audio data."""
    audio_bytes: bytes
    text: str
    chunk_index: int
    is_last: bool
    lip_sync_data: Optional[List[Tuple[float, float]]] = None


class TTSCache:
    """Simple file-based cache for TTS audio."""
    
    def __init__(self, cache_dir: str = "tts_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, text: str, voice_id: str, model: str) -> str:
        """Generate cache key from text and settings."""
        key = f"{text}:{voice_id}:{model}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.mp3")
    
    def get(self, text: str, voice_id: str, model: str) -> Optional[bytes]:
        """Get cached audio if exists."""
        cache_key = self._get_cache_key(text, voice_id, model)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"[TTS Cache] Error reading cache: {e}")
        return None
    
    def set(self, text: str, voice_id: str, model: str, audio_bytes: bytes):
        """Cache audio data."""
        cache_key = self._get_cache_key(text, voice_id, model)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                f.write(audio_bytes)
        except Exception as e:
            print(f"[TTS Cache] Error writing cache: {e}")
    
    def clear(self):
        """Clear all cached files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.mp3'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except Exception:
                    pass


class OptimizedMinimaxTTS:
    """
    Optimized Minimax TTS with:
    - Persistent WebSocket connection
    - Streaming audio chunks
    - Parallel lip sync analysis
    - Sentence-level chunking
    - Audio caching
    """
    
    WS_URL = "wss://api.minimax.io/ws/v1/t2a_v2"
    
    DEFAULT_CONFIG = {
        "model": "speech-2.8-turbo",
        "voice_id": "Malay_male_1_v1",
        "language_boost": "Malay",
        "pronunciation_dict": {"tone": ["uitm/UITM", "UiTM/UITM"]},
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        },
        "voice_setting": {
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0
        }
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = None,
        voice_id: str = None,
        language_boost: str = None,
        enable_cache: bool = True,
        enable_persistent_ws: bool = True
    ):
        self.api_key = api_key
        self.config = self.DEFAULT_CONFIG.copy()
        
        if model:
            self.config["model"] = model
        if voice_id:
            self.config["voice_id"] = voice_id
        if language_boost:
            self.config["language_boost"] = language_boost
            
        # Cache
        self.cache = TTSCache() if enable_cache else None
        
        # Persistent WebSocket
        self._persistent_ws = None
        self._persistent_ws_lock = asyncio.Lock()
        self._enable_persistent_ws = enable_persistent_ws
        self._ws_last_used = 0
        self._ws_timeout = 30  # Close WS after 30 seconds of inactivity
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Sentence delimiters for chunking
        self.sentence_delimiters = re.compile(r'([.!?。！？]+\s*)')
        
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for WebSocket connection."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context
    
    async def _get_persistent_ws(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get or create persistent WebSocket connection."""
        if not self._enable_persistent_ws:
            return None
            
        async with self._persistent_ws_lock:
            # Check if existing connection is still valid
            if self._persistent_ws:
                if time.time() - self._ws_last_used > self._ws_timeout:
                    # Connection timed out, close it
                    try:
                        await self._persistent_ws.close()
                    except Exception:
                        pass
                    self._persistent_ws = None
                else:
                    # Connection is valid, update last used time
                    self._ws_last_used = time.time()
                    return self._persistent_ws
            
            # Create new connection
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                ssl_context = self._create_ssl_context()
                
                ws = await websockets.connect(
                    self.WS_URL,
                    additional_headers=headers,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                # Wait for connection confirmation
                response = await ws.recv()
                connected = json.loads(response)
                
                if connected.get("event") == "connected_success":
                    self._persistent_ws = ws
                    self._ws_last_used = time.time()
                    print("[TTS] Persistent WebSocket connection established")
                    return ws
                else:
                    await ws.close()
                    return None
                    
            except Exception as e:
                print(f"[TTS] Failed to create persistent WebSocket: {e}")
                return None
    
    async def _start_tts_task(self, ws: websockets.WebSocketClientProtocol) -> bool:
        """Send task_start event."""
        start_msg = {
            "event": "task_start",
            "model": self.config["model"],
            "voice_setting": {
                "voice_id": self.config["voice_id"],
                **self.config["voice_setting"]
            },
            "audio_setting": self.config["audio_setting"],
            "pronunciation_dict": self.config["pronunciation_dict"]
        }
        
        if self.config.get("language_boost"):
            start_msg["language_boost"] = self.config["language_boost"]
        
        await ws.send(json.dumps(start_msg))
        
        response = json.loads(await ws.recv())
        return response.get("event") == "task_started"
    
    def split_into_sentences(self, text: str, max_chunk_size: int = 200) -> List[str]:
        """
        Split text into sentence chunks for parallel processing.
        
        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of sentence chunks
        """
        # Split by sentence delimiters
        parts = self.sentence_delimiters.split(text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""
            
            full_sentence = sentence + delimiter
            
            # If single sentence is too long, split by commas or just chunk it
            if len(full_sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence by commas
                comma_parts = full_sentence.split(',')
                for part in comma_parts:
                    if len(current_chunk) + len(part) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk += part + ","
                current_chunk = current_chunk.rstrip(",")
            else:
                if len(current_chunk) + len(full_sentence) > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = full_sentence
                else:
                    current_chunk += full_sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def generate_audio_streaming(
        self,
        text: str,
        on_chunk: Optional[Callable[[TTSChunk], None]] = None
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Generate TTS audio with streaming chunks.
        
        Args:
            text: Text to synthesize
            on_chunk: Optional callback for each chunk
            
        Yields:
            TTSChunk objects with audio data
        """
        if not text or not text.strip():
            return
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(text, self.config["voice_id"], self.config["model"])
            if cached:
                chunk = TTSChunk(
                    audio_bytes=cached,
                    text=text,
                    chunk_index=0,
                    is_last=True
                )
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
                return
        
        # Split into sentences for chunking
        sentences = self.split_into_sentences(text)
        
        if len(sentences) == 1:
            # Single sentence - use streaming
            async for chunk in self._generate_single_sentence_streaming(sentences[0], on_chunk):
                yield chunk
        else:
            # Multiple sentences - process in parallel
            async for chunk in self._generate_parallel_sentences(sentences, on_chunk):
                yield chunk
    
    async def _generate_single_sentence_streaming(
        self,
        text: str,
        on_chunk: Optional[Callable[[TTSChunk], None]]
    ) -> AsyncGenerator[TTSChunk, None]:
        """Generate TTS for a single sentence with streaming."""
        ws = None
        use_persistent = False
        
        try:
            # Try to use persistent connection
            ws = await self._get_persistent_ws()
            if ws:
                use_persistent = True
            else:
                # Fall back to new connection
                headers = {"Authorization": f"Bearer {self.api_key}"}
                ssl_context = self._create_ssl_context()
                ws = await websockets.connect(
                    self.WS_URL,
                    additional_headers=headers,
                    ssl=ssl_context
                )
                
                # Wait for connection confirmation
                response = await ws.recv()
                connected = json.loads(response)
                if connected.get("event") != "connected_success":
                    return
            
            # Start TTS task
            if not use_persistent:
                if not await self._start_tts_task(ws):
                    return
            
            # Send text
            await ws.send(json.dumps({"event": "task_continue", "text": text}))
            
            # Stream audio chunks
            chunk_index = 0
            all_audio = b""
            
            while True:
                response = json.loads(await ws.recv())
                event = response.get("event")
                
                if event == "task_failed":
                    break
                
                if "data" in response and response["data"]:
                    audio_hex = response["data"].get("audio", "")
                    if audio_hex:
                        audio_bytes = bytes.fromhex(audio_hex)
                        all_audio += audio_bytes
                        
                        chunk = TTSChunk(
                            audio_bytes=audio_bytes,
                            text=text,
                            chunk_index=chunk_index,
                            is_last=response.get("is_final", False)
                        )
                        
                        if on_chunk:
                            on_chunk(chunk)
                        yield chunk
                        
                        chunk_index += 1
                
                if response.get("is_final", False):
                    break
            
            # Cache complete audio
            if self.cache and all_audio:
                self.cache.set(text, self.config["voice_id"], self.config["model"], all_audio)
            
        finally:
            if ws and not use_persistent:
                try:
                    await ws.send(json.dumps({"event": "task_finish"}))
                    await ws.close()
                except Exception:
                    pass
    
    async def _generate_parallel_sentences(
        self,
        sentences: List[str],
        on_chunk: Optional[Callable[[TTSChunk], None]]
    ) -> AsyncGenerator[TTSChunk, None]:
        """Generate TTS for multiple sentences in parallel."""
        
        async def generate_sentence(sentence: str, index: int) -> Tuple[int, bytes, str]:
            """Generate audio for a single sentence."""
            audio = await self._generate_sentence_sync(sentence)
            return index, audio, sentence
        
        # Process all sentences in parallel
        tasks = [generate_sentence(sent, i) for i, sent in enumerate(sentences)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort by original index and yield
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"[TTS] Error generating sentence: {result}")
                continue
            valid_results.append(result)
        
        valid_results.sort(key=lambda x: x[0])
        
        for i, (index, audio, sentence) in enumerate(valid_results):
            chunk = TTSChunk(
                audio_bytes=audio,
                text=sentence,
                chunk_index=index,
                is_last=(i == len(valid_results) - 1)
            )
            
            if on_chunk:
                on_chunk(chunk)
            yield chunk
    
    async def _generate_sentence_sync(self, text: str) -> bytes:
        """Generate audio for a single sentence (used for parallel processing)."""
        # Check cache
        if self.cache:
            cached = self.cache.get(text, self.config["voice_id"], self.config["model"])
            if cached:
                return cached
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sentence_blocking,
            text
        )
    
    def _generate_sentence_blocking(self, text: str) -> bytes:
        """Blocking TTS generation for thread pool."""
        return asyncio.run(self._generate_sentence_async(text))
    
    async def _generate_sentence_async(self, text: str) -> bytes:
        """Async TTS generation for a single sentence."""
        ws = None
        audio_data = b""
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            ssl_context = self._create_ssl_context()
            ws = await websockets.connect(
                self.WS_URL,
                additional_headers=headers,
                ssl=ssl_context
            )
            
            # Wait for connection
            response = await ws.recv()
            connected = json.loads(response)
            if connected.get("event") != "connected_success":
                return b""
            
            # Start task
            start_msg = {
                "event": "task_start",
                "model": self.config["model"],
                "voice_setting": {
                    "voice_id": self.config["voice_id"],
                    **self.config["voice_setting"]
                },
                "audio_setting": self.config["audio_setting"],
                "pronunciation_dict": self.config["pronunciation_dict"]
            }
            if self.config.get("language_boost"):
                start_msg["language_boost"] = self.config["language_boost"]
            
            await ws.send(json.dumps(start_msg))
            response = json.loads(await ws.recv())
            if response.get("event") != "task_started":
                return b""
            
            # Send text
            await ws.send(json.dumps({"event": "task_continue", "text": text}))
            
            # Collect audio
            while True:
                response = json.loads(await ws.recv())
                
                if response.get("event") == "task_failed":
                    break
                
                if "data" in response and response["data"]:
                    audio_hex = response["data"].get("audio", "")
                    if audio_hex:
                        audio_data += bytes.fromhex(audio_hex)
                
                if response.get("is_final", False):
                    break
            
            # Cache result
            if self.cache and audio_data:
                self.cache.set(text, self.config["voice_id"], self.config["model"], audio_data)
            
        finally:
            if ws:
                try:
                    await ws.send(json.dumps({"event": "task_finish"}))
                    await ws.close()
                except Exception:
                    pass
        
        return audio_data
    
    async def close(self):
        """Close persistent WebSocket and cleanup."""
        if self._persistent_ws:
            try:
                await self._persistent_ws.close()
            except Exception:
                pass
            self._persistent_ws = None
        
        self._executor.shutdown(wait=False)


# Global instance for reuse
_tts_instance: Optional[OptimizedMinimaxTTS] = None


def get_tts_instance(
    api_key: str,
    model: str = "speech-2.8-turbo",
    voice_id: str = "Malay_male_1_v1",
    language_boost: str = "Malay"
) -> OptimizedMinimaxTTS:
    """Get or create global TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = OptimizedMinimaxTTS(
            api_key=api_key,
            model=model,
            voice_id=voice_id,
            language_boost=language_boost
        )
    return _tts_instance
