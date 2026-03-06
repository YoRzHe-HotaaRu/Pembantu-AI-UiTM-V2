"""
Minimax TTS Service Module
Handles Text-to-Speech synthesis using Minimax WebSocket API
"""

import asyncio
import websockets
import json
import ssl
from typing import Optional, Dict, Any


class MinimaxTTSError(Exception):
    """Custom exception for Minimax TTS errors"""
    def __init__(self, message: str, code: Optional[int] = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class MinimaxTTS:
    """
    Minimax Text-to-Speech client using WebSocket API.
    
    Configuration:
        - Model: speech-2.8-turbo
        - Voice: Malay_male_1_v1
        - Language: Malay (with language_boost)
        - Format: MP3
    """
    
    WS_URL = "wss://api.minimax.io/ws/v1/t2a_v2"
    
    # Default settings for UITM chatbot
    DEFAULT_CONFIG = {
        "model": "speech-2.8-turbo",
        "voice_id": "Malay_male_1_v1",
        "language_boost": "Malay",
        "pronunciation_dict": {"tone": ["uitm/UITM"]},
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
        language_boost: str = None
    ):
        """
        Initialize Minimax TTS client.
        
        Args:
            api_key: Minimax API key
            model: TTS model (default: speech-2.8-turbo)
            voice_id: Voice ID (default: Malay_male_1_v1)
            language_boost: Language for enhanced recognition (default: Malay)
        """
        self.api_key = api_key
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Override defaults if provided
        if model:
            self.config["model"] = model
        if voice_id:
            self.config["voice_id"] = voice_id
        if language_boost:
            self.config["language_boost"] = language_boost
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for WebSocket connection"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context
    
    async def _connect(self) -> websockets.WebSocketClientProtocol:
        """
        Establish WebSocket connection to Minimax API.
        
        Returns:
            WebSocket connection object
            
        Raises:
            MinimaxTTSError: If connection fails
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        ssl_context = self._create_ssl_context()
        
        try:
            ws = await websockets.connect(
                self.WS_URL,
                additional_headers=headers,
                ssl=ssl_context
            )
            
            # Wait for connection confirmation
            response = await ws.recv()
            connected = json.loads(response)
            
            if connected.get("event") != "connected_success":
                raise MinimaxTTSError(
                    f"Connection failed: {connected.get('base_resp', {}).get('status_msg', 'Unknown error')}",
                    code=connected.get("base_resp", {}).get("status_code")
                )
            
            return ws
            
        except websockets.exceptions.WebSocketException as e:
            raise MinimaxTTSError(f"WebSocket connection error: {str(e)}")
        except json.JSONDecodeError as e:
            raise MinimaxTTSError(f"Invalid JSON response: {str(e)}")
    
    async def _start_task(self, ws: websockets.WebSocketClientProtocol) -> bool:
        """
        Send task_start event and wait for confirmation.
        
        Args:
            ws: WebSocket connection
            
        Returns:
            True if task started successfully
            
        Raises:
            MinimaxTTSError: If task fails to start
        """
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
        
        # Add language_boost if specified
        if self.config.get("language_boost"):
            start_msg["language_boost"] = self.config["language_boost"]
        
        await ws.send(json.dumps(start_msg))
        
        response = json.loads(await ws.recv())
        
        if response.get("event") != "task_started":
            base_resp = response.get("base_resp", {})
            raise MinimaxTTSError(
                f"Task start failed: {base_resp.get('status_msg', 'Unknown error')}",
                code=base_resp.get("status_code")
            )
        
        return True
    
    async def _synthesize(
        self,
        ws: websockets.WebSocketClientProtocol,
        text: str
    ) -> bytes:
        """
        Send text for synthesis and collect audio chunks.
        
        Args:
            ws: WebSocket connection
            text: Text to synthesize
            
        Returns:
            Complete audio data as bytes
            
        Raises:
            MinimaxTTSError: If synthesis fails
        """
        # Send text for synthesis
        continue_msg = {
            "event": "task_continue",
            "text": text
        }
        await ws.send(json.dumps(continue_msg))
        
        # Collect audio chunks
        audio_data = b""
        
        while True:
            try:
                response = json.loads(await ws.recv())
                event = response.get("event")
                
                # Check for errors
                if event == "task_failed":
                    base_resp = response.get("base_resp", {})
                    raise MinimaxTTSError(
                        f"Synthesis failed: {base_resp.get('status_msg', 'Unknown error')}",
                        code=base_resp.get("status_code")
                    )
                
                # Collect audio data
                if "data" in response and response["data"]:
                    audio_hex = response["data"].get("audio", "")
                    if audio_hex:
                        audio_data += bytes.fromhex(audio_hex)
                
                # Check if synthesis is complete
                if response.get("is_final", False):
                    break
                    
            except json.JSONDecodeError as e:
                raise MinimaxTTSError(f"Invalid JSON response during synthesis: {str(e)}")
        
        return audio_data
    
    async def _finish_task(self, ws: websockets.WebSocketClientProtocol) -> None:
        """
        Send task_finish event and close connection.
        
        Args:
            ws: WebSocket connection
        """
        try:
            await ws.send(json.dumps({"event": "task_finish"}))
            await ws.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
    async def generate_audio_async(self, text: str) -> bytes:
        """
        Generate audio from text asynchronously.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (MP3 format)
            
        Raises:
            MinimaxTTSError: If synthesis fails
        """
        if not text or not text.strip():
            raise MinimaxTTSError("Text cannot be empty")
        
        # Truncate text if too long (Minimax limit: 10,000 characters)
        if len(text) > 10000:
            text = text[:9997] + "..."
        
        ws = None
        try:
            # Establish connection
            ws = await self._connect()
            
            # Start TTS task
            await self._start_task(ws)
            
            # Synthesize audio
            audio_data = await self._synthesize(ws, text)
            
            return audio_data
            
        except MinimaxTTSError:
            raise
        except Exception as e:
            raise MinimaxTTSError(f"Unexpected error: {str(e)}")
        finally:
            if ws:
                await self._finish_task(ws)
    
    def generate_audio(self, text: str) -> bytes:
        """
        Generate audio from text synchronously.
        
        This is a synchronous wrapper around the async WebSocket implementation.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (MP3 format)
            
        Raises:
            MinimaxTTSError: If synthesis fails
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context, need to run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.generate_audio_async(text))
                return future.result()
        else:
            # No running loop, use asyncio.run
            return asyncio.run(self.generate_audio_async(text))


# Convenience function for quick TTS
def text_to_speech(
    text: str,
    api_key: str,
    voice_id: str = "Malay_male_1_v1",
    model: str = "speech-2.8-turbo",
    language_boost: str = "Malay"
) -> bytes:
    """
    Convert text to speech using Minimax TTS.
    
    Args:
        text: Text to convert to speech
        api_key: Minimax API key
        voice_id: Voice ID (default: Malay_male_1_v1)
        model: TTS model (default: speech-2.8-turbo)
        language_boost: Language for enhanced recognition (default: Malay)
        
    Returns:
        Audio data as bytes (MP3 format)
    """
    tts = MinimaxTTS(
        api_key=api_key,
        model=model,
        voice_id=voice_id,
        language_boost=language_boost
    )
    return tts.generate_audio(text)