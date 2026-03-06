"""
Audio Converter for VTube Studio Lip Sync
Converts MP3 audio to WAV format for amplitude analysis.
"""

import io
import subprocess
import tempfile
import os
from typing import Optional, Tuple
from pathlib import Path


class AudioConverter:
    """
    Converts audio formats for lip sync analysis.
    Uses ffmpeg for MP3 to WAV conversion.
    """
    
    # Default sample rate for WAV output
    DEFAULT_SAMPLE_RATE = 16000  # Lower sample rate is sufficient for lip sync
    DEFAULT_CHANNELS = 1  # Mono is sufficient for lip sync
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize audio converter.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable (default: "ffmpeg" in PATH)
        """
        self.ffmpeg_path = ffmpeg_path
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            print("[AudioConverter] Warning: ffmpeg not found. MP3 to WAV conversion will not work.")
            print("[AudioConverter] Please install ffmpeg: https://ffmpeg.org/download.html")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if converter is available."""
        return self._ffmpeg_available
    
    def convert_mp3_to_wav(
        self, 
        mp3_data: bytes, 
        sample_rate: int = None,
        channels: int = None
    ) -> Optional[bytes]:
        """
        Convert MP3 audio bytes to WAV format.
        
        Args:
            mp3_data: MP3 audio as bytes
            sample_rate: Target sample rate (default: 16000)
            channels: Target channels (default: 1 = mono)
            
        Returns:
            WAV audio as bytes, or None if conversion failed
        """
        if not self._ffmpeg_available:
            return None
            
        sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        channels = channels or self.DEFAULT_CHANNELS
        
        try:
            # Create temp files for input and output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
                mp3_file.write(mp3_data)
                mp3_path = mp3_file.name
            
            wav_path = mp3_path.replace(".mp3", ".wav")
            
            try:
                # Run ffmpeg conversion
                result = subprocess.run(
                    [
                        self.ffmpeg_path,
                        "-y",  # Overwrite output file
                        "-i", mp3_path,  # Input file
                        "-ar", str(sample_rate),  # Sample rate
                        "-ac", str(channels),  # Channels
                        "-f", "wav",  # Output format
                        wav_path  # Output file
                    ],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"[AudioConverter] ffmpeg error: {result.stderr}")
                    return None
                
                # Read WAV output
                with open(wav_path, "rb") as wav_file:
                    wav_data = wav_file.read()
                
                return wav_data
                
            finally:
                # Cleanup temp files
                if os.path.exists(mp3_path):
                    os.unlink(mp3_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                    
        except Exception as e:
            print(f"[AudioConverter] Conversion error: {e}")
            return None
    
    def get_audio_duration(self, audio_data: bytes, format: str = "mp3") -> Optional[float]:
        """
        Get duration of audio in seconds.
        
        Args:
            audio_data: Audio bytes
            format: Audio format (mp3, wav, etc.)
            
        Returns:
            Duration in seconds, or None if failed
        """
        if not self._ffmpeg_available:
            return None
            
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as audio_file:
                audio_file.write(audio_data)
                audio_path = audio_file.name
            
            try:
                # Use ffprobe to get duration
                result = subprocess.run(
                    [
                        self.ffmpeg_path.replace("ffmpeg", "ffprobe"),
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        audio_path
                    ],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return float(result.stdout.strip())
                return None
                
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            print(f"[AudioConverter] Duration detection error: {e}")
            return None


# Global converter instance
_converter: Optional[AudioConverter] = None


def get_converter() -> AudioConverter:
    """Get or create the global audio converter instance."""
    global _converter
    if _converter is None:
        _converter = AudioConverter()
    return _converter