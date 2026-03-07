"""
Parallel Lip Sync Analyzer
Processes audio chunks in real-time using threading for CPU-bound analysis.
"""

import asyncio
import io
import struct
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class ParallelLipSyncAnalyzer:
    """
    Lip sync analyzer with parallel processing support.
    Can analyze audio chunks as they arrive (streaming) or full audio.
    """
    
    def __init__(self, 
                 target_fps: int = 30,
                 smoothing: float = 0.3,
                 sensitivity: float = 3.0,
                 min_threshold: float = 0.02,
                 max_workers: int = 4):
        """
        Initialize parallel lip sync analyzer.
        
        Args:
            target_fps: Frame rate for mouth updates
            smoothing: Smoothing factor to reduce jitter
            sensitivity: Amplitude multiplier
            min_threshold: Minimum amplitude threshold
            max_workers: Number of threads for parallel processing
        """
        self.target_fps = target_fps
        self.smoothing = smoothing
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def analyze_wav_bytes_parallel(
        self, 
        wav_data: bytes,
        on_chunk: Optional[Callable[[List[Tuple[float, float]]], None]] = None
    ) -> List[Tuple[float, float]]:
        """
        Analyze WAV audio with parallel processing.
        
        Args:
            wav_data: WAV file as bytes
            on_chunk: Optional callback for partial results
            
        Returns:
            List of (timestamp_seconds, mouth_value) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._analyze_wav_blocking,
            wav_data,
            on_chunk
        )
    
    def _analyze_wav_blocking(
        self,
        wav_data: bytes,
        on_chunk: Optional[Callable[[List[Tuple[float, float]]], None]]
    ) -> List[Tuple[float, float]]:
        """Blocking WAV analysis for thread pool."""
        try:
            sample_rate, audio_samples = self._parse_wav(wav_data)
            
            if sample_rate is None or audio_samples is None:
                return []
            
            return self._analyze_samples(audio_samples, sample_rate, on_chunk)
            
        except Exception as e:
            print(f"[ParallelLipSync] Error analyzing audio: {e}")
            return []
    
    def _parse_wav(self, wav_data: bytes) -> Tuple[Optional[int], Optional[List[float]]]:
        """Parse WAV file and extract audio samples."""
        try:
            with io.BytesIO(wav_data) as wav_file:
                # Read RIFF header
                riff = wav_file.read(4)
                if riff != b'RIFF':
                    return None, None
                
                wav_file.read(4)  # Skip file size
                
                # Read WAVE header
                wave = wav_file.read(4)
                if wave != b'WAVE':
                    return None, None
                
                # Read chunks
                fmt_chunk = None
                data_chunk = None
                
                while True:
                    chunk_id = wav_file.read(4)
                    if len(chunk_id) < 4:
                        break
                    
                    chunk_size = struct.unpack('<I', wav_file.read(4))[0]
                    
                    if chunk_id == b'fmt ':
                        fmt_chunk = wav_file.read(chunk_size)
                    elif chunk_id == b'data':
                        data_chunk = wav_file.read(chunk_size)
                    else:
                        wav_file.read(chunk_size)
                
                if fmt_chunk is None or data_chunk is None:
                    return None, None
                
                # Parse fmt chunk
                num_channels = struct.unpack('<H', fmt_chunk[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_chunk[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_chunk[14:16])[0]
                
                # Convert samples to float
                samples = []
                
                if bits_per_sample == 16:
                    num_samples = len(data_chunk) // (2 * num_channels)
                    for i in range(num_samples):
                        sample_sum = 0
                        for ch in range(num_channels):
                            offset = (i * num_channels + ch) * 2
                            sample = struct.unpack('<h', data_chunk[offset:offset+2])[0]
                            sample_sum += sample
                        samples.append(sample_sum / num_channels / 32768.0)
                        
                elif bits_per_sample == 8:
                    num_samples = len(data_chunk) // num_channels
                    for i in range(num_samples):
                        sample_sum = 0
                        for ch in range(num_channels):
                            offset = i * num_channels + ch
                            sample = data_chunk[offset] - 128
                            sample_sum += sample
                        samples.append(sample_sum / num_channels / 128.0)
                else:
                    return None, None
                
                return sample_rate, samples
                
        except Exception as e:
            print(f"[ParallelLipSync] WAV parsing error: {e}")
            return None, None
    
    def _analyze_samples(
        self, 
        audio: List[float], 
        sample_rate: int,
        on_chunk: Optional[Callable[[List[Tuple[float, float]]], None]] = None
    ) -> List[Tuple[float, float]]:
        """
        Analyze audio samples and generate lip sync data.
        
        Args:
            audio: Audio samples as list of floats
            sample_rate: Audio sample rate in Hz
            on_chunk: Optional callback for partial results
            
        Returns:
            List of (timestamp_seconds, mouth_value) tuples
        """
        samples_per_frame = int(sample_rate / self.target_fps)
        num_frames = len(audio) // samples_per_frame
        
        if num_frames == 0:
            return []
        
        results = []
        previous = 0.0
        
        # Process in chunks for callback support
        chunk_size = 30  # 1 second chunks at 30fps
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            chunk = audio[start:end]
            
            # Calculate RMS amplitude
            rms = (sum(s ** 2 for s in chunk) / len(chunk)) ** 0.5
            
            # Apply threshold
            if rms < self.min_threshold:
                value = 0.0
            else:
                value = min(1.0, rms * self.sensitivity)
            
            # Apply smoothing
            value = previous * self.smoothing + value * (1 - self.smoothing)
            previous = value
            
            timestamp = i / self.target_fps
            results.append((timestamp, value))
            
            # Call callback every chunk
            if on_chunk and (i + 1) % chunk_size == 0:
                chunk_start = (i // chunk_size) * chunk_size
                on_chunk(results[chunk_start:i+1])
        
        return results
    
    async def analyze_chunk_async(
        self,
        audio_chunk: bytes,
        sample_rate: int = 32000,
        start_time: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        Analyze a small audio chunk asynchronously.
        Useful for real-time streaming analysis.
        
        Args:
            audio_chunk: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate of audio
            start_time: Starting timestamp offset
            
        Returns:
            List of (timestamp, mouth_value) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._analyze_chunk_blocking,
            audio_chunk,
            sample_rate,
            start_time
        )
    
    def _analyze_chunk_blocking(
        self,
        audio_chunk: bytes,
        sample_rate: int,
        start_time: float
    ) -> List[Tuple[float, float]]:
        """Blocking chunk analysis."""
        # Convert bytes to samples
        samples = []
        for i in range(0, len(audio_chunk), 2):
            if i + 1 < len(audio_chunk):
                sample = struct.unpack('<h', audio_chunk[i:i+2])[0]
                samples.append(sample / 32768.0)
        
        if not samples:
            return []
        
        samples_per_frame = int(sample_rate / self.target_fps)
        num_frames = len(samples) // samples_per_frame
        
        results = []
        previous = 0.0
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame_samples = samples[start:end]
            
            rms = (sum(s ** 2 for s in frame_samples) / len(frame_samples)) ** 0.5
            
            if rms < self.min_threshold:
                value = 0.0
            else:
                value = min(1.0, rms * self.sensitivity)
            
            value = previous * self.smoothing + value * (1 - self.smoothing)
            previous = value
            
            timestamp = start_time + (i / self.target_fps)
            results.append((timestamp, value))
        
        return results
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)


# Global instance
_parallel_analyzer: Optional[ParallelLipSyncAnalyzer] = None


def get_parallel_analyzer() -> ParallelLipSyncAnalyzer:
    """Get or create global parallel analyzer instance."""
    global _parallel_analyzer
    if _parallel_analyzer is None:
        _parallel_analyzer = ParallelLipSyncAnalyzer()
    return _parallel_analyzer
