"""
Lip Sync Analyzer for VTube Studio
Analyzes audio data to generate mouth movement parameters.
"""

import io
from typing import List, Tuple, Optional
import struct


class LipSyncAnalyzer:
    """
    Analyzes audio waveforms to generate lip sync data.
    Converts audio amplitude to mouth open values for Live2D models.
    """
    
    # Use the default VTube Studio mouth tracking parameter
    # This is a standard tracking parameter (input) that controls mouth opening
    # In VTube Studio: Ensure "MouthOpen" (Input) is bound to "ParamMouthOpenY" (Output)
    PARAM_NAME = "MouthOpen"
    
    def __init__(self, 
                 target_fps: int = 30,
                 smoothing: float = 0.3,
                 sensitivity: float = 3.0,
                 min_threshold: float = 0.02):
        """
        Initialize the lip sync analyzer.
        
        Args:
            target_fps: Frame rate for mouth updates (default: 30)
            smoothing: Smoothing factor to reduce jitter (0.0-1.0)
            sensitivity: Amplitude multiplier
            min_threshold: Minimum amplitude to register as mouth movement
        """
        self.target_fps = target_fps
        self.smoothing = smoothing
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        self._previous_value = 0.0
        
    def analyze_wav_bytes(self, wav_data: bytes) -> List[Tuple[float, float]]:
        """
        Analyze WAV audio bytes and generate lip sync data.
        
        Args:
            wav_data: WAV file as bytes
            
        Returns:
            List of (timestamp_seconds, mouth_value) tuples
        """
        try:
            # Parse WAV header and extract audio data
            sample_rate, audio_samples = self._parse_wav(wav_data)
            
            if sample_rate is None or audio_samples is None:
                print("[LipSync] Error: Could not parse WAV data")
                return []
            
            return self._analyze_samples(audio_samples, sample_rate)
            
        except Exception as e:
            print(f"[LipSync] Error analyzing audio: {e}")
            return []
    
    def _parse_wav(self, wav_data: bytes) -> Tuple[Optional[int], Optional[List[float]]]:
        """
        Parse WAV file and extract audio samples.
        
        Args:
            wav_data: WAV file as bytes
            
        Returns:
            Tuple of (sample_rate, samples_list) or (None, None) on error
        """
        try:
            # Use io.BytesIO to handle the WAV data
            with io.BytesIO(wav_data) as wav_file:
                # Read RIFF header
                riff = wav_file.read(4)
                if riff != b'RIFF':
                    print(f"[LipSync] Invalid WAV: Expected RIFF, got {riff}")
                    return None, None
                
                # Skip file size
                wav_file.read(4)
                
                # Read WAVE header
                wave = wav_file.read(4)
                if wave != b'WAVE':
                    print(f"[LipSync] Invalid WAV: Expected WAVE, got {wave}")
                    return None, None
                
                # Read chunks until we find fmt and data
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
                    print("[LipSync] Invalid WAV: Missing fmt or data chunk")
                    return None, None
                
                # Parse fmt chunk
                audio_format = struct.unpack('<H', fmt_chunk[0:2])[0]
                num_channels = struct.unpack('<H', fmt_chunk[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_chunk[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_chunk[14:16])[0]
                
                # Convert samples to float
                samples = []
                
                if bits_per_sample == 16:
                    # 16-bit samples
                    num_samples = len(data_chunk) // (2 * num_channels)
                    for i in range(num_samples):
                        # For stereo, average channels
                        sample_sum = 0
                        for ch in range(num_channels):
                            offset = (i * num_channels + ch) * 2
                            sample = struct.unpack('<h', data_chunk[offset:offset+2])[0]
                            sample_sum += sample
                        samples.append(sample_sum / num_channels / 32768.0)
                        
                elif bits_per_sample == 8:
                    # 8-bit samples (unsigned)
                    num_samples = len(data_chunk) // num_channels
                    for i in range(num_samples):
                        sample_sum = 0
                        for ch in range(num_channels):
                            offset = i * num_channels + ch
                            sample = data_chunk[offset] - 128
                            sample_sum += sample
                        samples.append(sample_sum / num_channels / 128.0)
                        
                else:
                    print(f"[LipSync] Unsupported bits per sample: {bits_per_sample}")
                    return None, None
                
                return sample_rate, samples
                
        except Exception as e:
            print(f"[LipSync] WAV parsing error: {e}")
            return None, None
    
    def _analyze_samples(self, audio: List[float], sample_rate: int) -> List[Tuple[float, float]]:
        """
        Analyze audio samples and generate lip sync data.
        
        Args:
            audio: Audio samples as list of floats
            sample_rate: Audio sample rate in Hz
            
        Returns:
            List of (timestamp_seconds, mouth_value) tuples
        """
        # Calculate samples per frame
        samples_per_frame = int(sample_rate / self.target_fps)
        num_frames = len(audio) // samples_per_frame
        
        if num_frames == 0:
            return []
        
        results = []
        previous = 0.0
        
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
                # Normalize and apply sensitivity
                value = min(1.0, rms * self.sensitivity)
            
            # Apply smoothing
            value = previous * self.smoothing + value * (1 - self.smoothing)
            previous = value
            
            # Calculate timestamp
            timestamp = i / self.target_fps
            
            results.append((timestamp, value))
        
        return results
    
    def get_mouth_parameters(self, mouth_value: float) -> List[dict]:
        """
        Convert mouth value to VTS parameter format.
        
        Args:
            mouth_value: Mouth open value (0.0 to 1.0)
            
        Returns:
            List of parameter dicts for VTS
        """
        # Convert to native Python float for JSON serialization
        value = float(mouth_value)
        return [{"id": self.PARAM_NAME, "value": value, "weight": 1.0}]
    
    def reset(self):
        """Reset the analyzer state (smoothing memory)."""
        self._previous_value = 0.0


class LipSyncPlayer:
    """
    Plays back lip sync data by sending timed parameter updates.
    Coordinates with audio playback for synchronized mouth movement.
    """
    
    def __init__(self, analyzer: Optional[LipSyncAnalyzer] = None):
        """
        Initialize the lip sync player.
        
        Args:
            analyzer: LipSyncAnalyzer instance (creates default if None)
        """
        self.analyzer = analyzer or LipSyncAnalyzer()
        self._current_playback = None
        self._stop_flag = False
        
    async def play_lip_sync(self, vts_connector, lip_sync_data: List[Tuple[float, float]],
                            playback_speed: float = 1.0):
        """
        Play lip sync data with proper timing.
        
        Args:
            vts_connector: VTSConnector instance
            lip_sync_data: List of (timestamp, mouth_value) from analyzer
            playback_speed: Audio playback speed multiplier
        """
        import asyncio
        
        if not lip_sync_data:
            print("[LipSync] No lip sync data to play")
            return
            
        self._stop_flag = False
        loop = asyncio.get_running_loop()  # Get the loop running this coroutine
        start_time = loop.time()
        
        print(f"[LipSync] Playing {len(lip_sync_data)} frames, speed={playback_speed}")
        
        frame_count = 0
        for timestamp, mouth_value in lip_sync_data:
            if self._stop_flag:
                print("[LipSync] Stopped by flag")
                break
            
            # Adjust timestamp for playback speed
            adjusted_timestamp = timestamp / playback_speed
                
            # Wait until it's time to send this frame
            current_time = loop.time() - start_time
            wait_time = adjusted_timestamp - current_time
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Send mouth parameter
            params = self.analyzer.get_mouth_parameters(mouth_value)
            success = await vts_connector.set_parameters(params)
            
            # Log every 30 frames for debugging
            if frame_count % 30 == 0:
                print(f"[LipSync] Frame {frame_count}: value={mouth_value:.3f}, param={self.analyzer.PARAM_NAME}, success={success}")
            frame_count += 1
        
        # Close mouth when done
        params = self.analyzer.get_mouth_parameters(0.0)
        await vts_connector.set_parameters(params)
        print(f"[LipSync] Completed {frame_count} frames")
    
    def stop(self):
        """Stop current lip sync playback."""
        self._stop_flag = True


# Global instances
_analyzer: Optional[LipSyncAnalyzer] = None
_player: Optional[LipSyncPlayer] = None


def get_analyzer() -> LipSyncAnalyzer:
    """Get or create the global lip sync analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LipSyncAnalyzer()
    return _analyzer


def get_player() -> LipSyncPlayer:
    """Get or create the global lip sync player."""
    global _player
    if _player is None:
        _player = LipSyncPlayer(get_analyzer())
    return _player