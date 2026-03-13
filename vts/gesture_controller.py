"""
Gesture Controller for VTube Studio
Manages natural, organic head/body/brow/eye movements during speech.
Uses layered noise-like motion, smooth easing, and multi-channel blending
to produce human-like animation instead of robotic sine waves.
"""

import asyncio
import random
import math
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum


class EmotionType(Enum):
    """Emotion types for gesture mapping."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    THINKING = "thinking"
    SURPRISED = "surprised"
    CONFUSED = "confused"


@dataclass
class GestureConfig:
    """Configuration for natural gesture animations."""

    # --- Organic drift (slow wandering baseline) ---
    drift_amplitude: float = 8.0         # Max degrees for slow drift
    drift_speed: float = 0.2             # Slow wandering Hz

    # --- Speech rhythm (the "talking sway") ---
    rhythm_amplitude_y: float = 6.0      # Up-down bobbing (degrees)
    rhythm_amplitude_x: float = 4.0      # Left-right sway (degrees)
    rhythm_amplitude_z: float = 3.0      # Lean/tilt sway (degrees)
    rhythm_base_speed: float = 2.2       # Base speech rhythm Hz

    # --- Emphasis gestures (nods, tilts on punctuation) ---
    emphasis_enabled: bool = True
    emphasis_nod_strength: float = 12.0  # Strong nod (degrees)
    emphasis_tilt_strength: float = 8.0  # Strong tilt (degrees)
    emphasis_easing_speed: float = 10.0  # How fast emphasis snaps in/out
    emphasis_tilt_chance: float = 0.4    # Chance of tilt on emphasis (0-1)

    # --- Micro-expressions ---
    brow_emphasis_strength: float = 0.6  # Brow raise on emphasis (0-1)
    eye_drift_amplitude: float = 0.3     # Eye movement range
    eye_drift_speed: float = 0.5         # Eye drift Hz

    # --- Transitions ---
    start_ramp_duration: float = 0.5     # Seconds to ramp up at speech start
    stop_ramp_duration: float = 0.5      # Seconds to ramp down at speech end

    # --- Amplitude envelope (intensity varies over time) ---
    envelope_speed: float = 0.12         # Hz for amplitude modulation
    envelope_min: float = 0.6            # Minimum amplitude multiplier
    envelope_max: float = 1.0            # Maximum amplitude multiplier

    # --- Engagement tracking ---
    engagement_enabled: bool = True
    engagement_range: float = 10.0

    # --- Emotion-based positions ---
    emotion_positions: Dict[EmotionType, Dict[str, float]] = None

    def __post_init__(self):
        if self.emotion_positions is None:
            self.emotion_positions = {
                EmotionType.NEUTRAL: {"x": 0, "y": 0, "z": 0},
                EmotionType.HAPPY: {"x": 0, "y": 5, "z": 3},
                EmotionType.SAD: {"x": 0, "y": -8, "z": -3},
                EmotionType.EXCITED: {"x": 0, "y": 3, "z": 5},
                EmotionType.THINKING: {"x": -5, "y": 3, "z": -8},
                EmotionType.SURPRISED: {"x": 0, "y": -5, "z": 0},
                EmotionType.CONFUSED: {"x": 5, "y": 3, "z": -5},
            }


# Irrational frequency ratios that never repeat perfectly
_PHI = 1.6180339887       # Golden ratio
_SQRT2 = 1.4142135624     # √2
_SQRT3 = 1.7320508076     # √3


def _organic_noise(t: float, base_speed: float, amplitude: float = 1.0) -> float:
    """
    Generate organic, non-repeating noise using layered sine waves
    at irrational frequency ratios. Produces motion that feels natural.
    """
    # Layer 1: Primary motion
    v = math.sin(t * base_speed * 2.0 * math.pi) * 0.45
    # Layer 2: Golden ratio offset — never aligns with layer 1
    v += math.sin(t * base_speed * _PHI * 2.0 * math.pi) * 0.28
    # Layer 3: √2 ratio — different phase drift
    v += math.sin(t * base_speed * _SQRT2 * 2.0 * math.pi + 0.7) * 0.18
    # Layer 4: Very slow drift at √3 ratio
    v += math.sin(t * base_speed * 0.3 * _SQRT3 * 2.0 * math.pi + 2.1) * 0.09

    return v * amplitude


def _ease_toward(current: float, target: float, speed: float, dt: float) -> float:
    """
    Exponential ease toward a target value. Produces smooth, natural
    transitions instead of instant snaps.
    """
    factor = 1.0 - math.exp(-speed * dt)
    return current + (target - current) * factor


def _smoothstep(x: float) -> float:
    """Hermite smoothstep for natural ramp curves."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _amplitude_envelope(t: float, speed: float, min_val: float, max_val: float) -> float:
    """
    Slow amplitude modulation so gesture intensity varies over time
    instead of staying constant.
    """
    raw = _organic_noise(t, speed, 1.0)
    normalized = (raw + 1.0) * 0.5  # 0 to 1
    return min_val + normalized * (max_val - min_val)


class GestureController:
    """
    Manages expressive, natural gestures during speech.
    
    Architecture:
    - FaceAngleX/Y/Z are the ONLY head/body input parameters needed.
      VTS natively maps these to BOTH head rotation (ParamAngleX/Y/Z) 
      AND body rotation (ParamBodyAngleX/Y/Z) via the model's parameter bindings.
    - Brow and eye parameters are sent separately for micro-expressions.
    - All values are computed per-frame and sent in a single set_parameters call
      (merged with lip sync mouth value by LipSyncPlayer).
    """

    # Punctuation that triggers emphasis gestures
    EMPHASIS_PUNCTUATION = ['!', '?', '.']
    PAUSE_PUNCTUATION = [',', ';', ':']

    def __init__(self, vts_connector, config: Optional[GestureConfig] = None):
        self.vts = vts_connector
        self.config = config or GestureConfig()

        # State
        self._current_emotion = EmotionType.NEUTRAL
        self._is_speaking = False
        self._is_ramping_down = False
        self._speech_start_time: float = 0
        self._speech_stop_time: float = 0
        self._current_text = ""

        # --- Output values (sent to VTS each frame) ---
        self._head_x = 0.0
        self._head_y = 0.0
        self._head_z = 0.0
        self._brow_left = 0.0
        self._brow_right = 0.0
        self._eye_x = 0.0
        self._eye_y = 0.0

        # --- Emphasis state (smoothed toward targets) ---
        self._emphasis_target_y = 0.0
        self._emphasis_target_z = 0.0
        self._emphasis_brow = 0.0
        self._emphasis_current_y = 0.0
        self._emphasis_current_z = 0.0
        self._emphasis_brow_current = 0.0

        # --- Emotion base position (smoothed) ---
        self._emotion_target_x = 0.0
        self._emotion_target_y = 0.0
        self._emotion_target_z = 0.0
        self._emotion_current_x = 0.0
        self._emotion_current_y = 0.0
        self._emotion_current_z = 0.0

        # --- Random per-session offsets for variety ---
        self._phase_offset_x = 0.0
        self._phase_offset_y = 0.0
        self._phase_offset_z = 0.0

        # --- Sway direction (changes periodically for variety) ---
        self._sway_direction = 1.0       # Flips between -1 and 1
        self._next_sway_change = 0.0     # Time of next direction change

        # --- Activity level ---
        self._activity_level = 0.0  # 0 = idle, 1 = fully talking

        # Tasks
        self._gesture_task: Optional[asyncio.Task] = None
        self._update_task: Optional[asyncio.Task] = None

    async def start_speaking(self, text: str = "", emotion: EmotionType = EmotionType.NEUTRAL):
        """Start speaking with natural gestures."""
        self._is_speaking = True
        self._is_ramping_down = False
        self._current_text = text
        self._speech_start_time = asyncio.get_event_loop().time()
        self._current_emotion = emotion

        # Randomize phase offsets so each speech session feels unique
        self._phase_offset_x = random.uniform(0, 100)
        self._phase_offset_y = random.uniform(0, 100)
        self._phase_offset_z = random.uniform(0, 100)

        # Randomize initial sway direction
        self._sway_direction = random.choice([-1.0, 1.0])
        self._next_sway_change = random.uniform(2.0, 5.0)

        # Set emotion base position
        emotion_pos = self.config.emotion_positions.get(emotion, {})
        self._emotion_target_x = emotion_pos.get("x", 0)
        self._emotion_target_y = emotion_pos.get("y", 0)
        self._emotion_target_z = emotion_pos.get("z", 0)

        # Start emphasis analyzer
        if self.config.emphasis_enabled and text:
            self._gesture_task = asyncio.create_task(self._emphasis_loop(text))

        print(f"[GestureController] Started speaking with emotion: {emotion.value}")

    async def stop_speaking(self):
        """Stop speaking and smoothly ramp down gestures."""
        self._is_speaking = False
        self._is_ramping_down = True
        self._speech_stop_time = asyncio.get_event_loop().time()

        # Cancel emphasis task
        if self._gesture_task:
            self._gesture_task.cancel()
            self._gesture_task = None

        # Reset all targets (will ease to zero via ramp-down)
        self._emphasis_target_y = 0.0
        self._emphasis_target_z = 0.0
        self._emphasis_brow = 0.0
        self._emotion_target_x = 0.0
        self._emotion_target_y = 0.0
        self._emotion_target_z = 0.0

        print("[GestureController] Stopped speaking (ramping down)")

    async def update_emotion(self, emotion: EmotionType):
        """Update emotion during speech — smoothly transitions."""
        self._current_emotion = emotion
        emotion_pos = self.config.emotion_positions.get(emotion, {})
        self._emotion_target_x = emotion_pos.get("x", 0)
        self._emotion_target_y = emotion_pos.get("y", 0)
        self._emotion_target_z = emotion_pos.get("z", 0)

    async def trigger_emphasis(self, strength: float = 1.0):
        """
        Trigger a natural emphasis gesture (nod + optional tilt + brow raise).
        Uses smooth easing instead of instant position jumps.
        """
        if not self._is_speaking:
            return

        # Add randomized strength variation (±25%)
        jitter = random.uniform(0.75, 1.25)
        actual_strength = strength * jitter

        # Nod target (downward)
        self._emphasis_target_y = -self.config.emphasis_nod_strength * actual_strength

        # Tilt on emphasis (configurable chance)
        if random.random() < self.config.emphasis_tilt_chance:
            direction = random.choice([-1, 1])
            self._emphasis_target_z = direction * self.config.emphasis_tilt_strength * actual_strength * 0.7

        # Brow raise
        self._emphasis_brow = min(1.0, self.config.brow_emphasis_strength * actual_strength)

        # Schedule spring-back with randomized timing
        delay = random.uniform(0.12, 0.28)
        asyncio.get_event_loop().call_later(delay, self._release_emphasis)

    def _release_emphasis(self):
        """Release emphasis targets back to zero (will ease smoothly)."""
        self._emphasis_target_y = 0.0
        self._emphasis_target_z = 0.0
        asyncio.get_event_loop().call_later(0.08, self._release_brow_emphasis)

    def _release_brow_emphasis(self):
        """Release brow emphasis."""
        self._emphasis_brow = 0.0

    async def trigger_tilt(self, direction: str = "random", strength: float = 1.0):
        """Trigger a head tilt gesture with smooth easing."""
        if not self._is_speaking:
            return

        if direction == "random":
            direction = random.choice(["left", "right"])

        jitter = random.uniform(0.8, 1.2)
        tilt_value = self.config.emphasis_tilt_strength * strength * jitter
        if direction == "right":
            tilt_value = -tilt_value

        self._emphasis_target_z = tilt_value

        delay = random.uniform(0.25, 0.45)
        asyncio.get_event_loop().call_later(delay, self._release_tilt)

    def _release_tilt(self):
        """Release tilt emphasis."""
        self._emphasis_target_z = 0.0

    async def _emphasis_loop(self, text: str):
        """Analyze text and trigger emphasis gestures with natural timing."""
        words = text.split()
        word_index = 0
        # Track next "random emphasis" interval
        next_random_emphasis = random.randint(3, 6)

        try:
            while self._is_speaking and word_index < len(words):
                word = words[word_index]

                # Strong emphasis on sentence-ending punctuation
                if any(p in word for p in self.EMPHASIS_PUNCTUATION):
                    await self.trigger_emphasis(strength=1.0)
                    await asyncio.sleep(random.uniform(0.20, 0.40))
                # Medium emphasis on clause-separating punctuation
                elif any(p in word for p in self.PAUSE_PUNCTUATION):
                    await self.trigger_emphasis(strength=0.6)
                    await asyncio.sleep(random.uniform(0.12, 0.25))
                # Periodic subtle emphasis for natural cadence
                elif word_index == next_random_emphasis:
                    strength = random.uniform(0.3, 0.6)
                    await self.trigger_emphasis(strength=strength)
                    next_random_emphasis = word_index + random.randint(3, 6)

                word_index += 1
                # Natural word timing with variation
                await asyncio.sleep(random.uniform(0.13, 0.25))
        except asyncio.CancelledError:
            pass

    def _compute_frame(self, t: float, dt: float):
        """
        Compute all animation channels for one frame.
        This is the core animation engine — called once per lip sync frame.

        Args:
            t: Time elapsed since speech start (seconds)
            dt: Delta time since last frame (seconds)
        """
        cfg = self.config

        # Clamp dt to avoid huge jumps from lag spikes
        dt = min(dt, 0.1)

        # --- Activity level (smooth ramp up/down) ---
        if self._is_speaking:
            ramp = min(1.0, t / cfg.start_ramp_duration) if cfg.start_ramp_duration > 0 else 1.0
            target_activity = _smoothstep(ramp)
        elif self._is_ramping_down:
            time_since_stop = asyncio.get_event_loop().time() - self._speech_stop_time
            ramp = min(1.0, time_since_stop / cfg.stop_ramp_duration) if cfg.stop_ramp_duration > 0 else 1.0
            target_activity = 1.0 - _smoothstep(ramp)
            if target_activity < 0.01:
                self._is_ramping_down = False
                target_activity = 0.0
        else:
            target_activity = 0.0

        self._activity_level = _ease_toward(self._activity_level, target_activity, 8.0, dt)
        activity = self._activity_level

        # --- Amplitude envelope (slow intensity variation) ---
        envelope = _amplitude_envelope(t, cfg.envelope_speed, cfg.envelope_min, cfg.envelope_max)

        # --- Sway direction changes (adds variety to movement pattern) ---
        if t > self._next_sway_change:
            self._sway_direction *= -1
            self._next_sway_change = t + random.uniform(2.5, 6.0)

        # --- Organic drift (slow wandering baseline) ---
        drift_x = _organic_noise(t + self._phase_offset_x, cfg.drift_speed, cfg.drift_amplitude)
        drift_y = _organic_noise(t + self._phase_offset_y, cfg.drift_speed * 0.7, cfg.drift_amplitude * 0.4)
        drift_z = _organic_noise(t + self._phase_offset_z, cfg.drift_speed * 0.5, cfg.drift_amplitude * 0.35) * self._sway_direction

        # --- Speech rhythm (the core "talking sway") ---
        rhythm_y = _organic_noise(t + self._phase_offset_y + 50.0, cfg.rhythm_base_speed, cfg.rhythm_amplitude_y)
        rhythm_x = _organic_noise(t + self._phase_offset_x + 50.0, cfg.rhythm_base_speed * 0.6, cfg.rhythm_amplitude_x) * self._sway_direction
        rhythm_z = _organic_noise(t + self._phase_offset_z + 50.0, cfg.rhythm_base_speed * 0.4, cfg.rhythm_amplitude_z)

        # --- Emphasis (smooth eased nod/tilt) ---
        self._emphasis_current_y = _ease_toward(
            self._emphasis_current_y, self._emphasis_target_y,
            cfg.emphasis_easing_speed, dt
        )
        self._emphasis_current_z = _ease_toward(
            self._emphasis_current_z, self._emphasis_target_z,
            cfg.emphasis_easing_speed, dt
        )
        self._emphasis_brow_current = _ease_toward(
            self._emphasis_brow_current, self._emphasis_brow,
            cfg.emphasis_easing_speed * 0.8, dt
        )

        # --- Emotion base position (smooth transition) ---
        self._emotion_current_x = _ease_toward(self._emotion_current_x, self._emotion_target_x, 3.0, dt)
        self._emotion_current_y = _ease_toward(self._emotion_current_y, self._emotion_target_y, 3.0, dt)
        self._emotion_current_z = _ease_toward(self._emotion_current_z, self._emotion_target_z, 3.0, dt)

        # --- Combine all head channels ---
        # Apply activity level and envelope to organic motion
        motion_scale = activity * envelope

        self._head_x = (
            self._emotion_current_x
            + (drift_x + rhythm_x) * motion_scale
        )
        self._head_y = (
            self._emotion_current_y
            + (drift_y + rhythm_y) * motion_scale
            + self._emphasis_current_y * activity
        )
        self._head_z = (
            self._emotion_current_z
            + (drift_z + rhythm_z) * motion_scale
            + self._emphasis_current_z * activity
        )

        # --- Brow micro-expressions ---
        brow_base = _organic_noise(t + 200.0, 0.35, 0.15) * activity
        target_brow = brow_base + self._emphasis_brow_current
        self._brow_left = _ease_toward(self._brow_left, target_brow, 8.0, dt)
        self._brow_right = _ease_toward(self._brow_right, target_brow * 0.85, 7.0, dt)

        # --- Eye drift ---
        target_eye_x = _organic_noise(t + 300.0, cfg.eye_drift_speed, cfg.eye_drift_amplitude) * activity
        target_eye_y = _organic_noise(t + 400.0, cfg.eye_drift_speed * 0.7, cfg.eye_drift_amplitude * 0.6) * activity
        self._eye_x = _ease_toward(self._eye_x, target_eye_x, 6.0, dt)
        self._eye_y = _ease_toward(self._eye_y, target_eye_y, 6.0, dt)

    def get_current_position(self) -> Dict[str, float]:
        """Get current head position (backward compatible)."""
        return {
            "x": self._head_x,
            "y": self._head_y,
            "z": self._head_z,
        }

    def get_all_parameters(self) -> List[Dict]:
        """
        Get all animation parameters for the current frame.
        Returns VTS-format parameter list for head, brow, and eye.
        
        NOTE: We only send FaceAngleX/Y/Z for head input. The VTS model's
        parameter bindings automatically route these to BOTH head rotation
        (ParamAngleX/Y/Z) AND body rotation (ParamBodyAngleX/Y/Z) — so
        body follow happens natively through the model, no duplicate params needed.
        """
        params = [
            # Head rotation — VTS routes these to both head AND body via model bindings
            {"id": "FaceAngleX", "value": float(self._head_x), "weight": 1.0},
            {"id": "FaceAngleY", "value": float(self._head_y), "weight": 1.0},
            {"id": "FaceAngleZ", "value": float(self._head_z), "weight": 1.0},
        ]

        # Brow parameters (for emphasis raise / micro-expressions)
        if abs(self._brow_left) > 0.01 or abs(self._brow_right) > 0.01:
            params.append(
                {"id": "Brows", "value": float(self._brow_left), "weight": 0.6}
            )

        # Eye drift (subtle gaze movement)
        if abs(self._eye_x) > 0.005 or abs(self._eye_y) > 0.005:
            params.extend([
                {"id": "EyeRightX", "value": float(self._eye_x), "weight": 0.4},
                {"id": "EyeRightY", "value": float(self._eye_y), "weight": 0.4},
            ])

        return params

    async def _set_head_position(self, x: float, y: float, z: float):
        """Send all parameters to VTube Studio (backward compatible)."""
        if not self.vts or not self.vts.is_connected:
            return

        params = self.get_all_parameters()

        try:
            await self.vts.set_parameters(params)
        except Exception as e:
            print(f"[GestureController] Error setting parameters: {e}")

    async def update_loop(self):
        """
        Continuous update loop — call this regularly during speech.
        Computes all animation channels and sends to VTS each frame.
        """
        last_time = asyncio.get_event_loop().time()

        while self._is_speaking or self._is_ramping_down:
            try:
                now = asyncio.get_event_loop().time()
                dt = now - last_time
                last_time = now
                t = now - self._speech_start_time

                self._compute_frame(t, dt)
                await self._set_head_position(self._head_x, self._head_y, self._head_z)

                await asyncio.sleep(0.033)  # ~30fps
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[GestureController] Update loop error: {e}")
                await asyncio.sleep(0.033)


def detect_emotion_from_text(text: str) -> EmotionType:
    """Simple emotion detection from text."""
    text_lower = text.lower()

    if any(word in text_lower for word in ['happy', 'glad', 'great', 'excellent', 'wonderful', 'terbaik', 'gembira', 'senang']):
        return EmotionType.HAPPY

    if any(word in text_lower for word in ['sad', 'sorry', 'unfortunately', 'sedih', 'maaf']):
        return EmotionType.SAD

    if any(word in text_lower for word in ['excited', 'amazing', 'awesome', 'wow', 'hebat', 'mantap']):
        return EmotionType.EXCITED

    if any(word in text_lower for word in ['surprised', 'shocked', 'wow', 'oh', 'terkejut']):
        return EmotionType.SURPRISED

    if any(word in text_lower for word in ['think', 'consider', 'perhaps', 'maybe', 'fikir', 'mungkin']):
        return EmotionType.THINKING

    if any(word in text_lower for word in ['confused', 'unclear', 'what', 'huh', 'keliru']):
        return EmotionType.CONFUSED

    return EmotionType.NEUTRAL


# Global instance
_gesture_controller: Optional[GestureController] = None


def get_gesture_controller(vts_connector=None) -> GestureController:
    """Get or create the global gesture controller instance."""
    global _gesture_controller
    if _gesture_controller is None and vts_connector is not None:
        _gesture_controller = GestureController(vts_connector)
    return _gesture_controller
