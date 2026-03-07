"""
VTube Studio Integration Module
Enables Live2D avatar control via VTube Studio API for lip sync, expressions,
idle animations, and talking gestures.
"""

from .connector import VTSConnector, get_connector
from .lip_sync import LipSyncAnalyzer, LipSyncPlayer, get_analyzer, get_player
from .expressions import ExpressionMapper, get_mapper
from .audio_converter import AudioConverter
from .idle_animator import IdleAnimator, IdleConfig, get_idle_animator
from .gesture_controller import GestureController, GestureConfig, EmotionType, get_gesture_controller, detect_emotion_from_text
from .lip_sync_parallel import ParallelLipSyncAnalyzer, get_parallel_analyzer

__all__ = [
    'VTSConnector',
    'get_connector',
    'LipSyncAnalyzer',
    'LipSyncPlayer',
    'get_analyzer',
    'get_player',
    'ExpressionMapper',
    'get_mapper',
    'AudioConverter',
    'IdleAnimator',
    'IdleConfig',
    'get_idle_animator',
    'GestureController',
    'GestureConfig',
    'EmotionType',
    'get_gesture_controller',
    'detect_emotion_from_text',
    'ParallelLipSyncAnalyzer',
    'get_parallel_analyzer'
]