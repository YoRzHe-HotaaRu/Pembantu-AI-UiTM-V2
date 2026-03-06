"""
VTube Studio Integration Module
Enables Live2D avatar control via VTube Studio API for lip sync and expressions.
"""

from .connector import VTSConnector, get_connector
from .lip_sync import LipSyncAnalyzer, LipSyncPlayer, get_analyzer, get_player
from .expressions import ExpressionMapper, get_mapper
from .audio_converter import AudioConverter

__all__ = [
    'VTSConnector',
    'get_connector',
    'LipSyncAnalyzer',
    'LipSyncPlayer',
    'get_analyzer',
    'get_player',
    'ExpressionMapper',
    'get_mapper',
    'AudioConverter'
]