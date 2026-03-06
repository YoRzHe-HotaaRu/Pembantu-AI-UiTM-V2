"""
VTube Studio WebSocket Connector
Handles communication with VTube Studio API for Live2D model control.
Includes auto-reconnect functionality.
"""

import asyncio
import json
import os
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class VTSConnector:
    """
    WebSocket connector for VTube Studio Plugin API.
    Manages authentication, parameter injection, and expression triggers.
    Includes auto-reconnect on connection loss.
    """
    
    # Plugin identification
    PLUGIN_NAME = "UiTM AI Receptionist"
    PLUGIN_DEVELOPER = "UiTM"
    PLUGIN_ICON = None  # Optional: base64 encoded icon
    
    # Default mouth tracking parameter for TTS lip sync
    # NOTE: VTS doesn't allow injecting into Live2D parameters directly.
    # We use the standard "MouthOpen" tracking parameter (input) which should be
    # bound to "ParamMouthOpenY" (Live2D output) in VTube Studio.
    MOUTH_PARAM = "MouthOpen"
    
    def __init__(self, host: str = "localhost", port: int = 8001,
                 auto_reconnect: bool = True,
                 reconnect_interval: float = 5.0,
                 max_reconnect_attempts: int = 10):
        """
        Initialize VTS connector.
        
        Args:
            host: VTube Studio host (default: localhost)
            port: VTube Studio API port (default: 8001)
            auto_reconnect: Enable auto-reconnect on disconnect
            reconnect_interval: Seconds between reconnect attempts
            max_reconnect_attempts: Maximum reconnect attempts (0 = infinite)
        """
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.websocket = None
        self.authenticated = False
        self.auth_token = None
        self._request_id = 0
        self._token_path = Path(__file__).parent.parent / ".vts_token"
        self._load_token()
        
        # Connection state
        self._connecting = False
        self._reconnect_count = 0
        self._last_connect_time = 0
        
        # Event callbacks
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._on_error: Optional[Callable[[str], None]] = None
        
        # Background tasks
        self._reconnect_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        
    def _load_token(self):
        """Load saved authentication token if exists."""
        if self._token_path.exists():
            try:
                self.auth_token = self._token_path.read_text().strip()
                print(f"[VTS] Loaded saved auth token")
            except Exception as e:
                print(f"[VTS] Could not load token: {e}")
                
    def _save_token(self, token: str):
        """Save authentication token for future use."""
        try:
            self._token_path.write_text(token)
            print(f"[VTS] Auth token saved")
        except Exception as e:
            print(f"[VTS] Could not save token: {e}")
    
    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_id += 1
        return f"UiTM_VTS_{self._request_id}"
    
    async def _send_request(self, request_type: str, data: Dict = None) -> Dict:
        """
        Send a request to VTube Studio and wait for response.
        
        Args:
            request_type: VTS API request type
            data: Request-specific data
            
        Returns:
            Response data from VTS
        """
        if not self.websocket:
            raise ConnectionError("Not connected to VTube Studio")
            
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": request_type,
            "data": data or {}
        }
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)
    
    @property
    def is_available(self) -> bool:
        """Check if websockets library is available."""
        return WEBSOCKETS_AVAILABLE
    
    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self.websocket is not None and self.authenticated
    
    def on_connected(self, callback: Callable):
        """Set callback for connection established."""
        self._on_connected = callback
        
    def on_disconnected(self, callback: Callable):
        """Set callback for disconnection."""
        self._on_disconnected = callback
        
    def on_error(self, callback: Callable[[str], None]):
        """Set callback for errors."""
        self._on_error = callback
    
    async def connect(self) -> bool:
        """
        Connect to VTube Studio and authenticate.
        
        Returns:
            True if connection and authentication successful
        """
        if not WEBSOCKETS_AVAILABLE:
            print("[VTS] Error: websockets library not installed. Run: pip install websockets")
            return False
            
        if self._connecting:
            print("[VTS] Connection already in progress")
            return False
            
        self._connecting = True
        
        try:
            uri = f"ws://{self.host}:{self.port}"
            print(f"[VTS] Connecting to {uri}...")
            
            self.websocket = await websockets.connect(uri)
            print("[VTS] Connected to VTube Studio")
            
            # Reset reconnect count on successful connection
            self._reconnect_count = 0
            self._last_connect_time = time.time()
            
            # Authenticate
            success = await self._authenticate()
            
            if success:
                # Create mouth parameter if needed
                await self._ensure_mouth_parameter()
                
                # Start keepalive
                self._start_keepalive()
                
                # Notify callback
                if self._on_connected:
                    self._on_connected()
                    
            self._connecting = False
            return success
            
        except ConnectionRefusedError:
            print("[VTS] Error: VTube Studio is not running or API is disabled")
            print("[VTS] Please enable the API in VTube Studio: Settings → General Settings → Start API")
            self._connecting = False
            self.websocket = None
            return False
        except Exception as e:
            print(f"[VTS] Connection error: {e}")
            self._connecting = False
            self.websocket = None
            if self._on_error:
                self._on_error(str(e))
            return False
    
    async def _authenticate(self) -> bool:
        """
        Authenticate with VTube Studio.
        Uses saved token if available, otherwise requests new one.
        """
        # Try using saved token first
        if self.auth_token:
            print("[VTS] Attempting authentication with saved token...")
            response = await self._send_request("AuthenticationRequest", {
                "pluginName": self.PLUGIN_NAME,
                "pluginDeveloper": self.PLUGIN_DEVELOPER,
                "authenticationToken": self.auth_token
            })
            
            if response.get("data", {}).get("authenticated"):
                print("[VTS] Authentication successful!")
                self.authenticated = True
                return True
            else:
                print("[VTS] Saved token invalid, requesting new token...")
                self.auth_token = None
        
        # Request new token
        print("[VTS] Requesting authentication token...")
        print("[VTS] Please click 'Allow' in VTube Studio when prompted!")
        
        response = await self._send_request("AuthenticationTokenRequest", {
            "pluginName": self.PLUGIN_NAME,
            "pluginDeveloper": self.PLUGIN_DEVELOPER,
            "pluginIcon": self.PLUGIN_ICON
        })
        
        token = response.get("data", {}).get("authenticationToken")
        if not token:
            error = response.get("data", {}).get("message", "Unknown error")
            print(f"[VTS] Token request failed: {error}")
            return False
            
        self.auth_token = token
        self._save_token(token)
        
        # Now authenticate with the new token
        response = await self._send_request("AuthenticationRequest", {
            "pluginName": self.PLUGIN_NAME,
            "pluginDeveloper": self.PLUGIN_DEVELOPER,
            "authenticationToken": self.auth_token
        })
        
        if response.get("data", {}).get("authenticated"):
            print("[VTS] Authentication successful!")
            self.authenticated = True
            return True
        else:
            error = response.get("data", {}).get("message", "Unknown error")
            print(f"[VTS] Authentication failed: {error}")
            return False
    
    async def _ensure_mouth_parameter(self):
        """
        Ensure mouth parameter is available.
        For default tracking parameters like MouthOpen, we don't need to create them.
        For custom parameters, we create them as tracking parameters.
        """
        if not self.is_connected:
            return
        
        # MouthOpen is a standard default tracking parameter in VTube Studio
        # No need to create it - it already exists
        if self.MOUTH_PARAM == "MouthOpen":
            print(f"[VTS] Using default tracking parameter: {self.MOUTH_PARAM}")
            print(f"[VTS] Ensure '{self.MOUTH_PARAM}' (Input) is bound to 'ParamMouthOpenY' (Output) in VTS")
            return
            
        # For custom parameters, we need to create them
        try:
            response = await self._send_request("ParameterCreationRequest", {
                "parameterName": self.MOUTH_PARAM,
                "explanation": "Mouth open value for TTS lip sync",
                "min": 0.0,
                "max": 1.0,
                "defaultValue": 0.0
            })
            
            # Check if parameter was created or already exists
            if response.get("data", {}).get("parameterName"):
                print(f"[VTS] Created mouth parameter: {self.MOUTH_PARAM}")
            else:
                # Parameter might already exist (error 352)
                error_id = response.get("data", {}).get("errorID", 0)
                if error_id == 352:
                    print(f"[VTS] Mouth parameter already exists: {self.MOUTH_PARAM}")
                else:
                    print(f"[VTS] Could not create parameter: {response}")
        except Exception as e:
            print(f"[VTS] Error creating mouth parameter: {e}")
    
    def _start_keepalive(self):
        """Start keepalive task to maintain connection."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            
        async def keepalive():
            while self.websocket and self.authenticated:
                try:
                    # Send API state request every 30 seconds
                    await asyncio.sleep(30)
                    if self.websocket:
                        await self._send_request("APIStateRequest")
                except Exception:
                    break
                    
        self._keepalive_task = asyncio.create_task(keepalive())
    
    async def disconnect(self):
        """Disconnect from VTube Studio."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
            
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
            
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.authenticated = False
            print("[VTS] Disconnected from VTube Studio")
            
            if self._on_disconnected:
                self._on_disconnected()
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to VTube Studio.
        
        Returns:
            True if reconnection successful
        """
        if self.websocket:
            await self.disconnect()
            
        self._reconnect_count += 1
        
        if self.max_reconnect_attempts > 0 and self._reconnect_count > self.max_reconnect_attempts:
            print(f"[VTS] Max reconnect attempts ({self.max_reconnect_attempts}) reached")
            return False
            
        print(f"[VTS] Reconnecting (attempt {self._reconnect_count})...")
        await asyncio.sleep(self.reconnect_interval)
        
        return await self.connect()
    
    def start_auto_reconnect(self):
        """Start background auto-reconnect task."""
        if not self.auto_reconnect:
            return
            
        if self._reconnect_task:
            return
            
        async def auto_reconnect_loop():
            while self.auto_reconnect:
                await asyncio.sleep(self.reconnect_interval)
                
                if not self.is_connected and not self._connecting:
                    await self.reconnect()
                    
        self._reconnect_task = asyncio.create_task(auto_reconnect_loop())
    
    async def get_available_parameters(self) -> List[Dict]:
        """
        Get list of available parameters for the current model.
        
        Returns:
            List of parameter info dicts
        """
        if not self.is_connected:
            return []
            
        response = await self._send_request("InputParameterListRequest")
        return response.get("data", {}).get("customParameters", []) + \
               response.get("data", {}).get("defaultParameters", [])
    
    async def set_parameter(self, name: str, value: float, weight: float = 1.0) -> bool:
        """
        Set a single parameter value.
        
        Args:
            name: Parameter name (e.g., "MouthOpen", "ParamMouthOpenY")
            value: Parameter value (typically 0.0 to 1.0)
            weight: Blend weight (0.0 to 1.0)
            
        Returns:
            True if successful
        """
        return await self.set_parameters([{
            "id": name,
            "value": value,
            "weight": weight
        }])
    
    async def set_parameters(self, parameters: List[Dict]) -> bool:
        """
        Set multiple parameter values at once.
        
        Args:
            parameters: List of {"id": name, "value": float, "weight": float}
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            print(f"[VTS] Cannot set parameters - not connected")
            return False
            
        try:
            response = await self._send_request("InjectParameterDataRequest", {
                "faceFound": True,
                "mode": "set",
                "parameterValues": parameters
            })
            
            # Debug: Log response
            data = response.get("data", {})
            if "errorID" in data:
                print(f"[VTS] Parameter injection error: {data}")
                return False
            
            return True
        except Exception as e:
            print(f"[VTS] Error setting parameters: {e}")
            return False
    
    async def set_mouth(self, value: float) -> bool:
        """
        Set mouth open value.
        
        Args:
            value: Mouth open value (0.0 = closed, 1.0 = fully open)
            
        Returns:
            True if successful
        """
        return await self.set_parameter(self.MOUTH_PARAM, value)
    
    async def get_expressions(self) -> List[Dict]:
        """
        Get list of available expressions for the current model.
        
        Returns:
            List of expression info dicts
        """
        if not self.is_connected:
            return []
            
        response = await self._send_request("ExpressionStateRequest")
        return response.get("data", {}).get("expressions", [])
    
    async def set_expression(self, expression_file: str, active: bool = True) -> bool:
        """
        Activate or deactivate an expression.
        
        Args:
            expression_file: Expression filename (e.g., "happy.exp3.json")
            active: True to activate, False to deactivate
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            return False
            
        try:
            response = await self._send_request("ExpressionActivationRequest", {
                "expressionFile": expression_file,
                "active": active
            })
            return "errorID" not in response.get("data", {})
        except Exception as e:
            print(f"[VTS] Error setting expression: {e}")
            return False
    
    async def get_hotkeys(self) -> List[Dict]:
        """
        Get list of available hotkeys for the current model.
        
        Returns:
            List of hotkey info dicts
        """
        if not self.is_connected:
            return []
            
        response = await self._send_request("HotkeysInCurrentModelRequest")
        return response.get("data", {}).get("availableHotkeys", [])
    
    async def trigger_hotkey(self, hotkey_id: str) -> bool:
        """
        Trigger a hotkey action.
        
        Args:
            hotkey_id: Hotkey ID to trigger
            
        Returns:
            True if successful
        """
        if not self.is_connected:
            return False
            
        try:
            response = await self._send_request("HotkeyTriggerRequest", {
                "hotkeyID": hotkey_id
            })
            return "errorID" not in response.get("data", {})
        except Exception as e:
            print(f"[VTS] Error triggering hotkey: {e}")
            return False


# Global connector instance
_connector: Optional[VTSConnector] = None


def get_connector() -> VTSConnector:
    """Get or create the global VTS connector instance."""
    global _connector
    if _connector is None:
        _connector = VTSConnector()
    return _connector