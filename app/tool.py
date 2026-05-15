import requests
from requests.auth import HTTPDigestAuth
import base64

CAMERA_IP = "10.10.0.30"
CAMERA_USER = "admin"
CAMERA_PASS = "161982br"

def take_camera_snapshot() -> str:
    """Toma foto de la cámara IP y retorna base64."""
    url = f"http://{CAMERA_IP}/ISAPI/Streaming/channels/101/picture"
    r = requests.get(url, auth=HTTPDigestAuth(CAMERA_USER, CAMERA_PASS), timeout=5)
    r.raise_for_status()
    return base64.b64encode(r.content).decode()
