import requests
from requests.auth import HTTPDigestAuth
from requests_toolbelt.multipart.encoder import MultipartEncoder
import subprocess
import tempfile
import base64
import json
import os
import io
import uuid

import cv2, numpy as np
from PIL import Image

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "161982br")

# Ubicación -> IP del reloj
LOCATIONS = {
    "oficina": "10.10.0.12",
    "fabrica": "10.10.0.30",
    "lilser":  "10.10.0.92",
}
DEFAULT_LOCATION = "fabrica"

# Compat: snapshot por defecto
CAMERA_IP = os.getenv("CAMERA_IP", LOCATIONS[DEFAULT_LOCATION])
BASE = f"http://{CAMERA_IP}"
RTSP = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/Streaming/Channels/101"
JSON_HDR = {"Content-Type": "application/json"}


def _auth():
    return HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)


def resolve_location(loc: str) -> str:
    key = (loc or "").strip().lower()
    if key not in LOCATIONS:
        raise ValueError(f"Ubicación inválida: {loc}. Usá: {', '.join(LOCATIONS)}")
    return LOCATIONS[key]


def _base_for(ip: str) -> str:
    return f"http://{ip}"


SNAPSHOT_PATH = "/code/snapshots/foto.jpg"
SNAPSHOT_DIR = "/code/snapshots"


def take_camera_snapshot() -> bytes:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", RTSP,
         "-frames:v", "1", "-update", "1", "-q:v", "2", SNAPSHOT_PATH],
        check=True, timeout=15, capture_output=True,
    )
    with open(SNAPSHOT_PATH, "rb") as f:
        return f.read()


def read_snapshot() -> bytes:
    with open(SNAPSHOT_PATH, "rb") as f:
        return f.read()


def delete_snapshot() -> None:
    try:
        os.unlink(SNAPSHOT_PATH)
    except OSError:
        pass


def snapshot_b64() -> str:
    return base64.b64encode(take_camera_snapshot()).decode()


# def resize_face(jpg_bytes: bytes, max_side: int = 640, target_kb: int = 200) -> bytes:
#     arr = np.frombuffer(jpg_bytes, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("snapshot ilegible")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = _CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
#     if len(faces) == 0:
#         raise ValueError("no se detectó rostro en el snapshot")

#     x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

#     mx, my = int(w * 0.4), int(h * 0.4)
#     H, W = img.shape[:2]
#     x0, y0 = max(0, x - mx), max(0, y - my)
#     x1, y1 = min(W, x + w + mx), min(H, y + h + my)
#     crop = img[y0:y1, x0:x1]

#     pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     pil.thumbnail((max_side, max_side))

#     for q in (90, 80, 70, 60, 50):
#         buf = io.BytesIO()
#         pil.save(buf, "JPEG", quality=q)
#         if buf.tell() <= target_kb * 1024:
#             return buf.getvalue()
#     return buf.getvalue()

def resize_face(jpg_bytes: bytes, max_side: int = 480, target_kb: int = 60) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fin:
        fin.write(jpg_bytes); src = fin.name
    dst = src + ".small.jpg"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-vf", f"scale={max_side}:-1", "-q:v", "5", dst],
            check=True, timeout=15, capture_output=True,
        )
        with open(dst, "rb") as f:
            return f.read()
    finally:
        for p in (src, dst):
            try: os.unlink(p)
            except OSError: pass


def _post_json(url: str, body: dict, timeout: int = 15) -> dict:
    payload = json.dumps(body)
    r = requests.post(url, auth=_auth(), data=payload, headers=JSON_HDR, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.text else {}


def next_employee_no(ip: str = None) -> str:
    base = _base_for(ip) if ip else BASE
    url = f"{base}/ISAPI/AccessControl/UserInfo/Search?format=json"
    max_no = 0
    pos = 0
    sid = str(uuid.uuid4())[:8]
    while True:
        body = {"UserInfoSearchCond": {
            "searchID": sid,
            "searchResultPosition": pos,
            "maxResults": 30
        }}
        data = _post_json(url, body).get("UserInfoSearch", {})
        for u in data.get("UserInfo", []) or []:
            try:
                n = int(u.get("employeeNo", "0"))
                if n > max_no:
                    max_no = n
            except ValueError:
                pass
        if data.get("responseStatusStrg") != "MORE":
            break
        pos += data.get("numOfMatches", 30)
    return str(max_no + 1)


def create_employee(name: str, gender: str, location: str = DEFAULT_LOCATION, employee_no: str = None) -> tuple:
    ip = resolve_location(location)
    base = _base_for(ip)
    if employee_no is None:
        employee_no = next_employee_no(ip=ip)

    url = f"{base}/ISAPI/AccessControl/UserInfo/Record?format=json"
    payload = {"UserInfo": {
        "employeeNo": employee_no,
        "name": name,
        "userType": "normal",
        "gender": gender,
        "Valid": {
            "enable": True,
            "beginTime": "2025-01-01T00:00:00",
            "endTime": "2037-12-31T23:59:59",
            "timeType": "local"
        },
        "doorRight": "1",
        "RightPlan": [{"doorNo": 1, "planTemplateNo": "1"}],
        "userVerifyMode": "face",
        "localUIRight": False
    }}
    _post_json(url, payload)
    return employee_no, ip


def upload_face(employee_no: str, jpg_bytes: bytes, ip: str = None) -> dict:
    base = _base_for(ip) if ip else BASE
    face_record = {"faceLibType": "blackFD", "FDID": "1", "FPID": employee_no}
    url = f"{base}/ISAPI/Intelligent/FDLib/FaceDataRecord?format=json"

    enc = MultipartEncoder(fields={
        "FaceDataRecord": (None, json.dumps(face_record), "application/json"),
        "img": ("face.jpg", jpg_bytes, "image/jpeg"),
    })
    r = requests.post(
        url, auth=_auth(), data=enc,
        headers={"Content-Type": enc.content_type},
        timeout=30,
    )
    r.raise_for_status()
    return r.json() if r.text else {}