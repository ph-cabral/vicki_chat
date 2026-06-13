import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from io import BytesIO

import requests
from requests.auth import HTTPDigestAuth
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image

log = logging.getLogger("tool")

# Credenciales de los relojes/cámaras: CAMERA_* o, si no están, HIK_* (las que pasa docker-compose).
CAMERA_USER = os.getenv("CAMERA_USER") or os.getenv("HIK_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS") or os.getenv("HIK_PASS", "")

# Ubicación -> IP del reloj (override por env: LOCATIONS_JSON='{"oficina":"10.10.0.12",...}')
_DEFAULT_LOCATIONS = {
    "oficina": "10.10.0.12",
    "fabrica": "10.10.0.30",
    "lilser":  "10.10.0.92",
}
try:
    LOCATIONS: dict[str, str] = json.loads(os.environ["LOCATIONS_JSON"]) if os.getenv("LOCATIONS_JSON") else dict(_DEFAULT_LOCATIONS)
except (json.JSONDecodeError, TypeError):
    log.error("LOCATIONS_JSON inválido; uso defaults")
    LOCATIONS = dict(_DEFAULT_LOCATIONS)
DEFAULT_LOCATION = "oficina"

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


def take_camera_snapshot(ip: str = None) -> bytes:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    target_ip = ip or CAMERA_IP
    rtsp_url = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{target_ip}:554/Streaming/Channels/101"
    subprocess.run(
        ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp_url,
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
    for _ in range(5000):  # tope anti-loop si el firmware pagina mal
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


def create_employee_all(name: str, gender: str, employee_no: str) -> dict:
    res = {}
    for loc in LOCATIONS:
        try:
            create_employee(name=name, gender=gender, location=loc, employee_no=employee_no)
            res[loc] = "ok"
        except Exception as e:
            res[loc] = f"error: {e}"
    return res


def upload_face_all(employee_no: str, jpg_bytes: bytes) -> dict:
    res = {}
    for loc, ip in LOCATIONS.items():
        try:
            upload_face(employee_no, jpg_bytes, ip=ip)
            res[loc] = "ok"
        except Exception as e:
            res[loc] = f"error: {e}"
    return res

def _wait_user_committed(employee_no: str, ip: str, retries: int = 8, delay: float = 0.5) -> bool:
    base = _base_for(ip)
    url = f"{base}/ISAPI/AccessControl/UserInfo/Search?format=json"
    sid = str(uuid.uuid4())[:8]
    for _ in range(retries):
        body = {"UserInfoSearchCond": {
            "searchID": sid, "searchResultPosition": 0, "maxResults": 1,
            "EmployeeNoList": [{"employeeNo": employee_no}]
        }}
        try:
            data = _post_json(url, body).get("UserInfoSearch", {})
            if any(u.get("employeeNo") == employee_no for u in (data.get("UserInfo") or [])):
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def upload_face(employee_no: str, jpg_bytes: bytes, ip: str = None, retries: int = 3) -> dict:
    jpg_bytes = _shrink_jpg(jpg_bytes, max_kb=200, max_side=640)
    log.info(f"[face-upload] emp={employee_no} ip={ip} bytes={len(jpg_bytes)}")
    base = _base_for(ip) if ip else BASE

    _wait_user_committed(employee_no, ip or BASE.split("//")[1])

    face_record = {"faceLibType": "blackFD", "FDID": "1", "FPID": str(employee_no)}
    url_rec = f"{base}/ISAPI/Intelligent/FDLib/FaceDataRecord?format=json"
    url_mod = f"{base}/ISAPI/Intelligent/FDLib/FDModify?format=json&FDID=1&faceLibType=blackFD"

    last = None
    for i in range(retries):
        try:
            enc = MultipartEncoder(fields=[
                ("FaceDataRecord", (None, json.dumps(face_record), "application/json")),
                ("img", ("face.jpg", jpg_bytes, "image/jpeg")),
            ])
            body = enc.to_string()
            ctype = enc.content_type

            r = requests.post(
                url_rec,
                auth=HTTPDigestAuth(CAMERA_USER, CAMERA_PASS),
                data=body,
                headers={"Content-Type": ctype},
                timeout=30,
            )
            log.info(f"[face-upload] POST status={r.status_code} body={r.text[:300]}")

            need_modify = False
            if r.status_code >= 400:
                need_modify = True
            else:
                try:
                    j = r.json()
                    if j.get("statusCode") not in (1, None):
                        sub = (j.get("subStatusCode") or "").lower()
                        if "exist" in sub or "duplicate" in sub:
                            need_modify = True
                        else:
                            raise RuntimeError(f"FaceDataRecord failed: {j}")
                except ValueError:
                    pass

            if need_modify:
                enc2 = MultipartEncoder(fields=[
                    ("FaceDataRecord", (None, json.dumps(face_record), "application/json")),
                    ("img", ("face.jpg", jpg_bytes, "image/jpeg")),
                ])
                r = requests.put(
                    url_mod,
                    auth=HTTPDigestAuth(CAMERA_USER, CAMERA_PASS),
                    data=enc2.to_string(),
                    headers={"Content-Type": enc2.content_type},
                    timeout=30,
                )
                log.info(f"[face-upload] PUT status={r.status_code} body={r.text[:300]}")

            r.raise_for_status()
            return r.json() if r.text else {}
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last

def _deferred_upload_face(emp_no: str, ip: str, jpg: bytes, delay: int = 10):
    time.sleep(delay)
    try:
        upload_face(emp_no, jpg, ip=ip)
        delete_snapshot()
    except Exception as e:
        log.error(f"[face-upload] emp={emp_no} ip={ip} FAIL: {e}")
    else:
        log.info(f"[face-upload] emp={emp_no} ip={ip} OK")
        
def _shrink_jpg(jpg_bytes: bytes, max_kb: int = 200, max_side: int = 640) -> bytes:
    img = Image.open(BytesIO(jpg_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    for q in (75, 65, 55, 45, 35, 25):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        if buf.tell() <= max_kb * 1024:
            return buf.getvalue()
    return buf.getvalue()