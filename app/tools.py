from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from qdrant_client import QdrantClient
from app.config import config
import requests
from requests.auth import HTTPDigestAuth
import base64
import subprocess, base64, tempfile, os, time


CAMERA_IP = "10.10.0.30"
CAMERA_USER = "admin"
CAMERA_PASS = "161982br"
SNAPSHOT_PATH = "/code/snapshots/foto.jpg"
SNAPSHOT_DIR = "/code/snapshots"
PUBLIC_BASE = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")


def build_retriever_tool():
    embeddings = OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )

    return create_retriever_tool(
        retriever=retriever,
        name="buscar_cvs",
        description=(
            "Busca información en currículums de candidatos. "
            "Usá esta herramienta para encontrar perfiles de marketing, "
            "experiencia laboral, habilidades y seniority de candidatos."
        ),
    )

def take_camera_snapshot() -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    rtsp = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/Streaming/Channels/101"
    subprocess.run(
        ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp,
         "-frames:v", "1", "-update", "1", "-q:v", "2", SNAPSHOT_PATH],
        check=True, timeout=15, capture_output=True,
    )
    return f"{PUBLIC_BASE}/snapshots/foto.jpg"

def get_snapshot_b64() -> str | None:
    if not os.path.exists(SNAPSHOT_PATH):
        return None
    with open(SNAPSHOT_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()