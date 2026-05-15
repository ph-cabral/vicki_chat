from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from qdrant_client import QdrantClient
from app.config import config
import requests
from requests.auth import HTTPDigestAuth
import base64


CAMERA_IP = "10.10.0.30"
CAMERA_USER = "admin"
CAMERA_PASS = "161982br"

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
    """Toma foto de la cámara IP y retorna base64."""
    url = f"http://{CAMERA_IP}/ISAPI/Streaming/channels/101/picture"
    r = requests.get(url, auth=HTTPDigestAuth(CAMERA_USER, CAMERA_PASS), timeout=5)
    r.raise_for_status()
    return base64.b64encode(r.content).decode()