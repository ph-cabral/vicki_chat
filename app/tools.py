from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from qdrant_client import QdrantClient
from app.config import config
import os


CAMERA_IP = "10.10.0.12"
CAMERA_USER = "admin"
CAMERA_PASS = "161982br"
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
        content_payload_key="content",
        metadata_payload_key="metadata",
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
