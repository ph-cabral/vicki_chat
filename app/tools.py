"""Reemplazo de app/tools.py — búsqueda RAG multi-colección sobre Qdrant.

Velocidad:
- Cliente Qdrant y embeddings como singletons (se crean una sola vez).
- Lista de colecciones cacheada (TTL) para no consultar Qdrant en cada mensaje.
- El query se embebe UNA sola vez y se reusa en todas las colecciones.
- Si hay varias colecciones, se buscan en paralelo (ThreadPool).

Nota: usa qdrant_client directo (más liviano que el retriever de langchain).
Ya no se usa build_retriever_tool().
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from app.config import config

log = logging.getLogger("tools")

_embeddings: OpenAIEmbeddings | None = None
_client: QdrantClient | None = None
_cols_cache = {"ts": 0.0, "names": []}


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBED_MODEL)
    return _embeddings


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=config.QDRANT_TIMEOUT,
        )
    return _client


def list_collections() -> list[str]:
    """Nombres de colecciones en Qdrant, cacheado QDRANT_CACHE_TTL segundos."""
    now = time.time()
    if _cols_cache["names"] and now - _cols_cache["ts"] < config.QDRANT_CACHE_TTL:
        return _cols_cache["names"]
    try:
        names = [c.name for c in get_client().get_collections().collections]
        _cols_cache.update(ts=now, names=names)
    except Exception:
        log.exception("no pude listar colecciones Qdrant")
    return _cols_cache["names"]


def _format_hit(col: str, p) -> str:
    payload = p.payload or {}
    meta = payload.get("metadata", {}) or {}
    nombre = meta.get("candidato_nombre", "N/A")
    email = meta.get("candidato_email", "N/A")
    content = payload.get("content", "")
    empresas = meta.get("empresas", []) or []
    blk = [
        f"\n--- {nombre} (colección: {col}, relevancia: {(p.score or 0.0):.2f}) ---",
        f"Email: {email}",
        content,
    ]
    if empresas:
        blk.append("EXPERIENCIA LABORAL:")
        for e in empresas:
            blk.append(
                f"- {e.get('puesto','')} en {e.get('empresa','')} "
                f"({e.get('fecha_inicio','')} - {e.get('fecha_finalizacion','')})\n"
                f"  {e.get('descripcion','')}"
            )
    return "\n".join(blk)


def search_collections(query: str, collections: list[str], k: int | None = None) -> str:
    """Embebe el query una vez y busca en 1 o varias colecciones en paralelo.
    Devuelve contexto formateado, ordenado por score global."""
    k = k or config.TOP_K
    cols = [c for c in (collections or []) if c]
    if not cols:
        avail = list_collections()
        if config.QDRANT_COLLECTION in avail:
            cols = [config.QDRANT_COLLECTION]
        elif avail:
            cols = avail[:1]
        else:
            return ""

    vector = get_embeddings().embed_query(query)
    client = get_client()

    def _one(col: str):
        try:
            pts = client.query_points(col, query=vector, limit=k, with_payload=True).points
            return [(col, p) for p in pts]
        except Exception:
            log.exception(f"búsqueda falló en {col!r} (¿otra dimensión de embedding?)")
            return []

    hits: list = []
    if len(cols) == 1:
        hits = _one(cols[0])
    else:
        with ThreadPoolExecutor(max_workers=min(6, len(cols))) as ex:
            for part in ex.map(_one, cols):
                hits.extend(part)

    hits.sort(key=lambda cp: cp[1].score or 0.0, reverse=True)
    top = hits[: k if len(cols) == 1 else k * 2]
    return "\n".join(_format_hit(c, p) for c, p in top)
