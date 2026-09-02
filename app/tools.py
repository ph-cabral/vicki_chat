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
from qdrant_client import models as qm

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


# Tipos de documento que llegan desde ever /rrhh/puestos (ver
# lib/rrhh/documentosTipos.ts). `descripcion_puesto` es el perfil del puesto y
# se busca APARTE de los procedimientos — ver nodes.py::rag_search_node.
TIPO_DESCRIPCION_PUESTO = "descripcion_puesto"
TIPO_DOC_LABEL = {
    "procedimiento": "procedimiento",
    "instructivo": "instructivo",
    TIPO_DESCRIPCION_PUESTO: "descripción de puesto",
}


def _filtro_tipo_doc(tipos: list[str], excluir: bool = False) -> qm.Filter:
    """Filtro de payload por metadata.tipo_doc (match any / match none)."""
    cond = qm.FieldCondition(key="metadata.tipo_doc", match=qm.MatchAny(any=tipos))
    return qm.Filter(must_not=[cond]) if excluir else qm.Filter(must=[cond])


def _format_hit(col: str, p) -> str:
    payload = p.payload or {}
    meta = payload.get("metadata", {}) or {}
    # Hit de procedimiento/instructivo (colección PROC_COLLECTION) → otro formato.
    if meta.get("tipo_doc"):
        puestos = meta.get("puestos") or []
        etiqueta = TIPO_DOC_LABEL.get(meta.get("tipo_doc"), meta.get("tipo_doc"))
        return "\n".join([
            f"\n=== {meta.get('titulo', 'Sin título')} "
            f"[{etiqueta} v{meta.get('version', 1)}, "
            f"relevancia: {(p.score or 0.0):.2f}] ===",
            f"Puestos: {', '.join(puestos) if puestos else 'todos'}",
            payload.get("content", ""),
        ])
    nombre = " ".join(x for x in [meta.get("nombre"), meta.get("apellido")] if x) or "N/A"
    email = meta.get("email", "N/A")
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


def search_collections(
    query: str,
    collections: list[str],
    k: int | None = None,
    flt: qm.Filter | None = None,
    vector: list[float] | None = None,
    score_threshold: float | None = None,
) -> str:
    """Embebe el query una vez y busca en 1 o varias colecciones en paralelo.
    Devuelve contexto formateado, ordenado por score global.

    flt:    filtro de payload de Qdrant (ej. solo descripciones de puesto).
    vector: embedding ya calculado del query — para no pagar dos veces el
            embed cuando se hacen varias búsquedas con el MISMO query
            (CVs + descripción de puesto + procedimientos, ver
            nodes.py::rag_search_node).
    score_threshold: piso de similitud. Qdrant devuelve siempre los K mejores
            aunque sean malos; cuando el resultado es contexto de apoyo (no la
            respuesta), traer basura es peor que no traer nada.
    """
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

    if vector is None:
        vector = get_embeddings().embed_query(query)
    client = get_client()

    def _one(col: str):
        try:
            pts = client.query_points(
                col, query=vector, limit=k, with_payload=True, query_filter=flt,
                score_threshold=score_threshold,
            ).points
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


def embed_query(query: str) -> list[float]:
    """Embedding del query, para reusarlo entre varias búsquedas."""
    return get_embeddings().embed_query(query)


def search_descripcion_puesto(query: str, k: int | None = None, vector=None) -> str:
    """Busca SOLO descripciones de puesto (dentro de PROC_COLLECTION).

    Existe porque router_node excluye PROC_COLLECTION de las búsquedas de CVs
    (para que los procedimientos no ensucien los candidatos), pero el perfil del
    puesto SÍ tiene que llegar cuando se busca gente. Se trae aparte y filtrado.
    """
    return search_collections(
        query,
        [config.PROC_COLLECTION],
        k=k or config.PERFIL_TOP_K,
        flt=_filtro_tipo_doc([TIPO_DESCRIPCION_PUESTO]),
        vector=vector,
    )


def search_procedimientos(query: str, k: int | None = None, vector=None,
                          min_score: float | None = None) -> str:
    """Procedimientos e instructivos, SIN descripciones de puesto (si no,
    "¿qué procedimiento sigue X?" devuelve el perfil del puesto, que no tiene
    pasos).

    Dos usos:
    - intent=procedimiento → es LA respuesta: k normal, sin piso de score.
    - intent=search/ranking → es contexto de qué hace el puesto, al lado del
      perfil: k chico y min_score, porque ahí la query es de candidatos y sin
      piso entran procedimientos de cualquier otro puesto.
    """
    return search_collections(
        query,
        [config.PROC_COLLECTION],
        k=k,
        flt=_filtro_tipo_doc([TIPO_DESCRIPCION_PUESTO], excluir=True),
        vector=vector,
        score_threshold=min_score,
    )
