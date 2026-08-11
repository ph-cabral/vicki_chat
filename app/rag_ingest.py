"""Ingesta de procedimientos/instructivos a Qdrant (colección PROC_COLLECTION).

Lo llama `ever` (POST /rag/documento) al crear/editar/borrar un documento en
/rrhh/puestos. Flujo: chunking simple por párrafos → embeddings (mismo modelo
singleton de tools.py, así la dimensión siempre coincide con la búsqueda) →
delete de los puntos viejos del doc → upsert de los nuevos.

IDs de punto determinísticos (uuid5 de "doc:{id}:chunk:{i}") para poder
re-upsertear sin duplicar.
"""
import logging
import uuid

from qdrant_client import models as qm

from app.config import config
from app.tools import get_client, get_embeddings

log = logging.getLogger("rag_ingest")

_NAMESPACE = uuid.UUID("7d9c1e2a-5b4f-4c3d-9e8a-1f2b3c4d5e6f")

CHUNK_SIZE = 1200   # chars aprox por chunk
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[str]:
    """Corta por párrafos acumulando hasta ~CHUNK_SIZE chars, con overlap.
    Suficiente para procedimientos (documentos cortos y estructurados)."""
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        # párrafo gigante → cortarlo duro
        while len(p) > CHUNK_SIZE:
            head, p = p[:CHUNK_SIZE], p[CHUNK_SIZE - CHUNK_OVERLAP:]
            chunks.append((buf + "\n\n" + head).strip() if buf else head)
            buf = ""
        if len(buf) + len(p) + 2 > CHUNK_SIZE and buf:
            chunks.append(buf)
            buf = buf[-CHUNK_OVERLAP:] + "\n\n" + p  # overlap con la cola anterior
        else:
            buf = f"{buf}\n\n{p}".strip() if buf else p
    if buf:
        chunks.append(buf)
    return chunks or ([text.strip()] if (text or "").strip() else [])


def _point_id(doc_id: int, i: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"doc:{doc_id}:chunk:{i}"))


def _ensure_collection(client, dim: int):
    if not client.collection_exists(config.PROC_COLLECTION):
        client.create_collection(
            collection_name=config.PROC_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        log.info(f"colección {config.PROC_COLLECTION!r} creada (dim={dim})")


def delete_documento(doc_id: int) -> None:
    """Borra todos los chunks del documento (por metadata.doc_id)."""
    client = get_client()
    if not client.collection_exists(config.PROC_COLLECTION):
        return
    client.delete(
        collection_name=config.PROC_COLLECTION,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[
                qm.FieldCondition(key="metadata.doc_id", match=qm.MatchValue(value=doc_id))
            ])
        ),
        wait=True,
    )


def upsert_documento(doc: dict) -> int:
    """doc = {id, tipo, titulo, contenido, version, puestos: [nombres], vigente}.
    Si vigente=False solo borra. Devuelve cantidad de chunks indexados."""
    doc_id = int(doc["id"])
    if not doc.get("vigente", True):
        delete_documento(doc_id)
        return 0

    # El título + tipo + puestos van dentro del texto embebido: mejora el recall
    # cuando preguntan "procedimiento para X" sin palabras del cuerpo.
    header = (
        f"{doc.get('tipo', 'procedimiento').capitalize()}: {doc.get('titulo', '')}\n"
        f"Puestos: {', '.join(doc.get('puestos') or []) or 'todos'}"
    )
    chunks = chunk_text(doc.get("contenido", ""))
    if not chunks:
        delete_documento(doc_id)
        return 0
    texts = [f"{header}\n\n{c}" for c in chunks]

    vectors = get_embeddings().embed_documents(texts)
    client = get_client()
    _ensure_collection(client, len(vectors[0]))

    delete_documento(doc_id)  # limpiar versión anterior (puede tener más chunks)
    points = [
        qm.PointStruct(
            id=_point_id(doc_id, i),
            vector=vectors[i],
            payload={
                "content": chunks[i],
                "metadata": {
                    "doc_id": doc_id,
                    "tipo_doc": doc.get("tipo", "procedimiento"),
                    "titulo": doc.get("titulo", ""),
                    "version": int(doc.get("version", 1)),
                    "puestos": doc.get("puestos") or [],
                    "chunk": i,
                },
            },
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=config.PROC_COLLECTION, points=points, wait=True)
    log.info(f"doc {doc_id} ({doc.get('titulo')!r}) → {len(points)} chunks en {config.PROC_COLLECTION}")
    return len(points)
