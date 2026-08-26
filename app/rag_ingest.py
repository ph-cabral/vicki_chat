"""Ingesta de documentos por puesto a Qdrant (colección PROC_COLLECTION).

Tres tipos, distinguidos por metadata.tipo_doc: "procedimiento", "instructivo"
y "descripcion_puesto". Conviven en la misma colección y se separan con filtro
de payload al buscar (ver tools.py) — el perfil del puesto se usa al BUSCAR
CANDIDATOS y los otros dos al preguntar cómo se hace algo.

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
from app.tools import TIPO_DESCRIPCION_PUESTO, get_client, get_embeddings

log = logging.getLogger("rag_ingest")

_NAMESPACE = uuid.UUID("7d9c1e2a-5b4f-4c3d-9e8a-1f2b3c4d5e6f")

CHUNK_SIZE = 1200   # chars aprox por chunk
CHUNK_OVERLAP = 150


def _cola(buf: str) -> str:
    """Últimos ~CHUNK_OVERLAP chars de `buf` para usar como overlap, arrancando
    en un límite de renglón (o de palabra). Antes se hacía `buf[-CHUNK_OVERLAP:]`
    a secas y el chunk siguiente empezaba a mitad de palabra ("...localidades
    cerc" / "anas; si reside..."), que es basura para el embedding."""
    tail = buf[-CHUNK_OVERLAP:]
    if len(buf) <= CHUNK_OVERLAP:
        return tail.strip()
    corte = tail.find("\n")
    if corte == -1:
        corte = tail.find(" ")
    return (tail[corte + 1:] if corte != -1 else tail).strip()


def _split_parrafo_largo(p: str) -> list[str]:
    """Parte un párrafo más largo que CHUNK_SIZE RESPETANDO los renglones.

    Antes esto cortaba a `p[:CHUNK_SIZE]` a secas y partía palabras al medio
    ("...localidades cerc" / "anas; si reside fuera..."), lo que arruina tanto
    el embedding como el texto que después lee el modelo. Pasa siempre que una
    sección de viñetas (una descripción de puesto, un procedimiento largo) va
    sin línea en blanco adentro — o sea, casi siempre.

    El primer renglón se repite como encabezado en las partes siguientes: sin
    eso, la parte 2 de "REQUISITOS EXCLUYENTES" queda sin decir de qué habla.
    """
    lineas = p.split("\n")
    encabezado = lineas[0].strip() if len(lineas) > 1 and len(lineas[0]) < 120 else ""
    cont = f"{encabezado} (cont.)" if encabezado else ""

    partes: list[str] = []
    buf = ""

    def cerrar():
        nonlocal buf
        if buf.strip():
            partes.append(buf.strip())
        buf = cont

    for linea in lineas:
        # un solo renglón más largo que el chunk → no queda otra que cortarlo
        while len(linea) > CHUNK_SIZE:
            cerrar()
            partes.append(linea[:CHUNK_SIZE])
            linea = linea[CHUNK_SIZE - CHUNK_OVERLAP:]
        if buf and len(buf) + len(linea) + 1 > CHUNK_SIZE:
            cerrar()
        buf = f"{buf}\n{linea}" if buf else linea
    if buf.strip() and buf.strip() != cont:
        partes.append(buf.strip())
    return [x for x in partes if x.strip()]


def chunk_text(text: str) -> list[str]:
    """Corta por párrafos acumulando hasta ~CHUNK_SIZE chars, con overlap.
    Un párrafo más largo que CHUNK_SIZE se parte por renglones (ver
    _split_parrafo_largo), nunca a mitad de palabra."""
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_parrafo_largo(p))
            continue
        if len(buf) + len(p) + 2 > CHUNK_SIZE and buf:
            chunks.append(buf)
            cola = _cola(buf)  # overlap con la cola anterior, sin partir palabras
            buf = f"{cola}\n\n{p}" if cola else p
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
    # Índice de payload sobre tipo_doc: las búsquedas filtran por él
    # (descripciones de puesto ↔ procedimientos, ver tools.py). Sin índice
    # Qdrant filtra igual pero escaneando. Idempotente: si ya existe, tira y
    # se ignora — por eso va fuera del if (la colección ya existe en prod).
    try:
        client.create_payload_index(
            collection_name=config.PROC_COLLECTION,
            field_name="metadata.tipo_doc",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


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
    # Para una descripción de puesto el header se escribe con las palabras que
    # usa el reclutador ("perfil buscado", "se busca") — la consulta típica es
    # "buscá alguien para <puesto>", no "descripción de puesto de <puesto>".
    tipo = doc.get("tipo", "procedimiento")
    puestos_txt = ", ".join(doc.get("puestos") or []) or "todos"
    if tipo == TIPO_DESCRIPCION_PUESTO:
        header = (
            f"Descripción de puesto / perfil buscado para el puesto: "
            f"{doc.get('titulo', '')}\n"
            f"Puesto: {puestos_txt}\n"
            f"Se busca / se necesita una persona para: {puestos_txt}"
        )
    else:
        header = (
            f"{tipo.capitalize()}: {doc.get('titulo', '')}\n"
            f"Puestos: {puestos_txt}"
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
                    "tipo_doc": tipo,
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
