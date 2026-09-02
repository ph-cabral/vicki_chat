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

    top = _buscar_hits(query, cols, k, flt, vector, score_threshold)
    return "\n".join(_format_hit(c, p) for c, p in top)


def _buscar_hits(query, cols, k, flt, vector, score_threshold) -> list:
    """Parte "cruda" de search_collections: devuelve [(coleccion, point)]
    ordenado por score. Se separó del formateo porque la búsqueda de CVs
    además necesita la metadata de cada hit para armar la lista de candidatos
    que se muestra en la barra del chat — antes se perdía al concatenar todo
    en un string."""
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
    return hits[: k if len(cols) == 1 else k * 2]


def _filtro_descartes(candidato_ids: list[int] | None) -> qm.Filter | None:
    """Excluye candidatos descartados en la conversación (el "tacho" de la
    barra de CVs). Se filtra EN QDRANT y no después: si se filtrara acá, un
    descartado seguiría ocupando un lugar del top_k y el reclutador vería un
    candidato menos por cada uno que tiró. Sin descartes devuelve None → la
    búsqueda va sin filtro y no paga nada.

    Necesita índice de payload en metadata.candidato_id (lo crea
    ensure_indices_cv), si no Qdrant filtra escaneando."""
    ids = [int(c) for c in (candidato_ids or []) if c is not None]
    if not ids:
        return None
    return qm.Filter(must_not=[
        qm.FieldCondition(key="metadata.candidato_id", match=qm.MatchAny(any=ids))
    ])


def _candidato_de_hit(col: str, p) -> dict | None:
    """Datos mínimos del candidato desde el payload del punto. Devuelve None
    para los puntos que no son CVs (procedimientos y demás)."""
    meta = ((p.payload or {}).get("metadata") or {})
    if meta.get("tipo_doc"):
        return None
    nombre = " ".join(x for x in [meta.get("nombre"), meta.get("apellido")] if x).strip()
    cid = meta.get("candidato_id")
    hash_archivo = meta.get("hash_archivo")
    if not (cid or hash_archivo or nombre):
        return None
    return {
        "candidato_id": int(cid) if cid is not None else None,
        "nombre": meta.get("nombre") or nombre,
        "apellido": meta.get("apellido") or "",
        "nombre_completo": nombre or "Sin nombre",
        "email": meta.get("email") or "",
        "hash_archivo": hash_archivo,
        "score": round(float(p.score or 0.0), 4),
        "coleccion": col,
    }


def search_cvs(query: str, collections: list[str], k: int | None = None,
               vector: list[float] | None = None,
               descartados: list[int] | None = None,
               top_n: int | None = None) -> tuple[str, list[dict]]:
    """Búsqueda de CVs: devuelve (contexto formateado, candidatos).

    Los candidatos salen de los MISMOS hits que arma el contexto — no hay una
    segunda consulta a Qdrant ni una vuelta más al embedding. Vienen ordenados
    por relevancia y deduplicados: un CV largo entra con varios chunks y sin
    dedup la barra mostraría a la misma persona tres veces.

    SHORTLIST FIJA: se devuelven siempre las `top_n` (default
    config.CANDIDATOS_TOP_N) PERSONAS más cercanas, de mayor a menor. Antes se
    pedían TOP_K chunks y recién después se deduplicaba, así que la cantidad de
    personas dependía de cuán largos fueran los CVs: 8 chunks podían ser 2
    personas y la respuesta terminaba en "no tengo candidatos". Ahora se
    sobre-pide (top_n * chunks_por_candidato, una sola consulta a Qdrant, mismo
    vector) y se recorta a top_n personas con hasta CV_CHUNKS_POR_CANDIDATO
    chunks cada una.

    Sin piso de score a propósito: la shortlist es "lo más parecido que hay",
    no "lo que matchea". El encaje parcial lo explica el modelo, no se filtra
    acá — ver prompts.py::SHORTLIST_RULES.
    """
    top_n = top_n or config.CANDIDATOS_TOP_N
    por_cand = max(1, config.CV_CHUNKS_POR_CANDIDATO)
    # k cuenta CHUNKS: hay que pedir de más para que salgan top_n PERSONAS.
    k = max(k or config.TOP_K, top_n * por_cand)
    cols = [c for c in (collections or []) if c]
    if not cols:
        # mismo repliegue que search_collections: si el router se quedó sin
        # lista (Qdrant no respondió el listado), se usa la colección por
        # default antes que devolver "no hay candidatos"
        avail = [c for c in list_collections() if c != config.PROC_COLLECTION]
        if config.QDRANT_COLLECTION in avail:
            cols = [config.QDRANT_COLLECTION]
        elif avail:
            cols = avail[:1]
        else:
            return "", []
    hits = _buscar_hits(query, cols, k, _filtro_descartes(descartados), vector, None)

    # Agrupado por persona, respetando el orden por score de `hits`: el primer
    # chunk de alguien define su posición en la shortlist.
    candidatos: list[dict] = []
    vistos: dict = {}
    chunks: dict = {}  # clave → [(col, point)] (los mejores de esa persona)
    for col, p in hits:
        c = _candidato_de_hit(col, p)
        if not c:
            continue
        clave = c["candidato_id"] or c["hash_archivo"] or c["nombre_completo"].lower()
        if clave in vistos:
            # mismo candidato en otro chunk: queda el mejor score y, si todavía
            # hay cupo, el chunk suma contexto (otra parte del mismo CV)
            if c["score"] > vistos[clave]["score"]:
                vistos[clave]["score"] = c["score"]
            if len(chunks[clave]) < por_cand:
                chunks[clave].append((col, p))
            continue
        if len(candidatos) >= top_n:
            continue  # ya hay shortlist completa; el resto no entra al prompt
        vistos[clave] = c
        chunks[clave] = [(col, p)]
        c["posicion"] = len(candidatos) + 1
        candidatos.append(c)

    # El contexto va ordenado por candidato (no chunk por chunk intercalado):
    # el modelo tiene que poder leer a cada persona entera y de mayor a menor.
    partes: list[str] = []
    for c in candidatos:
        clave = c["candidato_id"] or c["hash_archivo"] or c["nombre_completo"].lower()
        partes.append(f"\n### Candidato #{c['posicion']} — {c['nombre_completo']}")
        partes.extend(_format_hit(col, p) for col, p in chunks[clave])
    return "\n".join(partes), candidatos


def cv_collections() -> list[str]:
    """Colecciones de CVs = todas menos la de procedimientos."""
    return [c for c in list_collections() if c != config.PROC_COLLECTION]


def diagnostico_cvs(collections: list[str] | None) -> str:
    """Por qué una búsqueda de CVs volvió VACÍA. Se llama sólo cuando no salió
    ningún candidato, así que no cuesta nada en el camino normal.

    Existe porque los tres modos de fallar terminaban en la misma respuesta
    ("no hay candidatos relevantes") y el reclutador no puede distinguir
    "no hay nadie parecido" de "el chat no está mirando ninguna colección":
      - Qdrant no responde       → list_collections() vuelve vacío
      - no existe la colección de CVs (ej. `cvs` nunca se creó porque todavía
        no entró ningún CV por vicki_mail, o el contenedor apunta a otro Qdrant)
      - la colección existe pero está vacía
    Devuelve "" si no hay nada raro que avisar (entonces sí: no hay parecidos).
    """
    try:
        todas = list_collections()
    except Exception:
        return "Qdrant no respondió el listado de colecciones."
    if not todas:
        return ("Qdrant no devolvió ninguna colección: el chat no tiene dónde "
                "buscar CVs (¿está caído o QDRANT_URL apunta a otro lado?).")
    cols = [c for c in (collections or []) if c] or cv_collections()
    if not cols:
        return (f"No existe ninguna colección de CVs en Qdrant "
                f"(hay: {', '.join(todas)}). Los CVs los ingesta vicki_mail en "
                f"{config.QDRANT_COLLECTION!r}.")
    try:
        total = sum(get_client().count(c, exact=False).count for c in cols)
    except Exception:
        log.exception("no pude contar puntos de las colecciones de CVs")
        return f"No pude leer las colecciones de CVs ({', '.join(cols)})."
    if total == 0:
        return f"Las colecciones de CVs ({', '.join(cols)}) están vacías: 0 CVs cargados."
    return ""


def ensure_indices_cv() -> None:
    """Índice de payload en metadata.candidato_id para las colecciones de CVs.
    Idempotente (si ya existe, Qdrant tira y se ignora). Sin esto, el filtro
    de descartados escanea la colección entera."""
    client = get_client()
    for col in list_collections():
        if col == config.PROC_COLLECTION:
            continue
        try:
            client.create_payload_index(
                collection_name=col,
                field_name="metadata.candidato_id",
                field_schema=qm.PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass


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
