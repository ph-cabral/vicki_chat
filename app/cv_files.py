"""Consultas y rutas del archivo de CV para la barra lateral del chat.

El archivo lo escribe vicki_mail (app/cv_store.py) en CV_STORE_DIR; acá se
monta de solo lectura. La ruta NO está en la base: se deriva del hash_archivo,
que ya es UNIQUE en rag_system.documento_aprobado.
"""
import logging
import os
import unicodedata

from app.config import config

log = logging.getLogger("cv_files")


def dir_hash(hash_archivo: str) -> str:
    return os.path.join(config.CV_STORE_DIR, hash_archivo[:2], hash_archivo)


def ruta_pdf(hash_archivo: str) -> str:
    return os.path.join(dir_hash(hash_archivo), "doc.pdf")


def ruta_thumb(hash_archivo: str) -> str:
    return os.path.join(dir_hash(hash_archivo), "thumb.jpg")


def ruta_original(hash_archivo: str) -> str | None:
    d = dir_hash(hash_archivo)
    try:
        for n in sorted(os.listdir(d)):
            if n.startswith("original"):
                return os.path.join(d, n)
    except OSError:
        pass
    return None


# Una sola consulta para todos los candidatos de la respuesta. DISTINCT ON no
# se usa: se traen las filas ordenadas y se arma el mapa en Python (son ≤ 20
# filas) para poder indexar además por hash_archivo, que es como vienen los
# puntos viejos de Qdrant que no tienen candidato_id en la metadata.
_SQL_DOCS = """
SELECT d.id, d.candidato_id, d.hash_archivo, d.nombre_archivo, d.mime_type,
       d.archivo_pdf, d.archivo_thumb, d.aprobado_at,
       c.nombre, c.apellido, c.email
  FROM rag_system.documento_aprobado d
  LEFT JOIN rag_system.candidato c ON c.id = d.candidato_id
 WHERE d.tipo = 'CV'
   AND (d.candidato_id = ANY($1::bigint[]) OR d.hash_archivo = ANY($2::text[]))
 ORDER BY d.aprobado_at DESC NULLS LAST
"""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


async def enriquecer_candidatos(pool, candidatos: list[dict], respuesta: str) -> list[dict]:
    """Le agrega a cada candidato el documento (para pedir el archivo) y si
    Vicki lo nombró en la respuesta.

    Una sola consulta por mensaje, por candidato_id (índice parcial
    documento_aprobado_candidato_cv_idx) o por hash_archivo (UNIQUE).
    """
    if not candidatos:
        return []
    ids = [c["candidato_id"] for c in candidatos if c.get("candidato_id")]
    hashes = [c["hash_archivo"] for c in candidatos if c.get("hash_archivo")]
    filas = []
    if ids or hashes:
        try:
            filas = await pool.fetch(_SQL_DOCS, ids, hashes)
        except Exception:
            log.exception("no pude enriquecer los candidatos con su documento")

    por_id: dict = {}
    por_hash: dict = {}
    for r in filas:  # ya vienen del más nuevo al más viejo
        if r["candidato_id"] is not None:
            por_id.setdefault(r["candidato_id"], r)
        if r["hash_archivo"]:
            por_hash.setdefault(r["hash_archivo"], r)

    respuesta_norm = _norm(respuesta)
    salida = []
    for c in candidatos:
        r = por_id.get(c.get("candidato_id")) or por_hash.get(c.get("hash_archivo"))
        nombre = c.get("nombre") or (r["nombre"] if r else "") or ""
        apellido = c.get("apellido") or (r["apellido"] if r else "") or ""
        completo = (f"{nombre} {apellido}").strip() or c.get("nombre_completo") or "Sin nombre"
        # "mencionado": el modelo solo puede nombrar candidatos que están en el
        # contexto (ver GROUNDING_RULES), así que alcanza con buscar el apellido
        # —o el nombre si no hay apellido— en el texto de la respuesta.
        clave = _norm(apellido) or _norm(nombre)
        salida.append({
            "candidato_id": c.get("candidato_id") or (r["candidato_id"] if r else None),
            "documento_id": r["id"] if r else None,
            "nombre": completo,
            "email": c.get("email") or (r["email"] if r else "") or "",
            "score": c.get("score"),
            "archivo": bool(r and r["archivo_pdf"]),
            "thumb": bool(r and r["archivo_thumb"]),
            "nombre_archivo": r["nombre_archivo"] if r else None,
            "mencionado": bool(clave and clave in respuesta_norm),
        })
    return salida


_SQL_DOC_ARCHIVO = """
SELECT hash_archivo, nombre_archivo, mime_type, texto_limpio, texto_raw
  FROM rag_system.documento_aprobado
 WHERE id = $1
"""


async def documento(pool, documento_id: int):
    return await pool.fetchrow(_SQL_DOC_ARCHIVO, documento_id)
