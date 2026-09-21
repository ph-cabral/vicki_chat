import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Falta la variable de entorno {name} (definila en .env)")
    return v


def _database_url() -> str:
    """DATABASE_URL directa, o armada desde DB_* (como pasa docker-compose)."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("DB_HOST")
    if host:
        user = os.getenv("DB_USER", "")
        pwd = os.getenv("DB_PASSWORD", "")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "")
        return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
    raise RuntimeError("Falta DATABASE_URL (o DB_HOST/DB_USER/DB_PASSWORD/DB_NAME) en el entorno")


class Config:
    HIK_USER: str = _required("HIK_USER")
    HIK_PASS: str = _required("HIK_PASS")
    HIK_IPS: list = [ip.strip() for ip in os.getenv("HIK_IPS", "").split(",") if ip.strip()]
    ANTHROPIC_KEY: str = os.getenv("ANTHROPIC_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://n8n_qdrant:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "postulantes")  # fallback si el router no elige
    # Colección de documentos por puesto (ingesta desde ever /rrhh/puestos):
    # procedimientos, instructivos Y descripciones de puesto. Se separan por
    # metadata.tipo_doc — ver tools.py::_filtro_tipo_doc.
    PROC_COLLECTION: str = os.getenv("PROC_COLLECTION", "procedimientos")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    TOP_K: int = int(os.getenv("TOP_K", "8"))
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "30"))
    CORS_ORIGINS: list = [o.strip() for o in os.getenv("CORS", "*").split(",") if o.strip()] or ["*"]
    TZ: str = os.getenv("TZ", "America/Argentina/Buenos_Aires")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Sin default con credenciales hardcodeadas: tiene que venir del entorno.
    DATABASE_URL: str = _database_url()
    # ── RAG / velocidad (nuevas) ──────────────────────────────────────────────
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "10"))
    QDRANT_CACHE_TTL: int = int(os.getenv("QDRANT_CACHE_TTL", "60"))  # cache de lista de colecciones
    ROUTER_MAX_TOKENS: int = int(os.getenv("ROUTER_MAX_TOKENS", "200"))
    # Cuántos chunks de descripción de puesto se inyectan al buscar candidatos.
    # Chico a propósito: es contexto de apoyo, no puede desplazar a los CVs.
    PERFIL_TOP_K: int = int(os.getenv("PERFIL_TOP_K", "4"))
    # Piso de relevancia de la descripción de puesto. SIN piso, Qdrant devuelve
    # igual la descripción más cercana aunque sea de otro puesto, y el modelo
    # rotulaba la respuesta con ese puesto: una búsqueda de "administración"
    # salió como "candidatas ... para el puesto de Responsable de RRHH". Mismo
    # criterio que PROC_CONTEXT_MIN_SCORE: si no se acerca, mejor sin perfil.
    PERFIL_MIN_SCORE: float = float(os.getenv("PERFIL_MIN_SCORE", "0.35"))
    # ── Shortlist de candidatos ───────────────────────────────────────────────
    # Cuántas PERSONAS distintas se devuelven siempre, ordenadas de mayor a
    # menor cercanía al puesto. TOP_K cuenta CHUNKS, no personas: un CV largo
    # entra con varios chunks y se comía el cupo, así que con TOP_K=8 podían
    # salir 2 personas (o ninguna nueva si las mejores compartían CV). Se
    # sobre-pide chunks y se deduplica a CANDIDATOS_TOP_N personas.
    CANDIDATOS_TOP_N: int = int(os.getenv("CANDIDATOS_TOP_N", "5"))
    # Chunks por persona que entran al contexto (los mejores). Acota el prompt:
    # sin tope, un CV de 12 chunks desplaza a los otros 4 candidatos.
    CV_CHUNKS_POR_CANDIDATO: int = int(os.getenv("CV_CHUNKS_POR_CANDIDATO", "3"))
    # Piso de encaje: por DEBAJO de este score el candidato sigue entrando a la
    # shortlist pero marcado como ENCAJE DÉBIL (ver tools.py::search_cvs y
    # prompts.py::ENCAJE_DEBIL_RULES). NO se filtra: la shortlist es de tamaño
    # fijo y, si se recortara, una búsqueda sin match perfecto volvería a
    # terminar en "no tengo candidatos" — que es justo lo que se quería evitar.
    # Lo que arregla es lo contrario: que un CV de limpieza aparezca presentado
    # como candidato válido a un puesto administrativo sólo por ocupar el 5º
    # lugar. Cosine sobre text-embedding-3-small: un CV del rubro correcto anda
    # por 0.40-0.55; abajo de 0.35 ya suele ser otro palo.
    CANDIDATO_MIN_SCORE: float = float(os.getenv("CANDIDATO_MIN_SCORE", "0.35"))
    # ── Paginado de la shortlist ("dame 5" → "ahora los otros 5") ────────────
    # CANDIDATOS_TOP_N es el default; si el usuario pide una cantidad ("dame
    # 10 perfiles", "otros 3") manda esa, topeada acá. El tope existe porque
    # cada candidato arrastra chunks de CV al prompt.
    CANDIDATOS_TOP_N_MAX: int = int(os.getenv("CANDIDATOS_TOP_N_MAX", "15"))
    # Cuántos candidatos ya mostrados se pueden excluir en Qdrant de una. Es el
    # techo del paginado: con 5 por página, 120 son 24 páginas. Se excluyen con
    # un must_not sobre metadata.candidato_id (indexado), así que la búsqueda no
    # se encarece de forma apreciable.
    MAX_EXCLUIR: int = int(os.getenv("MAX_EXCLUIR", "120"))
    # Cuántas respuestas de búsqueda hacia atrás se leen para saber a quién ya
    # vio (main.py::_mostrados). Tiene que dar para MAX_EXCLUIR: con 5 por
    # respuesta, 24 respuestas ≈ 120 candidatos.
    MOSTRADOS_MAX_RESPUESTAS: int = int(os.getenv("MOSTRADOS_MAX_RESPUESTAS", "24"))
    # ── Pool ampliado cuando se pide un recorte que la búsqueda NO aplica ─────
    # La búsqueda es similitud de texto: no filtra por género, zona, edad ni
    # estudios. Si se pide uno de esos recortes sobre la shortlist normal (5),
    # el recorte se aplica sobre 5 CVs y la respuesta termina en "no hay
    # ninguna" aunque más abajo en la lista sí haya. Cuando aparece un recorte
    # así se traen top_n * FACTOR candidatos (tope POOL_MAX) y el modelo filtra
    # sobre ese pool. Con el pool grande entra 1 chunk por CV para no reventar
    # el prompt.
    RECORTE_POOL_FACTOR: int = int(os.getenv("RECORTE_POOL_FACTOR", "5"))
    RECORTE_POOL_MAX: int = int(os.getenv("RECORTE_POOL_MAX", "30"))
    RECORTE_CHUNKS_POR_CANDIDATO: int = int(os.getenv("RECORTE_CHUNKS_POR_CANDIDATO", "1"))
    # ── Corte de conversación ─────────────────────────────────────────────────
    # El session_id es fijo por usuario (user_<uid>): la charla no termina
    # nunca y el modelo sigue leyendo lo que se habló días atrás. El corte lo
    # marca el botón «Nueva conversación» del chat; esto agrega ADEMÁS un corte
    # automático por inactividad. 0 = apagado (sólo corte manual). Nada se
    # borra en ningún caso: cambia hasta dónde mira el modelo, no el historial.
    CONVERSACION_HORAS: float = float(os.getenv("CONVERSACION_HORAS", "0"))
    # ── Procedimientos/instructivos como contexto al BUSCAR CANDIDATOS ────────
    # Además del perfil (qué se pide), se inyecta qué HACE el puesto. Va con
    # piso de score porque la query es de candidatos, no de procedimientos: sin
    # el piso, Qdrant devuelve igual los K mejores aunque no tengan nada que ver
    # y ensucian el contexto. TOP_K chico por lo mismo que PERFIL_TOP_K.
    PROC_CONTEXT_TOP_K: int = int(os.getenv("PROC_CONTEXT_TOP_K", "3"))
    PROC_CONTEXT_MIN_SCORE: float = float(os.getenv("PROC_CONTEXT_MIN_SCORE", "0.35"))
    PROC_CONTEXT_EN_BUSQUEDA: bool = os.getenv("PROC_CONTEXT_EN_BUSQUEDA", "1").lower() not in ("0", "false", "no")
    # ── Archivos de CV ────────────────────────────────────────────────────────
    # Store que escribe vicki_mail (original + PDF + miniatura), montado acá de
    # SOLO LECTURA. Layout: <hash[:2]>/<hash>/{original.ext,doc.pdf,thumb.jpg}
    # — ver vicki_mail/app/cv_store.py.
    CV_STORE_DIR: str = os.getenv("CV_STORE_DIR", "/data/cv_store")
    # ── Datos de ventas (intent "ventas") ─────────────────────────────────────
    # URL del servicio de red de mcp-magnus (streamable-http), corriendo en una
    # PC/VM Windows de la LAN con acceso al SQL Server de Magnus — ver
    # vicki/mcp/mcp-magnus/README.md "Modo servicio de red". Vacío = el intent
    # "ventas" queda deshabilitado (avisa en vez de fallar en cada mensaje).
    MAGNUS_MCP_URL: str = os.getenv("MAGNUS_MCP_URL", "")
    MAGNUS_MCP_TIMEOUT: float = float(os.getenv("MAGNUS_MCP_TIMEOUT", "20"))
    # Camino preferido (2026-09-06): endpoint JSON plano del MISMO server, sin
    # protocolo MCP. El transporte streamable-http del SDK MCP no funciona en
    # srv-active (mcp 1.29.1 + starlette 1.6.0: acepta el TCP y nunca responde,
    # también contra 127.0.0.1) — ver mcp-magnus/README.md "Modo endpoint HTTP".
    # Ej: http://10.10.0.232:8765/sql. Si está seteada, se usa esta y se ignora
    # MAGNUS_MCP_URL; si está vacía, se cae al camino MCP de antes.
    MAGNUS_SQL_URL: str = os.getenv("MAGNUS_SQL_URL", "")
    # Token compartido (header X-Api-Token) — tiene que coincidir con el
    # HTTP_API_TOKEN del .env de mcp-magnus. Vacío = el server no lo exige.
    MAGNUS_API_TOKEN: str = os.getenv("MAGNUS_API_TOKEN", "")


config = Config()
