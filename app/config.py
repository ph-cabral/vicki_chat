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
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "cvs")  # fallback si el router no elige
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    TOP_K: int = int(os.getenv("TOP_K", "5"))
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


config = Config()
