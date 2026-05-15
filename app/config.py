from dotenv import load_dotenv
import os


load_dotenv()

class Config:
    ANTHROPIC_KEY: str = os.getenv("ANTHROPIC_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    OPENAI_API_KEY: str = os.getenv("OPEN_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://n8n_qdrant:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "cvs")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "30"))
    CORS_ORIGINS: list = os.getenv("CORS", "*").split(",")
    TZ: str = os.getenv("TZ", "America/Argentina/Buenos_Aires")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://n8n:3v3rW3ar@n8n_sql:5432/n8n"
    )

config = Config()

