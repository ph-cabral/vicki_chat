from sqlalchemy import create_engine, text
from datetime import datetime
from app.config import config

engine = create_engine(config.DATABASE_URL)

def ensure_session(session_id: str, user_id: str):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO agent.chat_sessions (session_id, user_id)
            VALUES (:session_id, :user_id)
            ON CONFLICT DO NOTHING
        """), {"session_id": session_id, "user_id": user_id})

def save_message(session_id: str, user_id: str, role: str, content: str):
    ensure_session(session_id, user_id)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO agent.chat_messages (session_id, user_id, role, content)
            VALUES (:session_id, :user_id, :role, :content)
        """), {"session_id": session_id, "user_id": user_id, "role": role, "content": content})

def get_history(session_id: str, user_id: str, limit: int = 20) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT role, content FROM agent.chat_messages
            WHERE session_id = :session_id AND user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"session_id": session_id, "user_id": user_id, "limit": limit})
        rows = result.fetchall()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]

