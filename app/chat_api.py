from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
import psycopg2
import uuid
import os

app = FastAPI()

oa = OpenAI(
    api_key=os.getenv("ANTHROPIC_KEY"),
    base_url="https://api.anthropic.com"
)

# Cliente OpenAI para embeddings
oa_embeddings = OpenAI(
    api_key=os.getenv("OPEN_API_KEY")
)


qdrant = QdrantClient(url="http://n8n_qdrant:6333")

def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "n8n_sql"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "n8n"),
        user=os.getenv("DB_USER", "n8n"),
        password=os.getenv("DB_PASSWORD", "")
    )

SYSTEM_PROMPT = """Sos Viki, una asistente virtual especializada en selección de personal.
Tu rol es ayudar a encontrar candidatos ideales en la base de datos de CVs.
Cuando el usuario pida buscar candidatos para un puesto, usá el contexto de CVs que se te proporciona.
Respondé en español, de forma clara y profesional.
Si no encontrás candidatos relevantes, decilo honestamente.
Presentá los candidatos con nombre, experiencia relevante y por qué encajan en el puesto."""

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = "1"

class ChatResponse(BaseModel):
    response: str
    session_id: str

def ensure_session(session_id: str, user_id: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO agent.chat_sessions (session_id, user_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
    """, (session_id, user_id))
    db.commit()
    cur.close()
    db.close()

def save_message(session_id: str, user_id: str, role: str, content: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO agent.chat_messages (session_id, user_id, role, content)
        VALUES (%s, %s, %s, %s)
    """, (session_id, user_id, role, content))
    db.commit()
    cur.close()
    db.close()

def load_history(session_id: str, user_id: str) -> list:
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT role, content FROM agent.chat_messages
        WHERE session_id = %s AND user_id = %s
        ORDER BY created_at ASC
    """, (session_id, user_id))
    rows = cur.fetchall()
    cur.close()
    db.close()
    
    role_map = {"human": "user", "ai": "assistant"}
    return [{"role": role_map.get(r[0], r[0]), "content": r[1]} for r in rows]

def search_cvs(query: str, limit: int = 5) -> str:
    # resp = oa.embeddings.create(input=query, model="text-embedding-3-small")
    resp = oa_embeddings.embeddings.create(input=query, model="text-embedding-3-small")
    vector = resp.data[0].embedding
    results = qdrant.query_points("cvs", query=vector, limit=limit, with_payload=True)

    context = ""
    for r in results.points:
        nombre = r.payload.get("metadata", {}).get("candidato_nombre", "N/A")
        content = r.payload.get("content", "")
        score = r.score
        context += f"\n--- Candidato: {nombre} (relevancia: {score:.2f}) ---\n{content}\n"
    return context

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    uid = req.user_id

    ensure_session(sid, uid)

    history = load_history(sid, uid)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    cv_context = search_cvs(req.message)
    user_msg = req.message
    if cv_context.strip():
        user_msg += f"\n\n[CONTEXTO DE CVs ENCONTRADOS EN LA BASE DE DATOS]:\n{cv_context}"

    save_message(sid, uid, "human", req.message)

    messages.append({"role": "user", "content": user_msg})

    completion = oa.chat.completions.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        messages=messages,
        temperature=0.3
    )

    answer = completion.choices[0].message.content

    save_message(sid, uid, "ai", answer)

    return ChatResponse(response=answer, session_id=sid)

@app.get("/history/{session_id}")
def history(session_id: str, user_id: str = "1"):
    msgs = load_history(session_id, user_id)
    return {"history": msgs}

@app.get("/health")
def health():
    return {"status": "ok"}
