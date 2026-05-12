from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from anthropic import Anthropic
from openai import OpenAI
import psycopg2
import uuid
import os
import json
import logging



app = FastAPI()
log = logging.getLogger("uvicorn.error")

# Cliente principal: Anthropic
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

# Cliente fallback: OpenAI (también usado para embeddings)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# def search_cvs(query: str, limit: int = 5) -> str:
#     resp = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
#     vector = resp.data[0].embedding
#     results = qdrant.query_points("cvs", query=vector, limit=limit, with_payload=True)

#     context = ""
#     for r in results.points:
#         nombre = r.payload.get("metadata", {}).get("candidato_nombre", "N/A")
#         content = r.payload.get("content", "")
#         score = r.score
#         context += f"\n--- Candidato: {nombre} (relevancia: {score:.2f}) ---\n{content}\n"
#     return context
def search_cvs(query: str, limit: int = 5) -> str:
    import json
    resp = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
    vector = resp.data[0].embedding
    results = qdrant.query_points("cvs", query=vector, limit=limit, with_payload=True)

    context = ""
    for r in results.points:
        meta = r.payload.get("metadata", {})
        nombre = meta.get("candidato_nombre", "N/A")
        email = meta.get("candidato_email", "N/A")
        content = r.payload.get("content", "")
        empresas = meta.get("empresas", [])

        context += f"\n--- Candidato: {nombre} (relevancia: {r.score:.2f}) ---\n"
        context += f"Email: {email}\n"
        context += f"{content}\n"

        if empresas:
            context += "\nEXPERIENCIA LABORAL:\n"
            for e in empresas:
                context += (
                    f"- {e.get('puesto','')} en {e.get('empresa','')} "
                    f"({e.get('fecha_inicio','')} - {e.get('fecha_finalizacion','')})\n"
                    f"  {e.get('descripcion','')}\n"
                )
    return context

def llm_complete(system_prompt: str, messages: list) -> str:
    """Intenta Anthropic; si falla, cae a OpenAI."""
    try:
        msgs = [m for m in messages if m["role"] != "system"]
        resp = anthropic_client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            max_tokens=1024,
            system=system_prompt,
            messages=msgs,
            temperature=0.3,
        )
        return resp.content[0].text
    except Exception as e:
        log.warning(f"Anthropic falló, fallback OpenAI: {e}")

    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        log.error(f"OpenAI también falló: {e}")
        raise


def llm_stream(system_prompt: str, messages: list):
    """Generador que va emitiendo tokens. Anthropic con fallback a OpenAI."""
    # 1) Anthropic streaming
    try:
        msgs = [m for m in messages if m["role"] != "system"]
        with anthropic_client.messages.stream(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            max_tokens=1024,
            system=system_prompt,
            messages=msgs,
            temperature=0.3,
        ) as stream:
            for text in stream.text_stream:
                yield text
        return
    except Exception as e:
        log.warning(f"Anthropic stream falló, fallback OpenAI: {e}")

    # 2) OpenAI streaming
    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        log.error(f"OpenAI stream también falló: {e}")
        raise


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    uid = req.user_id

    ensure_session(sid, uid)
    history = load_history(sid, uid)

    cv_context = search_cvs(req.message)
    user_msg = req.message
    if cv_context.strip():
        user_msg += f"\n\n[CONTEXTO DE CVs ENCONTRADOS EN LA BASE DE DATOS]:\n{cv_context}"

    save_message(sid, uid, "human", req.message)

    messages = history + [{"role": "user", "content": user_msg}]
    answer = llm_complete(SYSTEM_PROMPT, messages)

    save_message(sid, uid, "ai", answer)
    return ChatResponse(response=answer, session_id=sid)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Endpoint con streaming SSE. El cliente recibe los tokens a medida que se generan."""
    sid = req.session_id or str(uuid.uuid4())
    uid = req.user_id

    ensure_session(sid, uid)
    history = load_history(sid, uid)

    cv_context = search_cvs(req.message)
    user_msg = req.message
    if cv_context.strip():
        user_msg += f"\n\n[CONTEXTO DE CVs ENCONTRADOS EN LA BASE DE DATOS]:\n{cv_context}"

    save_message(sid, uid, "human", req.message)
    messages = history + [{"role": "user", "content": user_msg}]

    def event_generator():
        # Enviar session_id primero
        yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"

        full_answer = ""
        try:
            for token in llm_stream(SYSTEM_PROMPT, messages):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Guardar respuesta completa al final
        try:
            save_message(sid, uid, "ai", full_answer)
        except Exception as e:
            log.error(f"Error guardando mensaje: {e}")

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # desactiva buffering en nginx
        },
    )


@app.get("/history/{session_id}")
def history(session_id: str, user_id: str = "1"):
    msgs = load_history(session_id, user_id)
    return {"history": msgs}

@app.get("/health")
def health():
    return {"status": "ok"}