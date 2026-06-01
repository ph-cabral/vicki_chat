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
import base64
import logging
import threading, time
from app.tool import take_camera_snapshot, create_employee, upload_face, resolve_location, read_snapshot, delete_snapshot, _deferred_upload_face


app = FastAPI()
log = logging.getLogger("uvicorn.error")

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(url="http://n8n_qdrant:6333")


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = "1"
    gender: str = None
    location: str = None
    retake: bool = False

def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "n8n_sql"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "n8n"),
        user=os.getenv("DB_USER", "n8n"),
        password=os.getenv("DB_PASSWORD", "")
    )


# ====== Tabla draft de empleado en creación ======
def ensure_employee_state_table():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent.employee_draft (
            session_id TEXT PRIMARY KEY,
            photo_b64 TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    db.commit()
    cur.close()
    db.close()

ensure_employee_state_table()


def save_draft(sid: str, photo_b64: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO agent.employee_draft (session_id, photo_b64)
        VALUES (%s, %s)
        ON CONFLICT (session_id) DO UPDATE
        SET photo_b64 = EXCLUDED.photo_b64, created_at = NOW()
    """, (sid, photo_b64))
    db.commit()
    cur.close()
    db.close()


def get_draft(sid: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT photo_b64 FROM agent.employee_draft WHERE session_id = %s", (sid,))
    row = cur.fetchone()
    cur.close()
    db.close()
    return row[0] if row else None


def del_draft(sid: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM agent.employee_draft WHERE session_id = %s", (sid,))
    db.commit()
    cur.close()
    db.close()


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
    gender: str = None
    location: str = None


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


# ====== Manejador del flujo "crear empleado" ======
# def handle_employee_flow(sid: str, uid: str, msg: str, gender: str = None, location: str = None):
#     """Devuelve un string si el mensaje pertenece al flujo, o None."""
#     text = msg.strip()
#     low = text.lower()

#     # Paso 1: disparador
#     if low.startswith("/crea un empleado") or low.startswith("/crea empleado") or low == "/crea":
#         try:
#             jpg = take_camera_snapshot()
#             b64 = base64.b64encode(jpg).decode()
#             save_draft(sid, b64)
#             # return (
#             #     "📸 Foto capturada del reloj:\n\n"
#             #     f"![foto](data:image/jpeg;base64,{b64})\n\n"
#             #     "Seleccioná sexo y ubicación, luego escribí el nombre."
#             # )
#             # return f" ✅ Foto tomada, ingresa datos... "
#             return (
#                 "📸 Foto tomada:\n\n"
#                 f"![foto](data:image/jpeg;base64,{b64})\n\n"
#                 "Seleccioná sexo y ubicación, luego escribí el nombre."
#             )
#         except Exception as e:
#             return f"❌ Error tomando foto del reloj: {e}"

#     # Paso 2: hay draft + viene metadata estructurada → crear directo
#     draft_b64 = get_draft(sid)
#     if draft_b64 and gender and location:
#         try:
#             g = (gender or "").strip().lower()
#             gender_norm = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(g)
#             if not gender_norm:
#                 return "❌ Sexo inválido."
#             name_part = text
#             if not name_part:
#                 return "❌ Falta el nombre."
#             try:
#                 resolve_location(location)
#             except ValueError as ve:
#                 return f"❌ {ve}"

#             emp_no, ip = create_employee(name=name_part, gender=gender_norm, location=location)
#             try:
#                 upload_face(emp_no, read_snapshot(), ip=ip)
#                 delete_snapshot()
#                 face_msg = "con foto cargada"
#             except Exception as fe:
#                 face_msg = f"⚠️ creado pero falló la foto: {fe}"

#             del_draft(sid)
#             # return f"✅ Empleado **{emp_no}** — {name_part} ({gender_norm}) @ {location.lower()} ({ip}) {face_msg}"
#             return (
#                 f"✅ {name_part} fué ingresado en el reloj de {location.lower()}\n\n"
#                 f"![foto](data:image/jpeg;base64,{draft_b64})"
#             )
#         except Exception as e:
#             return f"❌ Error creando empleado: {e}"

    # return None

def handle_employee_flow(sid: str, uid: str, msg: str, gender: str = None, location: str = None, retake: bool = False):
    """Devuelve un string si el mensaje pertenece al flujo, o None."""
    print(f"[EMP] msg={msg!r} gender={gender} location={location} retake={retake}", flush=True)
    text = msg.strip()
    low = text.lower()

    # Paso 1: disparador
    if low.startswith("/crea un empleado") or low.startswith("/crea empleado") or low == "/crea":
        try:
            jpg = take_camera_snapshot()
            b64 = base64.b64encode(jpg).decode()
            save_draft(sid, b64)
            return (
                "📸 Foto tomada:\n\n"
                f"![foto](data:image/jpeg;base64,{b64})\n\n"
                "Seleccioná sexo y ubicación, luego escribí el nombre."
            )
        except Exception as e:
            return f"❌ Error tomando foto del reloj: {e}"

    # Paso 1b: re-tomar foto (botón "Sacar de nuevo")
    if retake and get_draft(sid):
        try:
            ip = resolve_location(location) if location else None
            jpg = take_camera_snapshot(ip=ip)
            b64 = base64.b64encode(jpg).decode()
            save_draft(sid, b64)
            return (
                "📸 Foto tomada:\n\n"
                f"![foto](data:image/jpeg;base64,{b64})\n\n"
                "Seleccioná sexo y ubicación, luego escribí el nombre."
            )
        except Exception as e:
            return f"❌ Error tomando foto del reloj: {e}"


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    uid = req.user_id

    ensure_session(sid, uid)

    # Flujo crear empleado (intercepta antes del LLM)
    emp_answer = handle_employee_flow(sid, uid, req.message, req.gender, req.location, req.retake)
    # emp_answer = handle_employee_flow(sid, uid, req.message, req.gender, req.location)
    if emp_answer is not None:
        save_message(sid, uid, "human", req.message)
        save_message(sid, uid, "ai", emp_answer)
        return ChatResponse(response=emp_answer, session_id=sid)

    # Flujo normal CVs
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
    sid = req.session_id or str(uuid.uuid4())
    uid = req.user_id

    ensure_session(sid, uid)

    # Flujo crear empleado también en streaming
    emp_answer = handle_employee_flow(sid, uid, req.message, req.gender, req.location)
    if emp_answer is not None:
        save_message(sid, uid, "human", req.message)
        save_message(sid, uid, "ai", emp_answer)

        def emp_gen():
            yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': emp_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(emp_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "Connection": "keep-alive",
                                          "X-Accel-Buffering": "no"})

    history = load_history(sid, uid)
    cv_context = search_cvs(req.message)
    user_msg = req.message
    if cv_context.strip():
        user_msg += f"\n\n[CONTEXTO DE CVs ENCONTRADOS EN LA BASE DE DATOS]:\n{cv_context}"

    save_message(sid, uid, "human", req.message)
    messages = history + [{"role": "user", "content": user_msg}]

    def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"
        full_answer = ""
        try:
            for token in llm_stream(SYSTEM_PROMPT, messages):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
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
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
def history(session_id: str, user_id: str = "1"):
    msgs = load_history(session_id, user_id)
    return {"history": msgs}


@app.get("/health")
def health():
    return {"status": "ok"}