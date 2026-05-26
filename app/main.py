from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncpg
import base64
import traceback
import os
import threading, time


from langchain_core.messages import HumanMessage, AIMessage
from app.graph import build_graph
from app.config import config
from app.memory import build_checkpointer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.tools import SNAPSHOT_PATH
from app.tool import take_camera_snapshot, create_employee, upload_face, resolve_location, read_snapshot, delete_snapshot, _deferred_upload_face
from app.chat_api import del_draft


app = FastAPI(
    title="Chat CV Agent",
    description="Agente de selección de personal — Basdonax AI",
    version="1.0.0",
)


os.makedirs("/code/snapshots", exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="/code/snapshots"), name="snapshots")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


graph = None
db_pool = None


@app.get("/snapshot")
def snapshot():
    return FileResponse(SNAPSHOT_PATH, media_type="image/jpeg")


@app.on_event("startup")
async def startup():
    global db_pool, graph
    db_pool = await asyncpg.create_pool(config.DATABASE_URL)
    cp = await build_checkpointer()
    graph = build_graph().compile(checkpointer=cp)

    # Tabla para draft de empleado en creación
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS agent.employee_draft (
            session_id TEXT PRIMARY KEY,
            photo_b64 TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)


@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None


def user_id_from_session(session_id: str) -> int:
    try:
        return int(session_id.split("_")[1])
    except:
        return 1


@app.get("/history/{session_id}")
async def history(session_id: str):
    user_id = user_id_from_session(session_id)
    rows = await db_pool.fetch(
        "SELECT role, content FROM agent.chat_messages "
        "WHERE session_id = $1 AND user_id = $2 "
        "ORDER BY created_at ASC",
        session_id, user_id,
    )
    return {"history": [{"role": r["role"], "content": r["content"]} for r in rows]}


# ====== Flujo crear empleado ======
async def handle_employee_flow(session_id: str, message: str,
                               gender: Optional[str] = None,
                               location: Optional[str] = None):
    """Devuelve string si el mensaje cae en el flujo, o None."""
    text = message.strip()
    low = text.lower()

    triggers = ("/crea un empleado", "/crear un empleado", "/crea empleado",
                "/crear empleado", "/crea", "/crear")

    # Paso 1: disparador
    if any(low == t or low.startswith(t + " ") or low == t for t in triggers) or low in triggers:
        try:
            jpg = take_camera_snapshot()
            b64 = base64.b64encode(jpg).decode()
            await db_pool.execute("""
                INSERT INTO agent.employee_draft (session_id, photo_b64)
                VALUES ($1, $2)
                ON CONFLICT (session_id) DO UPDATE
                SET photo_b64 = EXCLUDED.photo_b64, created_at = NOW()
            """, session_id, b64)

            return (
            #     "📸 Foto capturada del reloj:\n\n"
            #     "Seleccioná sexo y ubicación, luego escribí el nombre."
            # )
            # f"✅ {name_part} fué ingresado en el reloj de {location.lower()}\n\n"
                f"![foto](data:image/jpeg;base64,{draft_b64})"
            )
            # return "✅ Foto tomada, ingresa datos..."
        except Exception as e:
            return f"❌ Error tomando foto del reloj: {e}"

    # Paso 2: hay draft + viene metadata estructurada → crear directo
    row = await db_pool.fetchrow(
        "SELECT photo_b64 FROM agent.employee_draft WHERE session_id = $1",
        session_id
    )
    if row and gender and location:
        try:
            g = (gender or "").strip().lower()
            gender_norm = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(g)
            if not gender_norm:
                return "❌ Sexo inválido."
            name_part = text
            if not name_part:
                return "❌ Falta el nombre."
            try:
                resolve_location(location)
            except ValueError as ve:
                return f"❌ {ve}"

            emp_no, ip = create_employee(name=name_part, gender=gender_norm, location=location)

            jpg = read_snapshot()
            # threading.Thread(
            #     # target=_deferred_upload_face,
            #     args=(emp_no, ip, jpg, 3),
            #     daemon=True,
            # ).start()
            try:
                upload_face(emp_no, jpg, ip=ip)
                delete_snapshot()
                foto_msg = "foto subida ✅"
            except Exception as e:
                foto_msg = f"foto pendiente ({e})"

            del_draft(session_id)
            return f"✅ {name_part} se creo en el reloj de {location.lower()}"
            # return (
            #     f"✅ {name_part} fué ingresado en el reloj de {location.lower()}\n\n"
            #     f"![foto](data:image/jpeg;base64,{draft_b64})"
            # )
            # return f"✅ Empleado **{emp_no}** — {name_part} ({gender_norm}) @ {location.lower()} ({ip}) — {foto_msg}"
        except Exception as e:
            return f"❌ Error creando empleado: {e}"

    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or "user_1"
    user_id = user_id_from_session(session_id)

    try:
        await db_pool.execute(
            """
            INSERT INTO agent.chat_sessions (session_id, user_id)
            VALUES ($1, $2)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id, user_id
        )

        # === INTERCEPT: crear empleado ===
        emp_answer = await handle_employee_flow(session_id, request.message, request.gender, request.location)
        if emp_answer is not None:
            await db_pool.execute(
                "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
                session_id, user_id, "human", request.message
            )
            await db_pool.execute(
                "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
                session_id, user_id, "ai", emp_answer
            )
            return ChatResponse(response=emp_answer, session_id=session_id, intent="employee")

        # === Flujo normal CVs ===
        await db_pool.execute(
            "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
            session_id, user_id, "human", request.message
        )

        rows = await db_pool.fetch(
            "SELECT role, content FROM agent.chat_messages WHERE user_id = $1 ORDER BY created_at ASC",
            user_id
        )
        history = []
        for r in rows:
            if r["role"] == "human":
                history.append(HumanMessage(content=r["content"]))
            else:
                history.append(AIMessage(content=r["content"]))

        graph_config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "messages": history,
            "session_id": session_id,
            "intent": None,
            "user_message": None,
            "retrieved_docs": None,
            "final_response": None,
        }

        result = await graph.ainvoke(initial_state, config=graph_config)
        answer = result["final_response"]

        await db_pool.execute(
            "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
            session_id, user_id, "ai", answer
        )

        return ChatResponse(
            response=answer,
            session_id=session_id,
            intent=result.get("intent"),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat-cv-agent"}