import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import config
from app.graph import build_graph
from app.summary import load_context, strip_b64, update_summary
from app.tool import (
    LOCATIONS,
    SNAPSHOT_PATH,
    create_employee_all,
    resolve_location,
    take_camera_snapshot,
    upload_face_all,
)
from app.user_registry import reserve_user_id

logger = logging.getLogger(__name__)

graph = None
db_pool: asyncpg.Pool | None = None
_bg_tasks: set = set()  # referencias fuertes para que el GC no cancele tareas en curso


def _spawn_bg(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global db_pool, graph
    try:
        db_pool = await asyncpg.create_pool(
            config.DATABASE_URL,
            min_size=2,
            max_size=10,
            timeout=30,
            command_timeout=60,
        )
        logger.info("✅ Pool de base de datos creado correctamente.")
        await db_pool.execute("CREATE SCHEMA IF NOT EXISTS agent")
        graph = build_graph().compile()
        logger.info("✅ Grafo compilado (sin checkpointer, historial vía load_context).")

        await db_pool.execute("""
            CREATE TABLE IF NOT EXISTS agent.employee_draft (
                session_id TEXT PRIMARY KEY,
                photo_b64 TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await db_pool.execute("""
            CREATE TABLE IF NOT EXISTS agent.chat_summary (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                summarized_through TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Tablas verificadas/creadas.")
    except Exception as e:
        logger.error(f"❌ Error crítico en startup: {e}")
        raise
    try:
        yield
    finally:
        if db_pool is not None:
            await db_pool.close()


app = FastAPI(
    title="Chat CV Agent",
    description="Agente de selección de personal — Basdonax AI",
    version="1.1.0",
    lifespan=lifespan,
)

os.makedirs("/code/snapshots", exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="/code/snapshots"), name="snapshots")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials="*" not in config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/snapshot")
def snapshot():
    if not os.path.exists(SNAPSHOT_PATH):
        raise HTTPException(404, "no hay snapshot disponible")
    return FileResponse(SNAPSHOT_PATH, media_type="image/jpeg")


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
    except (IndexError, ValueError):
        return 1


async def del_draft(session_id: str):
    await db_pool.execute(
        "DELETE FROM agent.employee_draft WHERE session_id = $1", session_id
    )


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


async def handle_employee_flow(session_id: str, message: str,
                               gender: Optional[str] = None,
                               location: Optional[str] = None):
    text = message.strip()
    low = text.lower()

    triggers = ("/crea un empleado", "/crear un empleado", "/crea empleado",
                "/crear empleado", "/crea", "/crear")

    # Paso 1: disparador → pedir ubicación (sin tomar foto)
    if any(low == t or low.startswith(t + " ") for t in triggers):
        return "📍 ¿Desde qué reloj querés sacar la foto?\n\n[LOC_PICK]"

    # Paso 2: viene location SIN draft → tomar foto desde ese reloj
    row = await db_pool.fetchrow(
        "SELECT photo_b64 FROM agent.employee_draft WHERE session_id = $1",
        session_id
    )
    if not row and location and not gender:
        try:
            ip = resolve_location(location)
        except ValueError as ve:
            return f"❌ {ve}"
        try:
            # la captura usa ffmpeg (bloqueante) → thread para no frenar el event loop
            jpg = await asyncio.to_thread(take_camera_snapshot, ip)
            b64 = base64.b64encode(jpg).decode()
            await db_pool.execute("""
                INSERT INTO agent.employee_draft (session_id, photo_b64)
                VALUES ($1, $2)
                ON CONFLICT (session_id) DO UPDATE
                SET photo_b64 = EXCLUDED.photo_b64, created_at = NOW()
            """, session_id, b64)
            return (
                f"📸 Foto tomada desde {location}.\n\n"
                f"![foto](data:image/jpeg;base64,{b64})\n\n"
                "Seleccioná sexo y escribí el nombre."
            )
        except Exception as e:
            logger.exception("error tomando foto")
            return f"❌ Error tomando foto del reloj: {e}"

    # Paso 3: hay draft + gender + location + nombre → crear
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

            async with db_pool.acquire() as conn:
                new_id = await reserve_user_id(conn, external_ref=f"vicki:{session_id}")

            emp_no = str(new_id)
            # alta en relojes: llamadas HTTP bloqueantes → thread
            cre = await asyncio.to_thread(
                create_employee_all, name_part, gender_norm, emp_no
            )

            SEXO_MAP = {"male": "M", "female": "F"}
            async with db_pool.acquire() as conn:
                await conn.execute(
                    'INSERT INTO everwear.legajo ("employeeNo", estado, nombre, sexo, "createdAt", "updatedAt") '
                    "VALUES ($1::text, 'activo', $2::text, $3::text, now(), now()) "
                    'ON CONFLICT ("employeeNo") DO NOTHING',
                    emp_no, name_part, SEXO_MAP[gender_norm],
                )

            # usar la foto del draft (por sesión) y no el archivo global compartido:
            # evita mezclar fotos si dos sesiones crean empleados a la vez
            jpg = base64.b64decode(row["photo_b64"])
            up = await asyncio.to_thread(upload_face_all, emp_no, jpg)

            await del_draft(session_id)
            ok = [l for l, r in cre.items() if r == "ok"]
            fail = {l: (cre[l], up.get(l)) for l in LOCATIONS if cre[l] != "ok" or up.get(l) != "ok"}
            msg = f"✅ {name_part} creado en: {', '.join(ok) or 'ninguno'} (ID {emp_no})"
            if fail:
                msg += f"\n⚠️ Revisar: {fail}"
            return msg
        except Exception as e:
            logger.exception("error creando empleado")
            await del_draft(session_id)
            return f"❌ Error creando empleado: {e}"

    return None


@app.get("/draft_status/{session_id}")
async def draft_status(session_id: str):
    row = await db_pool.fetchrow(
        "SELECT 1 FROM agent.employee_draft WHERE session_id = $1", session_id
    )
    return {"has_draft": bool(row)}


async def _update_summary_bg(session_id: str):
    """Resumen en segundo plano: no demora la respuesta ni rompe el request."""
    try:
        await update_summary(db_pool, session_id)
    except Exception:
        logger.exception(f"update_summary falló para {session_id}")


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
                session_id, user_id, "ai", strip_b64(emp_answer)
            )
            return ChatResponse(response=emp_answer, session_id=session_id, intent="employee")

        # === Flujo normal CVs ===
        await db_pool.execute(
            "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
            session_id, user_id, "human", request.message
        )

        history = await load_context(db_pool, session_id)

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
            session_id, user_id, "ai", strip_b64(answer)
        )

        # antes esto bloqueaba la respuesta (y un fallo daba 500 con la respuesta ya generada)
        _spawn_bg(_update_summary_bg(session_id))

        return ChatResponse(
            response=answer,
            session_id=session_id,
            intent=result.get("intent"),
        )

    except Exception as e:
        logger.exception("chat falló")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    db_ok = False
    try:
        if db_pool is not None:
            await db_pool.fetchval("SELECT 1")
            db_ok = True
    except Exception:
        logger.exception("health: DB caída")
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "service": "chat-cv-agent"}


@app.post("/cancel_employee/{session_id}")
async def cancel_employee(session_id: str):
    await del_draft(session_id)
    return {"ok": True}
