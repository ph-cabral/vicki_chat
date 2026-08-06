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
    delete_employee_all,
    find_employee,
    normalize_jpg,
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
            CREATE TABLE IF NOT EXISTS agent.asignar_foto_draft (
                session_id TEXT PRIMARY KEY,
                emp_no TEXT,
                emp_nombre TEXT,
                photo_b64 TEXT,
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


async def del_asignar_draft(session_id: str):
    await db_pool.execute(
        "DELETE FROM agent.asignar_foto_draft WHERE session_id = $1", session_id
    )


CONFIRM_WORDS = {"si", "sí", "ok", "dale", "confirmar", "guardar"}
CANCEL_WORDS = {"no", "cancelar", "cancela"}

MSG_PEDIR_FOTO = (
    "📷 Foto para **{nombre}** (ID {emp}).\n\n"
    "Elegí un reloj para sacarla, o subí una imagen.\n\n[LOC_PICK][UPLOAD_PICK]"
)
MSG_CONFIRMAR = (
    "![foto](data:image/jpeg;base64,{b64})\n\n"
    "¿Guardo esta foto de **{nombre}** (ID {emp}) en los 3 relojes?\n\n"
    "[LOC_PICK][UPLOAD_PICK][CONFIRM_PICK]"
)


async def handle_assign_photo_flow(session_id: str, message: str,
                                   location: Optional[str] = None):
    """Flujo /asignar foto: empleado existente → foto (reloj o imagen subida)
    → confirmación → upload_face_all (reemplaza rostro si ya tenía)."""
    text = (message or "").strip()
    low = text.lower()

    triggers = ("/asignar foto", "/asignar", "/foto")
    trigger = next((t for t in triggers if low == t or low.startswith(t + " ")), None)

    row = await db_pool.fetchrow(
        "SELECT emp_no, emp_nombre, photo_b64 FROM agent.asignar_foto_draft "
        "WHERE session_id = $1", session_id
    )

    # Trigger → (re)iniciar flujo
    if trigger:
        crear = await db_pool.fetchrow(
            "SELECT 1 FROM agent.employee_draft WHERE session_id = $1", session_id
        )
        if crear:
            return "❌ Hay un alta de empleado en curso. Terminala o cancelala antes de asignar una foto."
        await del_asignar_draft(session_id)
        resto = text[len(trigger):].strip()
        if not resto:
            await db_pool.execute(
                "INSERT INTO agent.asignar_foto_draft (session_id) VALUES ($1)",
                session_id,
            )
            return "👤 ¿A qué empleado? Escribí el **ID** o el **nombre**."
        # vino "/asignar foto <query>" → resolver directo
        await db_pool.execute(
            "INSERT INTO agent.asignar_foto_draft (session_id) VALUES ($1)",
            session_id,
        )
        row = {"emp_no": None, "emp_nombre": None, "photo_b64": None}
        text, low = resto, resto.lower()

    if not row:
        return None  # flujo no activo

    # Paso 1: falta elegir empleado → el mensaje es la búsqueda
    if not row["emp_no"]:
        matches = find_employee(text)
        if not matches:
            return f"❌ No encontré ningún empleado con «{text}». Probá con el ID o parte del nombre."
        if len(matches) > 1:
            lista = "\n".join(
                f"- **{m['emp_no']}** — {m['nombre'] or '(sin nombre)'} ({', '.join(m['relojes'])})"
                for m in matches[:8]
            )
            extra = "" if len(matches) <= 8 else f"\n… y {len(matches) - 8} más."
            return f"Encontré varios, escribí el ID exacto:\n\n{lista}{extra}"
        m = matches[0]
        await db_pool.execute(
            "UPDATE agent.asignar_foto_draft SET emp_no = $2, emp_nombre = $3 "
            "WHERE session_id = $1",
            session_id, m["emp_no"], m["nombre"] or m["emp_no"],
        )
        return MSG_PEDIR_FOTO.format(nombre=m["nombre"] or m["emp_no"], emp=m["emp_no"])

    emp_no, nombre = row["emp_no"], row["emp_nombre"] or row["emp_no"]

    # Cancelar / confirmar
    if low in CANCEL_WORDS:
        await del_asignar_draft(session_id)
        return "❌ Asignación de foto cancelada."
    if low in CONFIRM_WORDS:
        if not row["photo_b64"]:
            return "Todavía no hay foto. Elegí un reloj o subí una imagen.\n\n[LOC_PICK][UPLOAD_PICK]"
        jpg = base64.b64decode(row["photo_b64"])
        up = await asyncio.to_thread(upload_face_all, emp_no, jpg)
        await del_asignar_draft(session_id)
        ok = [l for l, r in up.items() if r == "ok"]
        fallas = {l: r for l, r in up.items() if r != "ok"}
        msg = f"✅ Foto de **{nombre}** (ID {emp_no}) guardada en: {', '.join(ok) or 'ninguno'}"
        if fallas:
            det = "\n".join(f"- {l}: {r}" for l, r in fallas.items())
            msg += f"\n⚠️ Falló en:\n{det}"
        return msg

    # Sacar foto con un reloj (botones [LOC_PICK] mandan location o el nombre)
    loc = (location or "").strip() or (text if low in LOCATIONS else "")
    if loc:
        try:
            ip = resolve_location(loc)
        except ValueError as ve:
            return f"❌ {ve}"
        try:
            jpg = await asyncio.to_thread(take_camera_snapshot, ip)
        except Exception as e:
            logger.exception("error tomando foto (asignar)")
            return f"❌ Error tomando foto del reloj: {e}"
        b64 = base64.b64encode(jpg).decode()
        await db_pool.execute(
            "UPDATE agent.asignar_foto_draft SET photo_b64 = $2 WHERE session_id = $1",
            session_id, b64,
        )
        return MSG_CONFIRMAR.format(b64=b64, nombre=nombre, emp=emp_no)

    return ("Elegí un reloj para sacar la foto, subí una imagen, "
            "o escribí **cancelar**.\n\n[LOC_PICK][UPLOAD_PICK]")


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

            if any(r != "ok" for r in up.values()):
                await asyncio.to_thread(delete_employee_all, emp_no)
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        'DELETE FROM everwear.legajo WHERE "employeeNo" = $1', emp_no
                    )
                await del_draft(session_id)
                return (
                    f"❌ Error subiendo la foto (acercate más a la cámara). "
                    f"Se revirtió el alta en relojes y legajo. Reintentá /crear.\n"
                    f"Detalle: {up}"
                )

            await del_draft(session_id)
            ok = [l for l, r in cre.items() if r == "ok"]
            msg = f"✅ {name_part} creado en: {', '.join(ok) or 'ninguno'} (ID {emp_no})"
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

        # === INTERCEPT: asignar foto (antes que crear: consume su propio draft) ===
        asig_answer = await handle_assign_photo_flow(session_id, request.message, request.location)
        if asig_answer is not None:
            await db_pool.execute(
                "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
                session_id, user_id, "human", request.message
            )
            await db_pool.execute(
                "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
                session_id, user_id, "ai", strip_b64(asig_answer)
            )
            return ChatResponse(response=asig_answer, session_id=session_id, intent="assign_photo")

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
    await del_asignar_draft(session_id)
    return {"ok": True}


class AsignarFotoUpload(BaseModel):
    session_id: str
    photo_b64: str  # dataURL o base64 pelado


@app.post("/asignar_foto_upload")
async def asignar_foto_upload(req: AsignarFotoUpload):
    """Recibe la imagen subida desde el chat para el flujo /asignar foto."""
    row = await db_pool.fetchrow(
        "SELECT emp_no, emp_nombre FROM agent.asignar_foto_draft WHERE session_id = $1",
        req.session_id,
    )
    if not row or not row["emp_no"]:
        raise HTTPException(400, "No hay una asignación de foto en curso. Escribí /asignar foto.")
    b64 = req.photo_b64.split(",", 1)[-1].strip()  # tolera dataURL
    try:
        raw = base64.b64decode(b64)
        if len(raw) > 8_000_000:
            raise ValueError("imagen demasiado grande (máx 8MB)")
        jpg = await asyncio.to_thread(normalize_jpg, raw)
    except Exception as e:
        raise HTTPException(400, f"Imagen inválida: {e}")

    b64_jpg = base64.b64encode(jpg).decode()
    await db_pool.execute(
        "UPDATE agent.asignar_foto_draft SET photo_b64 = $2 WHERE session_id = $1",
        req.session_id, b64_jpg,
    )
    nombre = row["emp_nombre"] or row["emp_no"]
    answer = MSG_CONFIRMAR.format(b64=b64_jpg, nombre=nombre, emp=row["emp_no"])
    user_id = user_id_from_session(req.session_id)
    await db_pool.execute(
        "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
        req.session_id, user_id, "human", "📎 (imagen subida)"
    )
    await db_pool.execute(
        "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
        req.session_id, user_id, "ai", strip_b64(answer)
    )
    return {"response": answer}
