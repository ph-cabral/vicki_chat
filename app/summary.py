import asyncio
import datetime as dt
import logging
import os
import re

from anthropic import Anthropic
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage

from app.config import config

log = logging.getLogger("summary")
KEEP_LAST = 10
_client = Anthropic(api_key=config.ANTHROPIC_KEY)
_openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Quita data-URIs base64 para no envenenar contexto/resumen
def strip_b64(s: str) -> str:
    # return re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "📸[foto]", s or "")
    s = s or ""
    s = re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "📸[foto]", s)   # markdown
    s = re.sub(r"data:image/[A-Za-z]+;base64,[A-Za-z0-9+/=\s]+", "📸[foto]", s)  # data-uri suelto
    s = re.sub(r"[A-Za-z0-9+/]{500,}={0,2}", "📸[blob]", s)          # base64 crudo largo
    return s

MAX_FOLD_CHARS = 60000 

async def inicio_conversacion(pool, session_id: str):
    """Arranque de la conversación EN CURSO dentro de la sesión.

    El session_id es fijo por usuario (user_<uid>), así que la charla no termina
    nunca: sin un corte, el modelo sigue leyendo lo que se habló días atrás — un
    "hola, necesito perfiles de depósito" se contestó con la productividad de
    agosto de la semana anterior.

    El corte lo marca `agent.chat_summary.conversacion_desde`, y lo pone el
    botón «Nueva conversación» del chat (POST /conversacion/{session_id}/nueva).
    NO SE BORRA NADA: los mensajes quedan todos en la base y se siguen pudiendo
    ver desde el chat; lo único que cambia es hasta dónde mira el modelo.

    `config.CONVERSACION_HORAS` agrega, además, un corte automático por
    inactividad. Viene en 0 = apagado: el corte es manual. Poniéndole un número
    de horas, una charla retomada después de ese tiempo arranca limpia sola.

    Devuelve el datetime de corte, o None si todavía no hay ninguno.
    """
    desde = await pool.fetchval(
        "SELECT conversacion_desde FROM agent.chat_summary WHERE session_id = $1",
        session_id,
    )
    horas = config.CONVERSACION_HORAS
    if horas > 0:
        ultimo = await pool.fetchval(
            "SELECT max(created_at) FROM agent.chat_messages WHERE session_id = $1",
            session_id,
        )
        if ultimo is not None:
            inactivo = dt.datetime.now(dt.timezone.utc) - ultimo
            if inactivo > dt.timedelta(hours=horas) and (desde is None or desde < ultimo):
                log.info(f"[CONV] {session_id}: corte automático ({horas} h sin mensajes)")
                return await marcar_conversacion_nueva(pool, session_id)
    return desde


async def marcar_conversacion_nueva(pool, session_id: str):
    """Abre una conversación nueva: corre el corte a ahora y limpia el resumen
    (si no, la charla nueva arrancaría con el resumen de la anterior). Los
    mensajes NO se tocan. Devuelve el corte."""
    return await pool.fetchval(
        """
        INSERT INTO agent.chat_summary
               (session_id, summary, summarized_through, conversacion_desde, updated_at)
        VALUES ($1, '', NOW(), NOW(), NOW())
        ON CONFLICT (session_id) DO UPDATE
        SET summary = '', summarized_through = NOW(),
            conversacion_desde = NOW(), updated_at = NOW()
        RETURNING conversacion_desde
        """,
        session_id,
    )


async def load_context(pool, session_id: str, desde=None):
    """Devuelve lista de mensajes LangChain: [resumen?] + últimos KEEP_LAST.

    `desde` acota el contexto a la conversación en curso (ver
    reiniciar_conversacion): con un valor, no se carga el resumen ni ningún
    mensaje anterior a esa hora."""
    srow = None if desde else await pool.fetchrow(
        "SELECT summary FROM agent.chat_summary WHERE session_id = $1", session_id
    )
    if desde:
        rows = await pool.fetch(
            "SELECT role, content FROM agent.chat_messages "
            "WHERE session_id = $1 AND created_at >= $2 "
            "ORDER BY created_at DESC LIMIT $3",
            session_id, desde, KEEP_LAST,
        )
    else:
        rows = await pool.fetch(
            "SELECT role, content FROM agent.chat_messages "
            "WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, KEEP_LAST,
        )
    rows = list(reversed(rows))

    msgs = []
    if srow and srow["summary"]:
        msgs.append(HumanMessage(content=f"[CONTEXTO PREVIO RESUMIDO]\n{srow['summary']}"))
    for r in rows:
        c = strip_b64(r["content"])
        msgs.append(HumanMessage(content=c) if r["role"] == "human" else AIMessage(content=c))
    return msgs


def _summarize_sync(prev_summary: str, fold_text: str) -> str:
    prompt = (
        "Sos un asistente de RRHH. Actualizá el RESUMEN de la conversación en español, "
        "máximo 150 palabras, conservando: puestos/perfiles buscados, candidatos mencionados, "
        "rankings, decisiones y datos pedidos. Sin saludos ni relleno.\n\n"
        f"RESUMEN ACTUAL:\n{prev_summary or '(vacío)'}\n\n"
        f"MENSAJES NUEVOS A INTEGRAR:\n{fold_text}\n\nRESUMEN ACTUALIZADO:"
    )
    try:
        resp = _openai_client.chat.completions.create(
            model=config.MODEL_NAME,
            max_tokens=400, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"OpenAI falló en resumen, fallback a Claude: {e}")
        resp = _client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL),
            max_tokens=400, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


async def update_summary(pool, session_id: str):
    """Pliega al resumen los mensajes que quedan fuera de los últimos KEEP_LAST.
    Nunca lanza: un fallo acá no debe romper el request de chat."""
    try:
        await _update_summary(pool, session_id)
    except Exception:
        log.exception(f"update_summary falló para {session_id}")


async def _update_summary(pool, session_id: str):
    srow = await pool.fetchrow(
        "SELECT summary, summarized_through FROM agent.chat_summary WHERE session_id = $1",
        session_id,
    )
    prev_summary = srow["summary"] if srow else ""
    prev_summary = (prev_summary or "")[:4000]

    watermark = srow["summarized_through"] if srow else None

    rows = await pool.fetch(
        """
        WITH ranked AS (
            SELECT role, content, created_at,
                   ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
            FROM agent.chat_messages WHERE session_id = $1
        )
        SELECT role, content, created_at FROM ranked
        WHERE rn > $2 AND ($3::timestamptz IS NULL OR created_at > $3)
        ORDER BY created_at ASC
        """,
        session_id, KEEP_LAST, watermark,
    )
    # inicio_conversacion() ya corrió la marca de agua a NOW() al abrir una
    # conversación nueva, así que acá nunca vuelven mensajes de la anterior.
    if not rows:
        return

    fold_text = "\n".join(
        f"{'Usuario' if r['role']=='human' else 'Asistente'}: {strip_b64(r['content'])}"
        for r in rows
    )
    if len(fold_text) > MAX_FOLD_CHARS:
        fold_text = fold_text[-MAX_FOLD_CHARS:]   
    new_summary = await asyncio.to_thread(_summarize_sync, prev_summary, fold_text)
    new_watermark = rows[-1]["created_at"]

    await pool.execute(
        """
        INSERT INTO agent.chat_summary (session_id, summary, summarized_through, updated_at)
        VALUES ($1, $2, $3, NOW())
        ON CONFLICT (session_id) DO UPDATE
        SET summary = EXCLUDED.summary,
            summarized_through = EXCLUDED.summarized_through,
            updated_at = NOW()
        """,
        session_id, new_summary, new_watermark,
    )
