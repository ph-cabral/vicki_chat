import asyncio, re, os
from anthropic import Anthropic
from langchain_core.messages import HumanMessage, AIMessage
from app.config import config

KEEP_LAST = 10
_client = Anthropic(api_key=config.ANTHROPIC_KEY)

# Quita data-URIs base64 para no envenenar contexto/resumen
def strip_b64(s: str) -> str:
    return re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "📸[foto]", s or "")


async def load_context(pool, session_id: str):
    """Devuelve lista de mensajes LangChain: [resumen?] + últimos KEEP_LAST."""
    srow = await pool.fetchrow(
        "SELECT summary FROM agent.chat_summary WHERE session_id = $1", session_id
    )
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
    resp = _client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL),
        max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


async def update_summary(pool, session_id: str):
    """Pliega al resumen los mensajes que quedan fuera de los últimos KEEP_LAST."""
    srow = await pool.fetchrow(
        "SELECT summary, summarized_through FROM agent.chat_summary WHERE session_id = $1",
        session_id,
    )
    prev_summary = srow["summary"] if srow else ""
    watermark = srow["summarized_through"] if srow else None

    # mensajes a plegar: anteriores a los últimos KEEP_LAST y posteriores al watermark
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
    if not rows:
        return

    fold_text = "\n".join(
        f"{'Usuario' if r['role']=='human' else 'Asistente'}: {strip_b64(r['content'])}"
        for r in rows
    )
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
