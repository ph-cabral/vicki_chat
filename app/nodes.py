"""Reemplazo de app/nodes.py

Cambios:
- router_node hace UNA sola llamada LLM que devuelve intent + colecciones (JSON).
- router_llm con max_tokens chico (rápido/barato). off_topic → general (conversa).
- rag_search_node busca en la(s) colección(es) elegidas (multi-colección).
"""
import json
import logging
import os
import re

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config
from app.graph_state import AgentState
from app.prompts import GROUNDING_RULES, ROUTER_PROMPT, SYSTEM_PROMPT
from app.tool import take_camera_snapshot
from app.tools import list_collections, search_collections

log = logging.getLogger("nodes")

# Extrae nombres de candidatos desde el bloque que arma _format_hit() en tools.py
# ("--- Nombre Apellido (colección: ..., relevancia: ...) ---").
_NAME_RE = re.compile(r"--- (.+?) \(colección:")


def _extract_names(docs: str) -> list[str]:
    seen: list[str] = []
    for m in _NAME_RE.finditer(docs or ""):
        n = m.group(1).strip()
        if n and n not in seen:
            seen.append(n)
    return seen


def _history_snippet(messages: list, max_pairs: int = 3) -> str:
    """Últimos mensajes (sin el actual) para que el router pueda reformular
    preguntas de seguimiento en una búsqueda autocontenida."""
    prev = messages[:-1][-(max_pairs * 2):]
    if not prev:
        return "(sin mensajes previos)"
    lines = []
    for m in prev:
        who = "Usuario" if isinstance(m, HumanMessage) else "Vicki"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


class LLMWithFallback:
    """OpenAI primario; si falla (ej. sin crédito), cae a Claude."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, messages):
        try:
            return self.primary.invoke(messages)
        except Exception as e:
            log.warning(f"OpenAI falló, fallback a Claude: {e}")
            return self.fallback.invoke(messages)


llm = LLMWithFallback(
    primary=ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
        max_tokens=1024,
        timeout=30,
        max_retries=1,
    ),
    fallback=ChatAnthropic(
        model=config.ANTHROPIC_MODEL,
        api_key=config.ANTHROPIC_KEY,
        temperature=0,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    ),
)

# LLM del router: respuesta corta → menor latencia y costo. OpenAI primario, Claude fallback.
router_llm = LLMWithFallback(
    primary=ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
        max_tokens=config.ROUTER_MAX_TOKENS,
        timeout=15,
        max_retries=1,
    ),
    fallback=ChatAnthropic(
        model=config.ANTHROPIC_MODEL,
        api_key=config.ANTHROPIC_KEY,
        temperature=0,
        max_tokens=config.ROUTER_MAX_TOKENS,
        timeout=15,
        max_retries=1,
    ),
)

VALID_INTENTS = {"search", "ranking", "camera", "general"}


def _safe_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def router_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1].content
    cols = list_collections()
    prompt = ROUTER_PROMPT.format(
        message=user_message,
        collections=", ".join(cols) if cols else "(ninguna)",
        history=_history_snippet(state["messages"]),
    )
    intent, collections, search_query = "general", [], user_message
    try:
        raw = router_llm.invoke([HumanMessage(content=prompt)]).content
        data = _safe_json(raw)
        intent = (data.get("intent") or "general").strip().lower()
        collections = [c for c in (data.get("collections") or []) if c in cols]
        # Query reformulada (autocontenida) para embeber. Si el router no la
        # devuelve, caemos al mensaje crudo (comportamiento anterior).
        search_query = (data.get("query") or "").strip() or user_message
    except Exception:
        log.exception("router falló; asumo general")

    if intent not in VALID_INTENTS:
        intent = "general"
    if intent in ("search", "ranking") and not collections:
        collections = [config.QDRANT_COLLECTION] if config.QDRANT_COLLECTION in cols else cols[:1]

    log.info(
        f"[ROUTER] intent={intent} cols={collections} "
        f"query={search_query[:120]!r} msg={user_message[:120]!r}"
    )
    return {
        **state,
        "intent": intent,
        "user_message": user_message,
        "search_query": search_query,
        "collections": collections,
    }


def general_node(state: AgentState) -> AgentState:
    """Respuesta conversacional sin RAG (saludos, dudas, temas generales)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_response": response.content,
    }


def rag_search_node(state: AgentState) -> AgentState:
    # search_query es la versión reformulada por el router (autocontenida);
    # si no vino, cae al mensaje crudo del usuario.
    query = state.get("search_query") or state["user_message"]
    try:
        docs = search_collections(query, state.get("collections") or [])
    except Exception:
        log.exception("rag_search falló")
        docs = ""
    log.info(f"[RAG] query={query[:120]!r} cols={state.get('collections')} {len(str(docs))} chars")
    return {**state, "retrieved_docs": docs}


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "search")
    ranking_instruction = (
        "Ordená los candidatos por: experiencia relevante al puesto, especialización, "
        "seniority y estabilidad laboral, explicando brevemente cada valoración."
        if intent == "ranking" else ""
    )
    docs = (state.get("retrieved_docs") or "").strip()
    cols = ", ".join(state.get("collections") or []) or "sin colección"
    names = _extract_names(docs)
    grounding = GROUNDING_RULES.format(
        names=", ".join(names) if names else "(ninguno — no hay CVs en el contexto)"
    )
    context_prompt = (
        f"## CVs encontrados ({cols}):\n"
        f"{docs if docs else '(no se encontraron CVs relevantes)'}\n\n"
        f"## Consulta del usuario:\n{state['user_message']}\n\n"
        f"{grounding}\n"
        f"Respondé apoyándote en los CVs de arriba. No inventes datos. "
        f"Si no hay CVs relevantes, decilo.\n{ranking_instruction}"
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"][:-1],
        HumanMessage(content=context_prompt),
    ]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_response": response.content,
    }


def camera_node(state):
    try:
        take_camera_snapshot()  # escribe el JPG en SNAPSHOT_PATH (servido por /snapshot)
        base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        url = f"{base}/snapshot"
        state["final_response"] = f"📸 Acá está la foto:\n\n![snapshot]({url})"
    except Exception as e:
        log.exception("camera_node falló")
        state["final_response"] = f"No pude acceder a la cámara: {e}"
    return state
