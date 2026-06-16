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
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.graph_state import AgentState
from app.prompts import ROUTER_PROMPT, SYSTEM_PROMPT
from app.tool import take_camera_snapshot
from app.tools import list_collections, search_collections

log = logging.getLogger("nodes")

# LLM principal (respuestas)
llm = ChatAnthropic(
    model=config.ANTHROPIC_MODEL,
    api_key=config.ANTHROPIC_KEY,
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

# LLM del router: respuesta corta → menor latencia y costo
router_llm = ChatAnthropic(
    model=config.ANTHROPIC_MODEL,
    api_key=config.ANTHROPIC_KEY,
    temperature=0,
    max_tokens=config.ROUTER_MAX_TOKENS,
    timeout=15,
    max_retries=1,
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
    )
    intent, collections = "general", []
    try:
        raw = router_llm.invoke([HumanMessage(content=prompt)]).content
        data = _safe_json(raw)
        intent = (data.get("intent") or "general").strip().lower()
        collections = [c for c in (data.get("collections") or []) if c in cols]
    except Exception:
        log.exception("router falló; asumo general")

    if intent not in VALID_INTENTS:
        intent = "general"
    if intent in ("search", "ranking") and not collections:
        collections = [config.QDRANT_COLLECTION] if config.QDRANT_COLLECTION in cols else cols[:1]

    log.info(f"[ROUTER] intent={intent} cols={collections} msg={user_message[:120]!r}")
    return {**state, "intent": intent, "user_message": user_message, "collections": collections}


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
    try:
        docs = search_collections(state["user_message"], state.get("collections") or [])
    except Exception:
        log.exception("rag_search falló")
        docs = ""
    log.info(f"[RAG] cols={state.get('collections')} {len(str(docs))} chars")
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
    context_prompt = (
        f"## CVs encontrados ({cols}):\n"
        f"{docs if docs else '(no se encontraron CVs relevantes)'}\n\n"
        f"## Consulta del usuario:\n{state['user_message']}\n\n"
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
