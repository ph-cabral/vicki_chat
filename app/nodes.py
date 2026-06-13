import logging
import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config
from app.graph_state import AgentState
from app.prompts import ROUTER_PROMPT, SYSTEM_PROMPT
from app.tool import take_camera_snapshot
from app.tools import build_retriever_tool

log = logging.getLogger("nodes")

llm = ChatAnthropic(
    model=config.ANTHROPIC_MODEL,
    api_key=config.ANTHROPIC_KEY,
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

retriever_tool = build_retriever_tool()

VALID_INTENTS = {"search", "ranking", "camera", "off_topic"}


def router_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1].content
    prompt = ROUTER_PROMPT.format(message=user_message)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()
    except Exception:
        log.exception("router LLM falló; asumo search")
        intent = "search"

    log.info(f"[ROUTER] intent={intent} msg={user_message[:120]!r}")

    if intent not in VALID_INTENTS:
        intent = "search"

    return {**state, "intent": intent, "user_message": user_message}


def off_topic_node(state: AgentState) -> AgentState:
    response = AIMessage(
        content="La consulta no corresponde a una búsqueda de "
                "perfiles en la base de datos."
    )
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_response": response.content,
    }


def rag_search_node(state: AgentState) -> AgentState:
    try:
        docs = retriever_tool.func(state["user_message"])
    except Exception as e:
        # sin Qdrant/embeddings no hay contexto, pero no tiramos abajo el chat
        log.exception("rag_search falló")
        docs = f"(no se pudo consultar la base de CVs: {e})"
    log.info(f"[RAG] {len(str(docs))} chars recuperados")
    return {**state, "retrieved_docs": docs}


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "search")
    ranking_instruction = (
        "Aplicá ranking por: años de experiencia relevante al puesto, "
        "especialización, seniority y estabilidad laboral."
        if intent == "ranking" else ""
    )

    context_prompt = (
        f"## Documentos encontrados en la base de datos:\n"
        f"{state.get('retrieved_docs', '')}\n\n"
        f"## Tipo de consulta identificada: {intent}\n\n"
        f"## Consulta del usuario:\n{state['user_message']}\n\n"
        f"Respondé basándote ÚNICAMENTE en los documentos proporcionados.\n"
        f"{ranking_instruction}"
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


def route_after_classification(state) -> Literal["rag_search", "off_topic", "camera"]:
    intent = state.get("intent", "search")
    if intent == "off_topic":
        return "off_topic"
    if intent == "camera":
        return "camera"
    return "rag_search"


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
