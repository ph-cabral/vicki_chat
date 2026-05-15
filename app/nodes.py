from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.config import config
from app.prompts import SYSTEM_PROMPT, ROUTER_PROMPT
from app.tools import build_retriever_tool, take_camera_snapshot
from app.graph_state import AgentState
from langchain_anthropic import ChatAnthropic
from app.config import config
from app.prompts import SYSTEM_PROMPT, ROUTER_PROMPT


llm = ChatAnthropic(
    model=config.ANTHROPIC_MODEL,
    api_key=config.ANTHROPIC_KEY,
    temperature=0,
)

retriever_tool = build_retriever_tool()


def router_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1].content
    prompt = ROUTER_PROMPT.format(message=user_message)
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    
    print(f"[ROUTER] mensaje: {user_message} → intent: {intent}", flush=True)  # agregá esto
    
    if intent not in ["search", "ranking", "camera", "off_topic"]:
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
    docs = retriever_tool.func(state["user_message"])
    return {**state, "retrieved_docs": docs}


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "search")
    ranking_instruction = (
        "Aplicá ranking por: años de experiencia en marketing, "
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
        url = take_camera_snapshot()
        state["final_response"] = f"📸 Acá está la foto:\n\n![snapshot]({url})"
    except Exception as e:
        state["final_response"] = f"No pude acceder a la cámara: {e}"
    return state