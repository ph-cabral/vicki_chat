# Reemplazo de app/graph.py
# off_topic → general. search/ranking → rag_search → response. camera → camera.
from langgraph.graph import StateGraph, END
from app.graph_state import AgentState
from app.nodes import (
    router_node,
    general_node,
    rag_search_node,
    response_node,
    camera_node,
    rrhh_node,
    ventas_node,
    compras_node,
)


def route_after_classification(state):
    intent = state.get("intent")
    if intent == "camera":
        return "camera"
    if intent == "ventas":
        return "ventas"
    if intent == "rrhh":
        return "rrhh"
    if intent == "compras":
        return "compras"
    if intent in ("search", "ranking", "procedimiento"):
        return "rag_search"
    return "general"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("general", general_node)
    builder.add_node("rag_search", rag_search_node)
    builder.add_node("response", response_node)
    builder.add_node("camera", camera_node)
    builder.add_node("ventas", ventas_node)
    builder.add_node("rrhh", rrhh_node)
    builder.add_node("compras", compras_node)
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "general": "general",
            "rag_search": "rag_search",
            "camera": "camera",
            "ventas": "ventas",
            "rrhh": "rrhh",
            "compras": "compras",
        },
    )
    builder.add_edge("rag_search", "response")
    builder.add_edge("response", END)
    builder.add_edge("general", END)
    builder.add_edge("camera", END)
    builder.add_edge("ventas", END)
    builder.add_edge("rrhh", END)
    builder.add_edge("compras", END)
    return builder
