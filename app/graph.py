from langgraph.graph import StateGraph, END
from app.graph_state import AgentState
from app.nodes import (
    router_node,
    off_topic_node,
    rag_search_node,
    response_node,
    camera_node,
)


def route_after_classification(state):
    intent = state.get("intent")
    if intent == "off_topic":
        return "off_topic"
    if intent == "camera":
        return "camera"
    return "rag_search"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("off_topic", off_topic_node)
    builder.add_node("rag_search", rag_search_node)
    builder.add_node("response", response_node)
    builder.add_node("camera", camera_node)              # ← agregar
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "off_topic": "off_topic",
            "rag_search": "rag_search",
            "camera": "camera",                          # ← agregar
        },
    )
    builder.add_edge("rag_search", "response")
    builder.add_edge("response", END)
    builder.add_edge("off_topic", END)
    builder.add_edge("camera", END)                      # ← agregar
    return builder