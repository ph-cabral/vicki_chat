# Reemplazo de app/graph_state.py
# 'collections': para intent search/ranking son TODAS las colecciones
# disponibles (ya no las elige el router — ver nodes.py::router_node).
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    user_message: Optional[str]
    search_query: Optional[str]          # ← query reformulada (autocontenida) por el router
    collections: Optional[list]          # ← todas las colecciones (search/ranking) o []
    retrieved_docs: Optional[str]
    perfil_docs: Optional[str]           # ← descripción del puesto buscado (intent search/ranking)
    final_response: Optional[str]
    session_id: Optional[str]
