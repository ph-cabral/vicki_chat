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
    proc_docs: Optional[str]             # ← procedimientos/instructivos del puesto (intent search/ranking)
    candidatos: Optional[list]           # ← candidatos de los hits de CVs (barra lateral del chat)
    cv_diag: Optional[str]               # ← por qué la búsqueda de CVs vino vacía (infra, no ausencia)
    descartados: Optional[list]          # ← candidato_id tirados al tacho en esta conversación
    final_response: Optional[str]
    session_id: Optional[str]
    # ── Intent "ventas" — ver app/ventas_tools.py ──────────────────────────
    # Resueltos por vicki_web contra la sesión autenticada (nunca por el
    # usuario ni por el LLM) y reenviados en cada /chat — ver
    # lib/ventas/vickiVentasAcceso.ts y app/main.py.
    ventas_habilitado: Optional[bool]     # False = el intent "ventas" no se ofrece
    ventas_admin: Optional[bool]          # True = sin filtro de vendedor (toda la empresa)
    ventas_vendedor_codigo: Optional[int] # código Magnus fijo para este usuario, o None si es admin
    # ── Intent "rrhh" — ver app/asistencia_tools.py ────────────────────────
    # Todo o nada: quien lo tiene ve la asistencia de TODA la empresa (no hay
    # equivalente al vendedorCodigo, un dato de RRHH parcial no sirve). Lo
    # resuelve vicki_web contra la sesión — ver lib/rrhh/vickiRrhhAcceso.ts.
    rrhh_habilitado: Optional[bool]
