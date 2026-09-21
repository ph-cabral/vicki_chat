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
    # ── No repetir candidatos ───────────────────────────────────────────────
    # `mostrados` son los candidato_id que ya se le mostraron al usuario en
    # esta conversación (salen de agent.chat_messages.metadata, ver main.py).
    # Se excluyen en Qdrant SOLO cuando pide gente distinta (`pide_otros`);
    # `sin_nuevos` marca que se excluyeron y no quedó nadie más cargado.
    mostrados: Optional[list]
    mostrados_nombres: Optional[list]
    pide_otros: Optional[bool]
    pide_recorte: Optional[bool]         # ← pidió zona/edad/estudios: la búsqueda no filtra por eso
    sin_nuevos: Optional[bool]
    excluidos_n: Optional[int]
    # ── Paginado de la shortlist ────────────────────────────────────────────
    # `top_n_pedido`: tamaño de página que pidió el usuario ("dame 10") o el
    # default. `pool_ampliado`: se trajo un pool más grande porque pidió un
    # recorte que la búsqueda no aplica (género, zona, edad). `pagina`: en qué
    # tanda va, contando los ya mostrados.
    top_n_pedido: Optional[int]
    pool_ampliado: Optional[bool]
    pagina: Optional[int]
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
    # ── Intent "compras" — ver app/compras_tools.py ────────────────────────
    # Todo o nada, sin filtro por persona: el que puede entrar a la vista
    # /compras ve por el chat lo mismo que ve por la vista. Lo resuelve
    # vicki_web contra la cookie de sesión — ver lib/compras/vickiComprasAcceso.ts.
    compras_habilitado: Optional[bool]
    # ── Intent "deposito" — ver app/deposito_tools.py ───────────────────────
    # Todo o nada, sin filtro por persona: el que puede entrar a la vista
    # /deposito ve por el chat lo mismo que ve por la vista. Lo resuelve
    # vicki_web contra la cookie de sesión — ver lib/deposito/vickiDepositoAcceso.ts.
    deposito_habilitado: Optional[bool]
