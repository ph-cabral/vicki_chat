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
from concurrent.futures import ThreadPoolExecutor

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config
from app.graph_state import AgentState
from app.prompts import (
    GROUNDING_RULES,
    PERFIL_BLOCK,
    PROC_CONTEXT_BLOCK,
    PROC_RESPONSE_PROMPT,
    ROUTER_PROMPT,
    SHORTLIST_RULES,
    SYSTEM_PROMPT,
)
from app.tool import take_camera_snapshot
from app.tools import (
    embed_query,
    list_collections,
    search_cvs,
    search_descripcion_puesto,
    search_procedimientos,
)

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

VALID_INTENTS = {"search", "ranking", "procedimiento", "camera", "general"}


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
        history=_history_snippet(state["messages"]),
    )
    intent, search_query = "general", user_message
    try:
        raw = router_llm.invoke([HumanMessage(content=prompt)]).content
        data = _safe_json(raw)
        intent = (data.get("intent") or "general").strip().lower()
        # Query reformulada (autocontenida) para embeber. Si el router no la
        # devuelve, caemos al mensaje crudo (comportamiento anterior).
        search_query = (data.get("query") or "").strip() or user_message
    except Exception:
        log.exception("router falló; asumo general")

    if intent not in VALID_INTENTS:
        intent = "general"
    # Buscar SIEMPRE en todas las colecciones disponibles para search/ranking.
    # Antes el LLM elegía la(s) colección(es) "más afín(es)" y esa elección
    # dependía de cómo estaba redactada la pregunta: "vendedor viajante para
    # zona de Córdoba" podía no elegir la colección correcta y devolver "no
    # tengo candidatos", mientras que "para viajante" sí la elegía y
    # aparecían los mismos candidatos que ya estaban cargados. Con pocas
    # colecciones de CVs, buscar en todas (en paralelo, ver tools.py) es más
    # barato que el riesgo de una respuesta contradictoria.
    # Excepciones: la colección de procedimientos (PROC_COLLECTION) queda FUERA
    # de las búsquedas de CVs, y el intent "procedimiento" busca SOLO ahí.
    #
    # OJO (2026-08-25): esa exclusión dejaba invisible la DESCRIPCIÓN DE PUESTO,
    # que vive en PROC_COLLECTION pero es justamente lo que hay que leer cuando
    # se busca gente. No se arregla metiendo PROC_COLLECTION acá (volverían los
    # procedimientos a ensuciar los CVs): se trae aparte y filtrada por
    # metadata.tipo_doc en rag_search_node → state["perfil_docs"].
    if intent == "procedimiento":
        collections = [config.PROC_COLLECTION]
    elif intent in ("search", "ranking"):
        collections = [c for c in cols if c != config.PROC_COLLECTION]
    else:
        collections = []

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
    intent = state.get("intent")

    # El embedding del query se calcula UNA vez y se reusa en las tres búsquedas
    # (CVs + descripción de puesto + procedimientos). Si falla, search_collections
    # lo recalcula por su cuenta.
    vector = None
    try:
        vector = embed_query(query)
    except Exception:
        log.exception("embed_query falló; cada búsqueda embebe por su cuenta")

    # ── procedimientos/instructivos: sin descripciones de puesto ──
    if intent == "procedimiento":
        try:
            docs = search_procedimientos(query, vector=vector)
        except Exception:
            log.exception("rag_search (procedimiento) falló")
            docs = ""
        log.info(f"[RAG] proc query={query[:120]!r} {len(str(docs))} chars")
        return {**state, "retrieved_docs": docs, "perfil_docs": "", "proc_docs": "",
                "candidatos": []}

    # ── búsqueda de candidatos ────────────────────────────────────────────
    # Tres consultas a Qdrant que comparten el MISMO embedding:
    #   cvs    → las personas
    #   perfil → descripción del puesto: qué se PIDE (criterio excluyente)
    #   proc   → procedimientos/instructivos: qué se HACE en el puesto
    # Van EN PARALELO porque son independientes; en serie la latencia se sumaba
    # una atrás de otra y esto corre en cada mensaje de búsqueda.
    # search_cvs devuelve (contexto, candidatos): los candidatos salen de los
    # mismos hits, para poder mostrar las miniaturas de los CVs al costado del
    # chat sin una segunda búsqueda. `descartados` son los que el reclutador
    # tiró al tacho en esta conversación: se excluyen en Qdrant.
    cvs_res: dict = {"texto": "", "candidatos": []}

    def _buscar_cvs():
        texto, cands = search_cvs(
            query,
            state.get("collections") or [],
            vector=vector,
            descartados=state.get("descartados") or [],
        )
        cvs_res["texto"], cvs_res["candidatos"] = texto, cands
        return texto

    tareas = {"cvs": _buscar_cvs}
    if intent in ("search", "ranking"):
        tareas["perfil"] = lambda: search_descripcion_puesto(query, vector=vector)
        if config.PROC_CONTEXT_EN_BUSQUEDA:
            tareas["proc"] = lambda: search_procedimientos(
                query,
                k=config.PROC_CONTEXT_TOP_K,
                vector=vector,
                min_score=config.PROC_CONTEXT_MIN_SCORE,
            )

    res: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(tareas)) as ex:
        futuros = {nombre: ex.submit(fn) for nombre, fn in tareas.items()}
        for nombre, fut in futuros.items():
            try:
                res[nombre] = fut.result() or ""
            except Exception:
                # el contexto de apoyo nunca puede voltear la búsqueda de CVs
                log.exception(f"búsqueda {nombre!r} falló")
                res[nombre] = ""

    docs, perfil, proc = res.get("cvs", ""), res.get("perfil", ""), res.get("proc", "")
    candidatos = cvs_res["candidatos"]
    log.info(
        f"[RAG] query={query[:120]!r} cols={state.get('collections')} "
        f"{len(docs)} chars cvs + {len(perfil)} chars perfil + {len(proc)} chars proc "
        f"+ {len(candidatos)} candidatos ({len(state.get('descartados') or [])} descartados)"
    )
    return {
        **state,
        "retrieved_docs": docs,
        "perfil_docs": perfil,
        "proc_docs": proc,
        "candidatos": candidatos,
    }


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "search")

    # Procedimientos/instructivos → prompt propio (sin reglas de CVs/candidatos).
    if intent == "procedimiento":
        docs = (state.get("retrieved_docs") or "").strip()
        context_prompt = PROC_RESPONSE_PROMPT.format(
            docs=docs if docs else "(no se encontró ningún procedimiento/instructivo relevante)",
            message=state["user_message"],
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

    ranking_instruction = (
        "Ordená los candidatos por: experiencia relevante al puesto, especialización, "
        "seniority y estabilidad laboral, explicando brevemente cada valoración."
        if intent == "ranking" else ""
    )
    docs = (state.get("retrieved_docs") or "").strip()
    cols = ", ".join(state.get("collections") or []) or "sin colección"
    names = _extract_names(docs)
    # Perfil del puesto (si hay uno cargado en /rrhh/puestos para lo buscado).
    # Va ANTES de los CVs y fuera de GROUNDING_RULES: es criterio, no candidato.
    perfil = (state.get("perfil_docs") or "").strip()
    perfil_block = PERFIL_BLOCK.format(perfil=perfil) + "\n" if perfil else ""
    # Procedimientos/instructivos del puesto: qué HACE la persona ahí. Contexto
    # para evaluar el encaje, no requisito ni candidato (ver PROC_CONTEXT_BLOCK).
    proc_ctx = (state.get("proc_docs") or "").strip()
    proc_block = PROC_CONTEXT_BLOCK.format(procedimientos=proc_ctx) + "\n" if proc_ctx else ""
    grounding = GROUNDING_RULES.format(
        names=", ".join(names) if names else "(ninguno — no hay CVs en el contexto)"
    )
    # La shortlist ya viene armada y ordenada desde search_cvs (los N más
    # cercanos, sin piso de score). Acá solo se le dice al modelo que la
    # presente entera y en orden: el filtro "¿califica o no?" era justamente lo
    # que hacía que una búsqueda sin match perfecto terminara en "no tengo nada".
    n_cands = len(state.get("candidatos") or []) or len(names)
    shortlist_block = SHORTLIST_RULES.format(n=n_cands) if n_cands else ""
    context_prompt = (
        f"{perfil_block}"
        f"{proc_block}"
        f"## Shortlist: los {n_cands} candidatos más cercanos ({cols}), "
        f"ordenados de mayor a menor:\n"
        f"{docs if docs else '(no hay ningún CV cargado que se acerque)'}\n\n"
        f"## Consulta del usuario:\n{state['user_message']}\n\n"
        f"{shortlist_block}\n"
        f"{grounding}\n"
        f"Respondé apoyándote en los CVs de arriba. No inventes datos. "
        f"Si de verdad no hay ningún CV en la shortlist, decilo y ofrecé ampliar "
        f"la búsqueda (otra zona, rubro afín, menos experiencia)."
        f"\n{ranking_instruction}"
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
