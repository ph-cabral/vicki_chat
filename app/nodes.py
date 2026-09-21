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
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config
from app.graph_state import AgentState
from app.prompts import (
    ENCAJE_DEBIL_RULES,
    ENCAJE_DEBIL_TODOS,
    FILTRO_NO_APLICADO_RULES,
    GENERAL_PROMPT,
    GROUNDING_RULES,
    NO_REPETIR_OK,
    NO_REPETIR_SIN_STOCK,
    PERFIL_BLOCK,
    PROC_CONTEXT_BLOCK,
    PROC_RESPONSE_PROMPT,
    ROUTER_PROMPT,
    SHORTLIST_RULES,
    SYSTEM_PROMPT,
    YA_MOSTRADOS_BLOCK,
    GENERO_FILTRADO_RULES,
)
from app.tool import take_camera_snapshot
from app.tools import (
    diagnostico_cvs,
    embed_query,
    list_collections,
    search_cvs,
    search_descripcion_puesto,
    search_procedimientos,
)

log = logging.getLogger("nodes")

# Tope de candidatos a excluir por pedido de "otros": un must_not con cientos
# de ids encarece la búsqueda y, pasados unos cuantos, ya no queda nadie nuevo.
# Es el techo del paginado (con páginas de 5, MAX_EXCLUIR=120 son 24 páginas).
MAX_EXCLUIR = config.MAX_EXCLUIR

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

# OJO: faltaban "compras" y "deposito" — las preguntas de esos módulos las
# clasificaba bien el router, pero acá se pisaban a "general" y terminaban en
# el nodo conversacional, que NO tiene datos. Ahí el modelo completaba: llegó a
# contestar "en agosto los preparadores hicieron 1.200 ítems y la mesa 950",
# todo inventado. Los intents tienen que estar los mismos que en graph.py.
VALID_INTENTS = {"search", "ranking", "procedimiento", "ventas", "rrhh",
                 "compras", "deposito", "camera", "general"}

# Intents que responden con datos reales de un módulo (nunca con el LLM).
INTENTS_DE_DATOS = ("ventas", "rrhh", "compras", "deposito")


def _norm(s: str) -> str:
    """minúsculas sin acentos, para que 'ítems' y 'items' sean lo mismo."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# Palabras que identifican de qué módulo sale un número. Es una RED DE
# SEGURIDAD, no el router: se usa solo para corregir al router cuando manda una
# pregunta de datos al nodo conversacional (que la contestaría inventando) o
# cuando la manda al módulo equivocado. El permiso lo sigue chequeando cada
# nodo, así que reencaminar acá no saltea ningún control.
_PALABRAS_MODULO = {
    "deposito": (
        r"\bitems?\b", r"\bunidades preparadas\b", r"picke", r"picking",
        r"\bpreparador", r"mesa de control", r"\bcontrolador",
        r"productividad", r"\bprepararon\b", r"\bpreparo\b",
    ),
    "compras": (
        r"faltante", r"\bfalto\b", r"orden(es)? de compra", r"\bocs?\b",
        r"\bingres(o|os|aron|ados?)\b", r"\bimportados?\b", r"\bnacionales?\b",
        r"sin cubrir", r"se cubrio",
    ),
    "rrhh": (
        r"asistencia", r"\bausenc", r"\bfalt(o|as|aron|ando)\b", r"horas? extra",
        r"\bferiado", r"vacacion", r"licencia", r"\bfichad", r"presentismo",
        r"\bpresent(e|es)\b", r"\bse present", r"\basisti(o|eron)\b",
        r"\bquien(es)? falt", r"\bfalt(o|aron)\b.{0,25}\b(a trabajar|al trabajo)\b",
        r"\bno vino\b", r"dias? de (falta|licencia|vacaciones)",
    ),
    "ventas": (
        r"factur", r"\bventas?\b", r"\bvendi(o|mos|eron|ste)?\b",
        r"\bcliente", r"ranking de vendedores", r"\bcomprobantes?\b",
    ),
}
_PALABRAS_MODULO_RE = {
    k: [re.compile(p) for p in pats] for k, pats in _PALABRAS_MODULO.items()
}


# Veto de la red de seguridad: pedidos que NOMBRAN palabras de un módulo pero
# no piden un número — un instructivo ("el procedimiento de picking") o gente a
# contratar ("candidatos para operario de depósito"). Sin esto, la corrección
# de intent los mandaba al módulo de datos.
_NO_ES_DATO_RE = [re.compile(p) for p in (
    r"procedimiento", r"instructivo", r"\bnorma\b", r"\bpaso a paso\b",
    r"\bcomo se (hace|carga|arma|prepara|completa|registra)\b",
    r"\bperfil(es)?\b", r"\bcandidat", r"\bcurriculum", r"\bcvs?\b",
    r"\bpostulante", r"\bcontratar\b", r"\bvacante\b", r"\bbusqueda de personal\b",
)]


def _puntaje_modulos(mensaje: str) -> dict:
    txt = _norm(mensaje)
    return {k: sum(1 for r in rs if r.search(txt))
            for k, rs in _PALABRAS_MODULO_RE.items()}


def _modulo_por_palabras(mensaje: str) -> tuple[str | None, dict]:
    """Módulo más probable según las palabras del mensaje, o None si empata
    o no hay ninguna."""
    sc = _puntaje_modulos(mensaje)
    txt = _norm(mensaje)
    if any(r.search(txt) for r in _NO_ES_DATO_RE):
        return None, sc
    mejor = max(sc, key=lambda k: sc[k])
    if sc[mejor] == 0:
        return None, sc
    if sum(1 for k, v in sc.items() if v == sc[mejor]) > 1:
        return None, sc  # empate: no tocar lo que dijo el router
    return mejor, sc


# "dame otros 5", "sin repetir", "perfiles distintos": el usuario pide gente
# que todavía no vio. Determinístico a propósito — de esto depende que se
# excluya a los ya mostrados, y no puede quedar librado a cómo redacte el LLM.
_PIDE_OTROS_RE = [re.compile(p) for p in (
    r"\botr[oa]s\b", r"\bdistint", r"\bdiferent", r"\bnuev[oa]s\b",
    r"sin repetir", r"no (me )?repit", r"\bmas perfiles\b", r"\bmas candidatos\b",
    r"(perfiles|candidatos|cvs?|curriculums?)\s+mas\b",
    r"que no (me )?(hayas|has|habias)\s+(pasado|dado|mostrado|planteado)",
    r"(ya )?(me )?(pasaste|diste|mostraste|planteaste|has pasado|has planteado)",
    # paginado explícito: "los siguientes 5", "la que sigue", "seguí", "el resto"
    r"\bsiguientes?\b", r"\bproximos?\b", r"\bque sigue", r"\bsegui(r|me)?\b",
    r"\bcontinua", r"\bel resto\b", r"\botra tanda\b", r"\bsegunda tanda\b",
    r"\bpagina \d", r"\b(dame|mostrame|pasame|traeme) mas\b", r"^mas\b",
)]


def _pide_otros(mensaje: str) -> bool:
    txt = _norm(mensaje)
    return any(r.search(txt) for r in _PIDE_OTROS_RE)


# Cuántos candidatos pidió ("dame 10 perfiles", "otros 3", "los siguientes
# cinco"). Determinístico, igual que _pide_otros: define el tamaño de la
# página, y si lo decidiera el LLM el paginado dejaría de ser reproducible.
_NUM_PALABRA = {
    "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "quince": 15, "veinte": 20,
}
_NUM = r"(\d{1,2}|" + "|".join(_NUM_PALABRA) + r")"
# Lo que sigue al número y hace que NO sea una cantidad de candidatos:
# "que tengan 5 años de experiencia", "3 meses", "2 turnos".
_NO_ES_CANTIDAD = r"(?!\s*(?:anos?|meses|mes\b|horas?|hs\b|km|turnos?|km/h))"
_CANTIDAD_RE = [
    # verbo de pedido + número: "dame 5", "mostrame otros tres", "top 10"
    re.compile(r"\b(?:dame|dame otr[oa]s|mostrame|pasame|traeme|buscame|quiero|"
               r"necesito|otr[oa]s|mas|siguientes|proximos|primeros|top|ver)\s+"
               + _NUM + r"\b" + _NO_ES_CANTIDAD),
    # número + sustantivo de candidato: "5 perfiles más", "diez CVs"
    re.compile(_NUM + r"\s+(?:perfiles?|candidat[oa]s|cvs?|curriculums?|"
               r"postulantes?|personas?|nombres?|opciones?)\b"),
]


def _cantidad_pedida(mensaje: str) -> int | None:
    """Tamaño de página que pidió el usuario, o None si no dijo ninguno."""
    txt = _norm(mensaje)
    for r in _CANTIDAD_RE:
        m = r.search(txt)
        if not m:
            continue
        crudo = m.group(1)
        n = int(crudo) if crudo.isdigit() else _NUM_PALABRA.get(crudo)
        if n and 1 <= n <= config.CANDIDATOS_TOP_N_MAX:
            return n
        if n and n > config.CANDIDATOS_TOP_N_MAX:
            return config.CANDIDATOS_TOP_N_MAX
    return None


# Recortes que el usuario pide pero que la búsqueda NO sabe aplicar (es
# similitud de texto, no filtros por campo). Cuando aparecen, se le avisa al
# modelo que no puede decir que filtró — llegó a contestar "los 5 perfiles
# nuevos, todos de San Francisco" sobre una lista que no se filtró por nada.
_PIDE_RECORTE_RE = [re.compile(p) for p in (
    r"que (sean|vivan|tengan|esten|cuenten|manejen|posean|residan)",
    r"\bzona\b", r"\blocalidad\b", r"\bciudad\b", r"\bviven? en\b",
    r"\bresiden", r"\bedad\b", r"\bmenores?\b", r"\bmayores?\b",
    r"\bsecundario\b", r"\btitulo\b", r"universitari", r"terciari",
    r"\bcarnet\b", r"registro de conducir", r"\bmovilidad\b",
    r"disponibilidad", r"\bturno",
)]

# El SEXO se trata aparte del resto de los recortes: es el único que se puede
# deducir de un dato que ya está en la metadata (el nombre de pila), así que lo
# aplica el código en tools.py::search_cvs y no el modelo leyendo los CVs.
# Mientras lo hacía el modelo, la respuesta era "no hay candidatas femeninas" y
# abajo cinco varones listados uno por uno — dos veces, con el pool chico y con
# el pool grande. Determinístico a propósito: un recorte que el LLM puede
# "interpretar" es un recorte que a veces no se aplica.
_GENERO_RE = [
    ("F", re.compile(r"\bfemenin|\bmujeres?\b|\bchicas\b|\bsenoritas?\b|"
                     r"\bcandidatas\b|\bpostulantes mujeres\b")),
    ("M", re.compile(r"\bmasculin|\bhombres?\b|\bvar[oo]nes?\b|\bchicos\b")),
]


def _genero_pedido(mensaje: str) -> str | None:
    """"F", "M" o None. Si el mensaje nombra los dos (o ninguno), None."""
    txt = _norm(mensaje)
    hits = [g for g, r in _GENERO_RE if r.search(txt)]
    return hits[0] if len(hits) == 1 else None


def _pide_recorte(mensaje: str) -> bool:
    txt = _norm(mensaje)
    return any(r.search(txt) for r in _PIDE_RECORTE_RE)


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

    # Red de seguridad sobre el router (que es un LLM y se equivoca): si las
    # palabras del mensaje apuntan claramente a un módulo de datos, va ahí.
    # Dos fallas reales que esto corrige: "cuántos ítems se hicieron en agosto"
    # cayó en "general" y el modelo inventó el número; "total de ítems de los
    # preparadores" fue a asistencia y devolvió horas trabajadas. Reencaminar
    # NO saltea permisos: el gate lo chequea cada nodo contra la sesión.
    sugerido, puntaje = _modulo_por_palabras(user_message)
    if sugerido and intent == "general":
        log.info(f"[ROUTER] general → {sugerido} por palabras {puntaje}")
        intent = sugerido
    elif (sugerido and sugerido != intent and intent in INTENTS_DE_DATOS
          and puntaje[sugerido] > puntaje.get(intent, 0)):
        log.info(f"[ROUTER] {intent} → {sugerido} por palabras {puntaje}")
        intent = sugerido
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
        # banderas determinísticas del pedido (no las decide el LLM):
        # si pide gente que todavía no vio, y si pide un recorte que la
        # búsqueda no sabe aplicar (localidad, edad, estudios, género).
        "pide_otros": _pide_otros(user_message),
        "pide_recorte": _pide_recorte(user_message),
        # sexo pedido ("perfiles femeninos"): lo aplica el código sobre el
        # nombre de pila, no el modelo — ver tools.py::search_cvs(sexo=...)
        "sexo_pedido": _genero_pedido(user_message),
        # tamaño de página pedido ("dame 10", "otros 3"); None = el default
        "top_n_pedido": _cantidad_pedida(user_message),
    }


def general_node(state: AgentState) -> AgentState:
    """Respuesta conversacional sin RAG (saludos, dudas, temas generales).

    Es el ÚNICO nodo que contesta sin ningún dato recuperado, así que lleva
    GENERAL_PROMPT encima: acá no hay documento que contradiga una respuesta
    inventada, y es donde salió el "1.200 ítems" de agosto que no existe."""
    messages = [SystemMessage(content=SYSTEM_PROMPT + GENERAL_PROMPT), *state["messages"]]
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

    # "dame otros 5", "sin repetir los que me pasaste": se excluyen EN QDRANT
    # los candidatos que ya se mostraron en esta conversación. Sin esto la
    # búsqueda devolvía siempre el mismo top-5 por más que el usuario lo
    # pidiera distinto (en una charla real, el mismo CV apareció en 5 de 11
    # respuestas seguidas). Se acota a los últimos MAX_EXCLUIR para no armar un
    # must_not gigante en una conversación larga.
    excluir = (state.get("mostrados") or [])[-MAX_EXCLUIR:] if state.get("pide_otros") else []

    # Tamaño de la PÁGINA: lo que pidió el usuario ("dame 10") o el default.
    top_n = min(state.get("top_n_pedido") or config.CANDIDATOS_TOP_N,
                config.CANDIDATOS_TOP_N_MAX)

    # POOL AMPLIADO. Cuando el pedido trae un recorte que la búsqueda NO sabe
    # aplicar (género, zona, edad, estudios, carnet), el filtro lo hace el
    # modelo leyendo los CVs — y si le mandamos sólo 5, filtra sobre 5. Caso
    # real: "perfiles femeninos para depósito" devolvió 5 CVs, los 5 de varones,
    # y la respuesta fue "no hay candidatas", cuando lo único cierto es que no
    # había ninguna entre las 5 más parecidas. Con el recorte se traen
    # top_n * RECORTE_POOL_FACTOR CVs (tope RECORTE_POOL_MAX) y 1 chunk por
    # persona, así el modelo filtra sobre una lista de verdad.
    pool_ampliado = bool(state.get("pide_recorte"))
    pool = min(top_n * config.RECORTE_POOL_FACTOR,
               config.RECORTE_POOL_MAX) if pool_ampliado else top_n
    por_cand = config.RECORTE_CHUNKS_POR_CANDIDATO if pool_ampliado else None
    # RECORTE POR SEXO: no va por el pool ampliado sino por dentro de
    # search_cvs, que descarta por nombre de pila mientras agrupa y devuelve
    # la página ya limpia. Así el modelo no puede listar a los que no cumplen:
    # no los tiene. `sexo_stats` es lo que revisó y descartó, para la respuesta.
    sexo = state.get("sexo_pedido")
    sexo_stats: dict = {}

    def _buscar_cvs():
        texto, cands = search_cvs(
            query,
            state.get("collections") or [],
            vector=vector,
            descartados=state.get("descartados") or [],
            excluir_ids=excluir,
            top_n=pool,
            chunks_por_candidato=por_cand,
            sexo=sexo,
            stats=sexo_stats if sexo else None,
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
    # Sin candidatos: averiguar si es "no hay nadie parecido" o un problema de
    # infraestructura (Qdrant caído, colección de CVs inexistente o vacía).
    # Antes las tres cosas terminaban en la misma respuesta y el reclutador
    # concluía que no había gente cargada.
    # Pidió gente nueva, se excluyó a los ya vistos y no quedó nadie: NO es una
    # falla, es la respuesta honesta (no hay más CVs cargados para ese puesto).
    # Se marca aparte para que el modelo lo diga en vez de reponer a los mismos
    # presentándolos como nuevos — ver prompts.py::NO_REPETIR_SIN_STOCK.
    sin_nuevos = bool(excluir) and not candidatos
    # Se pidió un sexo, se descartaron CVs por nombre de pila y no quedó
    # ninguno: la búsqueda funcionó, no hay nada que diagnosticar. Sin esto,
    # diagnostico_cvs metía un "AVISO TÉCNICO" en el prompt y la respuesta
    # culpaba a la ingesta de CVs.
    sin_del_sexo = bool(sexo) and not candidatos and bool(sexo_stats.get("descartados_sexo"))
    cv_diag = ""
    if not candidatos and not sin_nuevos and not sin_del_sexo:
        cv_diag = diagnostico_cvs(state.get("collections"))
        if cv_diag:
            log.error(f"[RAG] búsqueda de CVs vacía → {cv_diag}")
    log.info(
        f"[RAG] query={query[:120]!r} cols={state.get('collections')} "
        f"{len(docs)} chars cvs + {len(perfil)} chars perfil + {len(proc)} chars proc "
        f"+ {len(candidatos)} candidatos "
        f"(pagina={(len(excluir) // top_n) + 1 if excluir else 1} top_n={top_n} "
        f"pool={pool} ampliado={pool_ampliado} sexo={sexo or '-'} "
        f"desc_sexo={sexo_stats.get('descartados_sexo', 0)}, "
        f"{len(state.get('descartados') or [])} descartados, {len(excluir)} ya mostrados)"
    )
    return {
        **state,
        "retrieved_docs": docs,
        "perfil_docs": perfil,
        "proc_docs": proc,
        "candidatos": candidatos,
        "cv_diag": cv_diag,
        "sin_nuevos": sin_nuevos,
        "sin_del_sexo": sin_del_sexo,
        "sexo_stats": sexo_stats,
        "excluidos_n": len(excluir),
        "top_n_pedido": top_n,
        "pool_ampliado": pool_ampliado,
        # nº de página aproximado, para que la respuesta diga en qué va
        "pagina": (len(excluir) // top_n) + 1 if excluir else 1,
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
    cands = state.get("candidatos") or []
    n_cands = len(cands) or len(names)
    # n_pedido = tamaño de la página (lo que pidió el usuario o el default).
    # Cuando hubo pool ampliado, n_cands es el POOL (lo que hay para revisar) y
    # n_pedido es cuántos tiene que MOSTRAR de los que cumplan el recorte.
    n_pedido = min(state.get("top_n_pedido") or config.CANDIDATOS_TOP_N,
                   config.CANDIDATOS_TOP_N_MAX)
    pool_ampliado = bool(state.get("pool_ampliado")) and n_cands > n_pedido
    shortlist_block = SHORTLIST_RULES.format(n=n_cands) if n_cands else ""
    # Encaje débil: la shortlist es de tamaño fijo, así que cuando no hay gente
    # del rubro los últimos lugares se llenan con cualquier CV. search_cvs los
    # marca (score bajo config.CANDIDATO_MIN_SCORE) y acá se le dice al modelo
    # cómo presentarlos: aparte, sin buscarles el lado bueno. Si TODOS son
    # débiles, la respuesta arranca por "no hay candidatos".
    n_debiles = sum(1 for c in cands if c.get("encaje_debil"))
    if n_debiles and n_debiles == len(cands):
        shortlist_block += ENCAJE_DEBIL_TODOS.format(n=n_cands)
    elif n_debiles and not pool_ampliado:
        # Con pool ampliado no se agrega: ahí la consigna es mostrar sólo a los
        # que cumplen el recorte, y este bloque pide listar aparte a los flojos
        # (que en un pool de 30 son muchos y llenarían la respuesta).
        shortlist_block += ENCAJE_DEBIL_RULES.format(n_debiles=n_debiles, n=n_cands)
    # Pedido de "otros/sin repetir": o se excluyó a los ya vistos y estos son
    # nuevos de verdad, o no quedó nadie y hay que decirlo. Sin esto el modelo
    # volvía a listar a los mismos anunciándolos como "los 5 perfiles nuevos".
    if state.get("sin_nuevos"):
        shortlist_block += NO_REPETIR_SIN_STOCK.format(
            ya=len(state.get("mostrados") or []))
    elif state.get("excluidos_n") and n_cands:
        shortlist_block += NO_REPETIR_OK.format(
            n=n_pedido, pagina=state.get("pagina") or 2,
            ya=state.get("excluidos_n"))
    # El usuario pidió un recorte que la búsqueda no sabe aplicar (localidad,
    # edad, estudios, género): el filtro lo hace el modelo sobre el POOL, y
    # muestra sólo hasta n_pedido de los que cumplan. Reemplaza la regla de
    # "presentalos a todos" de SHORTLIST_RULES.
    if state.get("pide_recorte"):
        shortlist_block += FILTRO_NO_APLICADO_RULES.format(
            n_revisados=n_cands, n=n_pedido)
    # Recorte por SEXO: ya lo aplicó search_cvs sobre el nombre de pila, así
    # que los que no cumplen no están en `docs`. Acá sólo se le cuenta al
    # modelo qué se revisó y qué se descartó, para que no vuelva a redactar
    # "no hay candidatas" arriba de una lista de varones.
    if state.get("sexo_pedido"):
        st = state.get("sexo_stats") or {}
        shortlist_block += GENERO_FILTRADO_RULES.format(
            etiqueta="FEMENINO" if state["sexo_pedido"] == "F" else "MASCULINO",
            revisados=st.get("revisados", n_cands),
            descartados=st.get("descartados_sexo", 0),
            cumplen=st.get("cumplen", n_cands),
        )
    # Quiénes ya se mostraron en la conversación: para que no presente como
    # novedad a alguien que el usuario ya vio.
    ya_vistos = [n for n in (state.get("mostrados_nombres") or []) if n]
    ya_block = (
        YA_MOSTRADOS_BLOCK.format(nombres=", ".join(ya_vistos[-30:])) + "\n"
        if ya_vistos else ""
    )
    # Falla de infraestructura, no ausencia de gente: se lo decimos al modelo
    # para que no responda "no hay candidatos para ese puesto" cuando en
    # realidad no pudo mirar ningún CV.
    diag = (state.get("cv_diag") or "").strip()
    diag_block = (
        f"\n# AVISO TÉCNICO (no es que no haya gente)\n{diag}\n"
        f"Decíselo al usuario tal cual: la búsqueda no pudo leer CVs, así que no "
        f"podés afirmar que no hay candidatos para el puesto. Es un problema de "
        f"configuración/ingesta a revisar, no un resultado de la búsqueda.\n"
    ) if diag else ""
    if state.get("sin_nuevos"):
        vacio = ("(la búsqueda excluyó a los que ya se mostraron y no quedó "
                 "ningún candidato nuevo)")
    elif state.get("sin_del_sexo"):
        st = state.get("sexo_stats") or {}
        vacio = (f"(se revisaron los {st.get('revisados', 0)} CVs más parecidos "
                 f"al puesto y los {st.get('descartados_sexo', 0)} se "
                 f"descartaron por el sexo pedido: no quedó ninguno)")
    else:
        vacio = "(no hay ningún CV cargado que se acerque)"
    titulo_lista = (
        f"## {n_cands} CVs para revisar ({cols}), ordenados por cercanía al "
        f"puesto — de acá salen los que cumplan el recorte:\n"
        if pool_ampliado else
        f"## Shortlist: los {n_cands} candidatos más cercanos ({cols}), "
        f"ordenados de mayor a menor:\n"
    )
    context_prompt = (
        f"{perfil_block}"
        f"{proc_block}"
        f"{ya_block}"
        f"{titulo_lista}"
        f"{docs if docs else vacio}\n\n"
        f"## Consulta del usuario:\n{state['user_message']}\n\n"
        f"{shortlist_block}\n"
        f"{diag_block}"
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


async def ventas_node(state: AgentState) -> AgentState:
    """Intent "ventas": facturación/ranking propio. Toda la lógica de qué
    puede ver cada usuario vive en app/ventas_tools.py y en cómo llegó el
    estado (ver graph_state.py) — acá solo se valida el gate de habilitación
    y se delega. No pasa por response_node: la respuesta ya sale formateada
    con números reales, no hace falta que el LLM la "redacte" (y así no hay
    riesgo de que reescriba una cifra)."""
    from app.ventas_tools import responder_ventas

    habilitado = state.get("ventas_habilitado")
    es_admin = state.get("ventas_admin") or False
    vendedor_codigo = state.get("ventas_vendedor_codigo")

    if not habilitado:
        texto = (
            "No tenés habilitado el acceso a datos de ventas todavía. "
            "Pedile a un administrador que te lo active en Administración → "
            "Usuarios (y que te asigne tu código de vendedor, si todavía no "
            "lo tenés)."
        )
    else:
        texto = await responder_ventas(state["user_message"], vendedor_codigo, es_admin)

    response = AIMessage(content=texto)
    return {**state, "messages": state["messages"] + [response], "final_response": texto}


async def rrhh_node(state: AgentState) -> AgentState:
    """Intent "rrhh": asistencia (faltas, feriados, horas extra). Mismo patrón
    que ventas_node — el permiso llega resuelto por vicki_web contra la sesión
    (ver lib/rrhh/vickiRrhhAcceso.ts) y la respuesta sale formateada de
    app/asistencia_tools.py, sin pasar por el LLM: son números que RRHH usa
    para liquidar, no pueden salir redondeados por un modelo.

    A diferencia de ventas no hay filtro por persona: el permiso es todo o
    nada, porque un dato de asistencia "a medias" no sirve. Por eso el gate es
    lo único que se valida acá."""
    from app.asistencia_tools import responder_rrhh

    if not state.get("rrhh_habilitado"):
        texto = (
            "No tenés habilitado el acceso a los datos de asistencia. Pedile a "
            "un administrador que te lo active en Administración → Usuarios "
            "(columna «Vicki RRHH»)."
        )
    else:
        texto = await responder_rrhh(state["user_message"])

    response = AIMessage(content=texto)
    return {**state, "messages": state["messages"] + [response], "final_response": texto}


async def compras_node(state: AgentState) -> AgentState:
    """Intent "compras": el funnel del mes (faltó → tuvo OC → ingresó). Mismo
    patrón que ventas_node y rrhh_node — el permiso llega resuelto por
    vicki_web contra la sesión (ver lib/compras/vickiComprasAcceso.ts) y la
    respuesta sale ya formateada de app/compras_tools.py, sin pasar por el LLM.

    Como en rrhh, el permiso es todo o nada: no hay un equivalente al
    `vendedorCodigo` que recorte lo que se ve, porque un faltante "a medias" no
    sirve para decidir una compra. El criterio elegido es el más simple de
    explicar: si podés entrar a la vista /compras, el chat te contesta lo mismo
    que ya ves ahí."""
    from app.compras_tools import responder_compras

    if not state.get("compras_habilitado"):
        texto = (
            "No tenés acceso a los datos de compras. Se habilita con el permiso "
            "de la vista Compras — pedíselo a un administrador. (Si te lo "
            "acaban de dar, cerrá sesión y volvé a entrar.)"
        )
    else:
        texto = await responder_compras(state["user_message"])

    response = AIMessage(content=texto)
    return {**state, "messages": state["messages"] + [response], "final_response": texto}


async def deposito_node(state: AgentState) -> AgentState:
    """Intent "deposito": productividad del mes, por preparador (WMS) y por
    mesa de control (EVERWEAR). Mismo patrón que compras_node — el permiso
    llega resuelto por vicki_web contra la sesión (ver
    lib/deposito/vickiDepositoAcceso.ts) y la respuesta sale ya formateada de
    app/deposito_tools.py, sin pasar por el LLM.

    Todo o nada, igual que compras: si podés entrar a la vista /deposito, el
    chat te contesta lo mismo que ya ves ahí (no hay filtro por persona — acá
    no existe un "operario logueado" que recorte la vista)."""
    from app.deposito_tools import responder_deposito

    if not state.get("deposito_habilitado"):
        texto = (
            "No tenés acceso a los datos de depósito. Se habilita con el "
            "permiso de la vista Depósito — pedíselo a un administrador. (Si "
            "te lo acaban de dar, cerrá sesión y volvé a entrar.)"
        )
    else:
        texto = await responder_deposito(state["user_message"])

    response = AIMessage(content=texto)
    return {**state, "messages": state["messages"] + [response], "final_response": texto}


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
