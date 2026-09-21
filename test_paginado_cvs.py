"""Paginado de la shortlist de CVs y pool ampliado por recorte.

Corre sin Qdrant, sin OpenAI y sin base: stubea langchain/qdrant y reemplaza
search_cvs por una función que registra con qué parámetros la llamaron. Lo que
verifica:

- `_cantidad_pedida`: "dame 5", "otros tres", "los siguientes 10" → tamaño de
  página; y que NO confunda "5 años de experiencia" con una cantidad.
- `_pide_otros`: qué frases piden la página siguiente y cuáles no
  ("contame más del segundo" NO tiene que excluir a los ya mostrados, si no
  esconde justo a la persona por la que preguntan).
- `rag_search_node`: top_n, pool ampliado, chunks por candidato, cuántos se
  excluyen y en qué página va.
- `response_node`: qué bloques de prompt entran en cada caso.

Uso:  PYTHONPATH=. python3 test_paginado_cvs.py
"""
import os
import sys
import types

os.environ.setdefault("HIK_USER", "x")
os.environ.setdefault("HIK_PASS", "x")
os.environ.setdefault("DATABASE_URL", "postgresql://x:x@h/db")

ok = fail = 0


def chk(cond, etiqueta):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok  {etiqueta}")
    else:
        fail += 1
        print(f"  MAL {etiqueta}")


def _stub(nombre, **attrs):
    m = types.ModuleType(nombre)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[nombre] = m
    return m


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()


class Msg:
    def __init__(self, content="", **k): self.content = content


_stub("dotenv", load_dotenv=lambda *a, **k: None)
_stub("langchain_anthropic", ChatAnthropic=_Any)
_stub("langchain_openai", ChatOpenAI=_Any, OpenAIEmbeddings=_Any)
_stub("langchain_core")
_stub("langchain_core.messages", AIMessage=Msg, HumanMessage=Msg,
      SystemMessage=Msg, BaseMessage=Msg)
_stub("langgraph")
_stub("langgraph.graph")
_stub("langgraph.graph.message", add_messages=None)


class _Filter: pass


_qm = _stub("qdrant_client.models", Filter=_Filter, FieldCondition=_Any,
            MatchAny=_Any, PayloadSchemaType=_Any)
_stub("qdrant_client", QdrantClient=_Any, models=_qm)
_stub("app.tool", take_camera_snapshot=lambda *a, **k: None)

import app.nodes as N                       # noqa: E402
from app.config import config               # noqa: E402

MAX = config.CANDIDATOS_TOP_N_MAX

print("\n== cantidad pedida (tamaño de página) ==")
for txt, esp in [
    ("recomendame perfiles femeninos para deposito", None),
    ("dame 5 perfiles para deposito", 5),
    ("dame otros 5", 5),
    ("ahora los otros 5", 5),
    ("mostrame los siguientes 10", 10),
    ("pasame otros tres candidatos", 3),
    ("necesito 8 cvs de administracion", 8),
    ("dame 50 perfiles", MAX),                      # topeado
    ("busco alguien con 5 años de experiencia en wms", None),
    ("que tengan 3 meses de antiguedad", None),
    ("dame mas", None),                             # sin número → default
]:
    got = N._cantidad_pedida(txt)
    chk(got == esp, f"{txt!r} → {got} (esperado {esp})")

print("\n== pide la página siguiente ==")
for txt, esp in [
    ("recomendame perfiles femeninos para deposito", False),
    ("dame otros 5", True),
    ("mostrame los siguientes 5", True),
    ("los proximos 5", True),
    ("segui", True),
    ("dame mas", True),
    ("mas perfiles", True),
    ("sin repetir los que ya me pasaste", True),
    ("contame mas del segundo", False),             # NO: pregunta por uno ya visto
    ("busco un operario de deposito", False),
]:
    got = N._pide_otros(txt)
    chk(got == esp, f"{txt!r} → {got} (esperado {esp})")

print("\n== recorte que la búsqueda no aplica (dispara el pool ampliado) ==")
# El SEXO salió de esta lista: lo aplica el código por el nombre de pila
# (genero.py + search_cvs(sexo=...)), no el pool ampliado — ver test_genero_cvs.py.
for txt, esp in [
    ("recomendame perfiles femeninos para deposito, wms, autoelevador", False),
    ("dame otras 5 mujeres para deposito", False),
    ("que vivan en cordoba capital", True),
    ("mayores de 25 con secundario completo", True),
    ("busco un operario de deposito", False),
]:
    got = N._pide_recorte(txt)
    chk(got == esp, f"{txt!r} → {got} (esperado {esp})")

print("\n== rag_search_node: top_n / pool / exclusión / página ==")
_llamadas = []


def _fake_search_cvs(query, cols, vector=None, descartados=None, excluir_ids=None,
                     top_n=None, chunks_por_candidato=None, **k):
    _llamadas.append(dict(top_n=top_n, chunks=chunks_por_candidato,
                          excluir=len(excluir_ids or [])))
    return "docs", [{"candidato_id": i, "score": 0.5} for i in range(top_n or 5)]


N.search_cvs = _fake_search_cvs
N.embed_query = lambda q: [0.0]
N.search_descripcion_puesto = lambda *a, **k: ""
N.search_procedimientos = lambda *a, **k: ""
N.diagnostico_cvs = lambda *a, **k: ""


def _rag(**st):
    base = dict(intent="search", user_message="x", search_query="x",
                collections=["cvs"], mostrados=[], descartados=[])
    out = N.rag_search_node({**base, **st})
    return out, _llamadas[-1]

o, l = _rag()
chk(l["top_n"] == config.CANDIDATOS_TOP_N and l["chunks"] is None,
    f"sin pedido especial → página de {config.CANDIDATOS_TOP_N}, chunks default")
o, l = _rag(top_n_pedido=10)
chk(l["top_n"] == 10, "pidió 10 → página de 10")
o, l = _rag(pide_recorte=True)
chk(l["top_n"] == min(config.CANDIDATOS_TOP_N * config.RECORTE_POOL_FACTOR,
                      config.RECORTE_POOL_MAX)
    and l["chunks"] == config.RECORTE_CHUNKS_POR_CANDIDATO and o["pool_ampliado"],
    f"recorte → pool de {l['top_n']} CVs con {l['chunks']} chunk c/u")
o, l = _rag(pide_otros=True, mostrados=list(range(5)))
chk(l["excluir"] == 5 and o["pagina"] == 2, "pidió otros con 5 vistos → página 2")
o, l = _rag(pide_otros=True, mostrados=list(range(15)))
chk(o["pagina"] == 4, "15 vistos de a 5 → página 4")
o, l = _rag(pide_otros=True, pide_recorte=True, mostrados=list(range(5)))
chk(o["pool_ampliado"] and l["excluir"] == 5, "recorte + página 2 conviven")
o, l = _rag(pide_otros=True, mostrados=list(range(200)))
chk(l["excluir"] == config.MAX_EXCLUIR,
    f"tope de exclusión = {config.MAX_EXCLUIR}")

print("\n== response_node: bloques del prompt ==")


class _FakeLLM:
    ultimo = None

    def invoke(self, messages):
        _FakeLLM.ultimo = messages
        return Msg("respuesta")


N.llm = _FakeLLM()
_BASE = dict(messages=[Msg("hola")], intent="search", user_message="x",
             retrieved_docs="--- Persona 1 (colección: cvs, relevancia: 0.50) ---")


def _cand(i, score=0.5):
    return {"candidato_id": i, "nombre_completo": f"Persona {i}", "score": score,
            "encaje_debil": score < config.CANDIDATO_MIN_SCORE, "posicion": i}


def _prompt(**st):
    N.response_node({**_BASE, **st})
    return _FakeLLM.ultimo[-1].content

t = _prompt(candidatos=[_cand(i) for i in range(1, 6)])
chk("Shortlist:" in t and "ESTO REEMPLAZA" not in t and "Página" not in t,
    "página 1 normal → shortlist común")
t = _prompt(pide_recorte=True, pool_ampliado=True,
            candidatos=[_cand(i) for i in range(1, 26)])
chk("25 CVs para revisar" in t and "hasta 5 que CUMPLAN" in t,
    "recorte → 25 CVs para revisar, mostrar hasta 5 que cumplan")
t = _prompt(excluidos_n=5, pagina=2, candidatos=[_cand(i) for i in range(6, 11)])
chk("Página 2" in t, "página 2 → el prompt dice en qué página va")
t = _prompt(sin_nuevos=True, excluidos_n=10, mostrados=list(range(10)), candidatos=[])
chk("NO QUEDAN CANDIDATOS NUEVOS" in t, "sin stock nuevo → lo dice primero")
t = _prompt(pide_recorte=True, pool_ampliado=True, excluidos_n=5, pagina=2,
            top_n_pedido=3, candidatos=[_cand(i) for i in range(6, 21)])
chk("hasta 3 que CUMPLAN" in t and "Página 2" in t,
    "recorte + página 2 + cantidad pedida")

print(f"\n{ok} ok, {fail} fail")
sys.exit(1 if fail else 0)
