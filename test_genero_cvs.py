"""Recorte por sexo de la búsqueda de CVs ("recomendame perfiles femeninos").

Corre sin Qdrant, sin OpenAI y sin base: stubea langchain/qdrant y reemplaza
`_buscar_hits` por una lista de puntos falsos con nombres reales. Lo que
verifica:

- `genero.sexo_por_nombre`: nombres de pila F/M, compuestos (manda el primero),
  nombres que terminan en -a y son de varón (Luca, Bautista), y que lo que no
  se puede deducir vuelva None en vez de adivinar.
- `nodes._genero_pedido`: qué frases piden un sexo y cuáles no.
- `tools.search_cvs(sexo=...)`: que los del otro sexo NO lleguen al contexto,
  que los indeterminados sí (rotulados), que el limit de Qdrant se sobre-pida
  una sola vez, y que las estadísticas cuenten personas y no chunks.
- `nodes.rag_search_node` / `response_node`: que el sexo viaje a search_cvs,
  que "no quedó ninguna" no se confunda con falla de infraestructura y que el
  bloque de prompt del recorte entre con los números correctos.

Uso:  PYTHONPATH=. python3 test_genero_cvs.py
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

import app.tools as T                       # noqa: E402
import app.nodes as N                       # noqa: E402
from app import genero                      # noqa: E402
from app.config import config               # noqa: E402

print("\n== sexo por nombre de pila ==")
for nombre, esp in [
    ("Maria Laura Gomez", "F"),
    ("Antonella Ruiz", "F"),
    ("Micaela", "F"),
    ("Maria Jose Paz", "F"),          # compuesto: manda el primero
    ("Jose Maria Paz", "M"),          # compuesto al revés
    ("Emmanuel Lalicata", "M"),
    ("Lautaro Dominguez", "M"),
    ("Nahuel Nicolas Gauchat", "M"),
    ("Pablo Ezequiel Carabajal", "M"),
    ("Luca Ferrero", "M"),            # termina en -a y es varón
    ("Bautista Sosa", "M"),
    ("Lucas Cuello", "M"),            # -as no es femenino
    ("Matias Velazquez", "M"),
    ("Agustina Rossi", "F"),
    ("Agustin Rossi", "M"),
    ("Sin nombre", None),             # CV de la ingesta vieja
    ("", None),
    ("Alex Torres", None),            # ambiguo explícito
    ("Xhdjw Kfpqr", None),            # no se puede deducir → no se adivina
    # casos reales de la base (la metadata a veces viene apellido primero, y
    # hay apellidos que son nombre de pila: Paz, Luna, Rosa)
    ("Valenzuela Diego", "M"),        # el -a del apellido no manda
    ("Cabrera Marcelino", "M"),
    ("Alejo paz", "M"),               # Paz acá es apellido
    ("Rosa Benitez", "F"),            # Rosa primero sí es nombre
    ("Luna Gimenez", "F"),
    ("Clever Luna", None),            # no se deduce del apellido
]:
    got = genero.sexo_por_nombre(nombre)
    chk(got == esp, f"{nombre!r} → {got} (esperado {esp})")

print("\n== cumple_sexo: los indeterminados ENTRAN ==")
chk(genero.cumple_sexo("Maria Gomez", "F") == (True, "F"), "mujer con pedido F → entra")
chk(genero.cumple_sexo("Emmanuel Lalicata", "F")[0] is False, "varón con pedido F → fuera")
chk(genero.cumple_sexo("Sin nombre", "F") == (True, None), "indeterminado → entra sin afirmar")
chk(genero.cumple_sexo("Emmanuel Lalicata", None)[0] is True, "sin pedido → no filtra nada")

print("\n== qué frases piden un sexo ==")
for txt, esp in [
    ("recomendame pefiles femeninos para deposito, wms, autoelevador", "F"),
    ("dame 5 mujeres para embolsado", "F"),
    ("necesito candidatas para deposito", "F"),
    ("busco chicas para el turno noche", "F"),
    ("perfiles masculinos para deposito", "M"),
    ("dame varones con autoelevador", "M"),
    ("hombres para logistica", "M"),
    ("busco un operario de deposito", None),
    ("dame otros 5", None),
    ("mujeres y hombres, no importa", None),      # nombra los dos → no filtra
]:
    got = N._genero_pedido(txt)
    chk(got == esp, f"{txt!r} → {got} (esperado {esp})")

print("\n== search_cvs(sexo=...) sobre hits falsos ==")


class _P:
    def __init__(self, nombre, apellido, cid, score):
        self.score = score
        self.payload = {
            "content": f"CV de {nombre} {apellido}: deposito, wms, autoelevador",
            "metadata": {"nombre": nombre, "apellido": apellido,
                         "candidato_id": cid, "email": f"{cid}@x.com"},
        }


# 3 varones, 2 mujeres, 1 sin nombre. El primer varón entra con DOS chunks:
# sirve para comprobar que no se lo cuente dos veces entre los descartados.
_BASE_HITS = [
    ("Emmanuel", "Lalicata", 1, 0.90),
    ("Emmanuel", "Lalicata", 1, 0.88),
    ("Mariano", "Gomez", 2, 0.87),
    ("Antonella", "Ruiz", 3, 0.86),
    ("Lautaro", "Dominguez", 4, 0.85),
    ("", "", 5, 0.84),
    ("Micaela", "Paez", 6, 0.83),
]
_ks = []


def _fake_hits(query, cols, k, flt, vector, score_threshold):
    _ks.append(k)
    return [("cvs", _P(n, a, c, s)) for n, a, c, s in _BASE_HITS]


T._buscar_hits = _fake_hits
T.get_embeddings = lambda: _Any()

st = {}
texto, cands = T.search_cvs("deposito wms", ["cvs"], vector=[0.0], top_n=5,
                            sexo="F", stats=st)
nombres = [c["nombre_completo"] for c in cands]
chk("Antonella Ruiz" in nombres and "Micaela Paez" in nombres,
    f"quedan las mujeres: {nombres}")
chk(not any(x in nombres for x in
            ("Emmanuel Lalicata", "Mariano Gomez", "Lautaro Dominguez")),
    "los varones NO llegan a la lista de candidatos")
chk("Emmanuel" not in texto and "Lautaro" not in texto,
    "los varones NO llegan al contexto del prompt (el modelo no los tiene)")
chk(any(c["candidato_id"] == 5 for c in cands),
    "el CV sin nombre entra igual (indeterminado)")
chk("sexo NO DETERMINADO" in texto, "el indeterminado va rotulado en el contexto")
chk("nombre de pila femenino" in texto and "deducción" in texto,
    "el sexo va rotulado como deducción, no como dato del CV")
chk(st["descartados_sexo"] == 3 and st["cumplen"] == 3
    and st["indeterminados"] == 1 and st["revisados"] == 6,
    f"stats por PERSONA, no por chunk: {st}")
chk(_ks[-1] == min(5 * config.CV_CHUNKS_POR_CANDIDATO * config.GENERO_OVERFETCH,
                   config.GENERO_OVERFETCH_MAX) and len(_ks) == 1,
    f"una sola consulta a Qdrant, con limit sobre-pedido ({_ks[-1]})")

_ks.clear()
texto, cands = T.search_cvs("deposito wms", ["cvs"], vector=[0.0], top_n=5)
chk(len(cands) == 5 and _ks[-1] == max(config.TOP_K, 5 * config.CV_CHUNKS_POR_CANDIDATO),
    "sin pedido de sexo: no filtra nada (5 personas, tope de página) ni sobre-pide")

st = {}
texto, cands = T.search_cvs("deposito wms", ["cvs"], vector=[0.0], top_n=5,
                            sexo="M", stats=st)
chk(len(cands) == 4 and st["descartados_sexo"] == 2,
    f"pedido masculino: quedan los 3 varones + el sin nombre ({len(cands)})")

print("\n== los CVs sin nombre no le roban el lugar a las candidatas ==")
# En la colección real ~700 de 1800 CVs no tienen nombre en la metadata. Si
# entraran por orden de relevancia, una página de 2 serían dos "Sin nombre".
_BASE_HITS[:] = [
    ("", "", 10, 0.95),
    ("", "", 11, 0.94),
    ("", "", 12, 0.93),
    ("Emmanuel", "Lalicata", 13, 0.92),
    ("Antonella", "Ruiz", 14, 0.60),
    ("Micaela", "Paez", 15, 0.55),
]
st = {}
texto, cands = T.search_cvs("deposito", ["cvs"], vector=[0.0], top_n=2,
                            sexo="F", stats=st)
nombres = [c["nombre_completo"] for c in cands]
chk(nombres[:2] == ["Antonella Ruiz", "Micaela Paez"],
    f"las confirmadas primero, aunque tengan menos relevancia: {nombres}")
chk(len(cands) == 2 and [c["posicion"] for c in cands] == [1, 2],
    "la página respeta el tamaño pedido y se renumera")
texto, cands = T.search_cvs("deposito", ["cvs"], vector=[0.0], top_n=4,
                            sexo="F", stats=st)
chk(len(cands) == 4 and [c["nombre_completo"] for c in cands][:2] ==
    ["Antonella Ruiz", "Micaela Paez"] and st["indeterminados"] == 2,
    "si sobran lugares, se rellena con los indeterminados (rotulados)")

print("\n== rag_search_node: el sexo viaja y el vacío no es falla de infra ==")
_llamadas = []


def _fake_search_cvs(query, cols, vector=None, descartados=None, excluir_ids=None,
                     top_n=None, chunks_por_candidato=None, sexo=None, stats=None,
                     **k):
    _llamadas.append(dict(top_n=top_n, chunks=chunks_por_candidato, sexo=sexo))
    if sexo == "F" and stats is not None:
        # simula "revisé 30 y descarté los 30 por sexo"
        stats.update({"revisados": 30, "descartados_sexo": 30, "cumplen": 0,
                      "indeterminados": 0, "sexo_pedido": sexo})
        return "", []
    if stats is not None:
        stats.update({"revisados": top_n, "descartados_sexo": 0,
                      "cumplen": top_n, "indeterminados": 0})
    return "docs", [{"candidato_id": i, "score": 0.9} for i in range(top_n or 5)]


N.search_cvs = _fake_search_cvs
N.embed_query = lambda q: [0.0]
N.search_descripcion_puesto = lambda *a, **k: ""
N.search_procedimientos = lambda *a, **k: ""
_diag = []
N.diagnostico_cvs = lambda *a, **k: (_diag.append(1), "QDRANT CAIDO")[1]


def _rag(**st):
    base = dict(intent="search", user_message="x", search_query="x",
                collections=["cvs"], mostrados=[], descartados=[])
    out = N.rag_search_node({**base, **st})
    return out, _llamadas[-1]

o, l = _rag(sexo_pedido="F")
chk(l["sexo"] == "F", "el sexo pedido llega a search_cvs")
chk(o["sin_del_sexo"] is True and not _diag and not o["cv_diag"],
    "sin candidatas del sexo pedido → NO se diagnostica infraestructura")
chk(o["sexo_stats"]["descartados_sexo"] == 30, "las stats del recorte viajan al estado")
o, l = _rag(sexo_pedido="M")
chk(l["sexo"] == "M" and l["chunks"] is None and l["top_n"] == config.CANDIDATOS_TOP_N,
    "con recorte por sexo la página es normal: 5 CVs con los chunks de siempre")
o, l = _rag()
chk(l["sexo"] is None and o.get("sin_del_sexo") is False, "sin pedido de sexo, todo igual")

print("\n== response_node: bloque del recorte por sexo ==")


class _FakeLLM:
    ultimo = None

    def invoke(self, messages):
        _FakeLLM.ultimo = messages
        return Msg("respuesta")


N.llm = _FakeLLM()
_BASE = dict(messages=[Msg("hola")], intent="search",
             user_message="recomendame perfiles femeninos para deposito",
             retrieved_docs="### Candidato #1 — Antonella Ruiz (relevancia: 0.86)")


def _cand(i, nombre):
    return {"candidato_id": i, "nombre_completo": nombre, "score": 0.9,
            "encaje_debil": False, "posicion": i}


def _prompt(**st):
    N.response_node({**_BASE, **st})
    return _FakeLLM.ultimo[-1].content


t = _prompt(sexo_pedido="F",
            sexo_stats={"revisados": 30, "descartados_sexo": 27, "cumplen": 3,
                        "indeterminados": 1},
            candidatos=[_cand(1, "Antonella Ruiz"), _cand(2, "Micaela Paez"),
                        _cand(3, "Sin nombre")])
chk("YA ESTÁ APLICADO" in t and "FEMENINO" in t, "entra el bloque del recorte por sexo")
chk("revisó 30 CVs" in t and "descartó 27" in t,
    "el prompt lleva cuántos se revisaron y cuántos se descartaron")
chk("no hay candidatas, pero te detallo estos" in t,
    "el prompt prohíbe explícitamente la respuesta que venía dando")
t = _prompt(sexo_pedido="F", sin_del_sexo=True,
            sexo_stats={"revisados": 30, "descartados_sexo": 30, "cumplen": 0,
                        "indeterminados": 0},
            retrieved_docs="", candidatos=[])
chk("se descartaron por el sexo pedido" in t and "30 CVs más parecidos" in t,
    "lista vacía por el recorte → se lo dice al modelo, sin aviso técnico")
chk("AVISO TÉCNICO" not in t, "no se cuela un aviso de infraestructura")
t = _prompt(candidatos=[_cand(1, "Emmanuel Lalicata")])
chk("YA ESTÁ APLICADO" not in t, "sin pedido de sexo el bloque no entra")

print(f"\n{ok} ok, {fail} fail")
sys.exit(1 if fail else 0)
