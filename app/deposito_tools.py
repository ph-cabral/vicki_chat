"""Intent "deposito": productividad del mes — ítems por preparador (WMS) e
ítems por mesa de control (EVERWEAR), cada uno en SU PROPIA consulta agregada.

DE DÓNDE SALEN LOS NÚMEROS (dos fuentes, dos bases de Magnus — NO Postgres)

  · **Preparadores** = operarios de picking (WMS, base separada de EVERWEAR).
    Traduce a SQL lo mismo que ya usa /deposito → "Pedidos preparados"/
    "Tiempos de Picking":
      · vicki_web/indicadores-api/deposito.py::SQL_WMS_TODOS — mismo recorte
        (OT.OTEstado IN (2,3,4), OTFechaHoraEjecucion en rango, join a
        Personal por OTUsuarioGUID_Repositor).
      · La métrica es RECOLECTADOS (OTItemTipo=1 AND OTItemCantCumplida>0),
        NO el crudo "ITEMS pickeados" — ver
        [[deposito_pedidos_preparados_vs_tiempos_picking]]: es la que cierra
        con "Pedidos preparados", el ítem tocado pero no completado
        (faltante) no debe contar como productividad de nadie.
    Acá se agrega por operario en el propio SQL (SUM/GROUP BY) — nunca se
    trae una fila por OT a Python, serían miles por mes.

  · **Mesa de control** = controladores de la pestaña "Mesas de Control" de
    /deposito (contaduría los llama por CodCentroPrep = "Centro de
    Preparación"; en el día a día son "las mesas"). Traduce a SQL lo mismo
    que ya usa esa pestaña:
      · vicki_web/indicadores-api/mesa_control.py::SQL_RENGLONES_CONTROLADOS
        — join Ven_PedImpresoCP + venfer_pedidoReng por (NroMovVenta,
        CodCentroPrep), recorte por FechaControl (entero Clarion, días desde
        1800-12-28).
      · El crédito por controlador SE DEDUPLICA cuando CodControlador1 ==
        CodControlador2 (caso normal: el sistema duplica el mismo código en
        los dos campos) pero se acredita a los DOS si algún día son
        distintos. En Python (mesa_control.py) esto lo hacía un `set()` por
        fila; acá se resuelve en el propio SQL con UNION (no UNION ALL)
        sobre (NroMovVenta, NroRenglon, Codigo): dos filas idénticas
        colapsan solas, dos códigos distintos sobreviven los dos. Mismo
        resultado, una sola consulta agregada — ver `_SQL_MESA`.
      · El total "exacto" (renglones distintos, sin duplicar por recontrol
        ni por doble control) sale de la MISMA consulta como una fila
        sentinela ('~TOTAL'), evitando un segundo viaje a Magnus.

CÓMO SE LEE UN NÚMERO QUE NO CIERRA
  La suma de "mesa de control" por controlador puede dar un poco MÁS que el
  total exacto: si un pedido tuvo recontrol/reimpresión (mismo renglón
  controlado más de una vez en el mes), el total lo cuenta 1 vez pero cada
  evento de control se le acredita a quien lo controló — es crédito de
  productividad, no el universo de renglones. Ver el comentario de
  SQL_RENGLONES_CONTROLADOS en mesa_control.py, que documenta el mismo caso.
  Preparadores NO tiene esta trampa: un ítem recolectado es de UN operario.

RANGO DE FECHAS
  Siempre el MES CALENDARIO completo (se reusa `compras_tools._mes_calendario`,
  que a su vez usa `ventas_tools._parsear_rango`) — mismo criterio que
  "compras", para que "productividad de agosto" no dependa de en qué día del
  mes se pregunte.

SEGURIDAD
  Sin filtro por persona: el permiso es el de la vista /deposito (o ADMIN),
  todo o nada — mismo patrón que "compras" (lib/deposito/vickiDepositoAcceso.ts).
  Sin ese permiso, `deposito_node` ni llama a este módulo. Del mensaje del
  usuario no sale nada al SQL: sólo qué recorte mostrar (preparadores / mesa /
  los dos) y el mes, los dos por regex sobre listas cerradas.
"""
import logging
import re

from app.compras_tools import _ent, _mes_calendario
from app.ventas_tools import _FalloMagnus, _dias, _ejecutar_sql, _normalizar

log = logging.getLogger("deposito_tools")

_MSG_FALLO = (
    "No pude consultar los datos de depósito ahora mismo. Probá de nuevo en "
    "un rato; si sigue fallando, avisale a sistemas."
)

# ── Qué recorte pidió (preparadores / mesa / los dos) ─────────────────────────
_PAT_PREPARADORES = r"\bpreparador(es)?\b|\boperari[oa]s?\b|\bpicking\b|\bpicke[oa]d?[oa]?s?\b|\barmador(es)?\b"
_PAT_MESA = r"\bmesas?\b|\bcontrolador(es)?\b|\bmesas? de control\b|\bcontrol(es)?\b"


def _recorte_pedido(mensaje: str) -> str:
    """"preparadores" | "mesa" | "todos" — todos = default (los dos, cada uno
    por separado, que es lo que se pidió al principio)."""
    m = _normalizar(mensaje)
    tiene_prep = bool(re.search(_PAT_PREPARADORES, m))
    tiene_mesa = bool(re.search(_PAT_MESA, m))
    if tiene_prep and not tiene_mesa:
        return "preparadores"
    if tiene_mesa and not tiene_prep:
        return "mesa"
    return "todos"


# ── Preparadores (WMS) — items RECOLECTADOS por operario ─────────────────────
# Mismo recorte que SQL_WMS_TODOS de deposito.py (OTEstado 2/3/4, rango por
# OTFechaHoraEjecucion, join a Personal por OTUsuarioGUID_Repositor), agregado
# por operario en el propio SQL. RECOLECTADOS (cumplida>0), no el crudo — ver
# docstring del módulo.
_SQL_PREPARADORES = """
SELECT
    LTRIM(RTRIM(ISNULL(P_Repositor.PersonalNombre, '(sin operario)'))) AS OPERARIO,
    SUM(ISNULL(i.RECOLECTADOS, 0)) AS ITEMS
FROM OT
INNER JOIN Codot ON OT.CodotCodigo = Codot.CodotCodigo
LEFT JOIN Personal P_Repositor ON OT.OTUsuarioGUID_Repositor = P_Repositor.PersonalId
LEFT JOIN (
    SELECT OTId,
        SUM(CASE WHEN OTItemTipo = 1 AND OTItemCantCumplida > 0 THEN 1 ELSE 0 END) AS RECOLECTADOS
    FROM OTItem GROUP BY OTId
) i ON OT.OTId = i.OTId
WHERE OT.OTEstado IN (2, 3, 4)
  AND OT.OTFechaHoraEjecucion >= '{desde}'
  AND OT.OTFechaHoraEjecucion <= '{hasta}'
GROUP BY LTRIM(RTRIM(ISNULL(P_Repositor.PersonalNombre, '(sin operario)')))
HAVING SUM(ISNULL(i.RECOLECTADOS, 0)) > 0
ORDER BY ITEMS DESC
"""

# ── Mesa de control (EVERWEAR) — items controlados por controlador ───────────
# Mismo recorte que SQL_RENGLONES_CONTROLADOS de mesa_control.py. `creditos`
# reproduce el dedupe por fila (cod1==cod2 -> 1 solo crédito) con UNION en vez
# de un set() en Python. La fila '~TOTAL' es el número "exacto" (renglones
# distintos del mes, mismo criterio que por_mes/total_general en
# mesa_control.py) — sale de la MISMA consulta, sin un segundo viaje a Magnus.
_SQL_MESA = """
WITH renglones AS (
    SELECT reng.NroMovVenta, reng.NroRenglon, ped.CodControlador1, ped.CodControlador2
    FROM dbo.Ven_PedImpresoCP ped
    JOIN dbo.venfer_pedidoReng reng
      ON ped.NroMovVenta   = reng.NroMovVenta
     AND ped.CodCentroPrep = reng.CodCentroPrep
    WHERE (ped.CodControlador1 > 0 OR ped.CodControlador2 > 0)
      AND ped.FechaControl BETWEEN {d1} AND {d2}
),
creditos AS (
    SELECT NroMovVenta, NroRenglon, CodControlador1 AS Codigo FROM renglones WHERE CodControlador1 > 0
    UNION
    SELECT NroMovVenta, NroRenglon, CodControlador2 AS Codigo FROM renglones WHERE CodControlador2 > 0
)
SELECT '~TOTAL' AS Codigo, '' AS Controlador,
       COUNT(DISTINCT CAST(NroMovVenta AS varchar(20)) + '-' + CAST(NroRenglon AS varchar(20))) AS Items
FROM renglones
UNION ALL
SELECT CAST(c.Codigo AS varchar(20)) AS Codigo,
       LTRIM(RTRIM(ISNULL(u.Nombre, ''))) AS Controlador,
       COUNT(*) AS Items
FROM creditos c
LEFT JOIN dbo.Gen_Usuarios u ON u.Numero = c.Codigo
GROUP BY c.Codigo, LTRIM(RTRIM(ISNULL(u.Nombre, '')))
"""


def _parse_tsv(tsv: str) -> list[dict]:
    """TSV de magnus → filas. La última línea es "(N filas)" y se descarta —
    mismo parser que compras_tools/ventas_tools, sin duplicarlo porque acá
    alcanza con esto (no vale la pena importar uno ajeno por una función de
    5 líneas usada distinto en cada módulo)."""
    filas = []
    lineas = [l for l in (tsv or "").splitlines() if l.strip()]
    if not lineas:
        return filas
    cols = lineas[0].split("\t")
    for linea in lineas[1:]:
        if linea.startswith("(") and linea.rstrip().endswith("filas)"):
            continue
        partes = linea.split("\t")
        if len(partes) != len(cols):
            continue
        filas.append(dict(zip(cols, partes)))
    return filas


def _num(v) -> float:
    try:
        return float(str(v).strip() or 0)
    except ValueError:
        return 0.0


async def _preparadores(desde_hasta: tuple) -> list[dict]:
    desde, hasta = desde_hasta
    sql = _SQL_PREPARADORES.format(
        desde=f"{desde.isoformat()} 00:00:00",
        hasta=f"{hasta.isoformat()} 23:59:59",
    )
    tsv = await _ejecutar_sql(sql, db="WMS")
    filas = []
    for f in _parse_tsv(tsv):
        nombre = (f.get("OPERARIO") or "").strip()
        if not nombre:
            continue
        filas.append({"nombre": nombre, "items": _num(f.get("ITEMS"))})
    return filas


async def _mesa(desde_hasta: tuple) -> tuple[list[dict], float]:
    desde, hasta = desde_hasta
    sql = _SQL_MESA.format(d1=_dias(desde), d2=_dias(hasta))
    tsv = await _ejecutar_sql(sql, db="EVERWEAR")
    filas = []
    total = 0.0
    for f in _parse_tsv(tsv):
        codigo = (f.get("Codigo") or "").strip()
        if codigo == "~TOTAL":
            total = _num(f.get("Items"))
            continue
        nombre = (f.get("Controlador") or "").strip() or f"Controlador {codigo}"
        items = _num(f.get("Items"))
        if items <= 0:
            continue
        filas.append({"nombre": nombre, "items": items})
    filas.sort(key=lambda x: -x["items"])
    return filas, total


def _tabla(filas: list[dict]) -> str:
    if not filas:
        return "  (sin actividad registrada)"
    return "\n".join(f"  • {f['nombre']}: {_ent(f['items'])}" for f in filas)


async def responder_deposito(mensaje: str) -> str:
    """Punto de entrada del intent. Una respuesta ya formateada, sin pasar
    por el LLM — igual que ventas/rrhh/compras: son números operativos, no
    algo que un modelo deba redactar o redondear."""
    desde, hasta, etiqueta = _mes_calendario(mensaje)
    recorte = _recorte_pedido(mensaje)

    try:
        partes = [f"🏭 *Depósito — {etiqueta}*", ""]

        if recorte in ("preparadores", "todos"):
            prep = await _preparadores((desde, hasta))
            total_prep = sum(f["items"] for f in prep)
            partes.append(f"*Preparadores* — ítems recolectados (total: {_ent(total_prep)})")
            partes.append(_tabla(prep))
            partes.append("")

        if recorte in ("mesa", "todos"):
            mesa_filas, mesa_total = await _mesa((desde, hasta))
            partes.append(f"*Mesa de control* — ítems controlados (total: {_ent(mesa_total)})")
            partes.append(_tabla(mesa_filas))
            suma_creditos = sum(f["items"] for f in mesa_filas)
            if suma_creditos != mesa_total:
                partes.append("")
                partes.append(
                    "_El total es de renglones distintos (sin duplicar); la suma por "
                    "controlador puede dar un poco más si hubo recontrol/reimpresión "
                    "del mismo renglón — cada control se le acredita a quien lo hizo._"
                )
            partes.append("")
    except _FalloMagnus:
        return _MSG_FALLO

    while partes and partes[-1] == "":
        partes.pop()
    return "\n".join(partes)
