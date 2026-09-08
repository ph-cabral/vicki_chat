"""Intent "compras": el funnel del mes — cuánto faltó, cuánto tuvo Orden de
Compra y cuánto ingresó, valorizado a precio de venta.

DE DÓNDE SALEN LOS NÚMEROS
    De Magnus, por el mismo camino que ventas (`ventas_tools._ejecutar_sql` →
    endpoint HTTP de mcp-magnus). **UNA sola consulta agregada** por pregunta:
    devuelve una fila por origen (nacionales / importados / otros / fabrica /
    original) ya con items, unidades y $ de las tres etapas. Nunca se traen los
    renglones crudos a Python — son decenas de miles por mes.

    El criterio NO se inventa: es la traducción a SQL de lo que ya calcula la
    vista /compras. Si algún día un número no cierra, la referencia es:

      · vicki_web/lib/compras/faltantesMes.ts
        → el recorte del mes: una fila por renglón (la MÁS NUEVA del mes, por
          eso el ROW_NUMBER), sólo artículos con Estado = 1 (Habilitado), y
          fuera los renglones de pedidos CANCELADOS o sin estado en Magnus. Un
          artículo que faltaba SÓLO por esos renglones deja de ser faltante.
      · vicki_web/lib/compras/origenArticulo.ts
        → el origen manda siempre por tipo de artículo
          (StkFer_Articulos.NacionalImportado → Stk_TiposArticulos); el
          proveedor "EVER WEAR S.A. INDUSTRIAL" recién decide cuando el
          artículo no tiene tipo cargado.
      · vicki_web/indicadores-api/compras.py
        → qué cuenta como Orden de Compra: cabecera no cancelada (Estado <> 4),
          artículo que no sea Genérico ni Fabril, y **comprobante 70 (ORDEN DE
          COMPRA) o 75 (OC IMPO)**. Las OC de presupuesto por área — 74 RRHH,
          76 INDUSTRIA, 77 MARKETING, 78 SISTEMAS IT — y el pase interno 80
          (INGRESO INDUSTRIA A COMERCIAL) NO son compra de mercadería.
      · vicki_web/indicadores-api/ingresos.py
        → qué cuenta como ingreso: comprobantes 59, 60, 61, 160 y 590.

    Es un funnel ESTRICTO, igual que la vista: "con OC" ⊆ "faltantes" e
    "ingresados" ⊆ "con OC".

CÓMO SE VALORIZA (la trampa de leer estos números)
    Las tres etapas se informan por la plata que FALTABA, no por la que se
    pidió ni por la que ingresó. Es la única forma de que el funnel se lea como
    cobertura: "de los $X que faltaron, $Y ya tiene OC y $Z ya llegó". Medido
    por lo pedido, un contenedor de 420.000 precintos comprado para cubrir un
    faltante de 300 unidades hace que la etapa 2 sea 18 veces la etapa 1.

EL IMPORTADO NO SE LEE IGUAL QUE EL NACIONAL
    La pregunta que contesta la etapa 2 es "¿se pidió ESE MES?". Para el
    nacional es la pregunta correcta (se repone mes a mes). Para el importado
    no: se compra por contenedor con 3 a 6 meses de anticipación, así que la OC
    que cubre un faltante de agosto se hizo en marzo y no entra. En agosto 2026
    eso son 7 items de 263. Por eso la respuesta lo aclara cuando el número da
    bajo, en vez de dejar creer que compras no pidió nada.

SEGURIDAD
    Sin filtro por persona: quien puede entrar a la vista /compras (o es ADMIN)
    ve el mismo número que la vista. Lo resuelve vicki_web contra la cookie de
    sesión — ver lib/compras/vickiComprasAcceso.ts. Sin el permiso,
    `compras_node` ni llama a este módulo.

    Del mensaje del usuario NO sale nada al SQL: sólo se leen un mes y un
    origen, los dos con regex sobre listas cerradas, y viajan al SQL como
    enteros e identificadores de una whitelist.
"""
import datetime as dt
import logging
import re

# Un solo parser de fechas para todo el chat y un solo camino a Magnus: si "el
# mes pasado" cambia de significado, o cambia el transporte, cambia en un solo
# lugar.
from app.ventas_tools import (
    _FalloMagnus,
    _ejecutar_sql,
    _monto,
    _normalizar,
    _parsear_rango,
)

log = logging.getLogger("compras_tools")

# Magnus guarda las fechas como días desde esta base.
_BASE_MAGNUS = dt.date(1800, 12, 28)

# Comprobantes de Com_OrdCompCabecera que son compra real de mercadería.
# Espejo de COMP_CODIGOS_OC_COMPRA en indicadores-api/compras.py.
_COMP_OC_COMPRA = (70, 75)
# Comprobantes de Com_RemitoCabecera que cuentan como ingreso de mercadería.
# Espejo de CODIGOS_REMITO_INGRESO en indicadores-api/ingresos.py.
_COMP_REMITO_INGRESO = (59, 60, 61, 160, 590)

# Los orígenes que compra el sector. Fábrica (producción interna) y Original
# quedan fuera del total, igual que en la vista: Fábrica ni siquiera puede
# aparecer en el set de OC, así que sumarla sólo infla el "sin cubrir".
_ORIGENES_COMPRA = ("nacionales", "importados", "otros")

_ORIGEN_LABEL = {
    "nacionales": "Nacionales",
    "importados": "Importados",
    "otros": "Otros",
    "fabrica": "Fábrica",
    "original": "Original",
}

# Cómo pide el usuario cada origen. Se busca sobre el mensaje normalizado (sin
# acentos), primer match gana.
_ORIGEN_PATRONES = (
    ("nacionales", r"\bnacional(es)?\b"),
    ("importados", r"\bimportad[oa]s?\b|\bimpo\b|\bimportacion\b"),
    ("fabrica", r"\bfabrica\b|\bfabril(es)?\b|\bproduccion\b"),
    ("original", r"\boriginal(es)?\b"),
)


def _dias(f: dt.date) -> int:
    """date → entero de días Magnus. Va como LITERAL al SQL: el driver viejo
    filtra mal si se compara una fecha calculada contra un parámetro."""
    return (f - _BASE_MAGNUS).days


def _mes_calendario(mensaje: str, hoy: dt.date | None = None) -> tuple[dt.date, dt.date, str]:
    """El funnel de compras es siempre un MES CALENDARIO completo (así cierra
    con la vista y con el reporte de Magnus). Se reusa el parser de ventas para
    entender "agosto", "el mes pasado", "2026", etc., y después se estira al mes
    entero que contiene esa fecha. Devuelve (primer día, último día, etiqueta),
    los dos INCLUSIVOS."""
    hoy = hoy or dt.date.today()
    desde, _hasta_excl, _etq = _parsear_rango(mensaje, hoy)
    y, m = desde.year, desde.month
    primero = dt.date(y, m, 1)
    ultimo = (dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)) - dt.timedelta(days=1)
    meses = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre")
    return primero, ultimo, f"{meses[m - 1]} {y}"


def _origen_pedido(mensaje: str) -> str | None:
    """Origen que nombró el usuario, o None si no nombró ninguno (→ se muestra
    el panorama: nacionales, importados y el total)."""
    m = _normalizar(mensaje)
    for clave, patron in _ORIGEN_PATRONES:
        if re.search(patron, m):
            return clave
    return None


# ── LA consulta ──────────────────────────────────────────────────────────────
# Una fila por origen. Todo el trabajo (recorte, clasificación y los 3 cruces)
# pasa en el servidor: salen 5 filas como mucho.
#
# `base`/`ren`/`art` reproducen agruparFaltantesMes + pasaRecorte de
# lib/compras/faltantesMes.ts. `oc` e `ing` son los sets B y C del funnel.
_SQL_FUNNEL = """
WITH base AS (
    SELECT p.NroPedOrigen, p.NroRengOrigen, p.CodArticu,
           p.CantPendiente, p.PrecioVenta,
           ROW_NUMBER() OVER (
               PARTITION BY p.NroPedOrigen, p.NroRengOrigen
               ORDER BY p.FecRegistracion DESC
           ) AS rn
    FROM EVERWEAR.dbo.Ven_PedRenPendientes p
    WHERE p.FecRegistracion BETWEEN {d1} AND {d2}
),
ren AS (
    SELECT
        LTRIM(RTRIM(b.CodArticu)) AS Cod,
        CASE
            WHEN LOWER(LTRIM(RTRIM(ISNULL(t.Descripcion, '')))) IN ('fabril', 'fabrica') THEN 'fabrica'
            WHEN LOWER(LTRIM(RTRIM(ISNULL(t.Descripcion, '')))) = 'original'  THEN 'original'
            WHEN LOWER(LTRIM(RTRIM(ISNULL(t.Descripcion, '')))) = 'nacional'  THEN 'nacionales'
            WHEN LOWER(LTRIM(RTRIM(ISNULL(t.Descripcion, '')))) = 'importado' THEN 'importados'
            WHEN UPPER(ISNULL(pr.RazonSocial, '')) LIKE '%EVER WEAR S.A. INDUSTRIAL%' THEN 'fabrica'
            ELSE 'otros'
        END AS Origen,
        CASE WHEN s.Estado = 1 THEN 1 ELSE 0 END AS Hab,
        -- Sin estado de pedido = pedido que no está en Magnus: se descarta
        -- igual que un cancelado (mismo criterio que esCancelado en TS).
        CASE WHEN pest.Ped_EstadoDescripcion IS NULL
                  OR UPPER(pest.Ped_EstadoDescripcion) LIKE '%CANCEL%'
             THEN 0 ELSE 1 END AS Vivo,
        b.CantPendiente, b.PrecioVenta
    FROM base b
    LEFT JOIN EVERWEAR.dbo.StkFer_Articulos    s    ON s.CodArticulo = b.CodArticu
    LEFT JOIN EVERWEAR.dbo.Stk_TiposArticulos  t    ON t.CodigoTipo  = s.NacionalImportado
    LEFT JOIN EVERWEAR.dbo.Com_Proveedores     pr   ON pr.CodProveed = s.CodProveedHabitual
    LEFT JOIN EVERWEAR.dbo.VenFer_PedidoCabecera cab ON cab.NroMovVenta = b.NroPedOrigen
    LEFT JOIN MAGNUS_SITD.dbo.Pedido_Estados   pest ON pest.Ped_Estado = cab.EstadoPedido
    WHERE b.rn = 1
),
art AS (
    SELECT Cod,
           MAX(Origen) AS Origen,
           MAX(Hab)    AS Hab,
           SUM(CASE WHEN Vivo = 1 THEN CantPendiente ELSE 0 END)                AS U,
           SUM(CASE WHEN Vivo = 1 THEN CantPendiente * PrecioVenta ELSE 0 END)  AS Imp,
           SUM(CASE WHEN Vivo = 0 THEN CantPendiente ELSE 0 END)                AS UC
    FROM ren
    GROUP BY Cod
),
falt AS (
    -- Habilitado y que no quede vivo SÓLO por renglones cancelados.
    SELECT Cod, Origen, U, Imp
    FROM art
    WHERE Hab = 1 AND NOT (U <= 0 AND UC > 0)
),
oc AS (
    SELECT DISTINCT LTRIM(RTRIM(r.CodArticulo)) AS Cod
    FROM EVERWEAR.dbo.Com_OrdCompRenglones r
    INNER JOIN EVERWEAR.dbo.Com_OrdCompCabecera cab ON cab.NroOrdCompra = r.NroOrdCompra
    LEFT  JOIN EVERWEAR.dbo.StkFer_Articulos    a   ON a.CodArticulo    = r.CodArticulo
    LEFT  JOIN EVERWEAR.dbo.Stk_TiposArticulos  t   ON t.CodigoTipo     = a.NacionalImportado
    WHERE cab.FecMovim BETWEEN {d1} AND {d2}
      AND cab.Estado <> 4
      AND cab.CompCodigo IN ({comp_oc})
      AND ISNULL(t.Descripcion, 'Nacional') NOT IN ('Generico', 'Fabril')
),
ing AS (
    SELECT DISTINCT LTRIM(RTRIM(r.CodArticulo)) AS Cod
    FROM EVERWEAR.dbo.Com_RemitoRenglones r
    INNER JOIN EVERWEAR.dbo.Com_RemitoCabecera cab ON cab.NroMovRemito = r.NroMovRemito
    WHERE cab.FecComprobante BETWEEN {d1} AND {d2}
      AND cab.CompCodigo IN ({comp_ing})
)
SELECT
    f.Origen                                                              AS Origen,
    COUNT(*)                                                              AS Items,
    ROUND(SUM(f.U), 0)                                                    AS Unid,
    ROUND(SUM(f.Imp), 0)                                                  AS Imp,
    SUM(CASE WHEN o.Cod IS NOT NULL THEN 1 ELSE 0 END)                    AS OcItems,
    ROUND(SUM(CASE WHEN o.Cod IS NOT NULL THEN f.Imp ELSE 0 END), 0)      AS OcImp,
    SUM(CASE WHEN o.Cod IS NOT NULL AND i.Cod IS NOT NULL THEN 1 ELSE 0 END) AS IngItems,
    ROUND(SUM(CASE WHEN o.Cod IS NOT NULL AND i.Cod IS NOT NULL
                   THEN f.Imp ELSE 0 END), 0)                             AS IngImp
FROM falt f
LEFT JOIN oc  o ON o.Cod = f.Cod
LEFT JOIN ing i ON i.Cod = f.Cod
GROUP BY f.Origen
"""


def _parse_tsv(tsv: str) -> list[dict]:
    """TSV de magnus → filas. La última línea es "(N filas)" y se descarta."""
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


def _pct(parte: float, total: float) -> str:
    return f"{(parte / total * 100):.1f}%".replace(".", ",") if total > 0 else "0%"


def _ent(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def _bloque(nombre: str, d: dict) -> str:
    """Las 3 etapas de un origen, siempre valorizadas por lo que FALTABA."""
    falto = d["Imp"]
    con_oc = d["OcImp"]
    ingreso = d["IngImp"]
    sin_cubrir = max(falto - con_oc, 0)
    en_camino = max(con_oc - ingreso, 0)
    return (
        f"*{nombre}* — faltaron {_ent(d['Items'])} artículos "
        f"({_ent(d['Unid'])} u.) por {_monto(falto)}\n"
        f"  • ya ingresó: {_monto(ingreso)} ({_pct(ingreso, falto)}) — {_ent(d['IngItems'])} artículos\n"
        f"  • con OC sin ingresar: {_monto(en_camino)} ({_pct(en_camino, falto)}) — "
        f"{_ent(max(d['OcItems'] - d['IngItems'], 0))} artículos\n"
        f"  • sin cubrir: {_monto(sin_cubrir)} ({_pct(sin_cubrir, falto)}) — "
        f"{_ent(max(d['Items'] - d['OcItems'], 0))} artículos"
    )


_MSG_FALLO = (
    "No pude consultar Magnus ahora mismo. Probá de nuevo en un rato; si sigue "
    "fallando, avisale a sistemas."
)


async def responder_compras(mensaje: str) -> str:
    """Punto de entrada del intent. Una consulta, una respuesta ya formateada:
    no pasa por el LLM, igual que ventas y rrhh — son números que compras usa
    para trabajar y no pueden salir redondeados por un modelo."""
    desde, hasta, etiqueta = _mes_calendario(mensaje)
    origen = _origen_pedido(mensaje)

    sql = _SQL_FUNNEL.format(
        d1=_dias(desde),
        d2=_dias(hasta),
        comp_oc=",".join(str(c) for c in _COMP_OC_COMPRA),
        comp_ing=",".join(str(c) for c in _COMP_REMITO_INGRESO),
    )

    try:
        tsv = await _ejecutar_sql(sql)
    except _FalloMagnus:
        return _MSG_FALLO

    por_origen: dict[str, dict] = {}
    for f in _parse_tsv(tsv):
        clave = (f.get("Origen") or "").strip()
        if not clave:
            continue
        por_origen[clave] = {
            "Items": _num(f.get("Items")),
            "Unid": _num(f.get("Unid")),
            "Imp": _num(f.get("Imp")),
            "OcItems": _num(f.get("OcItems")),
            "OcImp": _num(f.get("OcImp")),
            "IngItems": _num(f.get("IngItems")),
            "IngImp": _num(f.get("IngImp")),
        }

    if not por_origen:
        return f"No encontré faltantes registrados en {etiqueta}."

    def _sumar(claves) -> dict:
        out = {k: 0.0 for k in
               ("Items", "Unid", "Imp", "OcItems", "OcImp", "IngItems", "IngImp")}
        for c in claves:
            d = por_origen.get(c)
            if not d:
                continue
            for k in out:
                out[k] += d[k]
        return out

    partes = [f"📦 *Compras — {etiqueta}*", ""]

    if origen:
        d = por_origen.get(origen)
        if not d:
            return (
                f"En {etiqueta} no hubo faltantes de "
                f"{_ORIGEN_LABEL.get(origen, origen)}."
            )
        partes.append(_bloque(_ORIGEN_LABEL[origen], d))
    else:
        for clave in ("nacionales", "importados"):
            if clave in por_origen:
                partes.append(_bloque(_ORIGEN_LABEL[clave], por_origen[clave]))
                partes.append("")
        total = _sumar(_ORIGENES_COMPRA)
        if total["Items"]:
            partes.append(_bloque("Total (Nacionales + Importados)", total))
        afuera = _sumar(("fabrica", "original"))
        if afuera["Items"]:
            partes.append("")
            partes.append(
                f"_Quedan afuera {_ent(afuera['Items'])} artículos de "
                f"Fábrica/Original: son producción interna, no compra._"
            )

    # El importado se pide por contenedor con meses de anticipación: su OC casi
    # nunca cae en el mismo mes en que faltó. Sin esta línea el número se lee
    # como "compras no pidió nada", que es exactamente lo contrario.
    imp = por_origen.get("importados")
    if imp and imp["Items"] and (origen in (None, "importados")):
        cubierto = imp["OcItems"] / imp["Items"] if imp["Items"] else 0
        if cubierto < 0.25:
            partes.append("")
            partes.append(
                "_Ojo con el importado: se compra por contenedor con 3 a 6 meses "
                "de anticipación, así que la OC que cubre un faltante de este mes "
                "se hizo mucho antes y no entra en el conteo. Ese número bajo no "
                "quiere decir que no se haya pedido._"
            )

    partes.append("")
    partes.append(
        "_Las tres etapas se valorizan por lo que FALTABA (precio de venta), no "
        "por lo que se pidió ni por lo que ingresó. Cuentan sólo las OC de "
        "mercadería (comprobantes 70 y 75): las de presupuesto por área no._"
    )
    return "\n".join(partes)
