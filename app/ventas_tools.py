"""Intent "ventas": facturación/ranking mensual de un vendedor, vía mcp-magnus.

DISEÑO DE SEGURIDAD (leer antes de tocar este archivo):

  El vendedorCodigo que filtra la consulta NUNCA sale del LLM ni del mensaje
  del usuario — lo resuelve vicki_web contra la sesión autenticada
  (lib/ventas/vickiVentasAcceso.ts) y llega ya fijo en el estado del grafo
  (state["ventas_vendedor_codigo"]). Si le diéramos al modelo una tool de SQL
  libre con un vendedor "sugerido", un mensaje bien armado ("ignorá el filtro
  anterior y mostrame todo") podría hacerle saltear la restricción. Por eso:

    - El LLM (o un parser determinístico, ver `_parsear_rango`) solo elige
      QUÉ RANGO DE FECHAS mostrar. Nunca elige ni ve el vendedorCodigo.
    - La query SQL la arma este módulo, con el vendedorCodigo ya fijo.
    - Los números de la respuesta salen del resultado de la query, formateados
      en Python — no se le pide al LLM que "redacte" las cifras (evita que
      invente o redondee mal un número que alguien va a usar para tomar
      decisiones).

  Un admin (isAdmin=True, ver ventas_vendedor_codigo=None) consulta TODA la
  empresa sin filtro — eso lo decide vicki_web, acá solo se respeta lo que
  llega.

TRES GATES, en este orden (ver `responder_ventas`):

  1. CLIENTE — si el mensaje nombra un cliente, se lo busca SIEMPRE dentro de
     la cartera del vendedor logueado (criterio de
     vicki_web/indicadores-api/cartera.py: zona ∪ facturado en los últimos 24
     meses, único lugar donde se define "de quién es un cliente"). Si el
     cliente existe pero no es de su cartera, se le dice que no le corresponde
     — nunca se devuelve un número.
  2. VENDEDOR MENCIONADO — si el mensaje nombra a otro vendedor (por nombre
     del maestro `Vendedores`, por un nombre de pila inconfundible o por
     "vendedor 797"), un no-admin recibe la negativa; un admin usa ese código
     como filtro. Se combina con el corte por línea: "ubaldo vendió bulones?"
     es vendedor + línea en la misma pregunta.
  3. LO PROPIO — sin cliente ni vendedor nombrado, sigue el camino de siempre:
     su facturación, filtrada por el código que llegó de la sesión.

  Todos los filtros se arman con ints (código de vendedor, código de cliente,
  Nivel1 del catálogo). El ÚNICO texto del usuario que entra al SQL es el
  término de búsqueda de cliente, saneado por `_literal_like`.
"""
import asyncio
import datetime as dt
import logging
import re

from app.config import config

log = logging.getLogger("ventas_tools")

_EPOCH = dt.date(1800, 12, 28)  # fechas de Magnus = días desde acá (ver mcp-magnus)

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _dias(fecha: dt.date) -> int:
    return (fecha - _EPOCH).days


def _primer_dia_mes_sig(anio: int, mes: int) -> dt.date:
    return dt.date(anio + 1, 1, 1) if mes == 12 else dt.date(anio, mes + 1, 1)


# ── Rangos de fecha ───────────────────────────────────────────────────────────
# Además del mes suelto se aceptan rangos explícitos: el reporte por vendedor
# se pide tanto por mes como por quincena o por un tramo cualquiera de días.
# Todo por regex (no LLM) por la razón del docstring de _parsear_rango.
_MESES_RE = "|".join(MESES)

# "del 1/8 al 15/8/2026", "01-08 a 15-08"
_PAT_RANGO_FECHAS = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\s*(?:al?|hasta(?:\s+el)?|y)\s*"
    r"(?:el\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", re.I)

# "del 5 de julio al 20 de agosto de 2026"
_PAT_RANGO_DIA_MES_DIA_MES = re.compile(
    rf"\b(\d{{1,2}})\s*(?:de\s*)?({_MESES_RE})\b(?:\s*(?:de[l]?\s*)?(20\d{{2}}))?\s*"
    rf"(?:al?|hasta(?:\s+el)?)\s*(?:el\s+)?(\d{{1,2}})\s*(?:de\s*)?({_MESES_RE})\b"
    rf"(?:\s*(?:de[l]?\s*)?(20\d{{2}}))?", re.I)

# "del 1 al 15 de agosto", "entre el 1 y el 15 de agosto de 2026"
_PAT_RANGO_DIAS_MES = re.compile(
    rf"\b(?:del|desde|entre)\s*(?:el\s+)?(\d{{1,2}})\s*(?:al?|hasta(?:\s+el)?|y)\s*"
    rf"(?:el\s+)?(\d{{1,2}})\s*de\s*({_MESES_RE})\b(?:\s*(?:de[l]?\s*)?(20\d{{2}}))?", re.I)

# "de enero a agosto", "entre marzo y junio de 2026"
_PAT_RANGO_MESES = re.compile(
    rf"\b(?:de|desde|entre)\s+({_MESES_RE})\s+(?:al?|hasta|y)\s+({_MESES_RE})\b"
    rf"(?:\s*(?:de[l]?\s*)?(20\d{{2}}))?", re.I)

# "el 15 de agosto" — un solo día
_PAT_DIA_MES = re.compile(
    rf"\b(\d{{1,2}})\s*de\s*({_MESES_RE})\b(?:\s*(?:de[l]?\s*)?(20\d{{2}}))?", re.I)

_PAT_ULTIMOS_DIAS = re.compile(r"\b[uú]ltim[oa]s?\s+(\d{1,3})\s+d[ií]as\b", re.I)
_PAT_ULTIMOS_MESES = re.compile(r"\b[uú]ltim[oa]s?\s+(\d{1,2})\s+meses\b", re.I)
_PAT_TRIMESTRE = re.compile(
    r"\b(primer|1er|1|segundo|2do|2|tercer|3er|3|cuarto|4to|4)\w*\s+trimestre\b", re.I)
_PAT_SEMESTRE = re.compile(r"\b(primer|1er|1|segundo|2do|2)\w*\s+semestre\b", re.I)
_ORDINALES = {"primer": 1, "1er": 1, "1": 1, "segundo": 2, "2do": 2, "2": 2,
              "tercer": 3, "3er": 3, "3": 3, "cuarto": 4, "4to": 4, "4": 4}


def _fecha(anio: int, mes: int, dia: int) -> dt.date | None:
    """None si la fecha no existe (31 de febrero, 15/13): el caller sigue
    probando los otros patrones en vez de romper la respuesta."""
    try:
        return dt.date(anio, mes, dia)
    except ValueError:
        return None


def _anio(txt: str | None, hoy: dt.date) -> int:
    """Año del texto ("2026" o "26"); si no se escribió, el año en curso."""
    if not txt:
        return hoy.year
    n = int(txt)
    return n if n >= 1000 else 2000 + n


def _rango_dias(d1: dt.date, d2: dt.date) -> tuple[dt.date, dt.date, str]:
    """Rango INCLUSIVO como lo dice el usuario → [desde, hasta_exclusivo) para
    el SQL. Se ordena solo, así "del 15 al 1" no devuelve vacío."""
    if d2 < d1:
        d1, d2 = d2, d1
    etiqueta = (f"el {d1.strftime('%d/%m/%Y')}" if d1 == d2 else
                f"el período {d1.strftime('%d/%m/%Y')} – {d2.strftime('%d/%m/%Y')}")
    return d1, d2 + dt.timedelta(days=1), etiqueta


def _parsear_rango(mensaje: str, hoy: dt.date | None = None) -> tuple[dt.date, dt.date, str]:
    """Determinístico a propósito (no vía LLM): un rango de fechas mal
    interpretado da una facturación mal calculada sin que nadie lo note. Mejor
    cubrir los casos comunes con certeza y caer a "este mes" que confiar en
    que el modelo entienda bien "el mes pasado" cada vez.

    Devuelve (desde, hasta_exclusivo, etiqueta) — hasta_exclusivo para poder
    filtrar con `>= desde AND < hasta` sin off-by-one.

    Formas reconocidas, en este orden (la primera que matchea gana):
      · "del 1/8 al 15/8/2026", "01-08 a 15-08"
      · "del 5 de julio al 20 de agosto"
      · "del 1 al 15 de agosto", "entre el 1 y el 15 de agosto de 2026"
      · "de enero a agosto", "entre marzo y junio de 2026"
      · "primer trimestre", "segundo semestre" (con año opcional)
      · "últimos 30 días", "últimos 3 meses"
      · "el 15 de agosto" (un día suelto)
      · mes suelto ("agosto", "agosto 2025"), hoy, ayer, esta semana,
        semana pasada, mes pasado, año / año NNNN
      · default: el mes en curso.
    """
    hoy = hoy or dt.date.today()
    m = (mensaje or "").lower()
    anio_match = re.search(r"\b(20\d{2})\b", m)
    anio = int(anio_match.group(1)) if anio_match else None

    # 1) dos fechas numéricas: "del 1/8 al 15/8/2026"
    g = _PAT_RANGO_FECHAS.search(m)
    if g:
        d1 = _fecha(_anio(g.group(3) or g.group(6), hoy), int(g.group(2)), int(g.group(1)))
        d2 = _fecha(_anio(g.group(6) or g.group(3), hoy), int(g.group(5)), int(g.group(4)))
        if d1 and d2:
            return _rango_dias(d1, d2)

    # 2) "del 5 de julio al 20 de agosto"
    g = _PAT_RANGO_DIA_MES_DIA_MES.search(m)
    if g:
        d1 = _fecha(_anio(g.group(3) or g.group(6), hoy), MESES[g.group(2)], int(g.group(1)))
        d2 = _fecha(_anio(g.group(6) or g.group(3), hoy), MESES[g.group(5)], int(g.group(4)))
        if d1 and d2:
            return _rango_dias(d1, d2)

    # 3) "del 1 al 15 de agosto"
    g = _PAT_RANGO_DIAS_MES.search(m)
    if g:
        y, mes = _anio(g.group(4), hoy), MESES[g.group(3)]
        d1, d2 = _fecha(y, mes, int(g.group(1))), _fecha(y, mes, int(g.group(2)))
        if d1 and d2:
            return _rango_dias(d1, d2)

    # 4) "de enero a agosto de 2026" — rango de meses completos
    g = _PAT_RANGO_MESES.search(m)
    if g:
        y = _anio(g.group(3), hoy)
        n1, n2 = MESES[g.group(1)], MESES[g.group(2)]
        etq1, etq2 = g.group(1), g.group(2)
        if n2 < n1:
            n1, n2, etq1, etq2 = n2, n1, etq2, etq1
        return dt.date(y, n1, 1), _primer_dia_mes_sig(y, n2), f"el período {etq1}–{etq2} {y}"

    # 5) trimestres y semestres
    g = _PAT_TRIMESTRE.search(m)
    if g:
        n = _ORDINALES[g.group(1).lower()]
        y = anio or hoy.year
        return dt.date(y, 3 * n - 2, 1), _primer_dia_mes_sig(y, 3 * n), f"el {n}º trimestre de {y}"
    g = _PAT_SEMESTRE.search(m)
    if g:
        n = _ORDINALES[g.group(1).lower()]
        y = anio or hoy.year
        return dt.date(y, 6 * n - 5, 1), _primer_dia_mes_sig(y, 6 * n), f"el {n}º semestre de {y}"

    # 6) ventanas móviles: "últimos 30 días" / "últimos 3 meses"
    g = _PAT_ULTIMOS_DIAS.search(m)
    if g:
        n = max(1, int(g.group(1)))
        return hoy - dt.timedelta(days=n - 1), hoy + dt.timedelta(days=1), f"los últimos {n} días"
    g = _PAT_ULTIMOS_MESES.search(m)
    if g:
        n = max(1, int(g.group(1)))
        y, mes = hoy.year, hoy.month - (n - 1)
        while mes <= 0:
            mes += 12
            y -= 1
        return dt.date(y, mes, 1), hoy + dt.timedelta(days=1), f"los últimos {n} meses"

    # 7) un día suelto: "el 15 de agosto"
    g = _PAT_DIA_MES.search(m)
    if g:
        d = _fecha(_anio(g.group(3), hoy), MESES[g.group(2)], int(g.group(1)))
        if d:
            return _rango_dias(d, d)

    # 8) un mes suelto — el que aparezca ANTES en el mensaje, no el primero del
    # calendario: "cuánto vendió Julio Blanco en agosto" ya no se lee como
    # julio sólo porque julio viene antes en el diccionario (el caller además
    # vuelve a parsear sin el nombre del vendedor, ver _sin_nombres_vendedor).
    mes_encontrado = min(
        (nombre for nombre in MESES if re.search(rf"\b{nombre}\b", m)),
        key=m.index, default=None,
    )
    if mes_encontrado:
        mes = MESES[mes_encontrado]
        y = anio or hoy.year
        return dt.date(y, mes, 1), _primer_dia_mes_sig(y, mes), f"{mes_encontrado} {y}"

    if "ayer" in m:
        ayer = hoy - dt.timedelta(days=1)
        return _rango_dias(ayer, ayer)

    if "hoy" in m:
        return hoy, hoy + dt.timedelta(days=1), f"hoy ({hoy.isoformat()})"

    if "semana pasada" in m or "semana anterior" in m:
        lunes = hoy - dt.timedelta(days=hoy.weekday() + 7)
        return _rango_dias(lunes, lunes + dt.timedelta(days=6))

    if "esta semana" in m:
        return _rango_dias(hoy - dt.timedelta(days=hoy.weekday()), hoy)

    if "mes pasado" in m or "mes anterior" in m:
        primero_mes_actual = hoy.replace(day=1)
        hasta = primero_mes_actual
        desde = dt.date(hoy.year - 1, 12, 1) if hoy.month == 1 else dt.date(hoy.year, hoy.month - 1, 1)
        return desde, hasta, "el mes pasado"

    if anio and "año" in m:
        return dt.date(anio, 1, 1), dt.date(anio + 1, 1, 1), f"el año {anio}"

    if "año" in m or "anual" in m:
        return dt.date(hoy.year, 1, 1), hoy + dt.timedelta(days=1), f"{hoy.year} (a la fecha)"

    if anio:
        return dt.date(anio, 1, 1), dt.date(anio + 1, 1, 1), f"el año {anio}"

    # default: mes en curso — el caso más común ("cómo vengo este mes")
    desde = hoy.replace(day=1)
    return desde, hoy + dt.timedelta(days=1), "este mes"



# ── LA OTRA SUB-EMPRESA: PRUEBA (2026-09-07) ─────────────────────────────────
# La venta de Ever Wear sale de DOS sub-empresas: MAGNUS (`Ven_*`) y PRUEBA
# (`PRU_Ven_*`). `PRU_` NO es una copia de prueba — son comprobantes reales,
# ~5% de la facturación (452.136.253 sobre 9.302.435.939 en ene-ago 2026), y
# el cubo del BI los suma. Leyendo sólo `Ven_*`, TODO lo que contestaba este
# módulo quedaba por debajo, y bastante más que un 5% en los vendedores con
# mucha bonificación: BECCARIA GERARDO 1-6/09/2026 daba 20.249.816 contra los
# 16.487.281 reales, porque PRUEBA le aporta +3,7M de facturas y -7,5M de NC.
# Ver `indicadores-api/subempresas.py`, que hace lo mismo del lado de la web.
#
# Los maestros de artículos y clientes son COMPARTIDOS (verificado sobre 2026:
# 0 renglones de PRUEBA sin `StkFer_*`, los 253 clientes en
# `MAGNUS_SITD.dbo.Clientes`), así que la gemela cambia SÓLO las tablas del
# circuito de ventas. `MAGNUS_SITD.dbo.*` no matchea ninguna de estas cadenas.
#
# Diferencia con la web: allá cada lista blanca de comprobantes hay que
# traducirla porque PRUEBA tiene su propio `PRU_Ven_CodCom`. Acá NO hace falta:
# el criterio de cabecera de este módulo (1,2,11 suman / 22,23,24,25 restan
# sobre Neto+NoGravado) es el que ya está conciliado contra el pivot del BI
# para PRUEBA — da 452.136.253 contra los 452.136.252 del cubo, $1 de redondeo
# (ver memoria 'facturacion-prueba-pivot'). Y las consultas por línea no
# filtran por comprobante: sólo miran el signo de `Ven_CodCom.DebitoCredito`.
# Si algún día se le agrega una lista blanca a este módulo, ahí SÍ hay que
# traducirla — los mismos números significan otra cosa en cada sub-empresa.
_TABLAS_PRUEBA = (
    "Ven_CompCabecera",
    "Ven_CompRenglon",
    "Ven_CodCom",
    "Ven_RenDebCre",
    "Ven_ConcDebCre",
    "Ven_Clientes",
)


def _a_prueba(sql: str) -> str:
    """La misma consulta, contra la sub-empresa PRUEBA. Ninguna de las tablas
    de `_TABLAS_PRUEBA` es prefijo de otra, así que el orden no importa.

    OJO: no pasarle una consulta que embeba `_sql_cartera()` — la cartera ya
    resuelve las dos sub-empresas por su cuenta y reescribirla la dejaría
    mirando sólo PRUEBA. Hoy ninguna de las que se unen acá la usa."""
    for tabla in _TABLAS_PRUEBA:
        sql = sql.replace(tabla, "PRU_" + tabla)
    return sql


def _dos_subempresas(sql: str, select: str, group_by: str, cola: str = "") -> str:
    """MAGNUS + PRUEBA en UNA sola consulta, re-agregando por fuera.

    `sql` es la consulta de MAGNUS ya agrupada y SIN ORDER BY ni TOP (una
    tabla derivada no los admite): el orden y el recorte van en `cola` y en
    `select`, sobre el resultado ya sumado — recortar adentro podría dejar
    afuera a alguien que es chico en MAGNUS y grande en PRUEBA.

    Re-agregar un agregado es correcto acá: SUM de SUM es SUM, y los COUNT se
    suman porque los universos son disjuntos (son tablas distintas, ningún
    NroMovVenta se repite entre las dos).

    Va en UNA consulta y no en dos como hace la web (`subempresas.filas_dos`)
    porque acá cada ida a Magnus es un round-trip HTTP contra el servicio de
    red: dos consultas serían el doble de latencia. Cada rama de la UNION
    conserva igual su propio plan (seek por fecha) y PRUEBA es chica."""
    return f"""
SELECT {select}
FROM (
{sql}
UNION ALL
{_a_prueba(sql)}
) u
GROUP BY {group_by}
{cola}""".strip()


def _sql_reporte_mensual(desde: dt.date, hasta_exclusivo: dt.date, vendedor_codigo: int | None) -> str:
    """SUM(Neto+NoGravado) con signo por CompCodigo (1,2,11 suman; 22,23,24,25
    restan — notas de crédito/devolución), agrupado por mes. Mismo criterio
    verificado contra Magnus en esta conversación (ver memoria
    'facturacion-prueba-pivot' para el porqué de Neto+NoGravado y no solo Neto).

    vendedor_codigo es SIEMPRE un int ya validado por el caller (o None para
    admin) — nunca un string armado con datos de fuera de este proceso, así
    que interpolarlo en el SQL es seguro (no hay forma de inyectar texto).
    """
    filtro_vendedor = f" AND Vendedor = {int(vendedor_codigo)}" if vendedor_codigo is not None else ""
    magnus = f"""
SELECT YEAR(DATEADD(DAY,FecMovim,'1800-12-28')) AS Anio,
  MONTH(DATEADD(DAY,FecMovim,'1800-12-28')) AS Mes,
  SUM(CASE WHEN CompCodigo IN (1,2,11) THEN Neto+NoGravado
           WHEN CompCodigo IN (22,23,24,25) THEN -(Neto+NoGravado)
           ELSE 0 END) AS Importe,
  COUNT(*) AS Comprobantes
FROM Ven_CompCabecera
WHERE FecMovim >= {_dias(desde)} AND FecMovim < {_dias(hasta_exclusivo)}{filtro_vendedor}
GROUP BY YEAR(DATEADD(DAY,FecMovim,'1800-12-28')), MONTH(DATEADD(DAY,FecMovim,'1800-12-28'))
""".strip()
    return _dos_subempresas(
        magnus,
        select="Anio, Mes, SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes",
        group_by="Anio, Mes",
        cola="ORDER BY 1,2",
    )


_NOMBRE_MES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

_PATRON_RANKING = re.compile(
    r"qu[ée] vendedor|cu[áa]l vendedor|qui[ée]n (vendi[óo]|factur[óo])|"
    r"mejor vendedor|ranking de vendedor|top de? vendedor|vendedores.{0,15}(m[áa]s|ranking)|"
    # "cuánto vendió cada vendedor en agosto" no matcheaba ninguno de los de
    # arriba: caía al reporte propio y a un admin (vendedor_codigo=None) le
    # contestaba la facturación de TODA la empresa, un número plausible que se
    # lee como si fuera el desglose pedido. Mismo modo de falla que el nombre
    # suelto (ver memoria 'ventas-chat-vendedor-nombre-suelto').
    r"cada vendedor|por vendedor(?:es)?\b|vendedor por vendedor|"
    r"todos? (?:los )?vendedores|de los vendedores|"
    r"cu[áa]nto (?:vendi[óo]|factur[óo]|vendieron|facturaron) cada|"
    r"cada uno de los vendedores|"
    r"(?:discrimin|desglos|abiert|apertur)\w*\s+(?:por|de|entre)\s+vendedor",
    re.I,
)

# "en total" / "sumados" pide UN número; "cada uno" / "discriminado" pide la
# lista. Se preguntan las dos cosas con las mismas palabras, así que el
# ranking siempre cierra con el total al pie y sólo el pedido explícito de
# total colapsa la respuesta a un número.
_PATRON_TOTAL = re.compile(
    r"\b(en total|el total|total(?:es)?|sumad[oa]s?|entre (?:los )?dos|"
    r"entre todos|juntos|consolidado|global)\b", re.I,
)
_PATRON_DISCRIMINADO = re.compile(
    r"cada vendedor|por vendedor|vendedor por vendedor|discrimin|desglos|"
    r"abiert[oa] por|apertura|uno por uno|cada uno|separad[oa]s?", re.I,
)
_PATRON_TODOS = re.compile(r"\b(todos?|todas?|cada|complet[oa]|lista(?:do)?)\b", re.I)


def _quiere_total(mensaje: str) -> bool:
    """True sólo si pidió el total Y no pidió el detalle: "el total de Ortiz y
    Beccaria" es un número, "el total de cada uno" son dos."""
    m = mensaje or ""
    return bool(_PATRON_TOTAL.search(m)) and not _PATRON_DISCRIMINADO.search(m)


def _quiere_todos(mensaje: str) -> bool:
    """"cada vendedor" / "todos los vendedores" → la lista completa, sin tope."""
    return bool(_PATRON_TODOS.search(mensaje or ""))


def _es_pedido_ranking(mensaje: str) -> bool:
    """Determinístico (no LLM) por la misma razón que _parsear_rango: si el
    pedido es "qué vendedor vendió más" hay que armar OTRA query (agrupada por
    vendedor, no por mes) y sobre todo aplicar OTRO gate de permisos — un
    no-admin no puede ver la facturación de sus compañeros, solo la propia."""
    return bool(_PATRON_RANKING.search(mensaje or ""))


def _sql_ranking_vendedores(
    desde: dt.date, hasta_exclusivo: dt.date, codigos: list[int] | None = None
) -> str:
    """Solo se llama para admins (ver responder_ventas) — sin filtro de
    vendedor, es justamente lo que se pide: comparar entre todos. Joinea
    contra el maestro `Vendedores` (MAGNUS_SITD.dbo, NO `Ped_Usu_Arma` — ver
    memoria 'magnus-codigos-vendedor') para el nombre.

    `codigos` (opcional) recorta a los vendedores que el usuario nombró:
    "cuánto vendieron Gómez y Pérez". Son ints del maestro, no texto del
    mensaje, así que interpolarlos es seguro."""
    filtro = f" AND c.Vendedor IN ({_in_codigos(codigos)})" if codigos else ""
    magnus = f"""
SELECT c.Vendedor AS VendedorCodigo, v.VendedorNombre,
  SUM(CASE WHEN c.CompCodigo IN (1,2,11) THEN c.Neto+c.NoGravado
           WHEN c.CompCodigo IN (22,23,24,25) THEN -(c.Neto+c.NoGravado)
           ELSE 0 END) AS Importe,
  COUNT(*) AS Comprobantes
FROM Ven_CompCabecera c
LEFT JOIN MAGNUS_SITD.dbo.Vendedores v ON v.VendedorCodigo = c.Vendedor
WHERE c.FecMovim >= {_dias(desde)} AND c.FecMovim < {_dias(hasta_exclusivo)}{filtro}
GROUP BY c.Vendedor, v.VendedorNombre
""".strip()
    # El maestro `Vendedores` es compartido, así que el nombre es el mismo en
    # las dos ramas y MAX() sólo lo arrastra.
    return _dos_subempresas(
        magnus,
        select=("VendedorCodigo, MAX(VendedorNombre) AS VendedorNombre, "
                "SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes"),
        group_by="VendedorCodigo",
        cola="ORDER BY Importe DESC",
    )


# ── Corte por línea de producto ───────────────────────────────────────────────
# "qué vendedor vendió más en bulones" = el mismo ranking pero acotado a una
# línea. Criterio idéntico al de /ventas/bulones (indicadores-api/bulones.py):
# la línea NO está en el comprobante sino en el artículo, vía
# StkFer_ArtParamet.Nivel1, y el nombre vive en Stk_Nivel1.Detalle (~82 filas).
# Allá la línea está fija en 'BULON%'; acá se resuelve la que nombre el usuario
# contra el catálogo, así sirve para cualquiera (mangueras, correas, etc.).
#
# OJO con la diferencia de criterio contra el ranking general: éste suma por
# RENGLÓN de artículo (Cantidad * PrecioVenta con signo por
# Ven_CodCom.DebitoCredito), no por cabecera, porque una línea solo existe a
# nivel artículo. Eso deja afuera las NC por concepto (bonificaciones, ajustes:
# no tienen artículo, viven en Ven_RenDebCre) — ~7% de la venta. Es correcto
# para un corte por línea (una bonificación no pertenece a ninguna), pero
# explica que el total por línea no cierre contra el ranking general.
_TTL_LINEAS = 3600
_lineas_cache: dict = {"t": 0.0, "datos": []}

_STOP_LINEA = {
    "linea", "lineas", "vendedor", "vendedores", "vendio", "vendieron",
    "venta", "ventas", "mejor", "mejores", "ranking", "factura", "facturo",
    "facturacion", "empresa", "cliente", "clientes", "cuanto", "cuanta",
    "quien", "cual", "cuales", "este", "esta", "mes", "meses", "anio", "año",
    "total", "totales", "mucho", "mas", "menos", "pasado", "actual", "curso",
    "producto", "productos", "articulo", "articulos", "rubro", "sector",
    "vendido", "vendidos", "vendida", "vendidas", "vendimos",
}


def _normalizar(s: str) -> str:
    """Minúsculas y sin acentos — el catálogo de Magnus escribe 'BULONERÍA'
    con tilde (por eso /ventas/bulones matchea con LIKE 'BULON%'), y el usuario
    escribe como se le ocurre."""
    s = (s or "").lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
                 ("ü", "u"), ("ñ", "n")):
        s = s.replace(a, b)
    return s


def _clave_linea(detalle: str) -> str:
    """Primera palabra significativa del nombre de la línea — la que el usuario
    va a nombrar. 'CORREAS EVER WEAR' → 'correas'; 'LÍNEA BUCO' → 'buco'
    (salteando la palabra genérica)."""
    for p in _normalizar(detalle).replace(".", " ").replace("/", " ").split():
        if len(p) >= 4 and p not in _STOP_LINEA:
            return p
    return ""


def _prefijo_comun(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def _catalogo_lineas() -> list[tuple[int, str]]:
    """(Nivel1, Detalle) de Stk_Nivel1, cacheado 1h. Son ~82 filas y no cambian
    casi nunca; sin cache pagaríamos una consulta extra por cada mensaje."""
    import time

    if _lineas_cache["datos"] and time.time() - _lineas_cache["t"] < _TTL_LINEAS:
        return _lineas_cache["datos"]
    tsv = await _ejecutar_sql(
        "SELECT Nivel1, LTRIM(RTRIM(Detalle)) AS Detalle FROM Stk_Nivel1 ORDER BY 2"
    )
    datos = []
    for l in (tsv or "").splitlines()[1:]:
        if l.startswith("("):
            break
        partes = l.split("\t")
        if len(partes) < 2:
            continue
        try:
            datos.append((int(partes[0]), partes[1].strip()))
        except ValueError:
            continue
    _lineas_cache.update({"t": time.time(), "datos": datos})
    return datos


def _detectar_lineas(mensaje: str, catalogo: list[tuple[int, str]]) -> list[tuple[str, list[int]]]:
    """Líneas nombradas en el mensaje → [(etiqueta, [Nivel1, ...]), ...], en
    orden de aparición. Determinístico (no LLM), misma razón que
    _parsear_rango: si el modelo elige mal la línea, el número sale mal y nadie
    lo nota.

    Devuelve un GRUPO de códigos por palabra, no uno solo: el catálogo tiene 7
    líneas de correas, 5 de filtros y 2 de mangueras (MANGUERAS y MANGUERAS
    NACIONALES). Quien pregunta "cuánto vendimos de correas" las quiere todas
    — es el mismo criterio de /ventas/bulones, que agrupa con LIKE 'BULON%' en
    vez de fijar un código.

    Match por prefijo común contra la primera palabra significativa del nombre,
    así 'bulones' encuentra BULONERÍA sin mantener a mano una tabla de
    sinónimos. Umbral: 5 caracteres (o la palabra entera si es más corta) y a
    lo sumo 4 de diferencia con ella."""
    tokens = [t for t in re.findall(r"[a-z]{4,}", _normalizar(mensaje)) if t not in _STOP_LINEA]
    if not tokens:
        return []
    claves: dict[str, list[tuple[int, str]]] = {}
    for nivel1, detalle in catalogo:
        k = _clave_linea(detalle)
        if k:
            claves.setdefault(k, []).append((nivel1, detalle))

    salida: list[tuple[str, list[int]]] = []
    usadas: set[str] = set()
    for tok in tokens:
        mejor = None
        for k in claves:
            p = _prefijo_comun(tok, k)
            if p >= min(5, len(k)) and p >= len(k) - 4:
                if mejor is None or p > _prefijo_comun(tok, mejor):
                    mejor = k
        if not mejor or mejor in usadas:
            continue
        usadas.add(mejor)
        grupo = claves[mejor]
        etiqueta = grupo[0][1] if len(grupo) == 1 else f"{mejor.upper()} ({len(grupo)} líneas)"
        salida.append((etiqueta, [n for n, _ in grupo]))
    return salida[:3]


_SQL_JOIN_LINEA = """
FROM Ven_CompCabecera vc
JOIN Ven_CodCom cc        ON cc.CompCodigo    = vc.CompCodigo
JOIN Ven_CompRenglon r    ON r.NroMovVenta    = vc.NroMovVenta
JOIN StkFer_Articulos s   ON s.CodArticulo    = r.CodArticu
JOIN StkFer_ArtParamet ap ON ap.ArticuloPatron = s.ArticuloPatron
"""

# Signo por Ven_CodCom.DebitoCredito (1 débito suma, 2 crédito resta) — mismo
# _MONTO que indicadores-api/bulones.py, para que los dos den lo mismo.
_MONTO_RENGLON = (
    "CASE cc.DebitoCredito WHEN 1 THEN (r.Cantidad * r.PrecioVenta) "
    "ELSE (r.Cantidad * r.PrecioVenta) * -1 END"
)


def _in_codigos(codigos: list[int]) -> str:
    """Los códigos de vendedor salen del maestro de Magnus (los resolvió
    `_detectar_vendedores_mencionados`), no del mensaje: son ints, no hay texto
    del usuario en el SQL."""
    return ",".join(str(int(c)) for c in codigos)


def _in_niveles(niveles: list[int]) -> str:
    """Los códigos salen del catálogo de Magnus, no del mensaje: son ints del
    propio catálogo, así que no hay texto del usuario en el SQL."""
    return ",".join(str(int(n)) for n in niveles)


def _sql_ranking_vendedores_linea(
    desde: dt.date, hasta_exclusivo: dt.date, niveles: list[int],
    codigos: list[int] | None = None,
) -> str:
    """Ranking entre vendedores acotado a una línea (solo admin, igual que
    _sql_ranking_vendedores). `codigos` recorta a los nombrados."""
    filtro = f" AND vc.Vendedor IN ({_in_codigos(codigos)})" if codigos else ""
    magnus = f"""
SELECT vc.Vendedor AS VendedorCodigo, v.VendedorNombre,
  SUM({_MONTO_RENGLON}) AS Importe,
  COUNT(DISTINCT vc.NroMovVenta) AS Comprobantes
{_SQL_JOIN_LINEA.strip()}
LEFT JOIN MAGNUS_SITD.dbo.Vendedores v ON v.VendedorCodigo = vc.Vendedor
WHERE vc.FecMovim >= {_dias(desde)} AND vc.FecMovim < {_dias(hasta_exclusivo)}
  AND ap.Nivel1 IN ({_in_niveles(niveles)}){filtro}
GROUP BY vc.Vendedor, v.VendedorNombre
""".strip()
    return _dos_subempresas(
        magnus,
        select=("VendedorCodigo, MAX(VendedorNombre) AS VendedorNombre, "
                "SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes"),
        group_by="VendedorCodigo",
        cola="ORDER BY Importe DESC",
    )


def _sql_total_linea(
    desde: dt.date, hasta_exclusivo: dt.date, niveles: list[int], vendedor_codigo: int | None
) -> str:
    """Total de UNA línea por mes — la versión de _sql_reporte_mensual acotada
    a la línea, para "cómo vengo en bulones". Con vendedor_codigo filtra por
    quién facturó el comprobante (mismo criterio que el resto de este módulo;
    /ventas/bulones en cambio corta por cartera, así que pueden no coincidir)."""
    filtro = f" AND vc.Vendedor = {int(vendedor_codigo)}" if vendedor_codigo is not None else ""
    magnus = f"""
SELECT YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Anio,
  MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Mes,
  SUM({_MONTO_RENGLON}) AS Importe,
  COUNT(DISTINCT vc.NroMovVenta) AS Comprobantes
{_SQL_JOIN_LINEA.strip()}
WHERE vc.FecMovim >= {_dias(desde)} AND vc.FecMovim < {_dias(hasta_exclusivo)}
  AND ap.Nivel1 IN ({_in_niveles(niveles)}){filtro}
GROUP BY YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')), MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28'))
""".strip()
    return _dos_subempresas(
        magnus,
        select="Anio, Mes, SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes",
        group_by="Anio, Mes",
        cola="ORDER BY 1,2",
    )


def _parsear_tsv_ranking(tsv: str) -> list[dict]:
    lineas = [l for l in (tsv or "").splitlines() if l.strip()]
    filas = []
    for l in lineas[1:]:
        if l.startswith("("):
            break
        partes = l.split("\t")
        if len(partes) < 4:
            continue
        try:
            filas.append({
                "codigo": int(partes[0]),
                "nombre": partes[1].strip() or f"(vendedor {partes[0]})",
                "importe": float(partes[2] or 0),
                "comprobantes": int(partes[3] or 0),
            })
        except ValueError:
            log.warning(f"fila TSV inesperada (ranking): {l!r}")
    return filas


def _parsear_tsv(tsv: str) -> list[dict]:
    """El tool `query` de mcp-magnus devuelve TSV + una línea final "(N filas)".
    Sin eso no hay forma de distinguir "no facturó nada" de "la query falló"."""
    lineas = [l for l in (tsv or "").splitlines() if l.strip()]
    if len(lineas) < 1:
        return []
    filas = []
    for l in lineas[1:]:
        if l.startswith("("):  # pie "(N filas)"
            break
        partes = l.split("\t")
        if len(partes) < 4:
            continue
        try:
            filas.append({
                "anio": int(partes[0]), "mes": int(partes[1]),
                "importe": float(partes[2] or 0), "comprobantes": int(partes[3] or 0),
            })
        except ValueError:
            log.warning(f"fila TSV inesperada de magnus: {l!r}")
    return filas


_query_tool = None


async def _get_query_tool():
    """Tool `query` de mcp-magnus, cacheada (evita reabrir sesión MCP en cada
    mensaje). Se conecta por streamable-http al servicio de red — ver
    vicki/mcp/mcp-magnus/README.md "Modo servicio de red": ese proceso corre
    en una PC/VM Windows con la impersonación AD ya resuelta; acá solo se
    habla el protocolo MCP por HTTP."""
    global _query_tool
    if _query_tool is not None:
        return _query_tool
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        "magnus": {"url": config.MAGNUS_MCP_URL, "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    tool = next((t for t in tools if t.name == "query"), None)
    if tool is None:
        raise RuntimeError("mcp-magnus no expone el tool 'query' (¿versión vieja del server?)")
    _query_tool = tool
    return tool


class _FalloMagnus(Exception):
    """Cualquier problema hablando con mcp-magnus (timeout, red caída, el
    server no responde). Un único tipo de error para que responder_ventas no
    tenga que repetir el mismo try/except en cada report."""


def _ejecutar_sql_http(sql: str) -> str:
    """POST /sql al servicio de magnus (endpoint JSON plano, sin MCP).

    Es el camino preferido desde 2026-09-06: el transporte streamable-http del
    SDK MCP no anda en srv-active (acepta el TCP y nunca responde, también
    contra 127.0.0.1 — no es red), así que el mismo server.py expone las
    consultas como JSON usando http.server de la stdlib. Sincrónico a
    propósito: lo llama _ejecutar_sql con asyncio.to_thread, y `requests` ya
    era dependencia. Ver mcp-magnus/README.md "Modo endpoint HTTP"."""
    import requests

    headers = {"Content-Type": "application/json"}
    if config.MAGNUS_API_TOKEN:
        headers["X-Api-Token"] = config.MAGNUS_API_TOKEN
    r = requests.post(
        config.MAGNUS_SQL_URL,
        json={"sql": sql, "db": "EVERWEAR", "max_rows": 60},
        headers=headers,
        timeout=config.MAGNUS_MCP_TIMEOUT,
    )
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"respuesta no-JSON de magnus (HTTP {r.status_code})")
    if not data.get("ok"):
        # El server distingue 400 (SQL rechazado / pedido mal armado) de 500
        # (falla contra el SQL Server). Los dos son bug nuestro, no del usuario:
        # que quede en el log con el detalle.
        raise RuntimeError(f"magnus HTTP {r.status_code}: {data.get('error')}")
    return data["tsv"]


async def _ejecutar_sql(sql: str) -> str:
    """Corre `sql` contra magnus con timeout duro (config.MAGNUS_MCP_TIMEOUT).

    Sin este timeout, si el servicio de red no contesta (firewall cerrado,
    la PC/VM Windows apagada, IP mal puesta) el pedido queda colgado hasta que
    lo aborta el FRONT — 60s en vicki_web — dejando al usuario 60 segundos
    esperando y sin un mensaje claro de qué pasó. Acá cortamos antes y
    devolvemos un error entendible."""
    try:
        async def _llamar():
            if config.MAGNUS_SQL_URL:
                return await asyncio.to_thread(_ejecutar_sql_http, sql)
            tool = await _get_query_tool()
            return await tool.ainvoke({"sql": sql, "db": "EVERWEAR", "max_rows": 60})

        return await asyncio.wait_for(_llamar(), timeout=config.MAGNUS_MCP_TIMEOUT)
    except asyncio.TimeoutError as e:
        destino = config.MAGNUS_SQL_URL or config.MAGNUS_MCP_URL
        log.error(f"magnus no respondió en {config.MAGNUS_MCP_TIMEOUT}s (destino={destino!r})")
        raise _FalloMagnus("timeout") from e
    except Exception as e:
        log.exception("consulta a magnus falló")
        raise _FalloMagnus(str(e)) from e


def _monto(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


def _comps(n: int) -> str:
    """"1 comprobante", no "1 comprobantes" — el corte por línea da muchos
    totales de un solo comprobante y quedaba mal escrito."""
    return f"{n} comprobante" if n == 1 else f"{n} comprobantes"


def _formatear_ranking(
    filas: list[dict], titulo: str, tope: int | None, sustantivo: str = "vendedores"
) -> str:
    """Formateo en Python, no vía LLM (ver docstring del módulo): estos números
    los usa alguien para decidir, no pueden salir redondeados de cualquier
    manera.

    `tope=None` lista a TODOS — es lo que corresponde cuando se pidió "cada
    vendedor" o se nombró a un grupo. El total del período va siempre al pie:
    la misma pregunta se hace de las dos formas ("discriminado" y "el total") y
    así una sola respuesta sirve para las dos."""
    if not filas:
        return f"{titulo}: no encontré facturación."
    filas = [f for f in filas if f["importe"] != 0]
    filas.sort(key=lambda f: f["importe"], reverse=True)
    if not filas:
        return f"{titulo}: no encontré facturación."
    mostradas = filas if tope is None else filas[:tope]
    out = [f"{titulo}:", ""]
    for i, f in enumerate(mostradas, start=1):
        out.append(
            f"{i}. {f['nombre']} (cód. {f['codigo']}): {_monto(f['importe'])} "
            f"({f['comprobantes']} comp.)"
        )
    if len(filas) > len(mostradas):
        out.append(f"… y {len(filas) - len(mostradas)} {sustantivo} más.")
    out.append("")
    out.append(
        f"Total de los {len(filas)} {sustantivo}: "
        f"{_monto(sum(f['importe'] for f in filas))} "
        f"({_comps(sum(f['comprobantes'] for f in filas))})."
    )
    return "\n".join(out)



# ── Gate 2: ¿el mensaje nombra a OTRO vendedor? ───────────────────────────────
# "cuánto vendió Juan Blanco en agosto" no es un ranking (no matchea
# _PATRON_RANKING) y hasta ahora caía al reporte propio: le devolvía SUS
# números diciendo "tu cartera". No filtraba nada, pero el usuario podía leer
# ese total como si fuera el de Juan. Ahora se detecta el nombre y se corta.
#
# Determinístico contra el maestro `Vendedores` (el mismo de cartera.py, NO
# `Ped_Usu_Arma` — ver memoria 'magnus-codigos-vendedor'). No se le pregunta al
# LLM: si el modelo "no ve" el nombre, el gate no se aplica y se filtra de más;
# acá el peor caso tiene que ser negar, no mostrar.
_TTL_VENDEDORES = 3600
_vendedores_cache: dict = {"t": 0.0, "datos": []}

# Palabras del maestro que no identifican a una persona (canales, zonas y
# agrupadores: MOSTRADORES, ZONA CBA, VIAJANTE ZONA ROSARIO…). Mismo criterio
# que _es_persona() en indicadores-api/ventas.py. Se sacan del match para que
# "zona" o "vendedor" en el mensaje no dispare el gate contra cualquiera.
_STOP_VENDEDOR = {
    "vendedor", "vendedora", "vendedores", "zona", "zonas", "viajante",
    "mostrador", "mostradores", "sin", "cero", "baja", "gerencia", "coop",
    "cooperativa", "comercio", "exterior", "empresa", "mercado", "libre",
    "atendidos", "por", "los", "las", "del", "san", "santa",
    # agrupadores del maestro que NO son personas: "VENDEDOR CLIENTES
    # INDUSTRIAS", "VENDEDOR AGROACTIVA 2026", "MANFREY COOP.DE TAMB.LTDA."
    "cliente", "clientes", "industrias", "agro", "agroact", "agroactiva",
    "expo", "comercial", "ltda", "tamb",
}

# "vendedor 797", "vendedor código 800", "los vendedores 797 y 800"
_PATRON_VENDEDORES_CODIGOS = re.compile(
    r"\bvendedor(?:a|es)?\s*(?:c[oó]digos?|cod\.?|nros?\.?|n[°º]|n[uú]meros?)?\s*"
    r"(\d{2,6}(?:\s*(?:,|y|/|-|\s)\s*\d{2,6})*)", re.I
)

# Marcas de razón social: si aparecen, el mensaje está hablando de una EMPRESA,
# no de una persona, y una sola parte del nombre no alcanza para disparar el
# gate (evita que "ferretería BLANCO" se lea como el vendedor Blanco, 797).
_PATRON_RAZON_SOCIAL = re.compile(
    r"\b(s\.?r\.?l|srl|s\.?a\.?s|sas|s\.?a|sa|ltda|hnos|hermanos|coop|"
    r"cooperativa|ferreteria|distribuidora|agropecuaria|agricola|talleres|"
    r"transportes|establecimiento|firma)\b",
    re.I,
)


async def _catalogo_vendedores() -> list[tuple[int, str]]:
    """(VendedorCodigo, VendedorNombre) del maestro, cacheado 1h. Son ~40
    filas y cambian con altas/bajas, nunca dentro de una charla."""
    import time

    if _vendedores_cache["datos"] and time.time() - _vendedores_cache["t"] < _TTL_VENDEDORES:
        return _vendedores_cache["datos"]
    tsv = await _ejecutar_sql(
        "SELECT VendedorCodigo, LTRIM(RTRIM(VendedorNombre)) AS nombre "
        "FROM MAGNUS_SITD.dbo.Vendedores ORDER BY 1"
    )
    datos = []
    for l in (tsv or "").splitlines()[1:]:
        if l.startswith("("):
            break
        partes = l.split("\t")
        if len(partes) < 2:
            continue
        try:
            datos.append((int(partes[0]), partes[1].strip()))
        except ValueError:
            continue
    _vendedores_cache.update({"t": time.time(), "datos": datos})
    return datos


def _partes_nombre(nombre: str) -> list[str]:
    return [
        p for p in re.findall(r"[a-z]{3,}", _normalizar(nombre))
        if p not in _STOP_VENDEDOR
    ]


def _partes_unicas(catalogo: list[tuple[int, str]]) -> set[str]:
    """Partes de nombre que aparecen en UN SOLO vendedor del maestro.

    Es lo que permite reconocer a alguien nombrado con una sola palabra sin
    mantener una lista de apodos: "ubaldo" identifica al 799 porque no hay otro
    UBALDO; "romero" no identifica a nadie porque hay dos (677 y 796) y ahí sí
    hacen falta dos partes."""
    cuenta: dict[str, int] = {}
    for _, nombre in catalogo:
        for p in set(_partes_nombre(nombre)):
            cuenta[p] = cuenta.get(p, 0) + 1
    return {p for p, n in cuenta.items() if n == 1}


def _detectar_vendedores_mencionados(
    mensaje: str, catalogo: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    """TODOS los vendedores nombrados en el mensaje, sin repetidos, el de match
    más fuerte primero.

    Formas de nombrarlos:
      · explícita — "vendedor 797", "los vendedores 797 y 800".
      · dos partes del nombre del maestro — "BLANCO JULIO" ← "cuánto vendió
        Julio Blanco".
      · UNA parte inconfundible — "ubaldo vendió bulones?". Alcanza una sola
        palabra si es de 5+ letras y aparece en un único vendedor del maestro
        (`_partes_unicas`). Antes se exigían siempre dos y por eso "ubaldo
        vendió bulonería?" no aplicaba ningún filtro: caía al reporte propio y
        un admin recibía el total de TODA la empresa como si fuera de él.
      · UNA parte + la palabra "vendedor", para los nombres de una sola palabra.

    Varios a la vez ("cuánto vendieron Gómez y Pérez en agosto") salen como
    lista y el caller los compara en UNA consulta con `IN`, en vez de contestar
    por uno solo y hacer pasar ese número por el de los dos.

    El riesgo de la parte suelta es al revés (negar de más, no mostrar de más):
    un apellido dentro de una razón social podría leerse como el vendedor. Se
    acota de dos maneras — la unicidad en el maestro, y `_PATRON_RAZON_SOCIAL`,
    que si detecta que se habla de una empresa vuelve a exigir dos partes. Y el
    gate de cliente corre ANTES que éste, así que "cliente Ferretería Blanco"
    ni llega acá.
    """
    por_codigo: dict[int, str] = {}
    for g in _PATRON_VENDEDORES_CODIGOS.finditer(mensaje or ""):
        for num in re.findall(r"\d{2,6}", g.group(1)):
            cod = int(num)
            por_codigo[cod] = next((n for c, n in catalogo if c == cod), f"vendedor {cod}")
    if por_codigo:
        return sorted(por_codigo.items())

    m = _normalizar(mensaje)
    tokens = set(re.findall(r"[a-z]{3,}", m))
    if not tokens:
        return []
    dice_vendedor = bool(re.search(r"\bvendedor(a|es)?\b", m))
    habla_de_empresa = bool(_PATRON_RAZON_SOCIAL.search(m))
    unicas = _partes_unicas(catalogo)

    candidatos: list[tuple[set, int, str]] = []
    for codigo, nombre in catalogo:
        partes = _partes_nombre(nombre)
        if not partes:
            continue
        encontradas = {p for p in partes if p in tokens}
        if not encontradas:
            continue
        if len(encontradas) >= 2:
            alcanza = True
        else:
            p = next(iter(encontradas))
            # `len(partes) >= 2` = el registro parece una PERSONA. Sin esto,
            # "VIAJANTE ZONA ROSARIO" (791) queda reducido a "rosario" y
            # cualquiera que escriba "mi zona de rosario" se comería la
            # negativa del gate. Los agrupadores de una sola palabra solo
            # matchean si además se dice "vendedor".
            alcanza = (
                (len(partes) >= 2 and len(p) >= 5 and p in unicas
                 and (dice_vendedor or not habla_de_empresa))
                or (dice_vendedor and len(partes) == 1)
            )
        if alcanza:
            candidatos.append((encontradas, codigo, nombre))

    # Un match que usa un subconjunto ESTRICTO de las palabras de otro es el
    # mismo pedido visto peor: "Julio Blanco" identifica al 797 con dos partes,
    # y un "BLANCO OTRO" que sólo matchea "blanco" sobra. Antes esto se
    # resolvía quedándose con el de más partes; ahora hay que conservar a los
    # demás, que pueden ser otra persona realmente nombrada.
    salida = [
        (len(tks), cod, nom) for tks, cod, nom in candidatos
        if not any(tks < otras for otras, _, _ in candidatos)
    ]
    salida.sort(key=lambda r: (-r[0], r[1]))
    return [(cod, nom) for _, cod, nom in salida]


def _detectar_vendedor_mencionado(
    mensaje: str, catalogo: list[tuple[int, str]]
) -> tuple[int, str] | None:
    """El vendedor nombrado (el match más fuerte) o None — la forma de un solo
    resultado de `_detectar_vendedores_mencionados`."""
    encontrados = _detectar_vendedores_mencionados(mensaje, catalogo)
    return encontrados[0] if encontrados else None


def _sin_nombres_vendedor(mensaje: str, vendedores: list[tuple[int, str]]) -> str:
    """El mensaje sin las palabras del nombre del vendedor que ADEMÁS son un
    mes ("Julio Blanco"), y sólo si queda otro mes en la frase.

    Sin esto, "cuánto vendió Julio Blanco en agosto" se leía como julio: el
    parser de rango ve "julio" antes que "agosto" y devuelve el mes equivocado
    con el vendedor correcto — el peor tipo de error, un número plausible."""
    del_nombre = {p for _, nombre in vendedores for p in _partes_nombre(nombre)} & set(MESES)
    if not del_nombre:
        return mensaje or ""
    m = _normalizar(mensaje)
    presentes = {n for n in MESES if re.search(rf"\b{n}\b", m)}
    if not (presentes - del_nombre):
        return mensaje or ""  # el único mes nombrado es el del nombre: se respeta
    salida = mensaje or ""
    for mes in del_nombre:
        salida = re.sub(rf"\b{mes}\b", " ", salida, flags=re.I)
    return salida



# ── Gate 1: clientes, siempre dentro de la cartera ────────────────────────────
# El criterio de "qué clientes son de un vendedor" NO se reinventa acá: es el
# de vicki_web/indicadores-api/cartera.py — zona (Clientes.Clasif_VendZona →
# Vendedor_Zona → Vendedores) UNIÓN historial (facturado por ese vendedor en
# los últimos CARTERA_MESES meses). Hacen falta los dos: hay vendedores activos
# sin zona cargada (Julio Blanco, 797) que con criterio de zona no verían
# ningún cliente, y clientes recién asignados que todavía no compraron.
#
# Si algún día cambia el criterio allá, cambiarlo TAMBIÉN acá: son dos procesos
# distintos (Next.js/pyodbc vs. este) y no comparten módulo.
CARTERA_MESES = 24

_DIA_CORTE_CARTERA = (
    f"DATEDIFF(day, '1800-12-28', DATEADD(month, -{CARTERA_MESES}, GETDATE()))"
)


def _sql_cartera(vendedor_codigo: int) -> str:
    """Subconsulta con los CodCliente de la cartera de UN vendedor.

    El historial mira LAS DOS SUB-EMPRESAS, igual que `cartera.py` del lado
    de la web: un cliente al que el vendedor sólo le facturó por PRUEBA es
    igual de suyo, y sin esta rama el gate lo trataba como cliente ajeno y le
    negaba sus propios números.

    Esta consulta NO se pasa nunca por `_a_prueba()` — ya resuelve las dos
    sub-empresas y reescribirla la dejaría mirando sólo PRUEBA."""
    v = int(vendedor_codigo)
    return f"""
SELECT c2.CodCliente
FROM MAGNUS_SITD.dbo.Clientes c2
JOIN MAGNUS_SITD.dbo.Vendedor_Zona vz ON vz.Clasif_VendZona = c2.Clasif_VendZona
JOIN MAGNUS_SITD.dbo.Vendedores v
  ON LTRIM(RTRIM(v.VendedorNombre)) = LTRIM(RTRIM(vz.Vendedor))
WHERE v.VendedorCodigo = {v}
UNION
SELECT DISTINCT vch.CodCliente
FROM Ven_CompCabecera vch
WHERE vch.vendedor = {v} AND vch.FecMovim >= {_DIA_CORTE_CARTERA}
UNION
SELECT DISTINCT vcp.CodCliente
FROM PRU_Ven_CompCabecera vcp
WHERE vcp.vendedor = {v} AND vcp.FecMovim >= {_DIA_CORTE_CARTERA}
""".strip()


_PATRON_CLIENTE = re.compile(r"\bclientes?\b", re.I)
_PATRON_CLIENTE_CODIGO = re.compile(
    r"\bclientes?\s*(?:nro\.?|n[°º]|numero|codigo|cod\.?)?\s*(\d{2,8})\b", re.I
)
_PATRON_RANKING_CLIENTES = re.compile(
    r"qu[ée] cliente|cu[áa]l cliente|mejor(es)? cliente|ranking de cliente|"
    r"top de? cliente|clientes.{0,15}(m[áa]s|ranking)|mis clientes",
    re.I,
)

# Relleno de una pregunta hablada: no aportan al nombre del cliente.
_STOP_CLIENTE = _STOP_LINEA | {
    "vendi", "vendio", "vendieron", "vendimos", "compro", "compra", "compras",
    "compraron", "facture", "facturamos", "facturado", "cuanto", "cuanta",
    "cuantos", "muestrame", "mostrame", "dame", "decime", "traeme", "quiero",
    "saber", "ver", "por", "para", "con", "los", "las", "del", "una", "uno",
    "que", "mis", "sus", "tus", "nos", "ese", "esa", "esos", "esas", "hoy",
    "ultimo", "ultima", "ultimos", "periodo", "desde", "hasta", "entre",
}


def _literal_like(s: str, max_len: int = 40) -> str:
    """Único punto donde texto del usuario entra al SQL (el término de
    búsqueda del cliente).

    Lista blanca, no lista negra: sobreviven letras, dígitos y espacio, todo
    lo demás pasa a ser un espacio. Sin comilla no se puede cerrar el literal;
    sin guion ni barra no se puede abrir un comentario (`--`, `/*`). Encima el
    server de magnus solo acepta SELECT/WITH (mcp-magnus/server.py `_check`).

    OJO: la comilla se DESCARTA, no se escapa — "D'Agostino" busca como
    "D Agostino". Con los LIKE en AND de `_where_like_cliente` eso igual
    matchea, y el tokenizador de `_detectar_cliente_pedido` ya parte por la
    comilla antes de llegar acá, así que en la práctica no cambia nada. El
    .replace() de abajo queda como segunda barrera por si algún día se
    ensancha la lista blanca."""
    limpio = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", " ", s or "")
    limpio = re.sub(r"\s+", " ", limpio).strip()[:max_len]
    return limpio.replace("'", "''")


def _detectar_cliente_pedido(mensaje: str) -> list[str] | None:
    """Tokens con los que buscar al cliente nombrado, o None si el mensaje no
    habla de un cliente puntual.

    Determinístico igual que el resto del módulo, pero acá la exactitud no es
    crítica para la seguridad: cualquier término que salga de esto se busca
    SIEMPRE dentro de la cartera (`_buscar_cliente_en_cartera`), así que un
    término mal extraído puede no encontrar al cliente — nunca encontrar uno
    ajeno."""
    if not _PATRON_CLIENTE.search(mensaje or ""):
        return None
    if _PATRON_RANKING_CLIENTES.search(mensaje or ""):
        return None  # "mis mejores clientes" → ranking, no un cliente puntual

    # 1) código explícito: "cliente 12345"
    cod = _PATRON_CLIENTE_CODIGO.search(mensaje)
    if cod:
        return [cod.group(1)]

    # 2) nombre entrecomillado: cliente "Ferretería del Centro"
    entre_comillas = re.search(r"[\"“”']([^\"“”']{2,60})[\"“”']", mensaje)
    cola = entre_comillas.group(1) if entre_comillas else re.split(
        r"\bclientes?\b", mensaje, maxsplit=1, flags=re.I
    )[-1]

    # 3) las palabras significativas que siguen a "cliente". Se buscan como
    # LIKE separados y en AND, no como una frase: "Ferretería del Centro" y
    # "FERRETERIA CENTRO SRL" tienen que matchear igual.
    tokens = []
    for t in re.findall(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ&]{3,}", cola):
        if _normalizar(t) in _STOP_CLIENTE or _normalizar(t) in MESES:
            continue
        tokens.append(t)
        if len(tokens) == 3:
            break
    return tokens or None


def _where_like_cliente(tokens: list[str]) -> str:
    """Cada token, un LIKE en AND. Si el token es todo dígitos también se
    prueba contra el código, que es como la gente nombra a un cliente
    ("el 4521")."""
    condiciones = []
    for t in tokens:
        lit = _literal_like(t)
        if not lit:
            continue
        if lit.isdigit():
            condiciones.append(
                f"(c.Cliente_Nombre LIKE '%{lit}%' "
                f"OR CAST(c.CodCliente AS varchar(20)) = '{lit}')"
            )
        else:
            condiciones.append(f"c.Cliente_Nombre LIKE '%{lit}%'")
    return " AND ".join(condiciones)


def _sql_buscar_cliente(tokens: list[str], vendedor_codigo: int | None) -> str:
    """Clientes que matchean, ACOTADOS a la cartera si hay vendedor (un admin
    va sin filtro). El JOIN contra la cartera va en el mismo SELECT: nunca se
    trae un cliente ajeno a memoria para descartarlo después."""
    where = _where_like_cliente(tokens)
    join = (
        f"JOIN ({_sql_cartera(vendedor_codigo)}) cart ON cart.CodCliente = c.CodCliente"
        if vendedor_codigo is not None else ""
    )
    return f"""
SELECT TOP 5 c.CodCliente, LTRIM(RTRIM(c.Cliente_Nombre)) AS Nombre
FROM MAGNUS_SITD.dbo.Clientes c
{join}
WHERE {where}
ORDER BY c.Cliente_Nombre
""".strip()


def _sql_existe_cliente(tokens: list[str]) -> str:
    """Sin filtro de cartera y solo para distinguir dos negativas MUY
    distintas: "ese cliente no es tuyo" vs. "no existe / lo escribiste
    distinto". Devuelve nada más que el nombre — nunca importes."""
    return f"""
SELECT TOP 3 LTRIM(RTRIM(c.Cliente_Nombre)) AS Nombre
FROM MAGNUS_SITD.dbo.Clientes c
WHERE {_where_like_cliente(tokens)}
ORDER BY c.Cliente_Nombre
""".strip()


def _parsear_tsv_clientes(tsv: str) -> list[tuple[int, str]]:
    filas = []
    for l in (tsv or "").splitlines()[1:]:
        if l.startswith("("):
            break
        partes = l.split("\t")
        if len(partes) < 2:
            continue
        try:
            filas.append((int(partes[0]), partes[1].strip()))
        except ValueError:
            continue
    return filas


def _parsear_tsv_nombres(tsv: str) -> list[str]:
    out = []
    for l in (tsv or "").splitlines()[1:]:
        if l.startswith("("):
            break
        if l.strip():
            out.append(l.split("\t")[0].strip())
    return out


def _sql_facturacion_cliente(
    desde: dt.date, hasta_exclusivo: dt.date, cod_cliente: int,
    vendedor_codigo: int | None, niveles: list[int] | None = None,
) -> str:
    """Facturación de UN cliente por mes. `cod_cliente` ya salió de la
    búsqueda filtrada por cartera, así que es un int del propio padrón.

    Con `vendedor_codigo` el número es "lo que VOS le facturaste" y no "lo que
    el cliente compró": un cliente de la cartera por zona puede tener
    comprobantes de otro vendedor, y esos no le corresponden. La etiqueta de
    la respuesta lo dice explícitamente para que nadie lea el número de más."""
    filtro_v = f" AND vc.Vendedor = {int(vendedor_codigo)}" if vendedor_codigo is not None else ""
    mensual = dict(
        select="Anio, Mes, SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes",
        group_by="Anio, Mes",
        cola="ORDER BY 1,2",
    )
    if niveles:
        magnus = f"""
SELECT YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Anio,
  MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Mes,
  SUM({_MONTO_RENGLON}) AS Importe,
  COUNT(DISTINCT vc.NroMovVenta) AS Comprobantes
{_SQL_JOIN_LINEA.strip()}
WHERE vc.FecMovim >= {_dias(desde)} AND vc.FecMovim < {_dias(hasta_exclusivo)}
  AND vc.CodCliente = {int(cod_cliente)}
  AND ap.Nivel1 IN ({_in_niveles(niveles)}){filtro_v}
GROUP BY YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')), MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28'))
""".strip()
        return _dos_subempresas(magnus, **mensual)
    magnus = f"""
SELECT YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Anio,
  MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28')) AS Mes,
  SUM(CASE WHEN vc.CompCodigo IN (1,2,11) THEN vc.Neto+vc.NoGravado
           WHEN vc.CompCodigo IN (22,23,24,25) THEN -(vc.Neto+vc.NoGravado)
           ELSE 0 END) AS Importe,
  COUNT(*) AS Comprobantes
FROM Ven_CompCabecera vc
WHERE vc.FecMovim >= {_dias(desde)} AND vc.FecMovim < {_dias(hasta_exclusivo)}
  AND vc.CodCliente = {int(cod_cliente)}{filtro_v}
GROUP BY YEAR(DATEADD(DAY,vc.FecMovim,'1800-12-28')), MONTH(DATEADD(DAY,vc.FecMovim,'1800-12-28'))
""".strip()
    return _dos_subempresas(magnus, **mensual)


def _sql_ranking_clientes(
    desde: dt.date, hasta_exclusivo: dt.date, vendedor_codigo: int | None
) -> str:
    """"Mis mejores clientes". Para un no-admin sale filtrado por su código —
    no hace falta el JOIN de cartera: lo que él facturó ES suyo por
    definición, y el JOIN costaría un scan más sin cambiar el resultado."""
    filtro = f" AND vc.Vendedor = {int(vendedor_codigo)}" if vendedor_codigo is not None else ""
    magnus = f"""
SELECT vc.CodCliente AS Codigo, LTRIM(RTRIM(c.Cliente_Nombre)) AS Nombre,
  SUM(CASE WHEN vc.CompCodigo IN (1,2,11) THEN vc.Neto+vc.NoGravado
           WHEN vc.CompCodigo IN (22,23,24,25) THEN -(vc.Neto+vc.NoGravado)
           ELSE 0 END) AS Importe,
  COUNT(*) AS Comprobantes
FROM Ven_CompCabecera vc
LEFT JOIN MAGNUS_SITD.dbo.Clientes c ON c.CodCliente = vc.CodCliente
WHERE vc.FecMovim >= {_dias(desde)} AND vc.FecMovim < {_dias(hasta_exclusivo)}{filtro}
GROUP BY vc.CodCliente, c.Cliente_Nombre
""".strip()
    # El TOP 15 pasa AFUERA a propósito: adentro recortaría cada sub-empresa
    # por separado y se podría perder un cliente que es chico en MAGNUS y
    # grande en PRUEBA. `Clientes` es el maestro compartido, así que el nombre
    # es el mismo en las dos ramas.
    return _dos_subempresas(
        magnus,
        select=("TOP 15 Codigo, MAX(Nombre) AS Nombre, "
                "SUM(Importe) AS Importe, SUM(Comprobantes) AS Comprobantes"),
        group_by="Codigo",
        cola="ORDER BY Importe DESC",
    )


_MSG_OTRO_VENDEDOR = (
    "No puedo darte datos de otro vendedor — solo tu propia facturación. "
    "Si necesitás comparar con el resto, pedíselo a un administrador."
)


def _msg_cliente_ajeno(nombre: str | None) -> str:
    quien = f'"{nombre}"' if nombre else "Ese cliente"
    return (
        f"{quien} no está en tu cartera, así que no puedo darte información "
        "suya. Si te lo asignaron hace poco y todavía no lo ves, avisale a un "
        "administrador."
    )


# Aclaración al pie de cualquier corte por línea: el número sale de los
# renglones de artículo, así que no incluye las NC/bonificaciones por concepto
# (no tienen artículo, ver el comentario de _MONTO_RENGLON). Sin esto, alguien
# compara contra el ranking general o contra /ventas/bulones y no cierra.
_PIE_LINEA = (
    "(Por línea sumo los renglones de artículo, así que no entran las "
    "bonificaciones ni las notas de crédito por concepto.)"
)

_MSG_FALLO_MAGNUS = (
    "No pude consultar la base de ventas ahora mismo (no responde el "
    "servicio de Magnus). Probá de nuevo en un rato; si sigue fallando, "
    "avisale a sistemas — puede ser que el servicio esté caído o que se haya "
    "cortado la red."
)


async def _responder_cliente(
    tokens: list[str],
    desde: dt.date,
    hasta: dt.date,
    etiqueta: str,
    vendedor_codigo: int | None,
    lineas_pedidas: list[tuple[str, list[int]]],
) -> str | None:
    """Facturación de UN cliente, con el gate de cartera aplicado en el SQL.

    Devuelve None (y el caller sigue con el flujo normal) sólo si el término
    quedó vacío después de sanear — nunca por falta de permiso: eso se responde
    con `_msg_cliente_ajeno`.

    Las tres salidas posibles cuando hay término:
      · está en su cartera        → el número.
      · existe pero no es suyo    → "no está en tu cartera".
      · no existe / mal escrito   → "no lo encontré".
    """
    if not _where_like_cliente(tokens):
        return None

    tsv = await _ejecutar_sql(_sql_buscar_cliente(tokens, vendedor_codigo))
    encontrados = _parsear_tsv_clientes(tsv)

    if not encontrados:
        # ¿Existe fuera de su cartera? Distinguirlo es lo que pidió el negocio:
        # "no te corresponde" y "no existe" son problemas distintos para el que
        # pregunta. Sólo se mira el NOMBRE, nunca un importe.
        ajenos = _parsear_tsv_nombres(await _ejecutar_sql(_sql_existe_cliente(tokens)))
        if ajenos:
            log.warning(
                f"[VENTAS] vendedor {vendedor_codigo} pidió el cliente "
                f"{ajenos[0]!r}, que no es de su cartera — denegado"
            )
            return _msg_cliente_ajeno(ajenos[0] if len(ajenos) == 1 else None)
        return (
            f"No encontré ningún cliente que coincida con \"{' '.join(tokens)}\". "
            "Probá con el nombre completo o con el código de cliente."
        )

    if len(encontrados) > 1:
        # Ambiguo: se listan los suyos y elige. Todos salieron del SELECT ya
        # filtrado por cartera, así que mostrarlos no revela nada ajeno.
        opciones = "\n".join(f"- {n} (cód. {c})" for c, n in encontrados)
        return f"Encontré más de un cliente tuyo con ese nombre:\n\n{opciones}\n\n¿Cuál de todos?"

    cod_cliente, nombre_cliente = encontrados[0]
    niveles = lineas_pedidas[0][1] if lineas_pedidas else None
    detalle_linea = lineas_pedidas[0][0] if lineas_pedidas else None

    tsv = await _ejecutar_sql(
        _sql_facturacion_cliente(desde, hasta, cod_cliente, vendedor_codigo, niveles)
    )
    filas = _parsear_tsv(tsv)

    quien = f"{nombre_cliente} (cód. {cod_cliente})"
    if detalle_linea:
        quien = f"{quien}, línea {detalle_linea}"
    # Para un vendedor el número es lo que ÉL facturó, no lo que el cliente
    # compró: un cliente de su zona puede tener comprobantes de otro vendedor.
    de_quien = "" if vendedor_codigo is None else " (lo que le facturaste vos)"
    pie = f"\n\n{_PIE_LINEA}" if detalle_linea else ""

    if not filas:
        return f"No encontré facturación de {quien} en {etiqueta}{de_quien}."

    total = sum(f["importe"] for f in filas)
    comprobantes = sum(f["comprobantes"] for f in filas)
    if len(filas) == 1:
        f = filas[0]
        return (
            f"Facturación de {quien} en {etiqueta}{de_quien}: {_monto(f['importe'])} "
            f"({_comps(f['comprobantes'])}).{pie}"
        )

    out = [f"Facturación de {quien}, {etiqueta}{de_quien}:", ""]
    for f in filas:
        out.append(
            f"- {_NOMBRE_MES[f['mes']]} {f['anio']}: {_monto(f['importe'])} "
            f"({f['comprobantes']} comp.)"
        )
    out.append("")
    out.append(f"Total del período: {_monto(total)} ({_comps(comprobantes)}).")
    return "\n".join(out) + pie


def _y(nombres: list[str]) -> str:
    """"A, B y C" — para rotular un grupo de vendedores en el título."""
    if len(nombres) <= 1:
        return nombres[0] if nombres else ""
    return ", ".join(nombres[:-1]) + " y " + nombres[-1]


async def _responder_ranking(
    desde: dt.date,
    hasta: dt.date,
    etiqueta: str,
    codigos: list[int] | None,
    lineas_pedidas: list[tuple[str, list[int]]],
    tope: int | None,
    titulo: str,
) -> str:
    """El desglose por vendedor: todos, o sólo `codigos` si el usuario nombró a
    algunos. El recorte va en el SQL (`IN`), nunca filtrando en Python: con
    varios meses de comprobantes traer todo para descartar sería un scan al
    pedo (mismo criterio que el gate de cartera)."""
    if lineas_pedidas:
        tope_l = tope if (tope is None or len(lineas_pedidas) == 1) else 5
        bloques = []
        for etiqueta_linea, niveles in lineas_pedidas:
            tsv = await _ejecutar_sql(
                _sql_ranking_vendedores_linea(desde, hasta, niveles, codigos)
            )
            bloques.append(_formatear_ranking(
                _parsear_tsv_ranking(tsv), f"{titulo} — {etiqueta_linea}, {etiqueta}", tope_l
            ))
        bloques.append(_PIE_LINEA)
        return "\n\n".join(bloques)

    tsv = await _ejecutar_sql(_sql_ranking_vendedores(desde, hasta, codigos))
    return _formatear_ranking(_parsear_tsv_ranking(tsv), f"{titulo}, {etiqueta}", tope)


async def _total_conjunto(
    desde: dt.date,
    hasta: dt.date,
    etiqueta: str,
    vendedores: list[tuple[int, str]],
    lineas_pedidas: list[tuple[str, list[int]]],
) -> str:
    """Los vendedores nombrados SUMADOS ("cuánto vendieron Gómez y Pérez en
    total"). Es la misma consulta recortada del desglose, agregada en Python
    sobre las pocas filas que vuelven — no hace falta ir de nuevo a Magnus."""
    codigos = [c for c, _ in vendedores]
    detalle = None
    if lineas_pedidas:
        detalle, niveles = lineas_pedidas[0]
        tsv = await _ejecutar_sql(_sql_ranking_vendedores_linea(desde, hasta, niveles, codigos))
    else:
        tsv = await _ejecutar_sql(_sql_ranking_vendedores(desde, hasta, codigos))
    filas = _parsear_tsv_ranking(tsv)
    quien = _y([n for _, n in vendedores])
    if detalle:
        quien = f"{quien}, línea {detalle}"
    if not filas:
        return f"No encontré facturación de {quien} en {etiqueta}."
    pie = f"\n\n{_PIE_LINEA}" if detalle else ""
    return (
        f"Facturación de {quien} en {etiqueta}, los {len(codigos)} sumados: "
        f"{_monto(sum(f['importe'] for f in filas))} "
        f"({_comps(sum(f['comprobantes'] for f in filas))}).{pie}"
    )


async def responder_ventas(mensaje: str, vendedor_codigo: int | None, es_admin: bool) -> str:
    """Punto de entrada del intent "ventas". `vendedor_codigo` ya viene
    resuelto y validado por el caller (nodes.py) — ver el docstring del
    módulo. Nunca levanta excepción hacia afuera: cualquier falla se convierte
    en un mensaje explicando qué pasó (infra vs. sin datos), mismo criterio
    que `diagnostico_cvs` en tools.py."""
    if not (config.MAGNUS_SQL_URL or config.MAGNUS_MCP_URL):
        return (
            "Todavía no tengo conectada la base de ventas (falta configurar "
            "MAGNUS_SQL_URL). Avisale a sistemas — no es que no haya datos, es "
            "que esta parte no está prendida."
        )

    desde, hasta, etiqueta = _parsear_rango(mensaje)
    # Sólo se completa cuando un ADMIN pide explícitamente por un vendedor:
    # cambia el "tu cartera" de la respuesta por el nombre de esa persona.
    etiqueta_vendedor: str | None = None
    # Varios vendedores nombrados en el mismo mensaje → comparación entre
    # ellos (no el reporte de uno), recortada con IN en el SQL.
    vendedores_pedidos: list[tuple[int, str]] = []

    # ¿Nombró alguna línea de producto? (bulones, mangueras, …). Si el catálogo
    # no se puede leer, se sigue sin corte por línea en vez de fallar: es mejor
    # dar el número general que no dar nada.
    lineas_pedidas: list[tuple[int, str]] = []
    try:
        lineas_pedidas = _detectar_lineas(mensaje, await _catalogo_lineas())
    except _FalloMagnus:
        return _MSG_FALLO_MAGNUS
    except Exception:
        log.exception("no pude leer el catálogo de líneas — sigo sin corte por línea")

    # ── GATE 1: cliente ──────────────────────────────────────────────────────
    # Va PRIMERO que el gate de vendedor a propósito: una razón social puede
    # contener el apellido de un vendedor ("FERRETERÍA BLANCO") y si corriera
    # antes el otro gate se negaría una consulta que sí le corresponde.
    #
    # Consecuencia conocida: un admin que pregunte "cuánto le vendió Blanco al
    # cliente Rossi" recibe el total del cliente entre TODOS los vendedores, no
    # el de Blanco. Es un número de menos precisión, no un dato de más — y
    # cruzar los dos gates traería de vuelta el falso positivo de la razón
    # social. Si algún día hace falta, resolverlo solo para es_admin.
    tokens_cliente = _detectar_cliente_pedido(mensaje)
    if tokens_cliente:
        try:
            respuesta = await _responder_cliente(
                tokens_cliente, desde, hasta, etiqueta, vendedor_codigo, lineas_pedidas
            )
        except _FalloMagnus:
            return _MSG_FALLO_MAGNUS
        if respuesta is not None:
            return respuesta
        # None = no se pudo armar el término de búsqueda; sigue el flujo normal.

    elif _PATRON_RANKING_CLIENTES.search(mensaje or ""):
        # "mis mejores clientes" / "qué cliente me compró más": el filtro por
        # vendedor ya deja adentro solo lo suyo.
        try:
            tsv = await _ejecutar_sql(_sql_ranking_clientes(desde, hasta, vendedor_codigo))
        except _FalloMagnus:
            return _MSG_FALLO_MAGNUS
        titulo = (
            f"Clientes de toda la empresa, {etiqueta}" if vendedor_codigo is None
            else f"Tus clientes, {etiqueta}"
        )
        return _formatear_ranking(
            _parsear_tsv_ranking(tsv), titulo, 15, sustantivo="clientes"
        )

    else:
        # ── GATE 2: ¿nombró a otro vendedor, o a varios? ─────────────────────
        # Un no-admin no puede ver a nadie más, ni siquiera preguntando por el
        # nombre en vez de pedir el ranking. Un admin sí: uno solo se usa como
        # filtro; dos o más son una comparación entre ellos, que sale por el
        # mismo camino que el ranking pero recortada con IN.
        try:
            mencionados = _detectar_vendedores_mencionados(mensaje, await _catalogo_vendedores())
        except _FalloMagnus:
            return _MSG_FALLO_MAGNUS
        except Exception:
            log.exception("no pude leer el maestro de vendedores")
            mencionados = []

        ajenos = [v for v in mencionados if v[0] != vendedor_codigo]
        if ajenos:
            if not es_admin:
                log.warning(
                    f"[VENTAS] vendedor {vendedor_codigo} pidió datos de "
                    + ", ".join(f"{c} ({n})" for c, n in ajenos) + " — denegado"
                )
                return _MSG_OTRO_VENDEDOR
            if len(mencionados) > 1:
                # varios: comparación entre ellos, no el reporte de uno
                vendedores_pedidos = mencionados
                log.info(f"[VENTAS] admin compara vendedores {[c for c, _ in mencionados]}")
            else:
                # admin: la consulta pasa a ser sobre ESE vendedor
                vendedor_codigo = mencionados[0][0]
                etiqueta_vendedor = f"{mencionados[0][1]} (cód. {mencionados[0][0]})"
                log.info(f"[VENTAS] admin consulta al vendedor {mencionados[0][0]}")

        # El nombre puede ser también un mes ("Julio Blanco"): se relee el
        # rango sin esa palabra — ver `_sin_nombres_vendedor`.
        if mencionados:
            limpio = _sin_nombres_vendedor(mensaje, mencionados)
            if limpio != (mensaje or ""):
                desde, hasta, etiqueta = _parsear_rango(limpio)

    # Comparación ENTRE vendedores: el ranking pedido explícitamente ("qué
    # vendedor vendió más", "cuánto vendió cada vendedor", "por vendedor") o
    # varios nombrados en el mismo mensaje. Solo admin — ver docstring del
    # módulo. Un no-admin que lo pida no se queda sin respuesta: se le aclara
    # el motivo y se le ofrece su propio dato en su lugar.
    if vendedores_pedidos or _es_pedido_ranking(mensaje):
        if not es_admin:
            return (
                "Ese dato es de toda la empresa y no te lo puedo mostrar — "
                "solo puedo darte TU propia facturación. Preguntame, por "
                f"ejemplo, \"cómo vengo {'' if etiqueta.split()[0] in ('este', 'el', 'hoy', 'los') else 'en '}{etiqueta}\"."
            )
        codigos = [c for c, _ in vendedores_pedidos] or None
        titulo = (_y([n for _, n in vendedores_pedidos]) if vendedores_pedidos
                  else "Ranking de vendedores")
        try:
            # "…en total" colapsa a un número; si no, la lista, que ya trae el
            # total al pie y responde las dos formas de la misma pregunta.
            if codigos and _quiere_total(mensaje):
                return await _total_conjunto(desde, hasta, etiqueta, vendedores_pedidos, lineas_pedidas)
            # Sin tope cuando se pidió "cada/todos" o cuando la lista son los
            # que nombró el usuario; si no, los 15 primeros.
            tope = None if (codigos or _quiere_todos(mensaje)) else 15
            return await _responder_ranking(
                desde, hasta, etiqueta, codigos, lineas_pedidas, tope, titulo
            )
        except _FalloMagnus:
            return _MSG_FALLO_MAGNUS

    if lineas_pedidas:
        # "cómo vengo en bulones": mismo reporte mensual pero por línea. El
        # gate de vendedor es el de siempre (vendedor_codigo ya viene fijo).
        detalle, niveles = lineas_pedidas[0]
        sql = _sql_total_linea(desde, hasta, niveles, vendedor_codigo)
    else:
        detalle = None
        sql = _sql_reporte_mensual(desde, hasta, vendedor_codigo)

    try:
        tsv = await _ejecutar_sql(sql)
    except _FalloMagnus:
        return _MSG_FALLO_MAGNUS

    filas = _parsear_tsv(tsv)
    if vendedor_codigo is None:
        quien = "toda la empresa"
    elif etiqueta_vendedor:
        quien = etiqueta_vendedor
    else:
        quien = f"tu cartera (vendedor {vendedor_codigo})"
    if detalle:
        quien = f"{quien}, línea {detalle}"
    if not filas:
        return f"No encontré facturación de {quien} en {etiqueta}."

    total = sum(f["importe"] for f in filas)
    comprobantes = sum(f["comprobantes"] for f in filas)
    pie = f"\n\n{_PIE_LINEA}" if detalle else ""

    if len(filas) == 1:
        f = filas[0]
        return (
            f"Facturación de {quien} en {etiqueta}: {_monto(f['importe'])} "
            f"({_comps(f['comprobantes'])}).{pie}"
        )

    lineas = [f"Facturación de {quien}, {etiqueta}:", ""]
    for f in filas:
        lineas.append(
            f"- {_NOMBRE_MES[f['mes']]} {f['anio']}: {_monto(f['importe'])} "
            f"({f['comprobantes']} comp.)"
        )
    mejor = max(filas, key=lambda f: f["importe"])
    lineas.append("")
    lineas.append(f"Total del período: {_monto(total)} ({_comps(comprobantes)}).")
    lineas.append(f"Mejor mes: {_NOMBRE_MES[mejor['mes']]} {mejor['anio']} con {_monto(mejor['importe'])}.")
    return "\n".join(lineas) + pie
