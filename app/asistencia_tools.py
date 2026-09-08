"""Intent "rrhh": asistencia — faltas, feriados y horas extras.

DE DÓNDE SALEN LOS NÚMEROS
    De Postgres (schemas `asistencia` y `everwear`), la MISMA base que usa
    /rrhh/asistencia en ever. No pasa por Magnus ni por el servicio HTTP de
    ventas: es la base local del propio contenedor (config.DATABASE_URL), así
    que una consulta acá cuesta milisegundos.

    El criterio de cálculo NO se inventa: es una traducción a SQL de lo que ya
    hace la pantalla, y tiene que seguir dando lo mismo. Dos archivos son la
    referencia y hay que mirarlos si algún día no cierra un número:

      · vicki_web/app/api/rrhh/asistencia/resumen/route.ts
        → cómo se arma el día de cada empleado a partir de los fichajes crudos:
          margen anti-duplicado de 5', descarte de la marca del medio cuando
          hay 3, impar=ingreso / par=egreso, minutos = suma de cada par
          (así un corte para almorzar no cuenta como trabajado), ajuste manual
          que pisa el fichaje, y el arrastre de `estado_diario.dias`.
      · vicki_web/lib/rrhh/asistenciaIndicadores.ts
        → tope diario por área (`horario_tipo`/`horario_area`, fallback
          "Estándar (Lun-Vie)"), feriado = tope 0, neto = fichado − novedad,
          extra = neto − tope, y el estado calculado por defecto.

    Todo eso vive en el CTE `_SQL_BASE`, que devuelve una fila por
    (empleado, día) ya con `neto`, `tope` y `estado` resueltos. Cada reporte es
    un GROUP BY sobre eso — nunca se traen los días crudos a Python.

SEGURIDAD
    A diferencia de ventas, acá NO hay filtro "por vendedor": quien tiene el
    permiso (`usuario.vickiRrhhAcceso`, o ser ADMIN — ver
    vicki_web/lib/rrhh/vickiRrhhAcceso.ts) ve la asistencia de TODA la empresa,
    porque es un dato de RRHH y no tiene sentido a medias. Por eso el permiso
    se da con cuentagotas. Sin el permiso, `rrhh_node` ni llama a este módulo.

    El ÚNICO texto del usuario que llega al SQL son los nombres de empleado, y
    van SIEMPRE como parámetro de asyncpg ($1, $2…), nunca interpolados.

QUÉ PREGUNTAS RESUELVE (todas detectadas por regex, no por el LLM — un rango o
un motivo mal interpretado da un número mal calculado sin que nadie lo note):
    · "¿cuántos días faltó Fulano?"        → `_reporte_faltas_persona`
    · "¿quién faltó más el mes pasado?"    → `_reporte_ranking_faltas`
    · "¿qué feriados registramos en julio?"→ `_reporte_feriados`
    · "¿quién hizo horas extras?"          → `_reporte_extras`
    · cualquier otra cosa del tema         → `_reporte_resumen` (el panorama)
"""
import datetime as dt
import logging
import re

from app.config import config
# Un solo parser de fechas para todo el chat: si "el mes pasado" cambia de
# significado, cambia en un solo lugar. Devuelve `hasta` EXCLUSIVO.
from app.ventas_tools import _normalizar, _parsear_rango

log = logging.getLogger("asistencia_tools")

_TZ = "America/Argentina/Buenos_Aires"

# ── Pool propio ──────────────────────────────────────────────────────────────
# main.py tiene su pool, pero importarlo desde acá es un ciclo (main → graph →
# nodes → este módulo). Uno chico y aparte: estas consultas son esporádicas.
_pool = None


async def _get_pool():
    global _pool
    if _pool is None:
        import asyncpg

        _pool = await asyncpg.create_pool(
            config.DATABASE_URL, min_size=1, max_size=3, timeout=15, command_timeout=30
        )
    return _pool


class _FalloBase(Exception):
    """Cualquier problema hablando con Postgres."""


_MSG_FALLO_BASE = (
    "No pude consultar la base de asistencia ahora mismo. Probá de nuevo en un "
    "rato; si sigue fallando, avisale a sistemas."
)


async def _consultar(sql: str, *params) -> list:
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(sql, *params)
    except Exception as e:
        log.exception("consulta de asistencia falló")
        raise _FalloBase(str(e)) from e


# ── El CTE base: una fila por (empleado, día) con neto/tope/estado ───────────
# {filtro_emp} se rellena con "" o con 'AND l."employeeNo" = $3'. Filtrar acá
# (y no al final) achica el CROSS JOIN de golpe: de 72 empleados × N días a
# 1 × N, y hace que el scan de `evento` sirva para una sola persona.
_SQL_BASE = """
WITH p AS (SELECT $1::date AS d1, $2::date AS d2),
bounds AS (
  SELECT (d1::timestamp) AT TIME ZONE '{tz}' AS lo,
         ((d2 + 1)::timestamp) AT TIME ZONE '{tz}' AS hi
  FROM p
),
dias AS (SELECT generate_series(d1, d2, interval '1 day')::date AS fecha FROM p),
act AS (
  SELECT l."employeeNo" AS employee_no,
         NULLIF(TRIM(l.nombre), '') AS employee_name,
         COALESCE(ar.nombre, l.sector) AS departamento
  FROM everwear.legajo l
  LEFT JOIN everwear.sector s ON s.id = l."sectorId"
  LEFT JOIN everwear.area   ar ON ar.id = s."areaId"
  WHERE l.estado = 'ACTIVO' AND l."employeeNo" IS NOT NULL
  {filtro_emp}
),
-- Legajos activos que comparten clave "sin ceros a la izquierda": "40" y
-- "00000040" son personas DISTINTAS y no se pueden fusionar (ver resumen/route.ts).
fuzzy_counts AS (
  SELECT ltrim(employee_no, '0') AS fuzzy_key, COUNT(*) AS n FROM act GROUP BY 1
),
ev_ids AS (
  SELECT DISTINCT e.employee_no AS raw_no
  FROM asistencia.evento e, bounds b
  WHERE e.event_time >= b.lo AND e.event_time < b.hi
),
emp_map AS (
  SELECT i.raw_no, COALESCE(exacto.employee_no, difuso.employee_no) AS employee_no
  FROM ev_ids i
  LEFT JOIN act exacto ON exacto.employee_no = i.raw_no
  LEFT JOIN LATERAL (
    SELECT a.employee_no FROM act a
    JOIN fuzzy_counts fc ON fc.fuzzy_key = ltrim(a.employee_no, '0')
    WHERE exacto.employee_no IS NULL
      AND ltrim(a.employee_no, '0') = ltrim(i.raw_no, '0')
      AND fc.n = 1
    LIMIT 1
  ) difuso ON true
),
raw_ev AS (
  SELECT m.employee_no AS emp_key,
         (e.event_time AT TIME ZONE '{tz}')::date AS fecha,
         e.event_time
  FROM asistencia.evento e
  JOIN emp_map m ON m.raw_no = e.employee_no
  CROSS JOIN bounds b
  WHERE e.event_time >= b.lo AND e.event_time < b.hi AND m.employee_no IS NOT NULL
),
gapped AS (
  SELECT r.*, r.event_time - LAG(r.event_time)
           OVER (PARTITION BY r.emp_key, r.fecha ORDER BY r.event_time) AS gap
  FROM raw_ev r
),
-- Dos toques del mismo reloj a menos de 5' son UNA marca (el reloj a veces
-- registra dos veces el mismo toque).
islas AS (
  SELECT g.*, SUM(CASE WHEN g.gap IS NULL OR g.gap >= interval '5 minutes' THEN 1 ELSE 0 END)
           OVER (PARTITION BY g.emp_key, g.fecha ORDER BY g.event_time) AS isla
  FROM gapped g
),
marcas AS (
  SELECT emp_key, fecha, isla, MIN(event_time) AS mark_time FROM islas GROUP BY 1, 2, 3
),
marcas_cnt AS (SELECT emp_key, fecha, COUNT(*) AS cnt FROM marcas GROUP BY 1, 2),
-- Un día completo tiene marcas PARES. Con exactamente 3, la del medio es
-- espuria: tomada como egreso restaría un tramo que sí se trabajó.
marcas_filtradas AS (
  SELECT m.emp_key, m.fecha, m.mark_time
  FROM marcas m
  JOIN marcas_cnt mc ON mc.emp_key = m.emp_key AND mc.fecha = m.fecha
  WHERE NOT (mc.cnt = 3 AND m.mark_time = (
    SELECT m2.mark_time FROM marcas m2
    WHERE m2.emp_key = m.emp_key AND m2.fecha = m.fecha
    ORDER BY m2.mark_time OFFSET 1 LIMIT 1
  ))
),
posn AS (
  SELECT emp_key, fecha, mark_time,
         ROW_NUMBER() OVER (PARTITION BY emp_key, fecha ORDER BY mark_time) AS posn,
         LAG(mark_time) OVER (PARTITION BY emp_key, fecha ORDER BY mark_time) AS prev_mark_time
  FROM marcas_filtradas
),
ev AS (
  SELECT emp_key, fecha,
         MIN(mark_time) FILTER (WHERE posn % 2 = 1) AS check_in,
         MAX(mark_time) FILTER (WHERE posn % 2 = 0) AS check_out,
         COALESCE(SUM(CASE WHEN posn % 2 = 0
           THEN GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (mark_time - prev_mark_time)) / 60))
           ELSE 0 END), 0)::int AS minutos
  FROM posn GROUP BY 1, 2
),
base AS (
  SELECT a.employee_no, a.employee_name, a.departamento, d.fecha,
         COALESCE(am.check_in, ev.check_in)   AS check_in,
         COALESCE(am.check_out, ev.check_out) AS check_out,
         CASE WHEN am.check_in IS NOT NULL OR am.check_out IS NOT NULL THEN
           GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (
             COALESCE(am.check_out, ev.check_out) - COALESCE(am.check_in, ev.check_in)
           )) / 60))::int
         ELSE COALESCE(ev.minutos, 0) END AS minutos,
         (fer.fecha IS NOT NULL) AS feriado,
         CASE WHEN ed.dias IS NOT NULL AND ed.dias <= 0 THEN NULL
              ELSE COALESCE(ed.estado, c.c_estado) END AS estado_guardado,
         COALESCE(nd.horas, 0) AS horas_novedad
  FROM dias d
  CROSS JOIN act a
  LEFT JOIN ev ON ev.emp_key = a.employee_no AND ev.fecha = d.fecha
  LEFT JOIN asistencia.ajuste_manual am ON am.employee_no = a.employee_no AND am.fecha = d.fecha
  LEFT JOIN asistencia.feriado fer ON fer.fecha = d.fecha
  LEFT JOIN asistencia.estado_diario ed ON ed.employee_no = a.employee_no AND ed.fecha = d.fecha
  -- Arrastre: "Vacaciones, 10 días" cargado una vez cubre los 10 días.
  LEFT JOIN LATERAL (
    SELECT ed2.estado AS c_estado
    FROM asistencia.estado_diario ed2
    WHERE ed.estado IS NULL AND ed2.employee_no = a.employee_no
      AND ed2.dias IS NOT NULL AND ed2.fecha < d.fecha
      AND ed2.fecha + (ed2.dias - 1) >= d.fecha
    ORDER BY ed2.fecha DESC LIMIT 1
  ) c ON true
  LEFT JOIN asistencia.novedad_diaria nd ON nd.employee_no = a.employee_no AND nd.fecha = d.fecha
),
calc AS (
  SELECT b.*,
         GREATEST(0, b.minutos - b.horas_novedad * 60) AS neto,
         -- Feriado pisa el tope a 0: nadie estaba programado para trabajar, así
         -- que lo trabajado ese día cuenta 100% como extra.
         CASE WHEN b.feriado THEN 0 ELSE (
           CASE EXTRACT(DOW FROM b.fecha)
             WHEN 0 THEN ht.tope_dom WHEN 1 THEN ht.tope_lun WHEN 2 THEN ht.tope_mar
             WHEN 3 THEN ht.tope_mie WHEN 4 THEN ht.tope_jue WHEN 5 THEN ht.tope_vie
             ELSE ht.tope_sab END
         ) END AS tope,
         COALESCE(b.estado_guardado, CASE
           WHEN b.check_in IS NULL THEN (CASE WHEN b.feriado THEN 'Feriado' ELSE 'Ausente' END)
           WHEN b.check_out IS NULL THEN (
             CASE WHEN b.fecha = (now() AT TIME ZONE '{tz}')::date THEN 'Presente' ELSE 'Revisar' END)
           WHEN b.minutos < 60 THEN 'Revisar'
           ELSE 'Normal' END) AS estado
  FROM base b
  LEFT JOIN asistencia.horario_area ha ON ha.departamento = b.departamento
  -- Área sin horario asignado → "Estándar (Lun-Vie)", igual que buildTopeResolver.
  LEFT JOIN LATERAL (
    SELECT * FROM asistencia.horario_tipo t
    WHERE t.id = COALESCE(ha.horario_tipo_id, (
      SELECT id FROM asistencia.horario_tipo WHERE nombre = 'Estándar (Lun-Vie)' ORDER BY id LIMIT 1))
  ) ht ON true
)
""".replace("{tz}", _TZ)


def _base(filtro_emp: str = "") -> str:
    return _SQL_BASE.replace("{filtro_emp}", filtro_emp)


_FILTRO_EMP = 'AND l."employeeNo" = $3'


# ── Estados ──────────────────────────────────────────────────────────────────
# "Normal"/"Presente" = trabajó. "Revisar" = fichada incompleta (falta el
# egreso): NO es una falta, es un dato a corregir — se informa aparte para que
# nadie lo lea como ausencia. Todo lo demás (Ausente, Vacaciones, Enfermedad,
# ART, …) es un día no trabajado, y se muestra desglosado por motivo.
#
# "Feriado" TAMPOCO es una falta, y hay que excluirlo a mano: un feriado
# cargado marcando a cada persona en la grilla (así quedó el 17/08) no entra en
# `asistencia.feriado`, así que ese día conserva tope > 0 y sin esta exclusión
# aparecía como un día no trabajado de las 74 personas. Por la misma razón se
# descuenta del denominador de jornadas hábiles.
_INCOMPLETOS = ("Revisar",)
_SIN_JUSTIFICAR = "Ausente"
_NO_ES_FALTA = "('Normal', 'Presente', 'Revisar', 'Feriado')"
_ES_JORNADA = "tope > 0 AND estado <> 'Feriado'"


# ── Detección de pedido (regex, no LLM) ──────────────────────────────────────
_PATRON_FERIADOS = re.compile(r"feriad", re.I)
_PATRON_EXTRAS = re.compile(r"horas?\s+extra|hs\.?\s*extra|extras?\b", re.I)
_PATRON_FALTAS = re.compile(
    r"falt[oó]|falt[eé]|faltas|ausen|inasisten|no vino|no fue a trabajar|"
    r"d[ií]as? (?:no )?trabajad|vacacion|licencia|enfermedad|carpeta",
    re.I,
)

def _hasta_inclusivo(hasta_exclusivo: dt.date, hoy: dt.date) -> dt.date:
    """`_parsear_rango` devuelve `hasta` exclusivo; acá los días se generan con
    generate_series(d1, d2), o sea inclusivo. Además se corta en HOY: un día
    futuro no tiene fichadas y aparecería como ausencia de todo el mundo."""
    return min(hasta_exclusivo - dt.timedelta(days=1), hoy)


# ── Catálogo de empleados (para reconocer a quién nombran) ───────────────────
_TTL_EMPLEADOS = 3600
_empleados_cache: dict = {"t": 0.0, "datos": []}

# Palabras que aparecen en la pregunta y no son parte de un nombre.
_STOP_PERSONA = {
    "cuantos", "cuantas", "dias", "falto", "falta", "faltas", "faltó", "empleado",
    "empleados", "persona", "personas", "gente", "mes", "meses", "pasado",
    "anterior", "este", "esta", "año", "anio", "horas", "extras", "extra",
    "hizo", "hicieron", "quien", "quienes", "cual", "cuales", "trabajo",
    "trabajó", "ausente", "ausencias", "ausencia", "feriado", "feriados",
    "registramos", "tiene", "tuvo", "vacaciones", "enfermedad", "licencia",
    "sector", "area", "total", "todos", "todas", "para", "por", "con", "del",
    "las", "los", "que", "una", "uno", "sus", "mis", "vino", "fue",
    "asistencia", "presente", "presentes", "julio", "agosto", "enero",
    "febrero", "marzo", "abril", "mayo", "junio", "septiembre", "setiembre",
    "octubre", "noviembre", "diciembre",
}


async def _catalogo_empleados() -> list[tuple[str, str]]:
    """(employeeNo, nombre) de los legajos ACTIVOS, cacheado 1h. Son ~72 filas
    y cambian con altas/bajas, nunca dentro de una charla."""
    import time

    if _empleados_cache["datos"] and time.time() - _empleados_cache["t"] < _TTL_EMPLEADOS:
        return _empleados_cache["datos"]
    filas = await _consultar(
        'SELECT l."employeeNo", TRIM(l.nombre) AS nombre FROM everwear.legajo l '
        "WHERE l.estado = 'ACTIVO' AND l.\"employeeNo\" IS NOT NULL "
        "AND NULLIF(TRIM(l.nombre), '') IS NOT NULL"
    )
    datos = [(r["employeeNo"], r["nombre"]) for r in filas]
    _empleados_cache.update({"t": time.time(), "datos": datos})
    return datos


def _partes_nombre(nombre: str) -> list[str]:
    return [p for p in re.findall(r"[a-z]{3,}", _normalizar(nombre)) if p not in _STOP_PERSONA]


def _partes_unicas(catalogo: list[tuple[str, str]]) -> set[str]:
    """Partes que identifican a UNA sola persona del legajo. Es lo que permite
    entender "cuántos días faltó Vaudagna" sin lista de apodos; un "Romero"
    repetido no alcanza y ahí se piden dos partes (o se pregunta cuál)."""
    cuenta: dict[str, int] = {}
    for _, nombre in catalogo:
        for p in set(_partes_nombre(nombre)):
            cuenta[p] = cuenta.get(p, 0) + 1
    return {p for p, n in cuenta.items() if n == 1}


def _detectar_personas(mensaje: str, catalogo: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Empleados nombrados en el mensaje → [(employee_no, nombre), ...].

    Igual que el gate de vendedores de ventas_tools, pero al revés en el
    riesgo: acá no hay dato ajeno que proteger (quien pregunta ya tiene el
    permiso de RRHH), así que el peor caso es no reconocer a alguien y contestar
    con el panorama general en vez de con su ficha.
    """
    tokens = set(re.findall(r"[a-z]{3,}", _normalizar(mensaje)))
    if not tokens:
        return []
    unicas = _partes_unicas(catalogo)
    encontrados: list[tuple[int, str, str]] = []
    for emp_no, nombre in catalogo:
        partes = _partes_nombre(nombre)
        if not partes:
            continue
        hits = [p for p in partes if p in tokens]
        if len(hits) >= 2 or (len(hits) == 1 and len(hits[0]) >= 4 and hits[0] in unicas):
            encontrados.append((len(hits), emp_no, nombre))
    encontrados.sort(key=lambda x: -x[0])
    return [(e, n) for _, e, n in encontrados[:3]]


# ── Formateo ─────────────────────────────────────────────────────────────────
_DIA_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_NOMBRE_MES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
               "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha_larga(f: dt.date) -> str:
    return f"{_DIA_SEMANA[f.weekday()]} {f.day}/{f.month:02d}"


def _horas(minutos: float) -> str:
    h = minutos / 60
    return f"{h:.1f} h".replace(".", ",")


def _dias(n: int) -> str:
    return "1 día" if n == 1 else f"{n} días"


# ── Reportes ─────────────────────────────────────────────────────────────────

async def _reporte_faltas_persona(
    emp_no: str, nombre: str, d1: dt.date, d2: dt.date, etiqueta: str
) -> str:
    """Días no trabajados de UNA persona, desglosados por motivo.

    Sólo cuenta jornadas hábiles (tope > 0): un domingo o un feriado no es una
    falta. `Revisar` va aparte — es una fichada incompleta, no una ausencia."""
    # Una sola pasada del CTE: el desglose por motivo y el total de jornadas
    # hábiles salen juntos (la subconsulta se evalúa sobre el mismo `calc`).
    filas = await _consultar(
        _base(_FILTRO_EMP) + f"""
SELECT estado, COUNT(*)::int AS dias,
       (array_agg(fecha ORDER BY fecha))[1:8] AS fechas,
       (SELECT COUNT(*)::int FROM calc WHERE {_ES_JORNADA}) AS habiles
FROM calc
WHERE {_ES_JORNADA} AND estado NOT IN ('Normal', 'Presente')
GROUP BY estado
ORDER BY dias DESC
""",
        d1, d2, emp_no,
    )
    if not filas:
        # Sin filas no hay de dónde sacar el denominador: se pide aparte (sólo
        # pasa cuando la persona no faltó ni un día, que es el caso feliz).
        habiles = await _consultar(
            _base(_FILTRO_EMP) + f"SELECT COUNT(*)::int AS n FROM calc WHERE {_ES_JORNADA}",
            d1, d2, emp_no,
        )
        n_habiles = habiles[0]["n"] if habiles else 0
    else:
        n_habiles = filas[0]["habiles"]

    faltas = [f for f in filas if f["estado"] not in _INCOMPLETOS]
    incompletos = sum(f["dias"] for f in filas if f["estado"] in _INCOMPLETOS)
    total = sum(f["dias"] for f in faltas)

    if not total:
        base = f"{nombre} no faltó ningún día en {etiqueta} ({n_habiles} jornadas hábiles)."
        if incompletos:
            base += (
                f"\n\nOjo: tiene {_dias(incompletos)} con la fichada incompleta "
                f'(estado "Revisar"), que no cuento como falta pero conviene corregir.'
            )
        return base

    sin_just = sum(f["dias"] for f in faltas if f["estado"] == _SIN_JUSTIFICAR)
    out = [
        f"{nombre} — {etiqueta}: {_dias(total)} no trabajados sobre "
        f"{n_habiles} jornadas hábiles"
        + (f", {sin_just} sin justificar." if sin_just else "."),
        "",
    ]
    for f in faltas:
        fechas = ", ".join(_fecha_larga(x) for x in f["fechas"])
        extra = f" — {fechas}" if f["dias"] <= 8 else f" — {fechas}, …"
        out.append(f"- {f['estado']}: {_dias(f['dias'])}{extra}")
    if incompletos:
        out.append("")
        out.append(
            f'Aparte: {_dias(incompletos)} con fichada incompleta ("Revisar"), '
            "que no cuento como falta."
        )
    return "\n".join(out)


async def _reporte_ranking_faltas(d1: dt.date, d2: dt.date, etiqueta: str) -> str:
    """Ranking por días no trabajados en jornada hábil ("quién faltó más")."""
    filas = await _consultar(
        _base() + f"""
SELECT employee_name,
       COUNT(*) FILTER (WHERE estado NOT IN {_NO_ES_FALTA})::int AS faltas,
       COUNT(*) FILTER (WHERE estado = 'Ausente')::int AS sin_justificar,
       string_agg(DISTINCT estado, ', ') FILTER (
         WHERE estado NOT IN ('Normal','Presente','Revisar','Feriado','Ausente')) AS motivos
FROM calc
WHERE {_ES_JORNADA}
GROUP BY employee_name
HAVING COUNT(*) FILTER (WHERE estado NOT IN {_NO_ES_FALTA}) > 0
ORDER BY faltas DESC, sin_justificar DESC
LIMIT 15
""",
        d1, d2,
    )
    if not filas:
        return f"No hay días no trabajados registrados en {etiqueta}."
    out = [f"Días no trabajados por persona — {etiqueta}:", ""]
    for i, f in enumerate(filas, start=1):
        det = []
        if f["sin_justificar"]:
            det.append(f"{f['sin_justificar']} sin justificar")
        if f["motivos"]:
            det.append(f["motivos"].lower())
        sufijo = f" ({'; '.join(det)})" if det else ""
        out.append(f"{i}. {f['employee_name']}: {_dias(f['faltas'])}{sufijo}")
    out.append("")
    out.append('No cuento los días "Revisar" (fichada incompleta) ni los feriados.')
    return "\n".join(out)


async def _reporte_extras(
    d1: dt.date, d2: dt.date, etiqueta: str, personas: list[tuple[str, str]]
) -> str:
    """Horas extra = lo trabajado por encima del tope del día. En un sábado sin
    horario asignado o en un feriado el tope es 0, así que TODO lo de ese día es
    extra — mismo criterio que la pestaña Horas Extras de /rrhh."""
    if personas:
        emp_no, nombre = personas[0]
        filas = await _consultar(
            _base(_FILTRO_EMP) + """
SELECT SUM(GREATEST(0, neto - tope))::int AS extra_min,
       COUNT(*) FILTER (WHERE neto > tope)::int AS dias
FROM calc
""",
            d1, d2, emp_no,
        )
        extra = filas[0]["extra_min"] or 0
        if not extra:
            return f"{nombre} no hizo horas extra en {etiqueta}."
        return (
            f"{nombre} — {etiqueta}: {_horas(extra)} extra en "
            f"{_dias(filas[0]['dias'])} con exceso sobre la jornada."
        )

    filas = await _consultar(
        _base() + """
SELECT employee_name, departamento,
       SUM(GREATEST(0, neto - tope))::int AS extra_min,
       COUNT(*) FILTER (WHERE neto > tope)::int AS dias
FROM calc
GROUP BY employee_name, departamento
HAVING SUM(GREATEST(0, neto - tope)) > 0
ORDER BY extra_min DESC
LIMIT 15
""",
        d1, d2,
    )
    if not filas:
        return f"No hay horas extra registradas en {etiqueta}."
    total = sum(f["extra_min"] for f in filas)
    out = [f"Horas extra — {etiqueta}:", ""]
    for i, f in enumerate(filas, start=1):
        area = f" · {f['departamento']}" if f["departamento"] else ""
        out.append(
            f"{i}. {f['employee_name']}{area}: {_horas(f['extra_min'])} "
            f"({_dias(f['dias'])})"
        )
    out.append("")
    out.append(f"Total del listado: {_horas(total)}.")
    return "\n".join(out)


async def _reporte_feriados(d1: dt.date, d2: dt.date, etiqueta: str) -> str:
    """Los feriados se cargan de DOS maneras y hay que mirar las dos:

      · `asistencia.feriado` — el botón "Feriados" de /rrhh/asistencia. Es un
        día no laborable para toda la empresa (pone el tope en 0).
      · `estado_diario.estado = 'Feriado'` — cuando se marca el día persona por
        persona (así quedó, por ejemplo, el 17/08). Ese camino NO toca la tabla
        de feriados, así que un reporte que mirara sólo la tabla diría que ese
        mes no hubo ninguno.
    """
    filas = await _consultar(
        """
SELECT fecha, bool_or(en_tabla) AS en_tabla, MAX(personas)::int AS personas
FROM (
  SELECT f.fecha, true AS en_tabla, 0 AS personas
  FROM asistencia.feriado f WHERE f.fecha BETWEEN $1 AND $2
  UNION ALL
  SELECT ed.fecha, false, COUNT(*)::int
  FROM asistencia.estado_diario ed
  WHERE ed.estado = 'Feriado' AND ed.fecha BETWEEN $1 AND $2
  GROUP BY ed.fecha
) x
GROUP BY fecha
ORDER BY fecha
""",
        d1, d2,
    )
    if not filas:
        return (
            f"No hay ningún feriado registrado en {etiqueta}. Se cargan en "
            "/rrhh/asistencia (botón «Feriados») o marcando el día como "
            "«Feriado» en la grilla."
        )
    out = [f"Feriados registrados en {etiqueta}: {len(filas)}.", ""]
    for f in filas:
        como = []
        if f["en_tabla"]:
            como.append("calendario de feriados")
        if f["personas"]:
            como.append(f"marcado a {f['personas']} personas")
        out.append(f"- {_fecha_larga(f['fecha'])} ({' + '.join(como)})")
    return "\n".join(out)


async def _reporte_resumen(d1: dt.date, d2: dt.date, etiqueta: str) -> str:
    """Panorama del período — lo que se responde cuando la pregunta es del tema
    pero no encaja en ningún reporte puntual."""
    filas = await _consultar(
        _base() + f"""
SELECT COUNT(DISTINCT employee_no)::int AS personas,
       SUM(LEAST(neto, tope))::int AS min_rrhh,
       SUM(GREATEST(0, neto - tope))::int AS min_extra,
       COUNT(*) FILTER (WHERE {_ES_JORNADA})::int AS jornadas,
       COUNT(*) FILTER (WHERE {_ES_JORNADA} AND estado NOT IN {_NO_ES_FALTA})::int AS faltas,
       COUNT(*) FILTER (WHERE {_ES_JORNADA} AND estado = 'Ausente')::int AS sin_justificar,
       COUNT(*) FILTER (WHERE {_ES_JORNADA} AND estado = 'Revisar')::int AS revisar
FROM calc
""",
        d1, d2,
    )
    r = filas[0]
    if not r["jornadas"]:
        return f"No hay jornadas hábiles con datos en {etiqueta}."
    pct = 100 * (r["faltas"] or 0) / r["jornadas"]
    return "\n".join([
        f"Asistencia — {etiqueta} ({r['personas']} personas activas):",
        "",
        f"- Horas trabajadas (con tope): {_horas(r['min_rrhh'] or 0)}",
        f"- Horas extra: {_horas(r['min_extra'] or 0)}",
        f"- Días no trabajados: {r['faltas']} sobre {r['jornadas']} jornadas hábiles "
        f"({pct:.1f}%)".replace(".", ","),
        f"- De esos, sin justificar: {r['sin_justificar']}",
        f"- Fichadas incompletas a corregir (\"Revisar\"): {r['revisar']}",
        "",
        "Preguntame por una persona, por los feriados o por las horas extra si "
        "querés el detalle.",
    ])


# ── Punto de entrada ─────────────────────────────────────────────────────────

async def responder_rrhh(mensaje: str) -> str:
    """Intent "rrhh". El permiso ya lo validó `rrhh_node` (ver graph_state y
    vicki_web/lib/rrhh/vickiRrhhAcceso.ts): acá se asume habilitado.

    Nunca levanta excepción hacia afuera — cualquier falla vuelve como un
    mensaje que explica qué pasó, mismo criterio que `responder_ventas`."""
    hoy = dt.date.today()
    desde, hasta_excl, etiqueta = _parsear_rango(mensaje, hoy)
    d1, d2 = desde, _hasta_inclusivo(hasta_excl, hoy)
    if d2 < d1:
        return f"El período que entendí ({etiqueta}) todavía no empezó — no tengo datos."

    try:
        personas = _detectar_personas(mensaje, await _catalogo_empleados())
    except _FalloBase:
        return _MSG_FALLO_BASE
    except Exception:
        log.exception("no pude leer el legajo — sigo sin detectar a la persona")
        personas = []

    try:
        # Orden a propósito: "feriados" antes que "faltas" porque un feriado es
        # justamente un día no trabajado y las dos regex matchean la misma
        # pregunta; y la persona nombrada gana sobre el ranking general.
        if _PATRON_FERIADOS.search(mensaje) and not personas:
            return await _reporte_feriados(d1, d2, etiqueta)

        if _PATRON_EXTRAS.search(mensaje):
            return await _reporte_extras(d1, d2, etiqueta, personas)

        if personas:
            if len(personas) > 1:
                nombres = ", ".join(n for _, n in personas)
                return f"¿Por cuál de estos preguntás? {nombres}"
            emp_no, nombre = personas[0]
            return await _reporte_faltas_persona(emp_no, nombre, d1, d2, etiqueta)

        # Sin persona nombrada, cualquier pregunta por faltas/ausencias se
        # responde con el ranking: sirve igual para "quién faltó más" que para
        # "cuántas faltas hubo" (el total sale de la misma lista).
        if _PATRON_FALTAS.search(mensaje):
            return await _reporte_ranking_faltas(d1, d2, etiqueta)

        return await _reporte_resumen(d1, d2, etiqueta)
    except _FalloBase:
        return _MSG_FALLO_BASE
