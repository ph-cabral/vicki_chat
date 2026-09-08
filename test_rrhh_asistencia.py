"""Verificación de las partes determinísticas de asistencia_tools (sin tocar
Postgres): a quién reconoce en el mensaje y qué reporte elige.

Correr: `python vicki_chat/test_rrhh_asistencia.py`. Las rutas salen de
__file__, así que anda desde cualquier directorio.

Lo que NO cubre: el SQL. Ese se verifica contra la base real (los números se
cruzan contra /rrhh/asistencia — ver el docstring del módulo).
"""
import os, sys, types, datetime as dt

_AQUI = os.path.dirname(os.path.abspath(__file__))

# stub de app.config para importar los módulos sueltos (asistencia_tools
# importa ventas_tools, que lee config al importarse)
app = types.ModuleType("app"); app.__path__ = [os.path.join(_AQUI, "app")]
cfg = types.ModuleType("app.config")
cfg.config = types.SimpleNamespace(
    MAGNUS_SQL_URL="", MAGNUS_MCP_URL="", MAGNUS_API_TOKEN="",
    MAGNUS_MCP_TIMEOUT=20, DATABASE_URL="postgresql://x/x")
sys.modules["app"] = app; sys.modules["app.config"] = cfg
sys.path.insert(0, _AQUI)
import app.asistencia_tools as at

# Espeja la forma del legajo real (nombres y legajos reales, tomados de
# everwear.legajo): apellido + nombres, un NOMBRE de pila inconfundible
# (Vladimir) y un apellido repetido (Pereyra) que por eso NO alcanza suelto.
LEGAJO = [
    ("00000631", "Vaudagna Joan"),
    ("00000624", "Cabral Pablo Hernan"),
    ("00000040", "Boscacci Vladimir"),
    ("00000041", "Pereyra Francisco"),
    ("00000042", "Pereyra Marcelo Ariel"),
    ("00000007", "Beccaria Gerardo Diego"),
]

ok = fail = 0
def check(nombre, got, want):
    global ok, fail
    if got == want: ok += 1; print(f"  ok  {nombre}")
    else: fail += 1; print(f"  FAIL {nombre}\n       got={got!r}\n       want={want!r}")


print("\n=== _detectar_personas ===")
casos = [
    # apellido único → alcanza suelto
    ("cuantos dias falto Vaudagna", [("00000631", "Vaudagna Joan")]),
    ("cuantas horas extras hizo beccaria el mes pasado",
     [("00000007", "Beccaria Gerardo Diego")]),
    # nombre de pila inconfundible: alcanza igual que un apellido
    ("cuantas faltas tiene vladimir en junio", [("00000040", "Boscacci Vladimir")]),
    # apellido repetido → hace falta el nombre
    ("cuantos dias falto Francisco Pereyra", [("00000041", "Pereyra Francisco")]),
    # preguntas generales: nadie nombrado
    ("quien hizo horas extras el mes pasado", []),
    ("que feriados registramos en julio", []),
    ("cuantas faltas hubo este mes", []),
]
for msg, want in casos:
    got = at._detectar_personas(msg, LEGAJO)
    # "pereyra" suelto matchea a los dos: se acepta cualquier orden
    check(msg, sorted(got), sorted(want))

# "pereyra" solo (ambiguo) tiene que devolver 0 personas, no una al azar:
# ninguna de las dos partes es única y no hay dos partes en el mensaje.
check("apellido ambiguo suelto", at._detectar_personas("dias que falto pereyra", LEGAJO), [])


print("\n=== _parsear_rango + _hasta_inclusivo ===")
HOY = dt.date(2026, 9, 7)
casos_rango = [
    ("cuantos dias falto vaudagna el mes pasado",
     (dt.date(2026, 8, 1), dt.date(2026, 8, 31), "el mes pasado")),
    ("que feriados registramos en julio",
     (dt.date(2026, 7, 1), dt.date(2026, 7, 31), "julio 2026")),
    # el mes en curso se corta en HOY, no en el 30: los días futuros no tienen
    # fichadas y aparecerían como ausencia de toda la empresa
    ("quien hizo horas extras", (dt.date(2026, 9, 1), HOY, "este mes")),
]
for msg, want in casos_rango:
    d, h, etiqueta = at._parsear_rango(msg, HOY)
    check(msg, (d, at._hasta_inclusivo(h, HOY), etiqueta), want)


print("\n=== detección de reporte ===")
reportes = [
    ("que feriados registramos el mes pasado", "feriados"),
    ("quien hizo horas extras el mes pasado", "extras"),
    ("cuantas horas extra hizo vaudagna", "extras"),
    ("cuantos dias falto vaudagna", "faltas"),
    ("quien falto mas en agosto", "faltas"),
    ("como viene la asistencia este mes", "resumen"),
]
for msg, want in reportes:
    if at._PATRON_FERIADOS.search(msg):
        got = "feriados"
    elif at._PATRON_EXTRAS.search(msg):
        got = "extras"
    elif at._PATRON_FALTAS.search(msg):
        got = "faltas"
    else:
        got = "resumen"
    check(msg, got, want)

print(f"\n{ok} ok, {fail} fail")
sys.exit(1 if fail else 0)
