"""Verificación de los gates de ventas_tools (sin tocar Magnus).

Correr desde cualquier lado: `python vicki_chat/test_gates_ventas.py`. Las
rutas salen de __file__ — antes estaban hardcodeadas a una sesión vieja y el
archivo no corría en otra máquina.
"""
import os, sys, types, asyncio, datetime as dt

_AQUI = os.path.dirname(os.path.abspath(__file__))

# stub de app.config para importar el módulo suelto
app = types.ModuleType("app"); app.__path__ = [os.path.join(_AQUI, "app")]
cfg = types.ModuleType("app.config")
cfg.config = types.SimpleNamespace(
    MAGNUS_SQL_URL="http://x/sql", MAGNUS_MCP_URL="", MAGNUS_API_TOKEN="",
    MAGNUS_MCP_TIMEOUT=20)
sys.modules["app"] = app; sys.modules["app.config"] = cfg
sys.path.insert(0, _AQUI)
import app.ventas_tools as vt

# Espeja la forma del maestro real: personas con 2+ partes, agrupadores de una
# sola palabra significativa, y un apellido repetido (ROMERO) que por eso NO
# alcanza suelto.
VEND = [(797, "BLANCO JULIO"), (800, "PEREZ MARIA"), (9000, "MOSTRADORES"),
        (814, "GOMEZ CARLOS"), (18200, "ZONA CBA"),
        (799, "UBALDO ANTONIO PALENCIA ANGULO"),
        (791, "VIAJANTE ZONA ROSARIO"),
        (677, "ROMERO ESTEBAN CESAR"), (796, "ROMERO RANDOLFO")]

ok = fail = 0
def check(nombre, got, want):
    global ok, fail
    if got == want: ok += 1; print(f"  ok  {nombre}")
    else: fail += 1; print(f"  FAIL {nombre}\n       got={got!r}\n       want={want!r}")

print("\n=== _detectar_vendedor_mencionado ===")
casos_v = [
    ("cuanto vendio Julio Blanco en agosto", (797, "BLANCO JULIO")),
    ("mostrame las ventas del vendedor 797", (797, "BLANCO JULIO")),
    ("ventas del vendedor codigo 800", (800, "PEREZ MARIA")),
    ("como vengo este mes", None),
    ("cuanto facture en agosto", None),
    ("cuanto vendi de bulones el mes pasado", None),
    ("ranking de vendedores de septiembre", None),
    ("cuanto vendio maria perez", (800, "PEREZ MARIA")),
    ("ventas del vendedor gomez carlos", (814, "GOMEZ CARLOS")),
    # parte suelta e inconfundible: el caso que antes no filtraba nada
    ("ubaldo ha vendido articulos de la linea buloneria?", (799, "UBALDO ANTONIO PALENCIA ANGULO")),
    ("ventas de gomez", (814, "GOMEZ CARLOS")),
    # apellido repetido en el maestro (677 y 796): suelto no identifica a nadie
    ("cuanto vendio romero", None),
    ("cuanto vendio romero randolfo", (796, "ROMERO RANDOLFO")),
    # razón social: vuelve a exigir dos partes, no se lee como el vendedor
    ("ventas de ferreteria gomez srl", None),
    # agrupador de una sola palabra: no se dispara con la palabra suelta
    ("como vienen mis ventas de la zona rosario", None),
    ("cuanto vendio el vendedor rosario", (791, "VIAJANTE ZONA ROSARIO")),
]
for msg, want in casos_v:
    check(msg, vt._detectar_vendedor_mencionado(msg, VEND), want)

print("\n=== _detectar_cliente_pedido ===")
casos_c = [
    ("cuanto le vendi al cliente Rossi este mes", ["Rossi"]),
    ("que compro el cliente 4521 en agosto", ["4521"]),
    ('ventas del cliente "Ferreteria del Centro"', ["Ferreteria", "Centro"]),
    ("mis mejores clientes de agosto", None),
    ("que cliente me compro mas", None),
    ("como vengo este mes", None),
    ("cuanto le vendi de bulones al cliente Rossi", ["Rossi"]),
]
for msg, want in casos_c:
    check(msg, vt._detectar_cliente_pedido(msg), want)

print("\n=== _literal_like (saneo) ===")
check("comilla", vt._literal_like("O'Brien"), "O Brien")
check("comentario", vt._literal_like("x'; DROP TABLE--"), "x DROP TABLE")
check("bloque", vt._literal_like("a/*b*/c"), "a b c")
check("largo", len(vt._literal_like("z" * 200)), 40)

print("\n=== SQL generado ===")
sql = vt._sql_buscar_cliente(["Rossi"], 797)
check("cartera en el SELECT", "cart.CodCliente = c.CodCliente" in sql, True)
check("vendedor interpolado", "VendedorCodigo = 797" in sql, True)
sql_admin = vt._sql_buscar_cliente(["Rossi"], None)
check("admin sin JOIN cartera", "cart.CodCliente" in sql_admin, False)
sql_fact = vt._sql_facturacion_cliente(dt.date(2026, 8, 1), dt.date(2026, 9, 1), 4521, 797)
check("cliente + vendedor en el WHERE",
      "vc.CodCliente = 4521" in sql_fact and "vc.Vendedor = 797" in sql_fact, True)
check("solo SELECT", all(s.strip().upper().startswith("SELECT")
                         for s in (sql, sql_admin, sql_fact)), True)

print("\n=== gates de responder_ventas (magnus mockeado) ===")
class Mock:
    def __init__(self): self.sqls = []
    async def __call__(self, sql):
        self.sqls.append(sql)
        if "FROM MAGNUS_SITD.dbo.Vendedores ORDER BY 1" in sql:
            return "VendedorCodigo\tnombre\n" + "\n".join(f"{c}\t{n}" for c, n in VEND) + "\n(5 filas)"
        if "FROM Stk_Nivel1" in sql:
            return "Nivel1\tDetalle\n1\tBULONERÍA\n(1 filas)"
        if "TOP 5 c.CodCliente" in sql:              # búsqueda en cartera
            return "CodCliente\tNombre\n(0 filas)"
        if "TOP 3 LTRIM" in sql:                     # existe fuera de la cartera
            return "Nombre\nFERRETERIA ROSSI SRL\n(1 filas)"
        return "Anio\tMes\tImporte\tComprobantes\n2026\t8\t1000\t3\n(1 filas)"

async def main():
    vt._vendedores_cache.update({"t": 0.0, "datos": []})
    vt._lineas_cache.update({"t": 0.0, "datos": []})
    m = Mock(); vt._ejecutar_sql = m

    r = await vt.responder_ventas("cuanto vendio Julio Blanco en agosto", 800, False)
    check("no-admin pide otro vendedor", r, vt._MSG_OTRO_VENDEDOR)
    check("no consultó facturación", any("Ven_CompCabecera" in s and "Neto" in s for s in m.sqls), False)

    m.sqls.clear()
    r = await vt.responder_ventas("cuanto le vendi al cliente Rossi", 800, False)
    check("cliente ajeno", "no está en tu cartera" in r, True)
    check("la búsqueda fue con cartera",
          any("VendedorCodigo = 800" in s for s in m.sqls), True)
    check("no trajo importes del ajeno",
          any("SUM(" in s and "CodCliente = " in s for s in m.sqls), False)

    m.sqls.clear()
    r = await vt.responder_ventas("cuanto facture en agosto", 800, False)
    check("lo propio sí responde", "Facturación de tu cartera (vendedor 800)" in r, True)
    check("filtró por su código", any("Vendedor = 800" in s for s in m.sqls), True)

    m.sqls.clear()
    r = await vt.responder_ventas("cuanto vendio Julio Blanco en agosto", None, True)
    check("admin ve a otro vendedor", "BLANCO JULIO (cód. 797)" in r, True)
    check("filtró por 797", any("Vendedor = 797" in s for s in m.sqls), True)

    m.sqls.clear()
    r = await vt.responder_ventas("ranking de vendedores de agosto", 800, False)
    check("ranking sigue vedado al no-admin", "no te lo puedo mostrar" in r, True)

    # vendedor + línea en la misma pregunta (el caso que devolvía el total de
    # toda la empresa como si fuera de la persona nombrada)
    m.sqls.clear()
    r = await vt.responder_ventas("ubaldo ha vendido articulos de buloneria?", None, True)
    check("admin: vendedor + línea", "UBALDO ANTONIO PALENCIA ANGULO (cód. 799)" in r, True)
    check("etiqueta con la línea", "línea BULONERÍA" in r, True)
    check("no dice toda la empresa", "toda la empresa" in r, False)
    check("SQL filtró por 799 y por Nivel1",
          any("vc.Vendedor = 799" in s and "ap.Nivel1 IN (1)" in s for s in m.sqls), True)

    m.sqls.clear()
    r = await vt.responder_ventas("ubaldo ha vendido buloneria?", 800, False)
    check("no-admin no puede preguntar por ubaldo", r, vt._MSG_OTRO_VENDEDOR)

    check("singular de comprobante", vt._comps(1), "1 comprobante")
    check("plural de comprobante", vt._comps(2), "2 comprobantes")

asyncio.run(main())
print(f"\n{ok} ok, {fail} fail")
sys.exit(1 if fail else 0)
