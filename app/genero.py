"""Sexo probable a partir del nombre de pila.

Por qué existe: la búsqueda de CVs es similitud de texto contra Qdrant y no
tiene ningún campo `sexo` en el payload (la ingesta de `vicki_mail` no lo
extrae). Cuando el reclutador pide "perfiles femeninos", el recorte lo hacía
el modelo leyendo los CVs, y el modelo terminaba listando varones igual
("no hay candidatas, pero te detallo estos cinco") o deduciendo en voz baja
("sexo no especificado, pero no se indica femenino").

Con esto el recorte lo hace el CÓDIGO antes de armar el prompt: los que no
cumplen NO entran al contexto, así que el modelo no puede listarlos.

Es una DEDUCCIÓN por el nombre de pila, no un dato del CV — quien la use
tiene que decirlo así (ver prompts.py::GENERO_FILTRADO_RULES). Cuando el
nombre no alcanza se devuelve None ("indeterminado") y esa persona se muestra
igual, rotulada: preferimos que el reclutador decida antes que esconder a
alguien por un nombre raro o por un CV cargado sin nombre.
"""
from __future__ import annotations

import re
import unicodedata

# Nombres de pila frecuentes en Argentina. No pretende ser exhaustivo: lo que
# no está cae en "indeterminado" y se muestra igual, que es el error barato.
_F = {
    "abigail", "adriana", "agostina", "agustina", "aida", "aixa", "alba",
    "alejandra", "alexa", "alicia", "alma", "amalia", "amanda", "ana",
    "anabel", "anabella", "analia", "andrea", "angela", "angeles", "anita",
    "antonella", "antonia", "araceli", "ariadna", "ayelen", "barbara",
    "beatriz", "belen", "berta", "bettina", "bianca", "blanca", "brenda",
    "brisa", "camila", "candela", "candelaria", "carina", "carla", "carlota",
    "carmen", "carolina", "catalina", "cecilia", "celeste", "celia", "chiara",
    "cintia", "clara", "claudia", "constanza", "cristina", "daiana", "daniela",
    "debora", "delfina", "diana", "dolores", "elba", "elena", "eliana",
    "elisa", "elizabeth", "eloisa", "elsa", "emilce", "emilia", "emma",
    "erica", "ernestina", "estefania", "estela", "ester", "eugenia", "eva",
    "evangelina", "evelyn", "ezequiela", "fabiana", "fatima", "fernanda",
    "flavia", "florencia", "fiorella", "francisca", "gabriela", "gimena",
    "gisela", "gladys", "gloria", "graciela", "griselda", "guadalupe",
    "guillermina", "haydee", "hilda", "iara", "ida", "ines", "ingrid",
    "irene", "irma", "isabel", "isabella", "ivana", "jazmin", "jesica",
    "jessica", "johana", "josefa", "josefina", "juana", "judith", "julia",
    "julieta", "karen", "karina", "laura", "lautara", "lea", "leila", "lena",
    "leonela", "leticia", "lia", "liliana", "lorena", "lourdes", "lucia",
    "luciana", "lucila", "luisa", "lujan", "luna", "luz", "macarena",
    "magali", "maia", "maira", "malena", "manuela", "mara", "marcela",
    "margarita", "maria", "mariana", "marianela", "maricel", "mariel",
    "mariela", "marina", "marisa", "marisol", "marta", "martina",
    "maria", "matilde", "maylen", "melania", "melina", "melisa", "melody",
    "mercedes", "micaela", "mila", "milagros", "mirna", "mirta", "monica",
    "morena", "nadia", "nahiara", "nancy", "naomi", "natalia", "natasha",
    "nelida", "nerina", "nicole", "nidia", "nilda", "noelia", "noemi",
    "nora", "norma", "nuria", "ofelia", "olga", "oriana", "pamela",
    "paola", "patricia", "paula", "paulina", "paz", "perla", "pia", "pilar",
    "priscila", "raquel", "rebeca", "regina", "renata", "rita", "rocio",
    "romina", "rosa", "rosana", "rosario", "roxana", "ruth", "sabrina",
    "salome", "samanta", "samantha", "sandra", "sara", "selena", "selva",
    "serena", "sofia", "sol", "solana", "soledad", "sonia", "stefania",
    "silvana", "silvia", "susana", "tamara", "tatiana", "teresa", "tiziana",
    "trinidad", "valentina", "valeria", "vanesa", "vanessa", "veronica",
    "vicenta", "victoria", "violeta", "virginia", "viviana", "wanda",
    "xiomara", "yamila", "yanina", "yesica", "yohana", "zoe", "zulema",
}

_M = {
    "abel", "abelardo", "abraham", "adolfo", "adrian", "agustin", "aldo",
    "alberto", "alejandro", "alexis", "alfredo", "alan", "alvaro", "amilcar",
    "anibal", "andres", "angel", "antonio", "ariel", "arnaldo", "arturo",
    "atilio", "augusto", "aurelio", "axel", "bautista", "benjamin",
    "bernardo", "braian", "brian", "bruno", "camilo", "carlos", "cesar",
    "cristian", "cristobal", "claudio", "clemente", "conrado", "damian",
    "daniel", "dante", "dario", "david", "delfor", "diego", "dylan",
    "domingo", "eduardo", "edgardo", "efrain", "elias", "eliseo", "emanuel",
    "emiliano", "emilio", "emmanuel", "enrique", "enzo", "eric", "erik",
    "ernesto", "esteban", "eugenio", "ezequiel", "fabian", "fabio",
    "facundo", "federico", "felipe", "felix", "fermin", "fernando",
    "fidel", "francisco", "franco", "gabriel", "gaston", "genaro", "geronimo",
    "gerardo", "german", "gilberto", "gonzalo", "gregorio", "guido",
    "guillermo", "gustavo", "hector", "heraldo", "hernan", "hipolito",
    "horacio", "hugo", "humberto", "ian", "ignacio", "ismael", "isidro",
    "ivan", "jacinto", "jaime", "javier", "jeremias", "jesus", "joaquin",
    "joel", "jonatan", "jonathan", "jorge", "jose", "juan", "julian",
    "julio", "kevin", "lautaro", "leandro", "leonardo", "leonel", "lisandro",
    "lorenzo", "lucas", "luca", "luciano", "lucio", "luis", "maico",
    "manuel", "marcelo", "marco", "marcos", "mariano", "mario", "martin",
    "mateo", "matias", "mauricio", "mauro", "maximiliano", "maximo",
    "miguel", "milton", "mirko", "moises", "nahuel", "napoleon", "nelson",
    "nestor", "nicolas", "norberto", "octavio", "omar", "orlando", "oscar",
    "osvaldo", "pablo", "patricio", "pedro", "raul", "ramiro", "ramon",
    "renzo", "ricardo", "roberto", "rodolfo", "rodrigo", "rogelio", "rolando",
    "roman", "ruben", "rufino", "sandro", "santiago", "santino", "saul",
    "sebastian", "sergio", "silvio", "simon", "teodoro", "thiago", "tiziano",
    "tomas", "tobias", "ulises", "valentin", "vicente", "victor", "walter",
    "wenceslao", "wilson", "yamil", "zacarias",
}

# Ambiguos de verdad en Argentina: no se deducen, van como indeterminado.
_AMBIGUO = {
    "alex", "cruz", "daysi", "guadalupe", "jesus", "jordan", "milagro",
    "noel", "renee", "reny", "sam", "yael", "ale", "cris", "gabi",
}

# Partículas y ruido que aparecen en el campo nombre de la ingesta.
_RUIDO = {
    "de", "del", "la", "las", "los", "da", "di", "van", "von", "mac", "mc",
    "san", "santa", "sin", "nombre", "cv", "curriculum", "sr", "sra", "srta",
    "documento", "na",
}

# Apellidos frecuentes que la heurística morfológica leería como nombre de pila
# ("Cabrera" termina en -a → F). Sólo importan cuando la ingesta cargó el campo
# al revés ("Cabrera Marcelino"): se saltean y se evalúa el token siguiente.
# No entran los que TAMBIÉN son nombre de pila (Luna, Paz, Rosa, Sol, Franco,
# Bruno, Marco): ahí el apellido y el nombre son la misma palabra y saltearlo
# perdería candidatos de verdad.
_APELLIDOS = {
    "acosta", "aguilera", "alvarado", "aranda", "arriola", "ayala", "barrera",
    "benitez", "cabrera", "carrizo", "castro", "ceballos", "cordoba",
    "cornejo", "correa", "costa", "cuello", "cuesta", "delgado", "escudero",
    "espinosa", "ferreyra", "figueroa", "gallardo", "gauna", "gimenez",
    "gomez", "gonzalez", "guzman", "herrera", "hidalgo", "ledesma", "leiva",
    "maldonado", "mansilla", "mendieta", "miranda", "molina", "montero",
    "mora", "moreno", "moyano", "navarro", "nieto", "ochoa", "ojeda",
    "olmedo", "oliva", "otero", "pardo", "peralta", "pereira", "pereyra",
    "pinto", "prieto", "quiroga", "ramirez", "rivera", "rodriguez", "romero",
    "russo", "saavedra", "salvo", "sanchez", "silva", "soto", "sosa",
    "taborda", "toledo", "urbina", "velazco", "vera", "villarreal", "zabala",
    "zapata",
}

# Varones cuyo nombre termina en -a: la heurística morfológica los daría F.
_M_TERMINA_EN_A = {"luca", "bautista", "cosme", "elisha", "iasha", "nicola"}

# Nombres de pila que en Argentina son MÁS usados como apellido (Paz, Luna,
# Rosa, Mercedes). Valen sólo si están PRIMEROS: en "Alejo Paz" el Paz es el
# apellido y contarlo daba "F" para un varón.
_SOLO_PRIMER_TOKEN = {
    "alma", "angeles", "asuncion", "carmen", "cruz", "dolores", "ida", "iara",
    "lea", "luna", "mercedes", "mila", "milagro", "milagros", "nieves", "paz",
    "perla", "rosa", "selva", "sol", "socorro", "vega",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def _norm(txt: str) -> str:
    s = unicodedata.normalize("NFKD", (txt or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _sexo_token(tok: str) -> str | None:
    if tok in _AMBIGUO or tok in _RUIDO or len(tok) < 3:
        return None
    if tok in _F:
        return "F"
    if tok in _M:
        return "M"
    # Heurística morfológica, sólo para lo que el diccionario no conoce y sólo
    # en los dos finales que en castellano son fiables. Todo lo demás queda
    # indeterminado a propósito: un falso "M" esconde a una candidata.
    if tok.endswith("a") and tok not in _M_TERMINA_EN_A:
        return "F"
    if tok.endswith("o") and not tok.endswith("io"):
        return "M"
    return None


def sexo_por_nombre(nombre: str) -> str | None:
    """"F", "M" o None (no se puede deducir del nombre).

    Manda el PRIMER token: el nombre viene de la metadata como `nombre` +
    `apellido`, así que el primero es el nombre de pila, y en los compuestos
    del castellano el primero gobierna igual ("José María" es varón, "María
    José" es mujer). El resto del nombre se mira sólo si el primero no dice
    nada — si no, un apellido como "Paz" o "Luna" daba F en "Alejo Paz".
    Los apellidos que la morfología leería como nombre ("Cabrera") se saltean:
    aparecen primeros cuando la ingesta cargó el campo al revés.
    """
    toks = [t for t in _TOKEN_RE.findall(_norm(nombre))
            if t not in _RUIDO and len(t) >= 3]
    utiles = [t for t in toks if t not in _APELLIDOS] or toks
    if not utiles:
        return None
    # 1) el primero, con diccionario y morfología
    if utiles[0] in _AMBIGUO:
        return None
    if utiles[0] in _F:
        return "F"
    if utiles[0] in _M:
        return "M"
    # 2) el DICCIONARIO en el resto le gana a la morfología del primero: un
    #    apellido que termina en -a o en -o no dice nada, un nombre de pila sí
    #    ("Valenzuela Diego" es varón, no mujer por el -a de Valenzuela).
    for t in utiles[1:]:
        if t in _SOLO_PRIMER_TOKEN:
            continue
        if t in _AMBIGUO:
            return None
        if t in _F:
            return "F"
        if t in _M:
            return "M"
    # 3) recién ahora la morfología del primero
    return _sexo_token(utiles[0])


def cumple_sexo(nombre: str, pedido: str | None) -> tuple[bool, str | None]:
    """(¿entra al contexto?, sexo deducido). `pedido` es "F", "M" o None.

    Los indeterminados ENTRAN: es un dato que el CV no tiene, y esconder a
    alguien por eso es peor que mostrarlo rotulado.
    """
    if not pedido:
        return True, None
    sexo = sexo_por_nombre(nombre)
    return (sexo is None or sexo == pedido), sexo
