import pandas as pd, numpy as np, random
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

ARCHIVO = "teams_match_features.csv"
FECHA_INICIO = "2010-01-01"
FECHA_CORTE = "2022-12-31"
N_SIMULACIONES = 100000
SEMILLA = 42

random.seed(SEMILLA)
np.random.seed(SEMILLA)

# ==========================================================
# CARGAR DATASET
# ==========================================================

df = pd.read_csv(ARCHIVO)
df["_date"] = pd.to_datetime(df["_date"])
df = df[df["_date"] >= pd.to_datetime(FECHA_INICIO)].copy()
df = df.sort_values("_date").reset_index(drop=True)


# ==========================================================
# VARIABLE OBJETIVO
# ==========================================================

def obtener_resultado(row):
    if row["home_goals"] > row["away_goals"]:
        return 0
    elif row["home_goals"] == row["away_goals"]:
        return 1
    else:
        return 2

df["resultado"] = df.apply(obtener_resultado, axis=1)

# ==========================================================
# VARIABLES DEL MODELO
# ==========================================================

variables = [
    "elo_diff", "overall_diff", "attack_diff", "defense_diff",
    "home_form_scored", "away_form_scored",
    "home_form_conceded", "away_form_conceded",
    "home_form_win_rate", "away_form_win_rate",
    "is_neutral", "is_world_cup", "is_continental",
]

# ==========================================================
# LIMPIAR DATOS
# ==========================================================

for variable in variables:
    df[variable] = pd.to_numeric(df[variable], errors="coerce")

df[variables] = df[variables].fillna(df[variables].median())

# ==========================================================
# TRAIN / TEST TEMPORAL
# ==========================================================

train = df[df["_date"] <= pd.to_datetime(FECHA_CORTE)].copy()
test = df[df["_date"] > pd.to_datetime(FECHA_CORTE)].copy()
X_train = train[variables]
X_test = test[variables]
y_train = train["resultado"]
y_test = test["resultado"]


# ==========================================================
# MODELO DE RESULTADO
# ==========================================================

modelo_resultado = RandomForestClassifier(
    n_estimators=700,
    max_depth=14,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEMILLA,
    n_jobs=-1
)

modelo_resultado.fit(
    X_train,
    y_train
)


# ==========================================================
# EVALUACIÓN
# ==========================================================

predicciones = modelo_resultado.predict(X_test)
probabilidades_test = modelo_resultado.predict_proba(X_test)

print()
print("==========================================")
print("EVALUACIÓN DEL MODELO")
print("==========================================")
print()
print("Partidos entrenamiento:", len(train))
print("Partidos prueba:", len(test))
print()
print("Accuracy:", round(accuracy_score(y_test, predicciones), 4))
print()
print("Log Loss:", round(log_loss(y_test, probabilidades_test, labels=[0, 1, 2]), 4))
print()
print("Informe de clasificación:")
print(classification_report(y_test, predicciones, target_names=["Victoria local", "Empate", "Victoria visitante"]))
print()
print("Matriz de confusión:")
print(confusion_matrix(y_test, predicciones))


# ==========================================================
# IMPORTANCIA DE VARIABLES
# ==========================================================

importancias = pd.DataFrame({"Variable": variables, "Importancia": modelo_resultado.feature_importances_})
importancias = importancias.sort_values("Importancia", ascending=False)

print()
print("==========================================")
print("IMPORTANCIA DE LAS VARIABLES")
print("==========================================")
print()
print(importancias.to_string(index=False))


# ==========================================================
# MODELOS DE GOLES
# ==========================================================

variables_goles = [
    "elo_diff", "overall_diff", "attack_diff", "defense_diff",
    "home_form_scored", "away_form_scored",
    "home_form_conceded", "away_form_conceded",
    "home_form_win_rate", "away_form_win_rate",
    "is_neutral", "is_world_cup", "is_continental",
]


modelo_goles_local = RandomForestRegressor(

    n_estimators=500,

    max_depth=14,

    min_samples_split=8,

    min_samples_leaf=3,

    max_features="sqrt",

    random_state=SEMILLA,

    n_jobs=-1

)


modelo_goles_visitante = RandomForestRegressor(

    n_estimators=500,

    max_depth=14,

    min_samples_split=8,

    min_samples_leaf=3,

    max_features="sqrt",

    random_state=SEMILLA,

    n_jobs=-1

)


modelo_goles_local.fit(

    train[variables_goles],

    train["home_goals"]

)


modelo_goles_visitante.fit(

    train[variables_goles],

    train["away_goals"]

)


# ==========================================================
# ÚLTIMAS ESTADÍSTICAS DE CADA EQUIPO
# ==========================================================

equipos = {}


for _, fila in df.sort_values("_date").iterrows():

    home = fila["_home_team"]

    away = fila["_away_team"]


    equipos[home] = {

        "elo": fila["home_elo"],

        "overall": fila["home_avg_overall"],

        "attack": fila["home_avg_attack"],

        "defense": fila["home_avg_defense"],

        "scored": fila["home_form_scored"],

        "conceded": fila["home_form_conceded"],

        "winrate": fila["home_form_win_rate"]

    }


    equipos[away] = {

        "elo": fila["away_elo"],

        "overall": fila["away_avg_overall"],

        "attack": fila["away_avg_attack"],

        "defense": fila["away_avg_defense"],

        "scored": fila["away_form_scored"],

        "conceded": fila["away_form_conceded"],

        "winrate": fila["away_form_win_rate"]

    }


# ==========================================================
# EQUIVALENCIAS DE NOMBRES
# ==========================================================

equivalencias = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
}


mapa = {}

for antiguo, nuevo in equivalencias.items():

    mapa[nuevo] = antiguo


# ==========================================================
# NORMALIZAR NOMBRE
# ==========================================================

def nom_dataset(equipo):

    if equipo in equipos:

        return equipo

    if equipo in mapa:

        return mapa[equipo]

    return equipo


# ==========================================================
# COMPROBAR EQUIPO
# ==========================================================

def comprobar_equipo(equipo):

    equipo_dataset = nom_dataset(equipo)

    if equipo_dataset not in equipos:

        raise ValueError(
            f"No hay datos para el equipo: {equipo}"
        )

    return equipo_dataset


# ==========================================================
# CREAR DATOS DE UN PARTIDO
# ==========================================================

def datos_partido(equipo1, equipo2):

    equipo1 = comprobar_equipo(equipo1)

    equipo2 = comprobar_equipo(equipo2)

    t1 = equipos[equipo1]

    t2 = equipos[equipo2]


    datos = pd.DataFrame([{

        "elo_diff":
            t1["elo"] - t2["elo"],

        "overall_diff":
            t1["overall"] - t2["overall"],

        "attack_diff":
            t1["attack"] - t2["attack"],

        "defense_diff":
            t1["defense"] - t2["defense"],

        "home_form_scored":
            t1["scored"],

        "away_form_scored":
            t2["scored"],

        "home_form_conceded":
            t1["conceded"],

        "away_form_conceded":
            t2["conceded"],

        "home_form_win_rate":
            t1["winrate"],

        "away_form_win_rate":
            t2["winrate"],

        "is_neutral":
            1,

        "is_world_cup":
            1,

        "is_continental":
            0

    }])


    return datos


# ==========================================================
# PROBABILIDADES DEL PARTIDO
# ==========================================================

def probabilidades(equipo1, equipo2):

    datos = datos_partido(
        equipo1,
        equipo2
    )


    probabilidades = (
        modelo_resultado
        .predict_proba(datos)[0]
    )


    return {

        "victoria_1":
            probabilidades[0],

        "empate":
            probabilidades[1],

        "victoria_2":
            probabilidades[2]

    }


# ==========================================================
# GOLES ESPERADOS
# ==========================================================

def goles_esperados(equipo1, equipo2):

    datos = datos_partido(
        equipo1,
        equipo2
    )


    goles1 = modelo_goles_local.predict(
        datos
    )[0]


    goles2 = modelo_goles_visitante.predict(
        datos
    )[0]


    goles1 = max(
        0.15,
        min(goles1, 5.0)
    )


    goles2 = max(
        0.15,
        min(goles2, 5.0)
    )


    return goles1, goles2


# ==========================================================
# SIMULAR MARCADOR
# ==========================================================

def simular_marcador(equipo1, equipo2, eliminatoria=False):
    p = probabilidades(equipo1, equipo2)
    lambda1, lambda2 = goles_esperados(equipo1, equipo2)
    ratio_1 = p["victoria_1"] / max(p["victoria_2"], 0.05)
    ratio_2 = p["victoria_2"] / max(p["victoria_1"], 0.05)

    if ratio_1 > 1:
        lambda1 *= 1 + 0.10 * min(ratio_1 - 1, 1)
    if ratio_2 > 1:
        lambda2 *= 1 + 0.10 * min(ratio_2 - 1, 1)

    lambda1 = min(lambda1, 5)
    lambda2 = min(lambda2, 5)

    goles1 = np.random.poisson(lambda1)
    goles2 = np.random.poisson(lambda2)

    # En fase de grupos los empates son válidos.
    if not eliminatoria:
        return goles1, goles2

    # En eliminatorias no puede acabar empatado.
    # Se resuelve mediante penaltis.
    if goles1 != goles2:
        return goles1, goles2

    # Penaltis
    if random.random() < p["victoria_1"] / (p["victoria_1"] + p["victoria_2"]):
        return goles1 + 1, goles2
    return goles1, goles2 + 1


# ==========================================================
# GRUPOS DEL MUNDIAL 2026
# ==========================================================

GRUPOS = {

    "A": [
        "Mexico",
        "South Africa",
        "Korea Republic",
        "Czechia"
    ],

    "B": [
        "Canada",
        "Bosnia and Herzegovina",
        "Qatar",
        "Switzerland"
    ],

    "C": [
        "Brazil",
        "Morocco",
        "Haiti",
        "Scotland"
    ],

    "D": [
        "USA",
        "Paraguay",
        "Australia",
        "Türkiye"
    ],

    "E": [
        "Germany",
        "Jamaica",
        "Côte d'Ivoire",
        "Ecuador"
    ],

    "F": [
        "Netherlands",
        "Japan",
        "Sweden",
        "Tunisia"
    ],

    "G": [
        "Belgium",
        "Egypt",
        "IR Iran",
        "New Zealand"
    ],

    "H": [
        "Spain",
        "Greece",
        "Saudi Arabia",
        "Uruguay"
    ],

    "I": [
        "France",
        "Senegal",
        "Iraq",
        "Norway"
    ],

    "J": [
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan"
    ],

    "K": [
        "Portugal",
        "Congo DR",
        "Uzbekistan",
        "Colombia"
    ],

    "L": [
        "England",
        "Croatia",
        "Ghana",
        "Panama"
    ]

}


# ==========================================================
# SIMULAR GRUPO
# ==========================================================

def simular_grupo(grupo):

    tabla = {}


    for equipo in grupo:

        tabla[equipo] = {

            "pts": 0,

            "gf": 0,

            "gc": 0

        }


    for i in range(
        len(grupo)
    ):

        for j in range(
            i + 1,
            len(grupo)
        ):

            e1 = grupo[i]

            e2 = grupo[j]


            g1, g2 = simular_marcador(
                e1,
                e2,
                False
            )


            tabla[e1]["gf"] += g1

            tabla[e1]["gc"] += g2

            tabla[e2]["gf"] += g2

            tabla[e2]["gc"] += g1


            if g1 > g2:

                tabla[e1]["pts"] += 3

            elif g2 > g1:

                tabla[e2]["pts"] += 3

            else:

                tabla[e1]["pts"] += 1

                tabla[e2]["pts"] += 1


    clasificacion = sorted(

        tabla.items(),

        key=lambda x: (

            x[1]["pts"],

            x[1]["gf"]
            -
            x[1]["gc"],

            x[1]["gf"]

        ),

        reverse=True

    )


    return clasificacion


# ==========================================================
# FASE DE GRUPOS
# ==========================================================

def fase_grupos(
    mostrar=False
):

    clasificados = []

    terceros = []

    posiciones = {}


    for letra, grupo in GRUPOS.items():

        resultado = simular_grupo(
            grupo
        )


        posiciones[letra] = resultado


        if mostrar:

            print()

            print(
                "================================"
            )

            print(
                f"GRUPO {letra}"
            )

            print(
                "================================"
            )


            for posicion, (
                equipo,
                datos
            ) in enumerate(
                resultado,
                start=1
            ):

                diferencia = (
                    datos["gf"]
                    -
                    datos["gc"]
                )


                print(

                    f"{posicion}. "
                    f"{equipo} | "
                    f"{datos['pts']} pts | "
                    f"DG {diferencia} | "
                    f"GF {datos['gf']}"

                )


        clasificados.append(
            resultado[0][0]
        )

        clasificados.append(
            resultado[1][0]
        )


        terceros.append({

            "grupo": letra,

            "equipo":
                resultado[2][0],

            "pts":
                resultado[2][1]["pts"],

            "dg":
                resultado[2][1]["gf"]
                -
                resultado[2][1]["gc"],

            "gf":
                resultado[2][1]["gf"]

        })


    # ======================================================
    # MEJORES TERCEROS
    # ======================================================

    terceros = sorted(

        terceros,

        key=lambda x: (

            x["pts"],

            x["dg"],

            x["gf"]

        ),

        reverse=True

    )


    mejores_terceros = terceros[:8]


    for tercero in mejores_terceros:

        clasificados.append(
            tercero["equipo"]
        )


    if mostrar:

        print()

        print(
            "================================"
        )

        print(
            "MEJORES TERCEROS"
        )

        print(
            "================================"
        )


        for tercero in mejores_terceros:

            print(

                f"{tercero['grupo']} - "
                f"{tercero['equipo']} | "
                f"{tercero['pts']} pts | "
                f"DG {tercero['dg']}"

            )


    return {

        "clasificados":
            clasificados,

        "posiciones":
            posiciones,

        "terceros":
            mejores_terceros

    }


# ==========================================================
# ASIGNAR TERCEROS A LOS CRUCES
# ==========================================================

def asignar_terceros(terceros):

    slots = [

        ("M74", ["A", "B", "C", "D", "F"]),

        ("M77", ["C", "D", "F", "G", "H"]),

        ("M79", ["C", "E", "F", "H", "I"]),

        ("M80", ["E", "H", "I", "J", "K"]),

        ("M81", ["B", "E", "F", "I", "J"]),

        ("M82", ["A", "E", "H", "I", "J"]),

        ("M85", ["E", "F", "G", "I", "J"]),

        ("M87", ["D", "E", "I", "J", "L"])

    ]


    grupos_terceros = [

        t["grupo"]

        for t in terceros

    ]


    # ======================================================
    # Buscar una asignación válida.
    # Se intenta respetar el cuadro oficial.
    # ======================================================

    def buscar(
        indice,
        usados,
        asignacion
    ):

        if indice == len(slots):

            return asignacion.copy()


        nombre_slot, posibles = slots[indice]


        for grupo in posibles:

            if grupo not in grupos_terceros:

                continue

            if grupo in usados:

                continue


            asignacion[nombre_slot] = grupo

            usados.add(grupo)


            resultado = buscar(
                indice + 1,
                usados,
                asignacion
            )


            if resultado is not None:

                return resultado


            usados.remove(grupo)

            del asignacion[nombre_slot]


        return None


    asignacion = buscar(
        0,
        set(),
        {}
    )


    if asignacion is None:

        raise RuntimeError(
            "No se pudo construir el cuadro de terceros."
        )


    tercero_por_grupo = {

        t["grupo"]:
            t["equipo"]

        for t in terceros

    }


    return {

        slot:
            tercero_por_grupo[grupo]

        for slot, grupo
        in asignacion.items()

    }


# ==========================================================
# CUADRO DE LA RONDA DE 32
# ==========================================================

def construir_ronda_32(posiciones, terceros):

    terceros_asignados = asignar_terceros(
        terceros
    )


    def primero(grupo):

        return posiciones[grupo][0][0]


    def segundo(grupo):

        return posiciones[grupo][1][0]


    partidos = [

        (
            "M73",
            segundo("A"),
            segundo("B")
        ),

        (
            "M74",
            primero("E"),
            terceros_asignados["M74"]
        ),

        (
            "M75",
            primero("F"),
            segundo("C")
        ),

        (
            "M76",
            primero("C"),
            segundo("F")
        ),

        (
            "M77",
            primero("I"),
            terceros_asignados["M77"]
        ),

        (
            "M78",
            segundo("E"),
            segundo("I")
        ),

        (
            "M79",
            primero("A"),
            terceros_asignados["M79"]
        ),

        (
            "M80",
            primero("L"),
            terceros_asignados["M80"]
        ),

        (
            "M81",
            primero("D"),
            terceros_asignados["M81"]
        ),

        (
            "M82",
            primero("G"),
            terceros_asignados["M82"]
        ),

        (
            "M83",
            segundo("K"),
            segundo("L")
        ),

        (
            "M84",
            primero("H"),
            segundo("J")
        ),

        (
            "M85",
            primero("B"),
            terceros_asignados["M85"]
        ),

        (
            "M86",
            primero("J"),
            segundo("H")
        ),

        (
            "M87",
            primero("K"),
            terceros_asignados["M87"]
        ),

        (
            "M88",
            segundo("D"),
            segundo("G")
        )

    ]


    return partidos


# ==========================================================
# SIMULAR ELIMINATORIA
# ==========================================================

def simular_eliminatoria(
    equipo1,
    equipo2,
    nombre=""
):

    g1, g2 = simular_marcador(

        equipo1,

        equipo2,

        True

    )


    if g1 > g2:

        ganador = equipo1

    else:

        ganador = equipo2


    if nombre:

        print(

            f"{nombre}: "
            f"{equipo1} "
            f"{g1}-{g2} "
            f"{equipo2} "
            f"-> {ganador}"

        )


    return ganador


# ==========================================================
# SIMULAR RONDA DE 32
# ==========================================================

def ronda_32(
    posiciones,
    terceros,
    mostrar=False
):

    partidos = construir_ronda_32(

        posiciones,

        terceros

    )


    ganadores = {}


    for numero, e1, e2 in partidos:

        ganador = simular_eliminatoria(

            e1,

            e2,

            numero if mostrar else ""

        )


        ganadores[numero] = ganador


    return ganadores


# ==========================================================
# OCTAVOS
# ==========================================================

def octavos(
    ganadores_32,
    mostrar=False
):

    cruces = [

        (
            "M89",
            "M74",
            "M77"
        ),

        (
            "M90",
            "M73",
            "M75"
        ),

        (
            "M91",
            "M76",
            "M78"
        ),

        (
            "M92",
            "M79",
            "M80"
        ),

        (
            "M93",
            "M83",
            "M84"
        ),

        (
            "M94",
            "M81",
            "M82"
        ),

        (
            "M95",
            "M86",
            "M88"
        ),

        (
            "M96",
            "M85",
            "M87"
        )

    ]


    ganadores = {}


    for numero, m1, m2 in cruces:

        e1 = ganadores_32[m1]

        e2 = ganadores_32[m2]


        ganador = simular_eliminatoria(

            e1,

            e2,

            numero if mostrar else ""

        )


        ganadores[numero] = ganador


    return ganadores


# ==========================================================
# CUARTOS
# ==========================================================

def cuartos(
    ganadores_16,
    mostrar=False
):

    cruces = [

        (
            "M97",
            "M89",
            "M90"
        ),

        (
            "M98",
            "M93",
            "M94"
        ),

        (
            "M99",
            "M91",
            "M92"
        ),

        (
            "M100",
            "M95",
            "M96"
        )

    ]


    ganadores = {}


    for numero, m1, m2 in cruces:

        e1 = ganadores_16[m1]

        e2 = ganadores_16[m2]


        ganador = simular_eliminatoria(

            e1,

            e2,

            numero if mostrar else ""

        )


        ganadores[numero] = ganador


    return ganadores


# SEMIFINALES

def semifinales(
    ganadores_cuartos,
    mostrar=False
):

    cruces = [

        (
            "M101",
            "M97",
            "M98"
        ),

        (
            "M102",
            "M99",
            "M100"
        )

    ]


    ganadores = {}


    for numero, m1, m2 in cruces:

        e1 = ganadores_cuartos[m1]

        e2 = ganadores_cuartos[m2]


        ganador = simular_eliminatoria(

            e1,

            e2,

            numero if mostrar else ""

        )


        ganadores[numero] = ganador


    return ganadores

# FINAL

def final(
    ganadores_semifinales,
    mostrar=False
):

    e1 = ganadores_semifinales["M101"]

    e2 = ganadores_semifinales["M102"]


    ganador = simular_eliminatoria(

        e1,

        e2,

        "FINAL" if mostrar else ""

    )


    return ganador

# MUNDIAL COMPLETO

def simular_mundial(
    mostrar=False
):

    grupos = fase_grupos(
        mostrar
    )


    r32 = ronda_32(

        grupos["posiciones"],

        grupos["terceros"],

        mostrar

    )


    r16 = octavos(

        r32,

        mostrar

    )


    qf = cuartos(

        r16,

        mostrar

    )


    sf = semifinales(

        qf,

        mostrar

    )


    campeon = final(

        sf,

        mostrar

    )


    return campeon

# SIMULACIÓN MASIVA

def simulaciones_mundial(
    numero_simulaciones=N_SIMULACIONES
):

    campeones = Counter()

    finales = Counter()

    semifinales_counter = Counter()

    cuartos_counter = Counter()

    octavos_counter = Counter()

    ronda32_counter = Counter()


    for i in range(
        numero_simulaciones
    ):

        grupos = fase_grupos(
            False
        )


        for equipo in grupos["clasificados"]:

            ronda32_counter[equipo] += 1


        r32 = ronda_32(

            grupos["posiciones"],

            grupos["terceros"],

            False

        )


        for equipo in r32.values():

            octavos_counter[equipo] += 1


        r16 = octavos(
            r32,
            False
        )


        for equipo in r16.values():

            cuartos_counter[equipo] += 1


        qf = cuartos(
            r16,
            False
        )


        for equipo in qf.values():

            semifinales_counter[equipo] += 1


        sf = semifinales(
            qf,
            False
        )


        for equipo in sf.values():

            finales[equipo] += 1


        campeon = final(
            sf,
            False
        )


        campeones[campeon] += 1


        if (
            (i + 1) % 10000 == 0
        ):

            print(
                f"Simulaciones: "
                f"{i + 1:,}"
            )


    equipos_resultados = set(

        list(campeones.keys())

        +
        list(ronda32_counter.keys())

    )


    resultados = []


    for equipo in equipos_resultados:

        resultados.append({

            "Equipo":
                equipo,

            "Ronda 32 %":
                ronda32_counter[equipo]
                /
                numero_simulaciones
                *
                100,

            "Octavos %":
                octavos_counter[equipo]
                /
                numero_simulaciones
                *
                100,

            "Cuartos %":
                cuartos_counter[equipo]
                /
                numero_simulaciones
                *
                100,

            "Semifinal %":
                semifinales_counter[equipo]
                /
                numero_simulaciones
                *
                100,

            "Final %":
                finales[equipo]
                /
                numero_simulaciones
                *
                100,

            "Campeón %":
                campeones[equipo]
                /
                numero_simulaciones
                *
                100

        })


    resultados = pd.DataFrame(
        resultados
    )


    resultados = resultados.sort_values(
        "Campeón %",
        ascending=False
    )


    return resultados

# SIMULACIÓN DESDE UNA FASE CONCRETA

def simular_desde_equipos(
    equipos_iniciales,
    numero_simulaciones,
    fase
):

    campeones = Counter()


    for _ in range(
        numero_simulaciones
    ):

        equipos_actuales = list(
            equipos_iniciales
        )


        while len(equipos_actuales) > 1:

            siguientes = []
            for i in range(
                0,
                len(equipos_actuales),
                2
            ):

                e1 = equipos_actuales[i]

                e2 = equipos_actuales[i + 1]

                ganador = simular_eliminatoria(
                    e1,
                    e2
                )
                siguientes.append(
                    ganador
                )
            equipos_actuales = siguientes
        campeones[
            equipos_actuales[0]
        ] += 1
    resultados = []

    for equipo, veces in campeones.most_common():

        resultados.append({

            "Equipo":
                equipo,

            "Campeón %":
                veces/numero_simulaciones * 100

        })


    return pd.DataFrame(
        resultados
    )

# CUADRO REAL DEL MUNDIAL 2026

REAL_DIECISEISAVOS = [

    ("South Africa", "Canada"),

    ("Germany", "Paraguay"),

    ("Netherlands", "Morocco"),

    ("Brazil", "Japan"),

    ("France", "Sweden"),

    ("Côte d'Ivoire", "Norway"),

    ("Mexico", "Ecuador"),

    ("England", "Congo DR"),

    ("USA", "Bosnia and Herzegovina"),

    ("Belgium", "Senegal"),

    ("Portugal", "Croatia"),

    ("Spain", "Austria"),

    ("Switzerland", "Algeria"),

    ("Argentina", "Cape Verde"),

    ("Colombia", "Ghana"),

    ("Australia", "Egypt")

]

# EQUIPOS REALES DE OCTAVOS

REAL_OCTAVOS = [

    ("Paraguay", "France"),

    ("Canada", "Morocco"),

    ("Brazil", "Norway"),

    ("Mexico", "England"),

    ("Portugal", "Spain"),

    ("USA", "Belgium"),

    ("Argentina", "Egypt"),

    ("Switzerland", "Colombia")

]

# EQUIPOS REALES DE CUARTOS

REAL_CUARTOS = [

    ("France", "Morocco"),

    ("Spain", "Belgium"),

    ("Norway", "England"),

    ("Argentina", "Switzerland")

]

# EQUIPOS REALES DE SEMIFINALES

REAL_SEMIFINALES = [

    ("France", "Spain"),

    ("Argentina", "England")

]

# FINAL REAL

REAL_FINAL = [

    ("Spain", "Argentina")
]

# SIMULAR UNA RONDA REAL

def simular_ronda_real(
    partidos,
    numero_simulaciones
):

    campeones = Counter()


    for _ in range(
        numero_simulaciones
    ):

        equipos = []


        for e1, e2 in partidos:

            ganador = simular_eliminatoria(

                e1,

                e2

            )


            equipos.append(
                ganador
            )


        while len(equipos) > 1:

            siguientes = []


            for i in range(
                0,
                len(equipos),
                2
            ):
                ganador = simular_eliminatoria(

                    equipos[i],

                    equipos[i + 1]
                )
                siguientes.append(
                    ganador
                )
            equipos = siguientes
        campeones[
            equipos[0]
        ] += 1
    resultados = []
    for equipo, veces in campeones.most_common():
        resultados.append({
            "Equipo":
                equipo,
            "Campeón %":
                veces/numero_simulaciones*100
        })
    return pd.DataFrame(
        resultados
    )

# MOSTRAR RESULTADOS

def mostrar_resultados(
    resultados,
    titulo
):
    print()
    print(
        "=========================================="
    )
    print(
        titulo
    )
    print(
        "=========================================="
    )
    print()
    print(

        resultados.to_string(
            index=False,
            formatters={
                columna:
                    lambda x:
                    f"{x:.2f}"
                    if isinstance(
                        x,
                        (float, np.floating)
                    )
                    else x

                for columna
                in resultados.columns
            }
        )

    )

# PROGRAMA PRINCIPAL

if __name__ == "__main__":
    print()
    print(
        "=========================================="
    )
    print(
        "IA PREDICTORA DEL MUNDIAL 2026"
    )
    print(
        "=========================================="
    )

    # MOSTRAR EVALUACIÓN

    print()
    print(
        "El modelo ha sido entrenado."
    )

    # EJEMPLO DE PREDICCIÓN

    equipo1 = "Spain"
    equipo2 = "Argentina"

    p = probabilidades(
        equipo1,
        equipo2
    )
    g1, g2 = goles_esperados(
        equipo1,
        equipo2
    )
    print()
    print(
        "=========================================="
    )
    print(
        f"PREDICCIÓN: "
        f"{equipo1} vs {equipo2}"
    )
    print(
        "=========================================="
    )
    print()
    print(
        f"Victoria {equipo1}: "
        f"{p['victoria_1'] * 100:.2f}%"
    )
    print(
        f"Empate: "
        f"{p['empate'] * 100:.2f}%"
    )
    print(
        f"Victoria {equipo2}: "
        f"{p['victoria_2'] * 100:.2f}%"
    )
    print()
    print(
        f"Goles esperados "
        f"{equipo1}: {g1:.2f}"
    )
    print(
        f"Goles esperados "
        f"{equipo2}: {g2:.2f}"
    )

    # SIMULACIÓN DE UN MUNDIAL

    print()
    print(
        "=========================================="
    )
    print(
        "SIMULANDO UN MUNDIAL"
    )
    print(
        "=========================================="
    )
    campeon = simular_mundial(
        mostrar=True
    )
    print()
    print(
        f"CAMPEÓN SIMULADO: "
        f"{campeon}"
    )

    # MONTE CARLO

    print()
    print(
        "=========================================="
    )
    print(
        f"SIMULANDO "
        f"{N_SIMULACIONES:,} MUNDIALES"
    )
    print(
        "=========================================="
    )
    resultados = simulaciones_mundial(
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados,
        "PROBABILIDADES DEL MUNDIAL COMPLETO"
    )

    # GUARDAR RESULTADOS

    resultados.to_csv(
        "resultados_simulaciones_mundial.csv",
        index=False
    )
    # SIMULACIONES DESDE LOS DIECISEISAVOS REALES
    print()
    print(
        "=========================================="
    )
    print(
        "SIMULACIÓN DESDE DIECISEISAVOS REALES"
    )
    print(
        "=========================================="
    )
    resultados_r32 = simular_ronda_real(
        REAL_DIECISEISAVOS,
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados_r32,
        "CAMPEÓN DESDE DIECISEISAVOS REALES"
    )
    resultados_r32.to_csv(
        "resultados_desde_dieciseisavos.csv",
        index=False
    )
    # DESDE OCTAVOS REALES
    print()
    print(
        "=========================================="
    )
    print(
        "SIMULACIÓN DESDE OCTAVOS REALES"
    )
    print(
        "=========================================="
    )
    resultados_octavos = simular_ronda_real(
        REAL_OCTAVOS,
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados_octavos,
        "CAMPEÓN DESDE OCTAVOS REALES"
    )
    resultados_octavos.to_csv(
        "resultados_desde_octavos.csv",
        index=False
    )
    # DESDE CUARTOS REALES
    print()
    print(
        "=========================================="
    )
    print(
        "SIMULACIÓN DESDE CUARTOS REALES"
    )
    print(
        "=========================================="
    )
    resultados_cuartos = simular_ronda_real(
        REAL_CUARTOS,
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados_cuartos,
        "CAMPEÓN DESDE CUARTOS REALES"
    )
    resultados_cuartos.to_csv(
        "resultados_desde_cuartos.csv",
        index=False
    )
    # DESDE SEMIFINALES REALES
    print()
    print(
        "=========================================="
    )
    print(
        "SIMULACIÓN DESDE SEMIFINALES REALES"
    )
    print(
        "=========================================="
    )
    resultados_semifinales = simular_ronda_real(
        REAL_SEMIFINALES,
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados_semifinales,
        "CAMPEÓN DESDE SEMIFINALES REALES"
    )
    resultados_semifinales.to_csv(
        "resultados_desde_semifinales.csv",
        index=False
    )
    # FINAL REAL
    print()
    print(
        "=========================================="
    )
    print(
        "SIMULACIÓN DE LA FINAL REAL"
    )
    print(
        "=========================================="
    )
    resultados_final = simular_ronda_real(
        REAL_FINAL,
        N_SIMULACIONES
    )
    mostrar_resultados(
        resultados_final,
        "RESULTADO DE LA FINAL REAL"
    )
    resultados_final.to_csv(
        "resultados_final.csv",
        index=False
    )
    print()
    print(
        "=========================================="
    )
    print(
        "PROGRAMA TERMINADO"
    )
    print(
        "=========================================="
    )