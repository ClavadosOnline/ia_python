import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# CARREGAR DATASET
# ==========================================================

df = pd.read_csv("teams_match_features.csv")

df = df[df["_date"] >= "2010-01-01"].copy()

# ==========================================================
# CREAR VARIABLE OBJECTIU
# ==========================================================

df["resultat"] = (
    df["home_goals"] >
    df["away_goals"]
).astype(int)

# ==========================================================
# VARIABLES DEL MODEL
# ==========================================================

variables = [
    "elo_diff",

    "overall_diff",

    "attack_diff",

    "defense_diff",

    "home_form_scored",
    "away_form_scored",

    "home_form_conceded",
    "away_form_conceded",

    "home_form_win_rate",
    "away_form_win_rate",

    "is_neutral"
]

X = df[variables].fillna(0)
y = df["resultat"]

# ==========================================================
# TRAIN / TEST
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================================
# RESULTATS
# ==========================================================

pred = model.predict(X_test)
print("\nPrecisió:")
print(

    accuracy_score(
        y_test,
        pred
    )

)

print("\nInforme:\n")

print(

    classification_report(
        y_test,
        pred
    )

)

# ==========================================================
# IMPORTÀNCIA VARIABLES
# ==========================================================

imp = pd.DataFrame({
    "Variable": variables,
    "Importancia": model.feature_importances_
})

imp = imp.sort_values(
    "Importancia",
    ascending=False
)

print("\nImportància variables:\n")
print(imp)

# ==========================================================
# ÚLTIMES ESTADÍSTIQUES DE CADA EQUIP
# ==========================================================

equips = {}

for _, fila in df.sort_values("_date").iterrows():

    home = fila["_home_team"]

    equips[home] = {
        "elo": fila["home_elo"],
        "overall": fila["home_avg_overall"],
        "attack": fila["home_avg_attack"],
        "defense": fila["home_avg_defense"],
        "scored": fila["home_form_scored"],
        "conceded": fila["home_form_conceded"],
        "winrate": fila["home_form_win_rate"]
    }

    away = fila["_away_team"]
    equips[away] = {
        "elo": fila["away_elo"],
        "overall": fila["away_avg_overall"],
        "attack": fila["away_avg_attack"],
        "defense": fila["away_avg_defense"],
        "scored": fila["away_form_scored"],
        "conceded": fila["away_form_conceded"],
        "winrate": fila["away_form_win_rate"]
    }

# ==========================================================
# EQUIVALÈNCIES DE NOMS
# ==========================================================

equivalencies = {
    "United States":"USA",
    "South Korea":"Korea Republic",
    "Iran":"IR Iran",
    "Turkey":"Türkiye",
    "Czech Republic":"Czechia",
    "DR Congo":"Congo DR",
    "Ivory Coast":"Côte d'Ivoire"
}

# invertir diccionari

mapa = {}
for antic, nou in equivalencies.items():
    mapa[nou] = antic
# ==========================================================
# NORMALITZAR NOM
# ==========================================================

def nom_dataset(equip):
    if equip in equips:
        return equip
    if equip in mapa:
        return mapa[equip]
    return equip

# ==========================================================
# PROBABILITAT DE VICTÒRIA
# ==========================================================

def probabilitat(equip1, equip2):

    equip1 = nom_dataset(equip1)
    equip2 = nom_dataset(equip2)

    t1 = equips[equip1]
    t2 = equips[equip2]

    dades = pd.DataFrame([{

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
            1

    }])

    return model.predict_proba(dades)[0][1]

# ==========================================================
# SIMULAR PARTIT
# ==========================================================

def simular_partit(equip1, equip2):

    p = probabilitat(equip1, equip2)

    if p > 0.80:

        marcador = random.choice([
            (3,0),
            (2,0),
            (3,1),
            (4,1)
        ])

    elif p > 0.65:

        marcador = random.choice([
            (2,0),
            (2,1),
            (1,0),
            (3,1)
        ])

    elif p > 0.55:

        marcador = random.choice([
            (2,1),
            (1,0),
            (1,1),
            (2,2)
        ])

    elif p > 0.45:

        marcador = random.choice([
            (1,1),
            (0,0),
            (2,1),
            (1,2)
        ])

    elif p > 0.35:

        marcador = random.choice([
            (0,1),
            (1,2),
            (0,2),
            (1,1)
        ])

    else:

        marcador = random.choice([
            (0,2),
            (0,3),
            (1,3),
            (1,2)
        ])

    return marcador

# ==========================================================
# GRUPS MUNDIAL
# ==========================================================

GRUPS = {
    "A": ["Mexico","South Africa","Korea Republic","Czechia"],
    "B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["USA","Paraguay","Australia","Türkiye"],
    "E": ["Germany","Jamaica","Côte d'Ivoire","Ecuador"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","IR Iran","New Zealand"],
    "H": ["Spain","Greece","Saudi Arabia","Uruguay"],
    "I": ["France","Senegal","Iraq","Norway"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","Congo DR","Uzbekistan","Colombia"],
    "L": ["England","Croatia","Ghana","Panama"]
}

# ==========================================================
# SIMULAR UN GRUP
# ==========================================================

def simular_grup(grup):
    taula = {}
    for equip in grup:
        taula[equip] = {
            "pts":0,
            "gf":0,
            "gc":0

        }

    for i in range(len(grup)):

        for j in range(i+1,len(grup)):
            e1 = grup[i]
            e2 = grup[j]

            g1,g2 = simular_partit(e1,e2)

            taula[e1]["gf"] += g1
            taula[e1]["gc"] += g2

            taula[e2]["gf"] += g2
            taula[e2]["gc"] += g1

            if g1 > g2:
                taula[e1]["pts"] += 3
            elif g2 > g1:
                taula[e2]["pts"] += 3
            else:
                taula[e1]["pts"] += 1
                taula[e2]["pts"] += 1

    classificacio = sorted(
        taula.items(),

        key=lambda x:(
            x[1]["pts"],
            x[1]["gf"]-x[1]["gc"],
            x[1]["gf"]
        ),
        reverse=True
    )

    return classificacio

# ==========================================================
# FASE DE GRUPS
# ==========================================================

def fase_grups():
    classificats = []
    tercers = []
    for lletra, grup in GRUPS.items():
        print(f"\n===== GRUP {lletra} =====")
        resultat = simular_grup(grup)
        for posicio, (equip, dades) in enumerate(resultat, start=1):
            dg = dades["gf"] - dades["gc"]
            print(
                f"{posicio}. {equip} | "
                f"{dades['pts']} pts | "
                f"DG {dg}"
            )
        classificats.append(resultat[0][0])
        classificats.append(resultat[1][0])
        tercers.append({
            "equip": resultat[2][0],
            "pts": resultat[2][1]["pts"],
            "dg": resultat[2][1]["gf"] - resultat[2][1]["gc"],
            "gf": resultat[2][1]["gf"]
        })
    tercers = sorted(
        tercers,
        key=lambda x:(
            x["pts"],
            x["dg"],
            x["gf"]
        ),
        reverse=True
    )
    millors = tercers[:8]
    print("\n===== MILLORS TERCERS =====")
    for t in millors:
        print(
            f"{t['equip']} ({t['pts']} pts)"
        )
        classificats.append(t["equip"])
    return classificats

# ==========================================================
# ELIMINATÒRIES
# ==========================================================

def ronda(equips, nom):
    print(f"\n===== {nom} =====")
    random.shuffle(equips)
    classificats = []
    for i in range(0, len(equips), 2):
        e1 = equips[i]
        e2 = equips[i+1]

        while True:
            g1, g2 = simular_partit(e1, e2)
            if g1 != g2:
                break
        if g1 > g2:
            guanyador = e1
        else:
            guanyador = e2
        print(
            f"{e1} {g1}-{g2} {e2} -> {guanyador}"
        )
        classificats.append(guanyador)
    return classificats

# ==========================================================
# MUNDIAL COMPLET
# ==========================================================

def simular_mundial():
    equips = fase_grups()
    equips = ronda(
        equips,
        "SETZENS"
    )
    equips = ronda(
        equips,
        "VUITENS"
    )
    equips = ronda(
        equips,
        "QUARTS"
    )
    equips = ronda(
        equips,
        "SEMIFINALS"
    )
    campio = ronda(
        equips,
        "FINAL"
    )[0]

    print("\n🏆 CAMPIÓ DEL MUNDIAL 2026")

    print(campio)

    return campio

# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    campio = simular_mundial()

from collections import Counter

N = 10000

campions = Counter()

for i in range(N):

    campio = simular_mundial()

    campions[campio] += 1

print("\n==========================")
print("PROBABILITAT DE SER CAMPIÓ")
print("==========================")

for equip, vegades in campions.most_common():

    print(
        f"{equip:20s} {vegades/N*100:.2f}%"
    )