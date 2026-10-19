import pandas as pd
import random
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# CARREGAR DATASET
# ==========================================================

df = pd.read_csv("teams_match_features.csv")

equivalencies = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Cabo Verde": "Cape Verde",
    "Cabo Verde Islands": "Cape Verde",
    "Cape Verde Islands": "Cape Verde"
}

for column in ["_home_team", "_away_team"]:
    df[column] = df[column].replace(equivalencies)

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
print(accuracy_score(y_test, pred))

# ==========================================================
# IMPORTÀNCIA VARIABLES
# ==========================================================

imp = pd.DataFrame({
    "Variable": variables,
    "Importancia": model.feature_importances_
})

imp = imp.sort_values("Importancia", ascending=False)

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

if "Cape Verde" not in equips:
    forma = pd.read_csv("teams_form.csv")
    forma["team"] = forma["team"].replace(equivalencies)
    forma_cape_verde = forma[forma["team"] == "Cape Verde"].sort_values("match_date")
    ultima_forma = forma_cape_verde.iloc[-1]
    equips["Cape Verde"] = {
        "elo": 1500.0,
        "overall": df["home_avg_overall"].mean(),
        "attack": df["home_avg_attack"].mean(),
        "defense": df["home_avg_defense"].mean(),
        "scored": ultima_forma["avg_goals_scored"],
        "conceded": ultima_forma["avg_goals_conceded"],
        "winrate": ultima_forma["win_rate"]
    }

# ==========================================================
# NORMALITZAR NOM
# ==========================================================

def nom_dataset(equip):
    return equivalencies.get(equip, equip)

# ==========================================================
# PROBABILITAT DE VICTÒRIA
# ==========================================================

def probabilitat(equip1, equip2):
    equip1 = nom_dataset(equip1)
    equip2 = nom_dataset(equip2)

    t1 = equips[equip1]
    t2 = equips[equip2]

    dades = pd.DataFrame([{
        "elo_diff": t1["elo"] - t2["elo"],
        "overall_diff": t1["overall"] - t2["overall"],
        "attack_diff": t1["attack"] - t2["attack"],
        "defense_diff": t1["defense"] - t2["defense"],
        "home_form_scored": t1["scored"],
        "away_form_scored": t2["scored"],
        "home_form_conceded": t1["conceded"],
        "away_form_conceded": t2["conceded"],
        "home_form_win_rate": t1["winrate"],
        "away_form_win_rate": t2["winrate"],
        "is_neutral": 1
    }])

    return model.predict_proba(dades)[0][1]

# ==========================================================
# SIMULAR PARTIT
# ==========================================================

def simular_partit(equip1, equip2):
    p = probabilitat(equip1, equip2)
    if p > 0.80:
        marcador = random.choice([
            (3, 0),
            (2, 0),
            (3, 1),
            (4, 1)
        ])
    elif p > 0.65:
        marcador = random.choice([
            (2, 0),
            (2, 1),
            (1, 0),
            (3, 1),
            (4, 2),
            (3, 2)
        ])
    elif p > 0.55:
        marcador = random.choice([
            (2, 1),
            (1, 0),
            (1, 1),
            (2, 2)
        ])
    elif p > 0.45:
        marcador = random.choice([
            (1, 1),
            (0, 0),
            (2, 1),
            (1, 2)
        ])
    elif p > 0.35:
        marcador = random.choice([
            (0, 1),
            (1, 2),
            (0, 2),
            (1, 1)
        ])
    else:
        marcador = random.choice([
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 2)
        ])
    return marcador

# ==========================================================
# GRUPS MUNDIAL
# ==========================================================

GRUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Jamaica", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}
SETZENS_MANUALS = [
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
VUITENS_MANUALS = [
    ("Paraguay", "France"),
    ("Canada", "Morocco"),
    ("Brazil", "Norway"),
    ("Mexico", "England"),
    ("Portugal", "Spain"),
    ("USA", "Belgium"),
    ("Argentina", "Egypt"),
    ("Switzerland", "Colombia")
]
QUARTS_MANUALS = [
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
    ("Argentina", "Switzerland")
]
SEMIFINALS_MANUALS = [("France", "Spain"), ("England", "Argentina")]
FINAL_MANUALS = [("Spain", "Argentina")]

PARTITS_PER_FASE = {
    "SETZENS": SETZENS_MANUALS,
    "VUITENS": VUITENS_MANUALS,
    "QUARTS": QUARTS_MANUALS,
    "SEMIFINALS": SEMIFINALS_MANUALS,
    "FINAL": FINAL_MANUALS
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
# INICIAR DESDE UNA RONDA CONCRETA
# ==========================================================

def ronda_manual(partits, nom):
    print(f"\n===== {nom} =====")
    classificats = []
    for e1, e2 in partits:
        while True:
            g1, g2 = simular_partit(e1, e2)
            if g1 != g2:
                break
        guanyador = e1 if g1 > g2 else e2
        print(f"{e1} {g1}-{g2} {e2} -> {guanyador}")
        classificats.append(guanyador)
    return classificats


def simular_mundial_desde(ronda_inici, equips_inicials=None, partits_manuals=None):
    """Simula el mundial començant des de la ronda indicada.
    ronda_inici pot ser: 'SETZENS', 'VUITENS', 'QUARTS', 'SEMIFINALS', 'FINAL'.
    Pots passar una llista manual per la fase concreta o una llista d'equips inicials.
    """
    ronda_ordres = ["SETZENS", "VUITENS", "QUARTS", "SEMIFINALS", "FINAL"]
    if ronda_inici not in ronda_ordres:
        raise ValueError(f"Ronda no vàlida: {ronda_inici}")

    inici = ronda_ordres.index(ronda_inici)

    if partits_manuals is not None:
        equips = ronda_manual(partits_manuals, ronda_inici)
    elif inici == 0:
        equips = ronda(fase_grups(), ronda_inici)
    elif equips_inicials is not None:
        equips = ronda(equips_inicials, ronda_inici)
    else:
        raise ValueError("Has de passar 'equips_inicials' o 'partits_manuals' per començar des d'aquesta ronda.")

    for nom in ronda_ordres[inici + 1:]:
        equips = ronda(equips, nom)

    return equips[-1]

# ==========================================================
# MUNDIAL COMPLET
# ==========================================================

def simular_mundial():
    equips = fase_grups()
    for nom in ["SETZENS", "VUITENS", "QUARTS", "SEMIFINALS", "FINAL"]:
        equips = ronda(equips, nom)
    campio = equips[0]
    print(f"\nCAMPIÓ DEL MUNDIAL 2026\n{campio}")
    return campio


def preguntar_entero(text, minim=1):
    while True:
        valor = input(text).strip()
        try:
            valor = int(valor)
            if valor >= minim:
                return valor
            print(f"Ha de ser un número major o igual a {minim}.")
        except ValueError:
            print("Escriu un número vàlid.")


def menu_simulacio():
    rondes = ["SETZENS", "VUITENS", "QUARTS", "SEMIFINALS", "FINAL"]

    while True:
        mode = input("Vols simular el mundial complet o començar des d'una fase? [C/F]: ").strip().upper()
        if mode in {"C", "F"}:
            break
        print("Opció no vàlida. Escriu 'C' o 'F'.")

    repeticions = preguntar_entero("Quantes vegades vols repetir la simulació? ", 1)

    if mode == "C":
        return "COMPLET", repeticions

    print("Escriu una fase exactament com apareix aquí, sense cometes.")
    print("Opcions: SETZENS, VUITENS, QUARTS, SEMIFINALS o FINAL")
    while True:
        fase = input("Fase inicial: ").strip().upper()
        if fase in rondes:
            return fase, repeticions
        print("Fase no vàlida. Torna-ho a provar.")


def executar_simulacio():
    eleccio, repeticions = menu_simulacio()
    campions = Counter()

    for _ in range(repeticions):
        if eleccio == "COMPLET":
            campio = simular_mundial()
        else:
            campio = simular_mundial_desde(
                eleccio,
                partits_manuals=PARTITS_PER_FASE.get(eleccio)
            )
        campions[campio] += 1

    print("\n==========================")
    print("PROBABILITAT DE SER CAMPIÓ")
    print("==========================")

    total = sum(campions.values())
    for equip, vegades in campions.most_common():
        print(f"{equip:20s} {(vegades / total) * 100:.2f}%")

    return campions

# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":
    executar_simulacio()