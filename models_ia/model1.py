import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================
# CARREGAR RESULTATS
# ==========================

results = pd.read_csv("results.csv")

results["date"] = pd.to_datetime(results["date"])

results = results[
    results["date"] >= "2018-01-01"
]

# ==========================
# CARREGAR RANKING FIFA
# ==========================

ranking = pd.read_csv("fifa_mens_rank.csv")

ranking["date"] = pd.to_datetime(
    ranking["date"]
)

ranking = ranking.sort_values("date")

# ==========================
# CORREGIR NOMS
# ==========================

equivalencies = {

    "United States":"USA",
    "Iran":"IR Iran",
    "South Korea":"Korea Republic",
    "North Korea":"Korea DPR",
    "Ivory Coast":"Côte d'Ivoire",
    "Czech Republic":"Czechia",
    "DR Congo":"Congo DR",
    "China":"China PR",
    "Turkey":"Türkiye"

}

results["home_team"] = results["home_team"].replace(equivalencies)
results["away_team"] = results["away_team"].replace(equivalencies)

# ==========================
# FUNCIÓ RANKING
# ==========================

def dades_fifa(equip,data):

    hist = ranking[
        (ranking["team"]==equip)
        &
        (ranking["date"]<=data)
    ]

    if len(hist)==0:

        return None

    fila = hist.iloc[-1]

    return fila

# ==========================
# CREAR DATASET
# ==========================

dataset=[]

for _,partit in results.iterrows():

    local=dades_fifa(
        partit["home_team"],
        partit["date"]
    )

    visitant=dades_fifa(
        partit["away_team"],
        partit["date"]
    )

    if local is None or visitant is None:
        continue

    dataset.append({

        "ranking_diff":
            visitant["rank"]
            -
            local["rank"],

        "points_diff":
            local["total.points"]
            -
            visitant["total.points"],

        "resultat":
            int(
                partit["home_score"]
                >
                partit["away_score"]
            )

    })

dataset=pd.DataFrame(dataset)

print("Partits:",len(dataset))

# ==========================
# VARIABLES
# ==========================

X=dataset[
[
"ranking_diff",
"points_diff"
]
]

y=dataset["resultat"]

# ==========================
# TRAIN TEST
# ==========================

X_train,X_test,y_train,y_test=train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ==========================
# MODEL
# ==========================

model=LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# ==========================
# RESULTATS
# ==========================

pred=model.predict(X_test)

print()

print("Precisió:")

print(
    accuracy_score(
        y_test,
        pred
    )
)