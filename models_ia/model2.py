import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# CARREGAR DATASET
# ==========================

df = pd.read_csv("teams_match_features.csv")

# ==========================
# VARIABLE OBJECTIU
# ==========================

df["resultat"] = (
    df["home_goals"] >
    df["away_goals"]
).astype(int)

# ==========================
# VARIABLES
# ==========================

X = df[[
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

    "is_neutral",
    "is_world_cup",
    "is_continental"
]]

y = df["resultat"]

# ==========================
# ELIMINAR NaN
# ==========================

X = X.fillna(0)

# ==========================
# TRAIN / TEST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

# ==========================
# MODEL
# ==========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# ==========================
# RESULTATS
# ==========================

pred = model.predict(X_test)

print()

print("Precisió:")

print(
    accuracy_score(
        y_test,
        pred
    )
)

print()

print(
    classification_report(
        y_test,
        pred
    )
)