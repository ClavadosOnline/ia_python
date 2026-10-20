import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# CARREGAR DATASETS
df = pd.read_csv("teams_match_features.csv", low_memory=False)
ranking = pd.read_csv("fifa_ranking-2024-06-20.csv")

df["_date"] = pd.to_datetime(df["_date"])
ranking["rank_date"] = pd.to_datetime(ranking["rank_date"])

# UNIFICAR ALGUNS NOMS DELS EQUIPS ENTRE ELS DATASETS
team_names = {
    "China": "China PR",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "East Timor": "Timor-Leste",
    "Gambia": "The Gambia",
    "Iran": "IR Iran",
    "Ireland": "Republic of Ireland",
    "Ivory Coast": "\ufffdte d'Ivoire",
    "Kyrgyzstan": "Kyrgyz Republic",
    "North Korea": "Korea DPR",
    "Saint Kitts and Nevis": "St Kitts and Nevis",
    "Saint Lucia": "St Lucia",
    "South Korea": "Korea Republic",
    "United States": "USA",
}
ranking = ranking[["country_full", "rank_date", "rank"]].sort_values("rank_date")

# AFEGIR A CADA PARTIT EL RÀNQUING MÉS RECENT PUBLICAT FINS A LA DATA DEL PARTIT
for team_column, rank_column in [
    ("_home_team", "home_fifa_rank"),
    ("_away_team", "away_fifa_rank"),
]:
    team_matches = df[["_date", team_column]].copy()
    team_matches["_match_index"] = df.index
    team_matches["country_full"] = team_matches[team_column].replace(team_names)
    team_matches = team_matches.sort_values("_date")

    matched_ranks = pd.merge_asof(
        team_matches,
        ranking,
        left_on="_date",
        right_on="rank_date",
        by="country_full",
        direction="backward",
    )
    df.loc[matched_ranks["_match_index"], rank_column] = matched_ranks["rank"].values

# VARIABLE OBJECTIU
df["resultat"] = (df["home_goals"] > df["away_goals"]).astype(int)

# UTILITZAR NOMÉS LA DIFERÈNCIA DEL RÀNQUING FIFA
df = df.dropna(subset=["home_fifa_rank", "away_fifa_rank"])
df["fifa_rank_diff"] = df["away_fifa_rank"] - df["home_fifa_rank"]
X = df[["fifa_rank_diff"]]
y = df["resultat"]

# TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# MODEL
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train,
    y_train
)

# IMPORTÀNCIA DE LES VARIABLES
imp = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": model.feature_importances_
})
imp = imp.sort_values("Importancia", ascending=False)

print("\nImportància variables:\n")
print(imp)

# RESULTATS
pred = model.predict(X_test)
print()
print("Precisió:")
print(accuracy_score(y_test, pred))
print()
print(classification_report(y_test, pred))