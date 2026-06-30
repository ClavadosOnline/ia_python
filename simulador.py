import pandas as pd
import random

def predir_prob(equip1, equip2, stats_df, model, fifa, date):

    row1 = stats_df.loc[equip1]
    row2 = stats_df.loc[equip2]

    dades = pd.DataFrame([{
        "elo_diff": row1["home_elo"] - row2["away_elo"],
        "overall_diff": row1["home_avg_overall"] - row2["away_avg_overall"],
        "attack_diff": row1["home_avg_attack"] - row2["away_avg_attack"],
        "defense_diff": row1["home_avg_defense"] - row2["away_avg_defense"],
        "home_form_win_rate": row1["home_form_win_rate"],
        "away_form_win_rate": row2["away_form_win_rate"],
        "is_neutral": 1
    }])

    return model.predict_proba(dades)[0][1]

def simular_partit(e1, e2, stats_df, model):

    p = predir_prob(e1, e2, stats_df, model, None, None)

    if p > 0.75:
        return random.choice([(3,0),(2,0),(3,1)])
    elif p > 0.60:
        return random.choice([(2,0),(2,1),(1,0)])
    elif p > 0.40:
        return random.choice([(1,1),(2,1),(1,2),(0,0)])
    else:
        return random.choice([(0,2),(1,2),(0,1)])
    
GRUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

def simular_grup(grup, stats_df, model):

    taula = {e: {"pts":0,"gf":0,"gc":0} for e in grup}

    for i in range(len(grup)):
        for j in range(i+1, len(grup)):

            e1, e2 = grup[i], grup[j]
            g1, g2 = simular_partit(e1, e2, stats_df, model)

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
        key=lambda x: (x[1]["pts"], x[1]["gf"]-x[1]["gc"], x[1]["gf"]),
        reverse=True
    )

    return classificacio

def fase_grups():

    classificats = []

    for g, equips in GRUPS.items():

        print(f"\n===== GRUP {g} =====")

        result = simular_grup(equips, stats_df, model)

        for i,(team,data) in enumerate(result,1):
            print(i, team, data)

        classificats += [result[0][0], result[1][0]]

    return classificats

def knockout_round(equips, stats_df, model):

    random.shuffle(equips)
    next_round = []

    for i in range(0, len(equips), 2):

        e1 = equips[i]
        e2 = equips[i+1]

        g1, g2 = simular_partit(e1, e2, stats_df, model)

        winner = e1 if g1 > g2 else e2
        next_round.append(winner)

        print(f"{e1} {g1}-{g2} {e2} -> {winner}")

    return next_round


def simular_mundial():

    equips = fase_grups()

    print("\n===== SETZENS =====")
    equips = knockout_round(equips, stats_df, model)

    print("\n===== VUITENS =====")
    equips = knockout_round(equips, stats_df, model)

    print("\n===== QUARTS =====")
    equips = knockout_round(equips, stats_df, model)

    print("\n===== SEMIS =====")
    equips = knockout_round(equips, stats_df, model)

    print("\n===== FINAL =====")
    campio = knockout_round(equips, stats_df, model)[0]

    print("\n🏆 CAMPION:", campio)

    return campio

campio = simular_mundial()