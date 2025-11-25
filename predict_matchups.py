# nfl_predictions_with_odds_abbr.py
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
import joblib

# --------------------------
# 1️⃣ Load historical data & model
# --------------------------
game_df = pd.read_csv("game_df_processed.csv", parse_dates=['game_date'])
model = load_model("nfl_model.h5")
scaler = joblib.load("scaler.save")

# --------------------------
# 2️⃣ Team mapping (full name → abbreviation)
# --------------------------
team_map = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS"
}

# --------------------------
# 3️⃣ Fetch upcoming NFL games from Odds API
# --------------------------
API_KEY = "API_KEY"
SPORT_KEY = "americanfootball_nfl"

def fetch_upcoming_games():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for game in data:
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_id = game["id"]
        game_date = game["commence_time"]

        if game["bookmakers"]:
            outcomes = game["bookmakers"][0]["markets"][0]["outcomes"]
            home_odds = next((o["price"] for o in outcomes if o["name"] == home_team), None)
            away_odds = next((o["price"] for o in outcomes if o["name"] == away_team), None)
        else:
            home_odds = away_odds = None

        records.append({
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_abbr": team_map.get(home_team),
            "away_team_abbr": team_map.get(away_team),
            "game_date": game_date,
            "home_decimal_odds": home_odds,
            "away_decimal_odds": away_odds
        })
    return pd.DataFrame(records)

week_df = fetch_upcoming_games()
if week_df.empty:
    raise ValueError("No upcoming games found from Odds API!")

# --------------------------
# 4️⃣ Feature function
# --------------------------
def get_features(row, game_df, N=5):
    home, away = row['home_team_abbr'], row['away_team_abbr']

    home_games = game_df[(game_df['home_team']==home) | (game_df['away_team']==home)].sort_values('game_date').tail(N)
    away_games = game_df[(game_df['home_team']==away) | (game_df['away_team']==away)].sort_values('game_date').tail(N)

    home_epa = home_games['home_rolling_epa'].mean() if not home_games.empty else 0
    away_epa = away_games['away_rolling_epa'].mean() if not away_games.empty else 0
    home_success = home_games['home_rolling_success'].mean() if not home_games.empty else 0
    away_success = away_games['away_rolling_success'].mean() if not away_games.empty else 0
    home_turnover = home_games['home_rolling_turnovers'].mean() if not home_games.empty else 0
    away_turnover = away_games['away_rolling_turnovers'].mean() if not away_games.empty else 0
    home_elo = home_games['home_elo_pre'].iloc[-1] if not home_games.empty else 1500
    away_elo = away_games['away_elo_pre'].iloc[-1] if not away_games.empty else 1500

    epa_diff = home_epa - away_epa
    success_diff = home_success - away_success
    turnover_diff = away_turnover - home_turnover
    elo_diff = home_elo - away_elo

    return [epa_diff, success_diff, turnover_diff, elo_diff]

# --------------------------
# 5️⃣ Predict probabilities
# --------------------------
X = np.array([get_features(row, game_df) for _, row in week_df.iterrows()])
X_scaled = scaler.transform(X)

probs = model.predict(X_scaled).flatten()
week_df['home_win_prob'] = probs
week_df['away_win_prob'] = 1 - probs

# --------------------------
# 6️⃣ Compute implied probabilities and edge
# --------------------------
week_df['home_implied_prob'] = 1 / week_df['home_decimal_odds']
week_df['away_implied_prob'] = 1 / week_df['away_decimal_odds']

week_df['home_edge'] = week_df['home_win_prob'] - week_df['home_implied_prob']
week_df['away_edge'] = week_df['away_win_prob'] - week_df['away_implied_prob']

# --------------------------
# 7️⃣ Save results
# --------------------------
week_df.to_csv("nfl_upcoming_predictions.csv", index=False)
print(week_df[['home_team','away_team','home_win_prob','away_win_prob','home_edge','away_edge']])
