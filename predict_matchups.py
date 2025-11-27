# nfl_predictions_with_odds_abbr.py
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
import joblib
import os

# Get Odds API key from environment variable
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    raise ValueError("Please set the ODDS_API_KEY environment variable!")
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
def get_features(row, game_df, N=3):
    home, away = row['home_team_abbr'], row['away_team_abbr']
    def last_mean(team, df, col_home, col_away, n=N):
        rows = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('game_date').tail(n)
        if rows.empty:
            return 0
        vals = np.where(rows['home_team'] == team, rows[col_home], rows[col_away])
        return np.nanmean(vals)

    def last_value(team, df, col_home, col_away):
        rows = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('game_date')
        if rows.empty:
            return None
        last = rows.iloc[-1]
        return last[col_home] if last['home_team'] == team else last[col_away]

    # basic rolling diffs (as before)
    home_epa = last_mean(home, game_df, 'home_rolling_epa', 'away_rolling_epa')
    away_epa = last_mean(away, game_df, 'home_rolling_epa', 'away_rolling_epa')
    home_success = last_mean(home, game_df, 'home_rolling_success', 'away_rolling_success')
    away_success = last_mean(away, game_df, 'home_rolling_success', 'away_rolling_success')
    home_turnover = last_mean(home, game_df, 'home_rolling_turnovers', 'away_rolling_turnovers')
    away_turnover = last_mean(away, game_df, 'home_rolling_turnovers', 'away_rolling_turnovers')

    # last elo pre-game
    home_elo = last_value(home, game_df, 'home_elo_pre', 'away_elo_pre') or 1500
    away_elo = last_value(away, game_df, 'home_elo_pre', 'away_elo_pre') or 1500

    # EWMA features (short and medium)
    home_ewma_3 = last_mean(home, game_df, 'home_ewma_epa_3', 'away_ewma_epa_3')
    away_ewma_3 = last_mean(away, game_df, 'home_ewma_epa_3', 'away_ewma_epa_3')
    home_ewma_7 = last_mean(home, game_df, 'home_ewma_epa_7', 'away_ewma_epa_7')
    away_ewma_7 = last_mean(away, game_df, 'home_ewma_epa_7', 'away_ewma_epa_7')

    # Rest days (most recent value)
    home_rest = last_value(home, game_df, 'home_days_rest', 'away_days_rest') or 7
    away_rest = last_value(away, game_df, 'home_days_rest', 'away_days_rest') or 7

    # Construct differences in the same order used for training
    epa_diff = home_epa - away_epa
    success_diff = home_success - away_success
    turnover_diff = away_turnover - home_turnover
    elo_diff = home_elo - away_elo
    epa_diff_ewm_3 = (home_ewma_3 or 0) - (away_ewma_3 or 0)
    epa_diff_ewm_7 = (home_ewma_7 or 0) - (away_ewma_7 or 0)
    rest_diff = (home_rest or 7) - (away_rest or 7)

    return [epa_diff, success_diff, turnover_diff, elo_diff, epa_diff_ewm_3, epa_diff_ewm_7, rest_diff]

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
week_df['home_implied_prob'] = week_df['home_decimal_odds'].apply(lambda x: 1 / x if x and x > 0 else np.nan)
week_df['away_implied_prob'] = week_df['away_decimal_odds'].apply(lambda x: 1 / x if x and x > 0 else np.nan)

week_df['home_edge'] = week_df['home_win_prob'] - week_df['home_implied_prob']
week_df['away_edge'] = week_df['away_win_prob'] - week_df['away_implied_prob']

# --------------------------
# 7️⃣ Save results
# --------------------------
week_df.to_csv("nfl_upcoming_predictions.csv", index=False)
print(week_df[['home_team','away_team','home_win_prob','away_win_prob','home_edge','away_edge']])
