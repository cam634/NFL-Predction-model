import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------
# STEP 1 — LOAD DATA
# --------------------------
pbp = pd.read_csv("pbp_2023_2024.csv", low_memory=False, parse_dates=['game_date'])
pbp = pbp[pbp['season_type'] == 'REG'].copy()

# --------------------------
# STEP 2 — CLEAN DATA
# --------------------------
pbp['team_offense'] = pbp['posteam']
pbp_team = pbp[~pbp['team_offense'].isna()].copy()
pbp_team['is_play'] = pbp_team['play_type'].notna() & (pbp_team['play_type'] != 'no_play')

# Correct turnover detection
pbp_team['turnover'] = pbp_team[['interception','fumble_lost']].fillna(0).astype(int).any(axis=1)

pbp_team['success'] = pbp_team['epa'] > 0

# --------------------------
# STEP 3 — AGGREGATE PER TEAM-GAME
# --------------------------
agg = pbp_team.groupby(['game_id','team_offense']).agg(
    plays=('is_play','sum'),
    epa_per_play=('epa','mean'),
    yards_per_play=('yards_gained','mean'),
    success_rate=('success','mean'),
    turnovers=('turnover','sum'),
    pass_attempts=('pass_attempt','sum'),
    rush_attempts=('rush_attempt','sum')
).reset_index()

agg = agg.rename(columns={'team_offense':'team'})

# --------------------------
# STEP 4 — FINAL SCORES
# --------------------------
games = pbp.sort_values(['game_id','game_seconds_remaining']).groupby('game_id').agg(
    home_team=('home_team','first'),
    away_team=('away_team','first'),
    home_score=('home_score','last'),
    away_score=('away_score','last'),
    season=('season','first'),
    game_date=('game_date','first')
).reset_index()

# --------------------------
# STEP 5 — MERGE HOME/AWAY FEATURES
# --------------------------
def prefix_cols(df, prefix, cols):
    return df.rename(columns={c: f"{prefix}_{c}" for c in cols})

cols_to_prefix = ['team','plays','epa_per_play','yards_per_play','success_rate','turnovers','pass_attempts','rush_attempts']

home = agg.merge(
    games[['game_id','home_team','away_team','home_score','away_score','season','game_date']],
    left_on=['game_id','team'],
    right_on=['game_id','home_team'],
    how='inner'
)
away = agg.merge(
    games[['game_id','home_team','away_team','home_score','away_score','season','game_date']],
    left_on=['game_id','team'],
    right_on=['game_id','away_team'],
    how='inner'
)

home = prefix_cols(home, 'home', cols_to_prefix).drop(columns=['home_team','away_team'])
away = prefix_cols(away, 'away', cols_to_prefix).drop(columns=['home_team','away_team'])

game_df = games.merge(home, on='game_id', how='left').merge(away, on='game_id', how='left')

# --------------------------
# STEP 6 — TARGET VARIABLES
# --------------------------
game_df['home_margin'] = game_df['home_score'] - game_df['away_score']
game_df['home_win'] = (game_df['home_margin'] > 0).astype(int)

# --------------------------
# STEP 7 — ROLLING FEATURES
# --------------------------
team_games = pd.concat([
    game_df[['game_id','home_team','home_epa_per_play','home_success_rate','home_turnovers','game_date']].rename(
        columns={'home_team':'team','home_epa_per_play':'epa_per_play','home_success_rate':'success_rate','home_turnovers':'turnovers'}
    ),
    game_df[['game_id','away_team','away_epa_per_play','away_success_rate','away_turnovers','game_date']].rename(
        columns={'away_team':'team','away_epa_per_play':'epa_per_play','away_success_rate':'success_rate','away_turnovers':'turnovers'}
    )
], ignore_index=True).sort_values(['team','game_date'])

N = 3  # last 5 games
team_games[['rolling_epa','rolling_success','rolling_turnovers']] = team_games.groupby('team')[
    ['epa_per_play','success_rate','turnovers']
].transform(lambda x: x.shift().rolling(N, min_periods=1).mean())

# --------------------------
# Additional recent-weighted features (EWMA) and rest days
# --------------------------
for span in [3, 7]:
    team_games[f'ewma_epa_{span}'] = team_games.groupby('team')['epa_per_play'].transform(
        lambda s: s.shift().ewm(span=span, adjust=False).mean()
    )
    team_games[f'ewma_success_{span}'] = team_games.groupby('team')['success_rate'].transform(
        lambda s: s.shift().ewm(span=span, adjust=False).mean()
    )
    team_games[f'ewma_turnovers_{span}'] = team_games.groupby('team')['turnovers'].transform(
        lambda s: s.shift().ewm(span=span, adjust=False).mean()
    )

# Rest days: days since previous game for each team (shifted so current game isn't included)
team_games = team_games.sort_values(['team','game_date'])
team_games['last_game_date'] = team_games.groupby('team')['game_date'].shift(1)
team_games['days_since_last'] = (team_games['game_date'] - team_games['last_game_date']).dt.days.fillna(7)

# Merge rolling features back
game_df = game_df.merge(
    team_games[['game_id','team','rolling_epa','rolling_success','rolling_turnovers']].rename(
        columns={'team':'home_team','rolling_epa':'home_rolling_epa','rolling_success':'home_rolling_success','rolling_turnovers':'home_rolling_turnovers'}
    ), on=['game_id','home_team'], how='left'
)
game_df = game_df.merge(
    team_games[['game_id','team','rolling_epa','rolling_success','rolling_turnovers']].rename(
        columns={'team':'away_team','rolling_epa':'away_rolling_epa','rolling_success':'away_rolling_success','rolling_turnovers':'away_rolling_turnovers'}
    ), on=['game_id','away_team'], how='left'
)

# Merge EWMA and rest-day features back for home/away
ewma_cols = ['ewma_epa_3','ewma_epa_7','ewma_success_3','ewma_success_7','ewma_turnovers_3','ewma_turnovers_7','days_since_last']
game_df = game_df.merge(
    team_games[['game_id','team'] + ewma_cols].rename(
        columns={
            'team':'home_team',
            'ewma_epa_3':'home_ewma_epa_3', 'ewma_epa_7':'home_ewma_epa_7',
            'ewma_success_3':'home_ewma_success_3','ewma_success_7':'home_ewma_success_7',
            'ewma_turnovers_3':'home_ewma_turnovers_3','ewma_turnovers_7':'home_ewma_turnovers_7',
            'days_since_last':'home_days_rest'
        }
    ), on=['game_id','home_team'], how='left'
)
game_df = game_df.merge(
    team_games[['game_id','team'] + ewma_cols].rename(
        columns={
            'team':'away_team',
            'ewma_epa_3':'away_ewma_epa_3', 'ewma_epa_7':'away_ewma_epa_7',
            'ewma_success_3':'away_ewma_success_3','ewma_success_7':'away_ewma_success_7',
            'ewma_turnovers_3':'away_ewma_turnovers_3','ewma_turnovers_7':'away_ewma_turnovers_7',
            'days_since_last':'away_days_rest'
        }
    ), on=['game_id','away_team'], how='left'
)

# Feature differences
game_df['epa_diff'] = game_df['home_rolling_epa'] - game_df['away_rolling_epa']
game_df['success_diff'] = game_df['home_rolling_success'] - game_df['away_rolling_success']
game_df['turnover_diff'] = game_df['away_rolling_turnovers'] - game_df['home_rolling_turnovers']

# EWMA differences (short and medium term)
game_df['epa_diff_ewm_3'] = game_df['home_ewma_epa_3'].fillna(0) - game_df['away_ewma_epa_3'].fillna(0)
game_df['epa_diff_ewm_7'] = game_df['home_ewma_epa_7'].fillna(0) - game_df['away_ewma_epa_7'].fillna(0)
game_df['success_diff_ewm_3'] = game_df['home_ewma_success_3'].fillna(0) - game_df['away_ewma_success_3'].fillna(0)
game_df['turnover_diff_ewm_3'] = game_df['away_ewma_turnovers_3'].fillna(0) - game_df['home_ewma_turnovers_3'].fillna(0)

# Rest difference (home rest - away rest); positive means home is more rested
game_df['rest_diff'] = game_df['home_days_rest'].fillna(7) - game_df['away_days_rest'].fillna(7)

# --------------------------
# STEP 8 — SIMPLE ELO RATINGS
# --------------------------
def run_elo(games, k=20, base=1500):
    teams = {}
    records = []
    games = games.sort_values('game_date')
    for _, r in games.iterrows():
        ht = r['home_team']; at = r['away_team']
        h_elo = teams.get(ht, base)
        a_elo = teams.get(at, base)
        home_adv = 65
        exp = 1 / (1 + 10 ** ((a_elo - (h_elo + home_adv)) / 400))
        result = r['home_win']
        teams[ht] = h_elo + k * (result - exp)
        teams[at] = a_elo + k * ((1 - result) - (1 - exp))
        records.append({'game_id': r['game_id'], 'home_elo_pre': h_elo, 'away_elo_pre': a_elo})
    return pd.DataFrame(records)

elo_df = run_elo(game_df)
game_df = game_df.merge(elo_df, on='game_id', how='left')
game_df['elo_diff'] = game_df['home_elo_pre'] - game_df['away_elo_pre']

# --------------------------
# STEP 9 — PREPARE FEATURES FOR TF
# --------------------------
# Expanded features: include EWMA short/medium-term diffs and rest days
feature_cols = [
    'epa_diff', 'success_diff', 'turnover_diff', 'elo_diff',
    'epa_diff_ewm_3', 'epa_diff_ewm_7', 'rest_diff'
]
# Use time-aware split: sort by date then split to avoid leakage
game_df_sorted = game_df.sort_values('game_date').reset_index(drop=True)
X_sorted = game_df_sorted[feature_cols].fillna(0).values
y_sorted = game_df_sorted['home_win'].values

split_idx = int(len(game_df_sorted) * 0.8)
X_train = X_sorted[:split_idx]
X_test = X_sorted[split_idx:]
y_train = y_sorted[:split_idx]
y_test = y_sorted[split_idx:]

# Fit scaler on training data only to avoid leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# STEP 10 — BUILD TENSORFLOW MODEL
# --------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# --------------------------
# STEP 11 — TRAIN
# --------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=2
)

# --------------------------
# STEP 12 — EVALUATE
# --------------------------
results = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", results[0])
print("Test AUC:", results[1])
game_df.to_csv("game_df_processed.csv", index=False)

# Predict probabilities
probs = model.predict(X_test).flatten()
model.save("nfl_model.h5")
joblib.dump(scaler, "scaler.save")
