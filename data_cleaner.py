import pandas as pd
pbp = pd.read_csv("pbp_2023_2024.csv", parse_dates=["game_date"])
pbp.info()
pbp.shape
pbp[['game_id','play_id','season','home_team','away_team','posteam','defteam','epa','wp']].head()
pbp.isna().sum().sort_values(ascending=False).head(20)
