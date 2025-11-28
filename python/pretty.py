# pretty_nfl.py
import pandas as pd
from pathlib import Path

# --------------------------
# Load predictions
# --------------------------
df = pd.read_csv("nfl_upcoming_predictions.csv")

# --------------------------
# Convert decimal odds to US odds
# --------------------------
def decimal_to_us(decimal_odds):
    if pd.isna(decimal_odds):
        return None
    if decimal_odds >= 2.0:
        return f"+{int(round((decimal_odds - 1) * 100))}"
    else:
        return f"-{int(round(100 / (decimal_odds - 1)))}"

df['home_us_odds'] = df['home_decimal_odds'].apply(decimal_to_us)
df['away_us_odds'] = df['away_decimal_odds'].apply(decimal_to_us)

# --------------------------
# Convert probabilities and edges to percentages
# --------------------------
percent_cols = ['home_win_prob','away_win_prob','home_implied_prob','away_implied_prob','home_edge','away_edge']
for col in percent_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') * 100

# --------------------------
# Highlight edges
# --------------------------
def highlight_edge(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "background-color: #c6f5c6"  # light green
    elif val < 0:
        return "background-color: #f5c6c6"  # light red
    return ""

# --------------------------
# Style dataframe
# --------------------------
styler = df[['home_team','away_team','home_us_odds','away_us_odds',
             'home_win_prob','away_win_prob','home_edge','away_edge']].style.format({
    'home_win_prob': '{:.1f}%',
    'away_win_prob': '{:.1f}%',
    'home_edge': '{:.1f}%',
    'away_edge': '{:.1f}%'
}).applymap(highlight_edge, subset=['home_edge','away_edge']) \
  .set_table_styles([
      {'selector':'th', 'props':[('background-color','#2c3e50'),
                                  ('color','white'),
                                  ('font-weight','bold'),
                                  ('text-align','center'),
                                  ('padding','8px')]},
      {'selector':'td', 'props':[('text-align','center'),
                                  ('padding','6px')]}
  ]) \
  .set_properties(**{'border':'1px solid #ddd'}) \
  .set_caption("NFL Upcoming Predictions & Betting Edge") \
  .hide(axis="index")  # removes row index

# --------------------------
# Save to HTML
# --------------------------
repo_root = Path(__file__).resolve().parents[1]
site_dir = repo_root / "site"
site_dir.mkdir(parents=True, exist_ok=True)

html_file = site_dir / "nfl_upcoming_predictions.html"
styler.to_html(str(html_file))
print(f"HTML output saved to {html_file}")
