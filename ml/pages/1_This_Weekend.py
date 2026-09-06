"""This Weekend — race briefing and weekend card."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.predict import list_calendar_races, next_upcoming_race, resolve_race_features
from f1ml.ui.charts import market_comparison
from f1ml.ui.components import render_accuracy_journal, render_prediction_block, sidebar_model_controls
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="This Weekend | Pitstop Oracle", layout="wide")
apply_theme()

st.title("This Weekend")
st.caption("Race briefing: win, podium, DNF, grid penalties, fantasy points, and accuracy journal.")

with st.sidebar:
  model_kind, normalize = sidebar_model_controls(show_lab=False)

season, round_num = next_upcoming_race()
cal = list_calendar_races()
meta_row = cal[(cal["season"] == season) & (cal["round"] == round_num)].iloc[0]

race_df, mode = resolve_race_features(season, int(round_num))
scored = render_prediction_block(
  race_df, meta_row, mode, model_kind, normalize, season=season, round_num=round_num
)

st.subheader("Weekend markets")
st.plotly_chart(market_comparison(scored, title="Top drivers"), use_container_width=True, key="weekend_markets")

display_cols = [
  "given_name", "family_name", "constructor_name", "quali_position", "grid", "grid_vs_quali",
  "win_prob", "podium_prob", "dnf_prob", "expected_finish", "expected_fantasy_pts",
]
available = [c for c in display_cols if c in scored.columns]
view = scored[available].copy()
for c in ("win_prob", "podium_prob", "dnf_prob"):
  if c in view.columns:
    view[c] = (view[c] * 100).round(1)
if "expected_finish" in view.columns:
  view["expected_finish"] = view["expected_finish"].round(1)
if "expected_fantasy_pts" in view.columns:
  view["expected_fantasy_pts"] = view["expected_fantasy_pts"].round(1)
if "grid_vs_quali" in view.columns:
  view["grid_vs_quali"] = view["grid_vs_quali"].fillna(0).astype(int)

st.dataframe(
  view.rename(
    columns={
      "given_name": "First",
      "family_name": "Last",
      "constructor_name": "Team",
      "quali_position": "Quali",
      "grid": "Grid",
      "grid_vs_quali": "Grid Δ",
      "win_prob": "Win %",
      "podium_prob": "Podium %",
      "dnf_prob": "DNF %",
      "expected_finish": "Exp finish",
      "expected_fantasy_pts": "Fantasy",
    }
  ),
  use_container_width=True,
  hide_index=True,
)

st.subheader("Accuracy journal")
render_accuracy_journal(12)

disclaimer_footer()
