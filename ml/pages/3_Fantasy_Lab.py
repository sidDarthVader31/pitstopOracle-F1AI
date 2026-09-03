"""Fantasy Lab — what-if rain and grid penalties."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.predict import (
  apply_what_if,
  list_calendar_races,
  model_pick,
  next_upcoming_race,
  resolve_race_features,
  score_race,
)
from f1ml.ui.charts import compare_scenarios, win_prob_bar
from f1ml.ui.components import sidebar_model_controls
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="Fantasy Lab | Pitstop Oracle", layout="wide")
apply_theme()

st.title("Fantasy Lab")
st.caption("Toggle rain or grid penalties and see how win probabilities shift.")

cal = list_calendar_races()
default_season, default_round = next_upcoming_race()
labels = cal["label"].tolist()
default_label = cal[
  (cal["season"] == default_season) & (cal["round"] == default_round)
]["label"].iloc[0]
default_idx = labels.index(default_label) if default_label in labels else 0
with st.sidebar:
  model_kind, normalize = sidebar_model_controls()
  picked_label = st.selectbox("Grand Prix", options=labels, index=default_idx)

meta = cal[cal["label"] == picked_label].iloc[0]
season, round_num = int(meta["season"]), int(meta["round"])

race_df, mode = resolve_race_features(season, round_num)
baseline_scored = score_race(race_df, kind=model_kind, normalize=normalize)

st.subheader(meta["race_name"])
col1, col2 = st.columns(2)
with col1:
  is_wet = st.toggle("Wet race forecast", value=False)
with col2:
  use_penalty = st.toggle("Apply grid penalty", value=False)

grid_overrides: dict[str, int] = {}
if use_penalty:
  race_df_sorted = race_df.sort_values("quali_position", na_position="last")
  options = {
    f"{r['given_name']} {r['family_name']}": r["driver_id"]
    for _, r in race_df_sorted.iterrows()
  }
  driver_label = st.selectbox("Driver", options=list(options.keys()))
  default_grid = 10
  row = race_df_sorted[race_df_sorted["driver_id"] == options[driver_label]].iloc[0]
  if pd.notna(row.get("grid")):
    default_grid = int(row["grid"])
  elif pd.notna(row.get("quali_position")):
    default_grid = int(row["quali_position"])
  new_grid = st.slider("Starting grid position", 1, 20, default_grid)
  grid_overrides[options[driver_label]] = new_grid

scenario_df = apply_what_if(
  race_df,
  is_wet=is_wet,
  grid_overrides=grid_overrides or None,
)
scenario_scored = score_race(scenario_df, kind=model_kind, normalize=normalize)

b_pick = model_pick(baseline_scored)
s_pick = model_pick(scenario_scored)
c1, c2 = st.columns(2)
c1.metric("Baseline pick", b_pick["driver_name"], f"{b_pick['win_prob'] * 100:.1f}%")
c2.metric("Scenario pick", s_pick["driver_name"], f"{s_pick['win_prob'] * 100:.1f}%")

st.plotly_chart(compare_scenarios(baseline_scored, scenario_scored), use_container_width=True)

tab1, tab2 = st.tabs(["Baseline", "Scenario"])
with tab1:
  st.plotly_chart(win_prob_bar(baseline_scored), use_container_width=True)
with tab2:
  st.plotly_chart(win_prob_bar(scenario_scored), use_container_width=True)

disclaimer_footer()
