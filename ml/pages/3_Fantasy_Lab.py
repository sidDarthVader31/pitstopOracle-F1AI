"""Fantasy Lab — what-if scenarios with expected fantasy points."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.fantasy import constructor_fantasy_points, oracle_xi
from f1ml.predict import (
  apply_what_if,
  list_calendar_races,
  model_pick,
  next_upcoming_race,
  resolve_race_features,
  weekend_card,
)
from f1ml.ui.charts import compare_scenarios, fantasy_pts_bar, win_prob_bar
from f1ml.ui.components import sidebar_model_controls
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="Fantasy Lab | Pitstop Oracle", layout="wide")
apply_theme()

st.title("Fantasy Lab")
st.caption("What-if rain or grid penalties — see win % and expected fantasy points shift.")

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
baseline_scored = weekend_card(race_df, race_mode=mode, kind=model_kind)
if not normalize:
  baseline_scored["win_prob"] = baseline_scored["win_prob_raw"]

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
scenario_scored = weekend_card(scenario_df, race_mode=mode, kind=model_kind)
if not normalize:
  scenario_scored["win_prob"] = scenario_scored["win_prob_raw"]

b_pick = model_pick(baseline_scored)
s_pick = model_pick(scenario_scored)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline pick", b_pick["driver_name"], f"{b_pick['win_prob'] * 100:.1f}%")
c2.metric("Scenario pick", s_pick["driver_name"], f"{s_pick['win_prob'] * 100:.1f}%")
b_fant = baseline_scored.loc[baseline_scored["driver_id"] == b_pick["driver_id"], "expected_fantasy_pts"]
s_fant = scenario_scored.loc[scenario_scored["driver_id"] == s_pick["driver_id"], "expected_fantasy_pts"]
c3.metric("Baseline fantasy pts", f"{b_fant.iloc[0]:.1f}" if len(b_fant) else "—")
c4.metric("Scenario fantasy pts", f"{s_fant.iloc[0]:.1f}" if len(s_fant) else "—")

st.plotly_chart(compare_scenarios(baseline_scored, scenario_scored), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Win %", "Fantasy points", "Oracle XI"])
with tab1:
  col_a, col_b = st.columns(2)
  with col_a:
    st.plotly_chart(win_prob_bar(baseline_scored), use_container_width=True)
  with col_b:
    st.plotly_chart(win_prob_bar(scenario_scored), use_container_width=True)
with tab2:
  col_a, col_b = st.columns(2)
  with col_a:
    st.plotly_chart(fantasy_pts_bar(baseline_scored), use_container_width=True)
  with col_b:
    st.plotly_chart(fantasy_pts_bar(scenario_scored), use_container_width=True)
with tab3:
  st.markdown("**Baseline Oracle XI** (greedy value picks)")
  st.dataframe(oracle_xi(baseline_scored), use_container_width=True, hide_index=True)
  st.markdown("**Scenario Oracle XI**")
  st.dataframe(oracle_xi(scenario_scored), use_container_width=True, hide_index=True)

st.subheader("Constructor expected points")
c_base = constructor_fantasy_points(baseline_scored)
c_scen = constructor_fantasy_points(scenario_scored)
st.dataframe(
  c_base.merge(
    c_scen[["constructor_id", "expected_fantasy_pts"]],
    on="constructor_id",
    suffixes=("_baseline", "_scenario"),
  ).rename(columns={"constructor_name": "Team"}),
  use_container_width=True,
  hide_index=True,
)

disclaimer_footer()
