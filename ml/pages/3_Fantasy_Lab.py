"""Fantasy Lab — budget optimizer and what-if scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.fantasy import constructor_fantasy_points, lineup_to_dataframe, optimize_fantasy_team
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
st.caption("Official-style $100M optimizer (5 drivers + 2 constructors) with what-if scenarios.")

cal = list_calendar_races()
default_season, default_round = next_upcoming_race()
labels = cal["label"].tolist()
default_label = cal[
  (cal["season"] == default_season) & (cal["round"] == default_round)
]["label"].iloc[0]
default_idx = labels.index(default_label) if default_label in labels else 0
with st.sidebar:
  model_kind, normalize = sidebar_model_controls(show_lab=True)
  picked_label = st.selectbox("Grand Prix", options=labels, index=default_idx)

meta = cal[cal["label"] == picked_label].iloc[0]
season, round_num = int(meta["season"]), int(meta["round"])

race_df, mode = resolve_race_features(season, round_num)
baseline_scored = weekend_card(race_df, race_mode=mode, kind=model_kind)
if not normalize:
  baseline_scored["win_prob"] = baseline_scored["win_prob_raw"]

st.subheader(meta["race_name"])

opt_tab, whatif_tab = st.tabs(["Team optimizer", "What-if scenarios"])

with opt_tab:
  mode_pick = st.selectbox(
    "Strategy",
    options=["balanced", "safe", "aggressive", "value"],
    format_func=lambda x: x.title(),
  )
  budget = st.slider("Budget ($M)", min_value=90.0, max_value=100.0, value=100.0, step=0.5)
  lineup = optimize_fantasy_team(
    baseline_scored,
    season,
    round_num,
    budget_m=budget,
    mode=mode_pick,
  )
  c1, c2, c3 = st.columns(3)
  c1.metric("Expected points", f"{lineup.expected_points:.1f}")
  c2.metric("Total cost", f"${lineup.total_cost_m:.1f}M")
  c3.metric("Strategy", lineup.mode.title())
  if lineup.drivers:
    st.dataframe(lineup_to_dataframe(lineup), use_container_width=True, hide_index=True)
    st.caption(
      "Add weekly prices to `ml/data/manual/fantasy_prices.csv` for official F1 Fantasy costs. "
      "Default costs used when file is empty."
    )
  else:
    st.warning("No valid lineup found within budget — widen budget or add price data.")

  st.subheader("Constructor expected points")
  st.dataframe(
    constructor_fantasy_points(baseline_scored).rename(columns={"constructor_name": "Team"}),
    use_container_width=True,
    hide_index=True,
  )

with whatif_tab:
  col1, col2 = st.columns(2)
  with col1:
    is_wet = st.toggle("Wet race forecast", value=False)
  with col2:
    use_penalty = st.toggle("Apply extra grid penalty", value=False)

  grid_overrides: dict[str, int] = {}
  if use_penalty:
    race_df_sorted = race_df.sort_values("quali_position", na_position="last")
    options = {
      f"{r['given_name']} {r['family_name']}": r["driver_id"]
      for _, r in race_df_sorted.iterrows()
    }
    driver_label = st.selectbox("Driver", options=list(options.keys()))
    row = race_df_sorted[race_df_sorted["driver_id"] == options[driver_label]].iloc[0]
    default_grid = int(row["grid"]) if pd.notna(row.get("grid")) else (
      int(row["quali_position"]) if pd.notna(row.get("quali_position")) else 10
    )
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
  c1, c2 = st.columns(2)
  c1.metric("Baseline pick", b_pick["driver_name"], f"{b_pick['win_prob'] * 100:.1f}%")
  c2.metric("Scenario pick", s_pick["driver_name"], f"{s_pick['win_prob'] * 100:.1f}%")
  st.plotly_chart(compare_scenarios(baseline_scored, scenario_scored), use_container_width=True, key="whatif_compare")
  col_a, col_b = st.columns(2)
  with col_a:
    st.plotly_chart(win_prob_bar(baseline_scored), use_container_width=True, key="whatif_win_base")
  with col_b:
    st.plotly_chart(win_prob_bar(scenario_scored), use_container_width=True, key="whatif_win_scenario")
  col_a, col_b = st.columns(2)
  with col_a:
    st.plotly_chart(fantasy_pts_bar(baseline_scored), use_container_width=True, key="whatif_fant_base")
  with col_b:
    st.plotly_chart(fantasy_pts_bar(scenario_scored), use_container_width=True, key="whatif_fant_scenario")

disclaimer_footer()
