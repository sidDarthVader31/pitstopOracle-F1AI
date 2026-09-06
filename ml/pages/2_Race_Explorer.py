"""Race Explorer — browse past races and compare model vs reality."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.predict import list_races, resolve_race_features
from f1ml.ui.components import render_prediction_block, render_result_banner, sidebar_model_controls
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="Race Explorer | Pitstop Oracle", layout="wide")
apply_theme()

st.title("Race Explorer")

races = list_races()
with st.sidebar:
  model_kind, normalize = sidebar_model_controls(show_lab=True)
  wet_only = st.checkbox("Wet races only", value=False)

if wet_only:
  from f1ml.predict import load_driver_race

  df = load_driver_race()
  wet_keys = (
    df.groupby(["season", "round"])["is_wet"]
    .max()
    .reset_index()
  )
  wet_keys = wet_keys[wet_keys["is_wet"] == 1][["season", "round"]]
  races = races.merge(wet_keys, on=["season", "round"], how="inner")

labels = races["label"].tolist()
picked_label = st.selectbox("Grand Prix", options=labels, index=len(labels) - 1)
meta = races[races["label"] == picked_label].iloc[0]
season, round_num = int(meta["season"]), int(meta["round"])

race_df, mode = resolve_race_features(season, round_num)
scored = render_prediction_block(race_df, meta, mode, model_kind, normalize)
render_result_banner(scored, race_df)

view = scored.copy()
view["Driver"] = view["given_name"] + " " + view["family_name"]
view["Win %"] = (view["win_prob"] * 100).round(1)
if "podium_prob" in view.columns:
  view["Podium %"] = (view["podium_prob"] * 100).round(1)
if "dnf_prob" in view.columns:
  view["DNF %"] = (view["dnf_prob"] * 100).round(1)
if "expected_fantasy_pts" in view.columns:
  view["Fantasy"] = view["expected_fantasy_pts"].round(1)
view["Won"] = view["won"].map({1: "Yes", 0: ""})
cols = ["Driver", "constructor_name", "quali_position", "grid", "Win %"]
for c in ("Podium %", "DNF %", "Fantasy", "Won"):
  if c in view.columns:
    cols.append(c)
st.dataframe(
  view[cols].rename(
    columns={
      "constructor_name": "Team",
      "quali_position": "Quali",
      "grid": "Grid",
    }
  ),
  use_container_width=True,
  hide_index=True,
)

disclaimer_footer()
