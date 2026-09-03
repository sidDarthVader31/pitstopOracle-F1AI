"""This Weekend — hero prediction for the next GP."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.predict import list_calendar_races, next_upcoming_race, resolve_race_features
from f1ml.ui.components import render_prediction_block, sidebar_model_controls
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="This Weekend | Pitstop Oracle", layout="wide")
apply_theme()

st.title("This Weekend")

with st.sidebar:
  model_kind, normalize = sidebar_model_controls()

season, round_num = next_upcoming_race()
cal = list_calendar_races()
meta_row = cal[(cal["season"] == season) & (cal["round"] == round_num)].iloc[0]

race_df, mode = resolve_race_features(season, int(round_num))
scored = render_prediction_block(race_df, meta_row, mode, model_kind, normalize)

st.dataframe(
  scored[
    [
      "given_name",
      "family_name",
      "constructor_name",
      "quali_position",
      "grid",
      "win_prob",
      "champ_position_before",
    ]
  ].assign(win_prob=lambda d: (d["win_prob"] * 100).round(1)),
  use_container_width=True,
  hide_index=True,
)

disclaimer_footer()
