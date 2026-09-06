#!/usr/bin/env python3
"""Pitstop Oracle — F1 weekend forecasts."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from f1ml.predict import next_upcoming_race
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(
  page_title="Pitstop Oracle",
  page_icon="🏎️",
  layout="wide",
  initial_sidebar_state="expanded",
)
apply_theme()

season, round_num = next_upcoming_race()

st.title("Pitstop Oracle")
st.markdown(
  f"""
  **Race weekend intelligence** — win probabilities, grid penalties, fantasy optimizer,
  and a public accuracy journal. Trained on 2022+ seasons.

  **Next up:** season **{season}**, round **{round_num}** — open **This Weekend** for the full briefing.
  """
)

st.markdown(
  """
  | Page | What you get |
  |---|---|
  | **This Weekend** | Model pick, grid penalties, markets, accuracy journal |
  | **Fantasy Lab** | $100M team optimizer + what-if rain/penalties |
  | **Race Explorer** | Archive any completed GP |
  | **Model Performance** | Log-loss, Brier, calibration vs pole baseline |
  """
)

st.sidebar.title("Pitstop Oracle")
st.sidebar.caption("Navigate via the pages menu above.")
st.sidebar.markdown("[Dataset schema](../docs/DATASET.md)")
disclaimer_footer()
