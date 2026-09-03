#!/usr/bin/env python3
"""Pitstop Oracle — F1 winner predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(
  page_title="Pitstop Oracle",
  page_icon="🏎️",
  layout="wide",
  initial_sidebar_state="expanded",
)
apply_theme()

st.title("Pitstop Oracle")
st.markdown(
  """
  F1 race-winner probabilities from historical results (2022+).

  **Get started:** open **This Weekend** for the next Grand Prix prediction,
  or explore past races, run what-if scenarios, and review model accuracy.
  """
)
st.sidebar.title("Pitstop Oracle")
st.sidebar.caption("Use the pages above to navigate.")
disclaimer_footer()
