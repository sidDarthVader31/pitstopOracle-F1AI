#!/usr/bin/env python3
"""Pitstop Oracle: unofficial F1 forecast and fantasy helper (Streamlit MVP, MVP_SPEC §4).

    python -m streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Pitstop Oracle", page_icon="🏁", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "": [
        st.Page("pages/home.py", title="This Weekend", icon="🏁", default=True),
        st.Page("pages/fantasy_helper.py", title="Fantasy Helper (unofficial)", icon="🧮"),
        st.Page("pages/track_record.py", title="Track Record", icon="📒"),
        st.Page("pages/how_it_works.py", title="How it works & limits", icon="ℹ️"),
    ],
    "Explore": [
        st.Page("pages/scenario_lab.py", title="Scenario Lab (hypothetical)", icon="🧪"),
        st.Page("pages/race_explorer.py", title="Race Explorer", icon="🗂️"),
    ],
}

st.navigation(PAGES).run()
