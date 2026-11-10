"""Cached data access for the MVP pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1ml import ledger
from f1ml.predict import next_upcoming_race, resolve_race_features


@st.cache_data(ttl=300, show_spinner=False)
def current_race() -> tuple[int, int, pd.DataFrame, str]:
    season, round_num = next_upcoming_race()
    race_df, mode = resolve_race_features(season, round_num)
    return season, round_num, race_df, mode


@st.cache_resource(show_spinner="Re-creating the simulated weekends…")
def snapshot_sim(snapshot_id: str, _payload: dict):
    """The exact simulated weekends behind a frozen snapshot (for team ranges)."""
    from f1ml.fantasy import sim_from_snapshot

    return sim_from_snapshot(_payload)


@st.cache_data(ttl=60, show_spinner=False)
def ledger_index() -> list[dict]:
    return ledger.read_index()


@st.cache_data(ttl=60, show_spinner=False)
def ledger_problems() -> list[str]:
    return ledger.verify_ledger()
