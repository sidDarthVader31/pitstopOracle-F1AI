"""Race Explorer: past races, model replay vs real pre-race forecast (MVP_SPEC S2)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1ml import ledger
from f1ml.predict import list_races, resolve_race_features, weekend_card
from f1ml.ui.mvp import PODIUM_BANDS, WIN_BANDS, band, page_setup, render_footer

page_setup("Race Explorer")
st.title("Race Explorer")

races = list_races()
races = races[races["has_winner"] == 1]
labels = races["label"].tolist()
picked = st.selectbox("Grand Prix", labels, index=len(labels) - 1)
meta = races[races["label"] == picked].iloc[0]
season, round_num = int(meta["season"]), int(meta["round"])

frozen = ledger.snapshots_for(season, round_num)
if frozen:
    st.success(f"A real pre-race forecast exists for this race ({len(frozen)} frozen snapshot(s)); see the Track Record.")
st.info("**Replay with today's model.** This is not what we published at the time: today's model was trained on "
        "data up to now, which can include this race. Use it to explore, not to judge accuracy (the Track Record does that).")

race_df, mode = resolve_race_features(season, round_num)
scored = weekend_card(race_df, race_mode=mode)
winner = race_df.loc[race_df["won"] == 1]
if not winner.empty:
    w = winner.iloc[0]
    st.markdown(f"**Winner:** {w['given_name']} {w['family_name']} ({w['constructor_name']}) · {meta['circuit_name']} · {meta['date']}")

res = race_df.set_index("driver_id")
view = pd.DataFrame({
    "Driver": scored["given_name"] + " " + scored["family_name"],
    "Team": scored["constructor_name"],
    "Grid": scored["grid"].astype("Int64"),
    "Win (replay)": [band(p, WIN_BANDS) for p in scored["win_prob"]],
    "Podium (replay)": [band(p, PODIUM_BANDS) for p in scored["podium_prob"]],
    "Expected finish": scored["expected_finish"].round(1),
    "Actual finish": [("P" + str(int(res.loc[d, "position"]))) if pd.notna(res.loc[d, "position"]) else "DNF"
                      for d in scored["driver_id"]],
})
st.dataframe(view, use_container_width=True, hide_index=True)
st.caption("Expected finish is the average finishing position when the driver is classified.")
render_footer()
