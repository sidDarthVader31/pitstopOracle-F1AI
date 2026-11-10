"""Scenario Lab (hypothetical): rain and grid-penalty what-ifs (MVP_SPEC S1).

Scenarios are never frozen in the ledger and never feed the Fantasy Helper or the Track Record.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1ml.fantasy import project_weekend
from f1ml.predict import WET_SCENARIO_TEMPERATURE_FACTOR, apply_what_if, inference_mode, load_weekend_models, weekend_card
from f1ml.ui.data import current_race
from f1ml.ui.mvp import PODIUM_BANDS, WIN_BANDS, band, page_setup, render_footer

page_setup("Scenario Lab")
season, round_num, race_df, mode = current_race()

st.title("Scenario Lab")
st.markdown('<div class="po-hypothetical">HYPOTHETICAL · What-ifs from the current model. Not a forecast, not saved, '
            'and not used by the Fantasy Helper or the Track Record.</div>', unsafe_allow_html=True)
st.caption(f"{race_df['race_name'].iloc[0]} · baseline = the model with the data we have now")

c1, c2 = st.columns(2)
with c1:
    is_wet = st.toggle("What if it rains?", value=False)
    if is_wet:
        st.caption("Assumption, not learned: the model has no wet-race data, so rain is simulated as a more random race "
                   "(gaps between drivers shrink). Outsiders gain; favourites lose a little.")
with c2:
    use_penalty = st.toggle("What if a driver takes a grid penalty?", value=False)

grid_overrides: dict[str, int] = {}
if use_penalty:
    ordered = race_df.sort_values("quali_position", na_position="last")
    options = {f"{r['given_name']} {r['family_name']}": r["driver_id"] for _, r in ordered.iterrows()}
    label = st.selectbox("Driver", list(options))
    row = ordered[ordered["driver_id"] == options[label]].iloc[0]
    default = int(row["grid"]) if pd.notna(row.get("grid")) else int(row["quali_position"]) if pd.notna(row.get("quali_position")) else 10
    grid_overrides[options[label]] = st.slider("Starts from grid position", 1, len(race_df), max(default, 1))

baseline = weekend_card(race_df, race_mode=mode)
scenario = weekend_card(apply_what_if(race_df, is_wet=is_wet, grid_overrides=grid_overrides or None), race_mode=mode)
temperature = load_weekend_models(inference_mode(mode)).temperature


def _weekend_points(scored: pd.DataFrame, t: float) -> pd.DataFrame:
    sim = project_weekend(scored, season=season, round_num=round_num, temperature=t, n_samples=3000)
    return sim.driver_summary()[["driver_id", "expected"]].rename(columns={"expected": "fx"})


baseline = baseline.merge(_weekend_points(baseline, temperature), on="driver_id")
scenario = scenario.merge(_weekend_points(scenario, temperature * (WET_SCENARIO_TEMPERATURE_FACTOR if is_wet else 1.0)),
                          on="driver_id")
merged = baseline[["driver_id", "given_name", "family_name", "constructor_name", "win_prob", "podium_prob",
                   "fx"]].merge(
    scenario[["driver_id", "win_prob", "podium_prob", "fx"]], on="driver_id", suffixes=("", "_s"))


def _move(a: float, b: float) -> str:
    if b > a * 1.15 + 0.005:
        return "▲ up"
    if b < a / 1.15 - 0.005:
        return "▼ down"
    return "≈"


view = pd.DataFrame({
    "Driver": merged["given_name"] + " " + merged["family_name"],
    "Team": merged["constructor_name"],
    "Win (baseline)": [band(p, WIN_BANDS) for p in merged["win_prob"]],
    "Win (scenario)": [band(p, WIN_BANDS) for p in merged["win_prob_s"]],
    "Change": [_move(a, b) for a, b in zip(merged["win_prob"], merged["win_prob_s"])],
    "Podium (scenario)": [band(p, PODIUM_BANDS) for p in merged["podium_prob_s"]],
    "Fantasy pts (baseline → scenario)": [f"{round(a)} → {round(b)}" for a, b in
                                          zip(merged["fx"], merged["fx_s"])],
})
st.dataframe(view.head(12), use_container_width=True, hide_index=True)
st.caption("Baseline here is the current model re-run on today's data, so it can differ slightly from the published "
           "forecast. A changed input is not a cause: the model only shows how its own numbers move.")
render_footer()
