"""Home: This Weekend (MVP_SPEC §4.2 page 0, wireframe §6.1)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1ml.predict import driver_insight_bullets
from f1ml.ui.data import current_race
from f1ml.ui.mvp import (
    PODIUM_BANDS, TOP10_BANDS, WIN_BAND_COPY, WIN_BANDS, band, band_cell, dev_mode, display_stage, drivers_frame,
    next_run_text, page_link, page_setup, points_range, render_footer, render_freshness, render_limits_box, render_lock_line,
    render_stage_chip, render_waitlist, weekend_state,
)
from f1ml.ui.theme import hero_card

page_setup("This Weekend")
season, round_num, race_df, data_stage = current_race()
state = weekend_state(season, round_num, data_stage)
meta = race_df.iloc[0]

st.title(f"{meta['race_name']}")
race_start = state.weekend.race_start if state.weekend else None
st.caption(f"{meta.get('circuit_name', '')} · race {race_start:%a %d %b %H:%M} UTC" if race_start else str(meta.get("circuit_name", "")))

with st.sidebar:
    render_waitlist("home")

if state.payload is None:
    render_stage_chip("not_published")
    render_lock_line(state)
    st.warning(f"The forecast for this weekend hasn't been published yet. {next_run_text(state)}")
    st.caption("Every forecast is frozen with a timestamp before the session it predicts, so it can be checked later.")
    render_limits_box()
    render_footer()
    st.stop()

payload = state.payload
stage = display_stage(payload)
render_stage_chip(stage)
render_lock_line(state)
render_freshness(payload)
if state.stale:
    st.warning(state.stale)
if state.lock_passed and stage != "pre_quali":
    st.info("Team lock has passed. Final Fix users: one change allowed before the race.")

df = drivers_frame(payload).sort_values("win_prob", ascending=False).reset_index(drop=True)
has_fx = "fx_expected" in df

c1, c2, c3 = st.columns(3)
with c1:
    if has_fx:
        top = df.loc[df["fx_expected"].idxmax()]
        hero_card("Top projected driver", top["name"], f"{round(top['fx_expected'])} pts (likely {points_range(top['fx_p10'], top['fx_p90'])})")
    else:
        hero_card("Top projected driver", "—", "Projections not in this snapshot")
with c2:
    fav = df.iloc[0]
    hero_card("Most likely winner", fav["name"], f"{band_cell(fav['win_prob'], WIN_BANDS)}")
with c3:
    grid_status = payload.get("grid_status")
    if grid_status == "official":
        pole = df.dropna(subset=["grid"]).sort_values("grid").iloc[0]
        hero_card("Starting grid", "Official grid in", f"Pole: {pole['name']}")
    elif grid_status == "quali_order":
        hero_card("Starting grid", "Qualifying done", "Official grid not published yet")
    else:
        when = f"{state.weekend.quali_start:%a %H:%M} UTC" if state.weekend and state.weekend.quali_start else "Saturday"
        hero_card("Starting grid", "Not set yet", f"Qualifying {when}")

st.subheader("Contenders")
view = pd.DataFrame({
    "#": range(1, len(df) + 1),
    "Driver": df["name"],
    "Team": df["constructor_name"],
    "Win": [band_cell(p, WIN_BANDS) for p in df["win_prob"]],
    "Podium": [band_cell(p, PODIUM_BANDS) for p in df["podium_prob"]],
    "Top 10": [band_cell(p, TOP10_BANDS) for p in df["points_prob"]],
})
if has_fx:
    view["Projected pts"] = df["fx_expected"].round().astype(int)
    view["Likely range"] = [points_range(a, b) for a, b in zip(df["fx_p10"], df["fx_p90"])]
if stage != "pre_quali":
    view.insert(3, "Grid", df["grid"].astype("Int64"))
show_all = st.toggle("Show all drivers", value=False)
st.dataframe(view if show_all else view.head(8), use_container_width=True, hide_index=True)
st.caption("Chance bands: Favourite = about 30% or more, Contender 10–30%, Outside chance 3–10%, Long shot under 3%. "
           "We show bands, not exact percentages, until our calibration checks pass.")

feats = race_df.set_index("driver_id")
cols = st.columns(3)
for col, (_, row) in zip(cols, df.head(3).iterrows()):
    with col:
        st.markdown(f"**Why {row['family_name']}?** · {WIN_BAND_COPY.get(band(row['win_prob'], WIN_BANDS), '')}")
        if row["driver_id"] in feats.index:
            for b in driver_insight_bullets(feats.loc[row["driver_id"]]):
                st.markdown(f"- {b}")

for w in payload.get("warnings", []):
    if "Rain" in w:
        st.warning(w)

render_limits_box()

l1, l2 = st.columns(2)
with l1:
    page_link("pages/fantasy_helper.py", "Fantasy Helper → projected points and Check my team", "🧮")
with l2:
    page_link("pages/track_record.py", "Track Record → every forecast we've published, scored", "📒")

if dev_mode():
    with st.expander("Developer details"):
        st.json({k: v for k, v in payload.items() if k not in ("drivers", "constructors")})
        st.dataframe(df[["driver_id", "win_prob", "podium_prob", "points_prob", "dnf_prob", "expected_finish"]])

render_footer()
