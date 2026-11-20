"""How it works & limits (MVP_SPEC §4.2 page 3, wireframe §6.5)."""
from __future__ import annotations

import streamlit as st

from f1ml.fantasy_rules import load_rules
from f1ml.ui.data import current_race
from f1ml.ui.mvp import page_setup, render_footer, render_waitlist

page_setup("How it works")
season, *_ = current_race()
rules = load_rules(season)

st.title("How it works & limits")

st.subheader("In one minute")
st.markdown(
    "We rate each car and driver from recent races, add qualifying and the starting grid when they're known, then "
    "**simulate the race 6,000 times**. Win and podium chances are how often each driver wins or finishes top 3 in "
    "those simulations. Fantasy points are counted in every simulated weekend: **projected** is the average, and the "
    "**likely range** covers 8 in 10 simulations. Every forecast is frozen with a timestamp before the session it "
    "predicts and published on the Track Record, hits and misses alike."
)

st.subheader("What we can't do (yet)")
st.markdown(
    "- **No practice or weather data.** Rain and big setup changes surprise the model.\n"
    "- **After qualifying, the starting grid is hard to beat.** On races the model hadn't seen we've been better than "
    "the grid on average, but not every race; the Track Record shows the comparison.\n"
    "- **Retirements are a team's season rate, not a prediction.** The model can't tell who will retire.\n"
    "- **Qualifying before it happens is estimated** from the same pace ratings (there's no separate qualifying model), "
    "so qualifying points before the lock are rough.\n"
    "- **Overtakes, fastest lap and pit stops are estimates; Driver of the Day isn't modelled.**\n"
    "- **New-season surprises.** Early-season forecasts lean on last year until new results come in.\n"
    "- We don't claim our projections pick a better fantasy team than simple rules of thumb. We publish the record "
    "so you can judge."
)

st.subheader("Glossary")
st.markdown(
    "- **Projection:** our average fantasy points over the simulated weekends.\n"
    "- **Likely range:** the middle 8 in 10 simulated outcomes (10th to 90th percentile). A wide range means an "
    "uncertain weekend; a negative low end is usually the retirement case (−20).\n"
    "- **Chance band:** Favourite (about 30% or more to win), Contender (10–30%), Outside chance (3–10%), Long shot "
    "(under 3%). We show bands instead of exact percentages until our calibration checks pass.\n"
    "- **Baseline:** a simple rule to compare against, like trusting the starting grid or a driver's last 3 races.\n"
    "- **Walk-forward:** testing on each past race using only races before it, so the model never sees the answer.\n"
    "- **Log-loss:** our *surprise score* when the result comes in. Lower is better."
)

st.subheader("Data and credits")
st.markdown(
    f"Race results, qualifying and standings: [Jolpica F1 API](https://github.com/jolpica/jolpica-f1). Practice and "
    f"session data: [OpenF1](https://openf1.org). Weather: [Open-Meteo](https://open-meteo.com). "
    f"Fantasy scoring: rules v{rules.version}, our reading of the published game rules. Prices are entered by hand "
    f"each week."
)

st.subheader("Notices")
st.markdown(
    "Pitstop Oracle is **unofficial**. It is not affiliated with, endorsed by or connected to Formula 1, the Formula One "
    "group (FOM), the FIA, any team, or the F1 Fantasy game. F1, FORMULA 1, FORMULA ONE, GRAND PRIX and related marks "
    "are trade marks of Formula One Licensing B.V. Projections are for fun and information, **not advice, and not "
    "betting advice**.\n\n"
    "**Privacy.** This site has no accounts. Check my team runs in your session and nothing about your team is saved. "
    "The email sign-up below is a placeholder and stores nothing yet; when it goes live we'll ask only for your email, "
    "use it only for the weekly pre-lock email, and let you unsubscribe any time."
)

st.subheader("Get the weekly pre-lock email")
render_waitlist("how")
render_footer()
