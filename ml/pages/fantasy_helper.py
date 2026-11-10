"""Fantasy Helper (unofficial): projected points, Check my team, Oracle team (MVP_SPEC §6.2–6.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from f1ml.fantasy import check_my_team, check_prices
from f1ml.fantasy_rules import load_rules
from f1ml.ui.data import current_race, snapshot_sim
from f1ml.ui.mvp import (
    Q3_BANDS, TOP10_BANDS, band, display_stage, drivers_frame, fmt_utc, next_run_text, page_setup, points_range,
    render_footer, render_freshness, render_limits_box, render_lock_line, render_stage_chip, retirement_text,
    weekend_state,
)

page_setup("Fantasy Helper")
season, round_num, race_df, data_stage = current_race()
state = weekend_state(season, round_num, data_stage)
rules = load_rules(season)

st.title("Fantasy Helper (unofficial)")
st.caption(
    "Not affiliated with Formula 1 or the official F1 Fantasy game. "
    f"Scoring: {season} rules v{rules.version} as we understand them; check the official game."
)
if rules.is_placeholder:
    st.warning(f"The {season} scoring rules aren't confirmed yet: projections use last season's values.")

if state.payload is None:
    render_stage_chip("not_published")
    render_lock_line(state)
    st.warning(f"Projections for {race_df['race_name'].iloc[0]} haven't been published yet. {next_run_text(state)}")
    render_footer()
    st.stop()

payload = state.payload
render_stage_chip(display_stage(payload), race_df["race_name"].iloc[0])
render_lock_line(state)
render_freshness(payload)
if state.stale:
    st.warning(state.stale)

df = drivers_frame(payload)
if "fx_expected" not in df:
    st.error("This snapshot has no fantasy projections.")
    render_footer()
    st.stop()

prices = check_prices(season, round_num, race_df, rules)
dp, cp = prices.driver_price(), prices.constructor_price()
if not prices.ok:
    st.warning(
        "**Prices for this round haven't been entered yet** (or don't check out), so prices, points per \\$M, "
        "the budget check and the Oracle team are hidden. Projected points below are still valid."
        + ("" if prices.status == "missing" else "  \nProblems: " + "; ".join(prices.problems))
    )

names = dict(zip(df["driver_id"], df["name"]))
teams = {c["constructor_id"]: c.get("constructor_name") or c["constructor_id"] for c in payload["constructors"]}
retire = df.groupby("constructor_id")["dnf_prob"].mean().to_dict()

tab_d, tab_c, tab_team, tab_oracle = st.tabs(["Drivers", "Constructors", "Check my team", "Oracle team"])

stage_pre = payload["stage"] == "pre_quali"

with tab_d:
    sort = st.radio("Sort by", ["Projected pts", "Pts per \\$M"] if prices.ok else ["Projected pts"], horizontal=True)
    view = pd.DataFrame({
        "Driver": df["name"],
        "Team": df["constructor_name"],
        "Projected": df["fx_expected"].round().astype(int),
        "Likely range": [points_range(a, b) for a, b in zip(df["fx_p10"], df["fx_p90"])],
        "Q3": ([band(p, Q3_BANDS) for p in df["fx_q3_prob"]] if stage_pre
               else [f"P{int(q)}" if pd.notna(q) else "—" for q in df["quali_position"]]),
        "Top 10": [band(p, TOP10_BANDS) for p in df["points_prob"]],
        "Team retirement rate": [retirement_text(p) for p in df["dnf_prob"]],
    })
    if not stage_pre:
        view = view.rename(columns={"Q3": "Qualified"})
    if prices.ok:
        view.insert(2, "$M", df["driver_id"].map(dp).map(lambda x: f"{x:.1f}"))
        view.insert(5, "Pts/$M", (df["fx_expected"] / df["driver_id"].map(dp)).round(2))
    key = "Pts/$M" if sort.startswith("Pts per") else "Projected"
    order = view[key].sort_values(ascending=False).index
    st.dataframe(view.loc[order], use_container_width=True, hide_index=True)
    st.download_button("⬇ Download CSV", view.loc[order].to_csv(index=False).encode(),
                       file_name=f"pitstop-oracle-{season}-R{round_num:02d}-drivers.csv", mime="text/csv")
    with st.expander('How is "Projected" worked out?'):
        pick = st.selectbox("Driver", df.sort_values("fx_expected", ascending=False)["driver_id"],
                            format_func=lambda d: names[d])
        b = df.set_index("driver_id").loc[pick, "breakdown"]
        labels = {"qualifying": "Qualifying", "race_finish": "Race finish", "positions": "Places gained/lost",
                  "overtakes": "Overtakes (estimate)", "fastest_lap": "Fastest lap (estimate)",
                  "retirement": "Retirement risk", "sprint": "Sprint"}
        st.markdown(" · ".join(f"{labels.get(k, k)} {v:+.1f}" for k, v in b.items() if abs(v) >= 0.05))
        st.caption("We simulate the weekend 6,000 times and count fantasy points in each one. Projected = the average; "
                   "the likely range covers 8 in 10 simulations. Before qualifying, qualifying points come from a "
                   "simulated qualifying order. Driver of the Day isn't modelled.")

with tab_c:
    cons = pd.DataFrame(payload["constructors"])
    cview = pd.DataFrame({
        "Constructor": cons["constructor_id"].map(teams),
        "Projected": cons["expected"].round().astype(int),
        "Likely range": [points_range(a, b) for a, b in zip(cons["p10"], cons["p90"])],
        "Qualifying bonus": [round(b.get("qualifying_bonus", 0), 1) for b in cons["breakdown"]],
        "Team retirement rate": [retirement_text(retire.get(c)) for c in cons["constructor_id"]],
    })
    if prices.ok:
        cview.insert(1, "$M", cons["constructor_id"].map(cp).map(lambda x: f"{x:.1f}"))
        cview["Pts/$M"] = (cons["expected"] / cons["constructor_id"].map(cp)).round(2)
    st.dataframe(cview, use_container_width=True, hide_index=True)
    st.caption(f"Both drivers' points plus the qualifying bonus, plus a rough pit-stop estimate "
               f"(+{rules['estimates']['pit_stop_points_flat']:.0f} for every team: we have no pit-stop time data).")

with tab_team:
    st.markdown("**Check my team** · nothing is saved")
    by_pts = df.sort_values("fx_expected", ascending=False)
    my_d = st.multiselect("Drivers (pick 5)", by_pts["driver_id"], max_selections=5, format_func=lambda d: names[d])
    my_c = st.multiselect("Constructors (pick 2)", list(teams), max_selections=2, format_func=lambda c: teams[c])
    c1, c2 = st.columns(2)
    free = c1.radio("Free transfers left", [2, 1, 0], horizontal=True)
    budget = c2.number_input("Your cost cap (\\$M)", min_value=50.0, max_value=150.0,
                             value=float(rules["team"]["budget_m"]), step=0.1, disabled=not prices.ok)
    if len(my_d) == 5 and len(my_c) == 2:
        sim = snapshot_sim(payload["snapshot_id"], payload)
        boost_x = int(rules["team"]["boost_multiplier"])
        if prices.ok:
            res = check_my_team(sim, prices, rules, my_d, my_c, free_transfers=free, budget_m=budget)
            cur, rec = res["current"], res["plans"][res["recommended"]]
            left, right = st.columns(2)
            with left:
                st.metric("Your team: projected", f"{round(cur['expected'])} pts")
                st.write(f"Likely {points_range(cur['p10'], cur['p90'])} · cost \\${cur['cost_m']:.1f}M")
                if res["budget_raised_to_team_value"]:
                    st.caption(f"Your team is worth more than \\${budget:.1f}M, so suggestions keep it within its current "
                               f"value (\\${res['budget_m']:.1f}M). Enter your game budget (team value + bank) to change this.")
                st.write(f"**2X Boost:** {names[cur['boost']]} (highest projection on your team)")
            with right:
                st.markdown(f"**Suggested changes** (up to {free} free)")
                if rec["n_transfers"] == 0:
                    st.write("No change projects better than your current team.")
                for i, s in enumerate(rec["swaps"], 1):
                    nm = names if s["kind"] == "driver" else teams
                    st.write(f"{i}. {nm[s['out']]} → {nm[s['in']]}  {s['gain']:+.0f} pts, cost {s['cost_change']:+.1f}M")
                if rec["n_transfers"]:
                    pen = f", after a {rec['transfer_penalty']:.0f} transfer penalty" if rec["transfer_penalty"] else ""
                    st.write(f"After changes: {round(rec['expected'])} pts (likely {points_range(rec['p10'], rec['p90'])}){pen}.")
                over = [p for p in res["plans"] if p["n_transfers"] > free]
                if over and rec["n_transfers"] <= free:
                    st.caption(f"An extra transfer costs {rules['team']['extra_transfer_penalty']} pts: "
                               "not worth it unless it gains more than that.")
        else:
            boost = max(my_d, key=lambda d: float(df.set_index("driver_id").loc[d, "fx_expected"]))
            dist = sim.team_distribution(my_d, my_c, boost, boost_x)
            st.metric("Your team: projected", f"{round(dist.mean())} pts")
            st.write(f"Likely {points_range(*np.percentile(dist, [10, 90]))} · "
                     f"2X Boost: {names[boost]}. Transfers and the budget check need this round's prices.")
    else:
        st.caption("Pick 5 drivers and 2 constructors to see your projection.")
    st.caption("Projections, not advice. Chips (Wildcard, Limitless, Final Fix…) are not modelled.")

with tab_oracle:
    team = payload.get("oracle_team")
    if not team:
        st.info("The Oracle team is published with the pre-lock forecast once this round's prices are entered.")
    else:
        rows = [{"Role": "Driver" + (" (2X)" if d == team["boost"] else ""), "Name": names.get(d, d),
                 "$M": f"{dp[d]:.1f}" if d in dp else "—"} for d in team["drivers"]]
        rows += [{"Role": "Constructor", "Name": teams.get(c, c), "$M": f"{cp[c]:.1f}" if c in cp else "—"}
                 for c in team["constructors"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.write(f"Projected **{round(team['expected'])} pts** (likely {points_range(team['p10'], team['p90'])}) · "
                 f"cost \\${team['cost_m']:.1f}M · published {fmt_utc(payload['generated_at'])}")
        st.caption("The highest-projected team under the budget. A projection, not advice; scored on the Track Record.")

render_limits_box()
st.caption(f"Projections, not advice. Ranges cover 8 in 10 simulated weekends. Scoring rules v{rules.version}; "
           f"prices {prices.version or 'not entered'}.")
render_footer()
