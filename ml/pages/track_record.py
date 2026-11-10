"""Track Record: every frozen forecast, scored (MVP_SPEC §4.2 page 2, wireframe §6.4).

Reads only the ledger (frozen snapshots + post-race scores). Never re-scores with a newer model.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from f1ml import ledger
from f1ml.paths import REPORTS
from f1ml.predict import load_metrics
from f1ml.ui.data import ledger_index, ledger_problems
from f1ml.ui.mvp import WIN_BANDS, band, drivers_frame, fmt_utc, page_setup, render_footer, STAGE_LABELS

page_setup("Track Record")
st.title("Track Record")
st.markdown("Every forecast is saved **before** the session it predicts and never edited. Here is every one, scored.")

index = ledger_index()
problems = ledger_problems()
scores = ledger.read_scores()
if problems:
    st.error("Ledger check failed: " + "; ".join(problems))
elif index:
    st.caption(f"✓ Ledger verified: {len(index)} frozen forecasts; every file matches the SHA-256 published in the "
               "index, and the index hash chain is intact. Check it yourself: `python race_weekend.py verify`.")


def _race_rows() -> pd.DataFrame:
    rows = []
    for e in index:
        s = scores.get(e["snapshot_id"], {})
        rows.append({
            "season": e["season"], "round": e["round"], "snapshot_id": e["snapshot_id"], "stage": e["stage"],
            "generated_at": e["generated_at"], "on_time": e.get("published_before_lock"),
            "model_version": e["model_version"], "sha256": e["sha256"], **s,
        })
    return pd.DataFrame(rows)


rows = _race_rows()
scored = rows[rows["log_loss"].notna()] if "log_loss" in rows else rows.iloc[0:0]
pre = rows[rows["stage"] == "pre_quali"] if not rows.empty else rows
pre_by_race = pre.sort_values("generated_at").groupby(["season", "round"]).tail(1) if not pre.empty else pre
pre_scored = scored[scored["stage"] == "pre_quali"].sort_values("generated_at").groupby(["season", "round"]).tail(1) \
    if not scored.empty else scored
post_scored = scored[scored["stage"] == "post_quali"].sort_values("generated_at").groupby(["season", "round"]).tail(1) \
    if not scored.empty else scored

c1, c2, c3, c4 = st.columns(4)
on_time = int(pre_by_race["on_time"].fillna(False).astype(bool).sum()) if not pre_by_race.empty else 0
c1.metric("On time", f"{on_time} / {len(pre_by_race)} races", help="Pre-lock forecast frozen at least 1 hour before the team lock")
if not pre_scored.empty:
    top2 = sum(band(p, WIN_BANDS) in ("Favourite", "Contender") for p in pre_scored["winner_prob"])
    c2.metric("Winner in our top 2 bands", f"{top2} of {len(pre_scored)}", help="Pre-lock: the winner was a Favourite or Contender")
else:
    c2.metric("Winner in our top 2 bands", "—", help="Fills in after the first scored race")
if "baseline_last3_mae" in pre_scored and pre_scored["baseline_last3_mae"].notna().any():
    better = int((pre_scored["fantasy_mae"] < pre_scored["baseline_last3_mae"]).sum())
    c3.metric("Projected points vs last-3 baseline", f"{better} of {len(pre_scored)} races closer")
else:
    c3.metric("Projected points vs last-3 baseline", "—")
if not post_scored.empty:
    diff = float((post_scored["log_loss"] - post_scored["grid_baseline_log_loss"]).mean())
    words = "slightly better" if -0.15 < diff < -0.02 else "better" if diff <= -0.15 else "level" if abs(diff) <= 0.02 else "worse"
    c4.metric("Vs starting grid (after quali)", words.capitalize(), help=f"Average log-loss difference {diff:+.2f} over {len(post_scored)} races (lower is better)")
else:
    c4.metric("Vs starting grid (after quali)", "—")

st.subheader("Race by race")
if rows.empty:
    st.info("No live forecasts in the ledger yet. The first one appears here as soon as the next pre-lock forecast "
            "is frozen. Until then, see the back-tested results below.")
else:
    table = []
    for _, r in rows.sort_values(["season", "round", "generated_at"], ascending=[False, False, True]).iterrows():
        payload = ledger.load_snapshot(next(e for e in index if e["snapshot_id"] == r["snapshot_id"]))
        df = drivers_frame(payload).sort_values("win_prob", ascending=False)
        fav = df.iloc[0]
        winner = r.get("actual_winner") if pd.notna(r.get("actual_winner", None)) else None
        wrow = df[df["driver_id"] == winner].iloc[0] if winner is not None and (df["driver_id"] == winner).any() else None
        stage = "grid_set" if payload.get("grid_status") == "official" and r["stage"] == "post_quali" else r["stage"]
        table.append({
            "Race": f"{r['season']} R{r['round']:02d} {payload['event']['race_name'].replace(' Grand Prix', '')}",
            "Snapshot (UTC)": fmt_utc(r["generated_at"]) + (" 🔒" if r["on_time"] else ""),
            "Stage": STAGE_LABELS.get(stage, stage),
            "Our favourite": f"{fav['name']} ({band(fav['win_prob'], WIN_BANDS)})",
            "Winner": wrow["name"] if wrow is not None else "pending",
            "Winner's band": band(wrow["win_prob"], WIN_BANDS) if wrow is not None else "",
            "Pick": ("✓" if r.get("pick_correct") == 1 else "✗") if wrow is not None else "",
            "Oracle team (proj → actual)": (f"{r['oracle_team_projected_driver_points']:.0f} → {r['oracle_team_actual_driver_points']:.0f}"
                                            if pd.notna(r.get("oracle_team_actual_driver_points", None)) else ""),
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption("🔒 = frozen at least one hour before the team lock. The Oracle team figures count only the "
               "categories we can score from official results (qualifying, finish, places, retirements, sprint).")
    st.download_button("⬇ Download CSV", rows.to_csv(index=False).encode(), file_name="pitstop-oracle-track-record.csv",
                       mime="text/csv")

    with st.expander("Open a frozen forecast as published"):
        pick = st.selectbox("Snapshot", [e["snapshot_id"] for e in reversed(index)])
        entry = next(e for e in index if e["snapshot_id"] == pick)
        payload = ledger.load_snapshot(entry)
        st.caption(f"File `ledger/{entry['path']}` · SHA-256 `{entry['sha256']}` · model {payload['model_version']} · "
                   f"rules {payload.get('scoring_rules_version')}")
        df = drivers_frame(payload).sort_values("win_prob", ascending=False)
        show = pd.DataFrame({"Driver": df["name"], "Win": [band(p, WIN_BANDS) for p in df["win_prob"]]})
        if "fx_expected" in df:
            show["Projected pts"] = df["fx_expected"].round().astype(int)
        st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("How to read this"):
    st.markdown(
        "- **Pre-lock** forecasts are made before qualifying, when F1 Fantasy teams lock. **Post-qualifying / grid "
        "set** forecasts come after.\n"
        "- **Winner's band** is the band we gave the eventual winner. A good forecaster's Favourites win often, and its "
        "Long shots rarely.\n"
        "- **Log-loss** is our *surprise score*: lower means the winner was given a higher chance. We compare it with "
        "simply trusting the starting grid.\n"
        "- Misses stay on this page exactly like hits."
    )

with st.expander("Back-tested 2022–2026 (simulated, not live)"):
    metrics = load_metrics()
    wf = pd.DataFrame(metrics.get("walk_forward", []))
    if not wf.empty:
        st.caption("Each past race was forecast using only earlier races (walk-forward). These were not published in "
                   "advance, so they are a back-test, not a live record.")
        by = wf.groupby("test_season").agg(
            races=("test_round", "count"),
            model_post=("post_log_loss", "mean"), grid=("post_grid_log_loss", "mean"),
            model_pre=("pre_log_loss", "mean"), prior_form=("pre_prior_form_log_loss", "mean"),
        ).reset_index()
        by.columns = ["Season", "Races", "After quali: model", "After quali: grid", "Before quali: model",
                      "Before quali: recent form"]
        st.dataframe(by.round(3), use_container_width=True, hide_index=True)
        st.caption("Log-loss, lower is better.")

with st.expander("Model versions"):
    if index:
        mv = pd.DataFrame(index).groupby("model_version").agg(first=("generated_at", "min"), last=("generated_at", "max"),
                                                              snapshots=("snapshot_id", "count")).reset_index()
        st.dataframe(mv, use_container_width=True, hide_index=True)
    st.caption("Recipe since 2026-11-10: pace model trained on classified finishers; confidence fitted on every "
               "earlier out-of-sample forecast. Retrained every race weekend.")

legacy = REPORTS / "live_tracking.csv"
if legacy.exists():
    with st.expander("Before the ledger (2026, saved as plain CSV files)"):
        st.caption("Forecasts saved before the hash-checked ledger existed. They were made before the race, but their "
                   "files can't be verified the same way.")
        st.dataframe(pd.read_csv(legacy), use_container_width=True, hide_index=True)

render_footer()
