from __future__ import annotations

import streamlit as st

from f1ml.predict import (
    ModelKind,
    accuracy_journal,
    champ_leader_pick,
    driver_insight_bullets,
    forecast_tracking,
    model_pick,
    safe_pole_pick,
    weekend_card,
    weekend_phase,
    weekend_phase_label,
)
from f1ml.paths import REPORTS
from f1ml.starting_grid import grid_metadata, penalty_callouts
from f1ml.ui.charts import fantasy_pts_bar, podium_strip, win_prob_bar
from f1ml.ui.theme import hero_card, phase_chip, status_badge
import pandas as pd


def sidebar_model_controls(show_lab: bool = False) -> tuple[ModelKind, bool]:
    if show_lab:
        model_kind = st.radio(
            "Model",
            options=["hgb", "logreg"],
            format_func=lambda x: "Gradient boosting (primary)" if x == "hgb" else "Logistic regression",
        )
    else:
        model_kind = "hgb"
    normalize = st.checkbox("Normalize win probabilities within race", value=True)
    return model_kind, normalize


def render_grid_banner(season: int, round_num: int, race_df: pd.DataFrame) -> None:
    meta = grid_metadata(season, round_num)
    callouts = penalty_callouts(race_df)
    if meta.get("has_grid"):
        src = meta.get("source", "unknown")
        as_of = meta.get("as_of", "")
        st.success(f"Official starting grid loaded ({src})" + (f" · {as_of[:16]} UTC" if as_of else ""))
    elif race_df["quali_position"].notna().any():
        st.warning("Qualifying set — awaiting official starting grid (penalties not applied yet).")
    if callouts:
        with st.expander(f"Grid penalties & promotions ({len(callouts)})", expanded=True):
            for c in callouts:
                st.markdown(f"- **{c['driver_name']}**: {c['note']}")


def render_prediction_block(
    race_df: pd.DataFrame,
    meta: pd.Series,
    mode: str,
    model_kind: ModelKind,
    normalize: bool,
    season: int | None = None,
    round_num: int | None = None,
) -> pd.DataFrame:
    season_i = int(season or meta.get("season", race_df["season"].iloc[0]))
    round_i = int(round_num or meta.get("round", race_df["round"].iloc[0]))
    phase = weekend_phase(race_df, season_i, round_i)

    scored = weekend_card(race_df, race_mode=mode, kind=model_kind)
    if not normalize:
        scored["win_prob"] = scored["win_prob_raw"]
    pick = model_pick(scored)
    pole = safe_pole_pick(race_df)
    champ = champ_leader_pick(race_df)
    inference = scored["inference_mode"].iloc[0] if "inference_mode" in scored.columns else "pre_quali"

    st.subheader(meta["race_name"])
    st.caption(f"{meta['circuit_name']} · {meta['date']}")
    phase_chip(phase)
    status_badge(mode)
    render_grid_banner(season_i, round_i, race_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hero_card("Model pick", pick["driver_name"], f"{pick['win_prob'] * 100:.1f}% win")
    with c2:
        if pole is not None:
            hero_card("Pole pick", pole["driver_name"], f"P{int(pole['quali_position'])}")
        else:
            hero_card("Pole pick", "Awaiting qualifying", "Run again after Saturday")
    with c3:
        pts = champ.get("champ_points")
        sub = f"P{int(champ['champ_position'])} · {pts:.0f} pts" if pd.notna(pts) else ""
        hero_card("Championship leader", champ["driver_name"], sub)
    with c4:
        hero_card("Weekend", weekend_phase_label(phase), inference.replace("_", " ").title())

    if mode == "pre_quali":
        st.info("Pre-qualifying forecast — form and standings only.")
    elif mode == "grid_set":
        st.success("Grid-set forecast — official starting positions and penalties applied.")
    elif mode == "post_quali":
        st.success("Post-qualifying forecast — quali included; grid may still update.")
    elif mode == "post_sprint":
        st.success("Post-sprint forecast — qualifying, official grid, and completed sprint included.")

    bullets = driver_insight_bullets(scored.iloc[0])
    if bullets:
        st.markdown("**Why the model likes the pick**")
        for b in bullets:
            st.markdown(f"- {b}")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(win_prob_bar(scored), use_container_width=True)
    with col_b:
        st.plotly_chart(podium_strip(scored), use_container_width=True)

    st.caption("Finish-position probabilities and expected finish are conditional on being classified. DNF probability is shown separately.")

    if "expected_fantasy_pts" in scored.columns:
        st.plotly_chart(fantasy_pts_bar(scored), use_container_width=True)

    tracking = forecast_tracking(race_df, scored)
    if tracking.get("status") == "complete":
        if tracking["correct"]:
            st.success(f"Forecast tracked: **{tracking['model_pick']}** called it ({tracking['model_win_prob']*100:.1f}%)")
        elif tracking.get("in_top3"):
            st.warning(
                f"Forecast tracked: winner **{tracking['actual_winner']}** was in top 3 "
                f"(picked **{tracking['model_pick']}**)"
            )
        else:
            st.error(
                f"Forecast tracked: actual **{tracking['actual_winner']}** — "
                f"model picked **{tracking['model_pick']}**"
            )

    return scored


def render_accuracy_journal(n_races: int = 10) -> None:
    """Honest track record: walk-forward results (each race predicted from earlier races only)."""
    from f1ml.predict import load_metrics

    wf = [r for r in load_metrics().get("walk_forward", []) if "post_hit" in r][-n_races:]
    if not wf:
        st.info("Accuracy journal fills in after `python run_pipeline.py --skip-ingest` writes walk-forward results.")
        return
    frame = pd.DataFrame(wf)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model winner hit rate", f"{frame['post_hit'].mean():.0%}")
    c2.metric("Pole baseline", f"{frame['post_pole_hit'].mean():.0%}")
    c3.metric("Model top-3", f"{frame['post_top3'].mean():.0%}")
    st.caption(
        f"Last {len(frame)} races, post-qualifying. Each race was predicted using only races before it "
        "(walk-forward), so these are honest out-of-sample numbers. Log-loss: lower is better."
    )
    view = pd.DataFrame(
        {
            "Season": frame["test_season"],
            "Round": frame["test_round"],
            "Date": frame.get("test_date"),
            "Model hit": frame["post_hit"].map({1.0: "✓", 0.0: "✗"}),
            "Pole hit": frame["post_pole_hit"].map({1.0: "✓", 0.0: "✗"}),
            "Winner in model top-3": frame["post_top3"].map({1.0: "✓", 0.0: "✗"}),
            "Model log-loss": frame["post_log_loss"].round(3),
            "Grid baseline log-loss": frame["post_grid_log_loss"].round(3),
        }
    ).iloc[::-1]
    st.dataframe(view, use_container_width=True, hide_index=True)
    live = REPORTS / "live_tracking.csv"
    if live.exists():
        st.markdown("**Live forecasts** (saved before each race by `race_weekend.py before`, scored by `after`)")
        st.dataframe(pd.read_csv(live), use_container_width=True, hide_index=True)


def render_result_banner(scored: pd.DataFrame, race_df: pd.DataFrame) -> None:
    tracking = forecast_tracking(race_df, scored)
    if tracking.get("status") != "complete":
        return
    if tracking["correct"]:
        st.success(f"Model called it — **{tracking['actual_winner']}** won")
    elif tracking.get("in_top3"):
        st.warning(
            f"Winner **{tracking['actual_winner']}** was in the model's top 3 "
            f"(picked **{tracking['model_pick']}**)"
        )
    else:
        st.error(f"Actual winner **{tracking['actual_winner']}** — model picked **{tracking['model_pick']}**")
