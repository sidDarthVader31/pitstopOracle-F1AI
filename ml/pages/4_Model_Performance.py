"""Model Performance — accuracy metrics and evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.paths import REPORTS
from f1ml.predict import load_metrics
from f1ml.ui.charts import accuracy_bars, calibration_chart, log_loss_bars
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="Model Performance | Pitstop Oracle", layout="wide")
apply_theme()

st.title("Model Performance")
st.caption(
    "v3 ranker — in-sample train metrics plus 2026 walk-forward out-of-sample evaluation."
)

metrics = load_metrics()
eda = metrics.get("eda", {})
oos = metrics.get("oos", {})
train_pre = metrics.get("train_pre_quali", {})
train_post = metrics.get("train_post_quali", {})
wf26 = metrics.get("walk_forward_2026", [])
transfer = metrics.get("transfer_2026", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Races in dataset", eda.get("n_races", "—"))
c2.metric("Pole win rate", f"{eda.get('pole_win_rate', 0):.1%}")
c3.metric(
    "Post-quali hit (WF 2026)",
    f"{oos.get('oos_post_hit', metrics.get('test_rf_hit', 0)):.1%}",
)
c4.metric("Post log-loss (WF 2026)", f"{oos.get('oos_post_log_loss', 0):.3f}")

tab_pre, tab_post, tab_wf = st.tabs(["Pre-quali", "Post-quali", "Walk-forward"])

with tab_pre:
    st.subheader("Pre-qualifying model (no quali/grid features)")
    st.plotly_chart(accuracy_bars(metrics, mode="pre_quali"), use_container_width=True, key="pre_accuracy")
    st.plotly_chart(log_loss_bars(metrics, mode="pre_quali"), use_container_width=True, key="pre_logloss")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            | Method | Train (in-sample) | Walk-forward 2026 |
            |---|---:|---:|
            | Model winner hit | {train_pre.get('train_pre_quali_hit', 0):.1%} | {oos.get('oos_pre_hit', 0):.1%} |
            | Model top-3 | {train_pre.get('train_pre_quali_top3', 0):.1%} | {oos.get('oos_pre_top3', 0):.1%} |
            | Prior-form baseline log-loss | — | {oos.get('oos_pre_prior_form_log_loss', 0):.3f} |
            """
        )
        st.caption("Walk-forward: each 2026 race predicted using only earlier races. Train numbers are optimistic.")
    with col_b:
        st.markdown(
            f"""
            | Metric | Walk-forward 2026 |
            |---|---:|
            | Log-loss | {oos.get('oos_pre_log_loss', 0):.3f} |
            | Races | {oos.get('oos_n_races', 0)} |
            """
        )

with tab_post:
    st.subheader("Post-qualifying model (quali + grid included)")
    st.plotly_chart(accuracy_bars(metrics, mode="post_quali"), use_container_width=True, key="post_accuracy")
    st.plotly_chart(log_loss_bars(metrics, mode="post_quali"), use_container_width=True, key="post_logloss")
    cal = train_post.get("train_post_quali_calibration", [])
    if cal:
        st.plotly_chart(calibration_chart(cal, "Post-quali calibration (train)"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            | Method | Train | Walk-forward 2026 |
            |---|---:|---:|
            | Model | {train_post.get('train_post_quali_hit', 0):.1%} | {oos.get('oos_post_hit', 0):.1%} |
            | Pole | {train_post.get('train_post_quali_pole_hit', 0):.1%} | {oos.get('oos_post_pole_hit', 0):.1%} |
            | Quali-gap | — | LL {oos.get('oos_post_quali_gap_log_loss', 0):.3f} |
            """
        )
    with col_b:
        disagree_n = train_post.get("train_post_quali_disagree_n_races", 0)
        st.markdown(
            f"""
            | Metric | Value |
            |---|---:|
            | OOS log-loss | {oos.get('oos_post_log_loss', 0):.3f} |
            | OOS quali-gap LL | {oos.get('oos_post_quali_gap_log_loss', 0):.3f} |
            | Top-3 hit (train) | {train_post.get('train_post_quali_top3', 0):.1%} |
            | Model ≠ pole races | {disagree_n} |
            | Hit when ≠ pole | {train_post.get('train_post_quali_disagree_hit', 0):.1%} |
            """
        )

with tab_wf:
    st.subheader("Walk-forward 2026 (primary shipping metric)")
    if wf26:
        st.dataframe(
            [
                {
                    "Round": f.get("test_round", "—"),
                    "Pre hit": f"{f.get('pre_hit', 0):.1%}",
                    "Pre log-loss": f"{f.get('pre_log_loss', 0):.3f}",
                    "Post hit": f"{f.get('post_hit', 0):.1%}" if "post_hit" in f else "—",
                    "Post log-loss": f"{f.get('post_log_loss', 0):.3f}" if "post_log_loss" in f else "—",
                    "Quali-gap LL": f"{f.get('post_quali_gap_log_loss', 0):.3f}"
                    if "post_quali_gap_log_loss" in f
                    else "—",
                }
                for f in wf26
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            f"**Mean post-quali hit:** {oos.get('oos_post_hit', 0):.1%} · "
            f"**mean log-loss:** {oos.get('oos_post_log_loss', 0):.3f} "
            f"(quali-gap baseline: {oos.get('oos_post_quali_gap_log_loss', 0):.3f})"
        )
    else:
        st.info("Walk-forward 2026 results appear after training.")

    if transfer.get("transfer_n_test_races", 0) > 0:
        st.subheader("Transfer diagnostic (train <2026 → test 2026)")
        st.markdown(
            f"""
            | Split | Races | Post hit | Post log-loss | Quali-gap LL |
            |---|---:|---:|---:|---:|
            | Train ≤2025 | {transfer.get('transfer_n_train_races', 0)} | — | — | — |
            | Test 2026 | {transfer.get('transfer_n_test_races', 0)} | {transfer.get('transfer_post_hit', 0):.1%} | {transfer.get('transfer_post_log_loss', 0):.3f} | {transfer.get('transfer_quali_gap_log_loss', 0):.3f} |
            """
        )

    wf_all = metrics.get("walk_forward", [])
    if wf_all:
        with st.expander("All seasons (season-start folds)"):
            st.dataframe(
                [
                    {
                        "Season": f["test_season"],
                        "Pre hit": f"{f.get('pre_hit', 0):.1%}",
                        "Pre log-loss": f"{f.get('pre_log_loss', 0):.3f}",
                        "Post hit": f"{f.get('post_hit', 0):.1%}" if "post_hit" in f else "—",
                        "Post log-loss": f"{f.get('post_log_loss', 0):.3f}" if "post_log_loss" in f else "—",
                    }
                    for f in wf_all
                ],
                use_container_width=True,
                hide_index=True,
            )

pole_img = REPORTS / "pole_win_rate.png"
dnf_img = REPORTS / "dnf_by_constructor.png"
if pole_img.exists():
    st.image(str(pole_img), caption="How often pole wins (2022+)")
if dnf_img.exists():
    st.image(str(dnf_img), caption="DNF rate by constructor")

eval_md = REPORTS / "EVAL.md"
if eval_md.exists():
    with st.expander("Full evaluation report"):
        st.markdown(eval_md.read_text())

st.markdown("**Re-train:** `python run_pipeline.py`")

disclaimer_footer()
