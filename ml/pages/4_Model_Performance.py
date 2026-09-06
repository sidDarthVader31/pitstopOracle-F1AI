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
st.caption("Dual-mode evaluation — pre-quali and post-quali with calibrated probabilities.")

metrics = load_metrics()
eda = metrics.get("eda", {})
pre_test = metrics.get("test_pre_quali", {})
post_test = metrics.get("test_post_quali", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Races in dataset", eda.get("n_races", "—"))
c2.metric("Pole win rate", f"{eda.get('pole_win_rate', 0):.1%}")
c3.metric("Post-quali hit (test)", f"{post_test.get('test_post_quali_hit', 0):.1%}")
c4.metric("Pre-quali hit (test)", f"{pre_test.get('test_pre_quali_hit', 0):.1%}")

tab_pre, tab_post, tab_wf = st.tabs(["Pre-quali", "Post-quali", "Walk-forward"])

with tab_pre:
  st.subheader("Pre-qualifying model (no quali/grid features)")
  st.plotly_chart(accuracy_bars(metrics, mode="pre_quali"), use_container_width=True, key="pre_accuracy")
  st.plotly_chart(log_loss_bars(metrics, mode="pre_quali"), use_container_width=True, key="pre_logloss")
  cal = pre_test.get("test_pre_quali_calibration", [])
  if cal:
    st.plotly_chart(calibration_chart(cal, "Pre-quali calibration"), use_container_width=True)
  col_a, col_b = st.columns(2)
  with col_a:
    st.markdown(
      f"""
      | Method | Train hit | Test hit |
      |---|---:|---:|
      | Oracle | {metrics.get('train_pre_quali', {}).get('train_pre_quali_hit', 0):.1%} | {pre_test.get('test_pre_quali_hit', 0):.1%} |
      | Equal field | — | {pre_test.get('test_pre_quali_equal_hit', 0):.1%} |
      | Champ leader | — | {pre_test.get('test_pre_quali_champ_hit', 0):.1%} |
      """
    )
  with col_b:
    st.markdown(
      f"""
      | Metric | Test |
      |---|---:|
      | Log-loss | {pre_test.get('test_pre_quali_log_loss', 0):.3f} |
      | Brier | {pre_test.get('test_pre_quali_brier', 0):.3f} |
      | Rank correlation | {pre_test.get('test_pre_quali_rank_corr', 0):.3f} |
      | Finish MAE | {pre_test.get('test_pre_quali_finish_mae', 0):.2f} |
      """
    )

with tab_post:
  st.subheader("Post-qualifying model (quali + grid included)")
  st.plotly_chart(accuracy_bars(metrics, mode="post_quali"), use_container_width=True, key="post_accuracy")
  st.plotly_chart(log_loss_bars(metrics, mode="post_quali"), use_container_width=True, key="post_logloss")
  cal = post_test.get("test_post_quali_calibration", [])
  if cal:
    st.plotly_chart(calibration_chart(cal, "Post-quali calibration"), use_container_width=True)
  col_a, col_b = st.columns(2)
  with col_a:
    st.markdown(
      f"""
      | Method | Train hit | Test hit |
      |---|---:|---:|
      | Oracle | {metrics.get('train_post_quali', {}).get('train_post_quali_hit', 0):.1%} | {post_test.get('test_post_quali_hit', 0):.1%} |
      | Pole | — | {post_test.get('test_post_quali_pole_hit', 0):.1%} |
      | Champ leader | — | {post_test.get('test_post_quali_champ_hit', 0):.1%} |
      """
    )
  with col_b:
    st.markdown(
      f"""
      | Metric | Test |
      |---|---:|
      | Log-loss | {post_test.get('test_post_quali_log_loss', 0):.3f} |
      | Brier | {post_test.get('test_post_quali_brier', 0):.3f} |
      | Top-3 hit | {post_test.get('test_post_quali_top3', 0):.1%} |
      | Finish MAE | {post_test.get('test_post_quali_finish_mae', 0):.2f} |
      | Ranker log-loss | {post_test.get('test_post_quali_ranker_log_loss', 0):.3f} |
      | Ranker Brier | {post_test.get('test_post_quali_ranker_brier', 0):.3f} |
      """
    )

with tab_wf:
  st.subheader("Walk-forward by season")
  wf = metrics.get("walk_forward", [])
  if wf:
    st.dataframe(
      [
        {
          "Season": f["test_season"],
          "Pre hit": f"{f.get('pre_hit', 0):.1%}",
          "Pre log-loss": f"{f.get('pre_log_loss', 0):.3f}",
          "Post hit": f"{f.get('post_hit', 0):.1%}" if "post_hit" in f else "—",
          "Post log-loss": f"{f.get('post_log_loss', 0):.3f}" if "post_log_loss" in f else "—",
        }
        for f in wf
      ],
      use_container_width=True,
      hide_index=True,
    )
  else:
    st.info("Walk-forward results appear after training.")

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
