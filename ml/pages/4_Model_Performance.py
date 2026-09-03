"""Model Performance — accuracy metrics and evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.paths import REPORTS
from f1ml.predict import load_metrics
from f1ml.ui.charts import accuracy_bars
from f1ml.ui.theme import apply_theme, disclaimer_footer

st.set_page_config(page_title="Model Performance | Pitstop Oracle", layout="wide")
apply_theme()

st.title("Model Performance")
st.caption("Honest evaluation — pole sitter is still a strong baseline.")

metrics = load_metrics()
eda = metrics.get("eda", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Races in dataset", eda.get("n_races", "—"))
c2.metric("Pole win rate", f"{eda.get('pole_win_rate', 0):.1%}")
c3.metric("Test RF hit rate", f"{metrics.get('test_rf_hit', 0):.1%}")
c4.metric("Test pole hit rate", f"{metrics.get('test_pole_hit', 0):.1%}")

st.plotly_chart(accuracy_bars(metrics), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
  st.subheader("Train vs test")
  st.markdown(
    f"""
    | Method | Train | Test |
    |---|---:|---:|
    | Pole | {metrics['train_pole_hit']:.1%} | {metrics['test_pole_hit']:.1%} |
    | Champ leader | {metrics['train_champ_hit']:.1%} | {metrics['test_champ_hit']:.1%} |
    | Logistic regression | {metrics['train_logreg_hit']:.1%} | {metrics['test_logreg_hit']:.1%} |
    | Random forest | {metrics['train_rf_hit']:.1%} | {metrics['test_rf_hit']:.1%} |
    """
  )
with col_b:
  st.subheader("Top-3 hit rate")
  st.markdown(
    f"""
    - Train RF top-3: **{metrics['train_rf_top3']:.1%}**
    - Test RF top-3: **{metrics['test_rf_top3']:.1%}**
    - Test logreg top-3: **{metrics['test_logreg_top3']:.1%}**
    """
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
