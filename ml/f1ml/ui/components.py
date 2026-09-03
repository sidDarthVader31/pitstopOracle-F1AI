from __future__ import annotations

import streamlit as st

from f1ml.predict import ModelKind, champ_leader_pick, model_pick, safe_pole_pick, score_race
from f1ml.ui.charts import podium_strip, win_prob_bar
from f1ml.ui.theme import hero_card, status_badge
import pandas as pd


def sidebar_model_controls() -> tuple[ModelKind, bool]:
  model_kind = st.radio(
    "Model",
    options=["rf", "logreg"],
    format_func=lambda x: "Random Forest" if x == "rf" else "Logistic Regression",
  )
  normalize = st.checkbox("Normalize probabilities within race", value=True)
  return model_kind, normalize


def render_prediction_block(
  race_df: pd.DataFrame,
  meta: pd.Series,
  mode: str,
  model_kind: ModelKind,
  normalize: bool,
) -> pd.DataFrame:
  scored = score_race(race_df, kind=model_kind, normalize=normalize)
  pick = model_pick(scored)
  pole = safe_pole_pick(race_df)
  champ = champ_leader_pick(race_df)

  st.subheader(meta["race_name"])
  st.caption(f"{meta['circuit_name']} · {meta['date']}")
  status_badge(mode)

  c1, c2, c3 = st.columns(3)
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

  if mode == "pre_quali":
    st.info("Qualifying not available yet — prediction uses form and standings only.")

  col_a, col_b = st.columns([2, 1])
  with col_a:
    st.plotly_chart(win_prob_bar(scored), use_container_width=True)
  with col_b:
    st.plotly_chart(podium_strip(scored), use_container_width=True)

  return scored


def render_result_banner(scored: pd.DataFrame, race_df: pd.DataFrame) -> None:
  winners = race_df[race_df["won"] == 1]
  if winners.empty:
    return
  actual = winners.iloc[0]
  actual_name = f"{actual['given_name']} {actual['family_name']}"
  model_top = scored.iloc[0]
  model_name = f"{model_top['given_name']} {model_top['family_name']}"
  top3_ids = set(scored.head(3)["driver_id"])
  if actual["driver_id"] == model_top["driver_id"]:
    st.success(f"Model called it — **{actual_name}** won (P{int(actual['position'])})")
  elif actual["driver_id"] in top3_ids:
    st.warning(f"Winner **{actual_name}** was in the model's top 3 (picked **{model_name}**)")
  else:
    st.error(f"Actual winner **{actual_name}** — model picked **{model_name}**")
