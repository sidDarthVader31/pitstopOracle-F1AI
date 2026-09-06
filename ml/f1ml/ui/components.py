from __future__ import annotations

import streamlit as st

from f1ml.predict import ModelKind, champ_leader_pick, forecast_tracking, model_pick, safe_pole_pick, weekend_card
from f1ml.ui.charts import fantasy_pts_bar, podium_strip, win_prob_bar
from f1ml.ui.theme import hero_card, status_badge
import pandas as pd


def sidebar_model_controls() -> tuple[ModelKind, bool]:
  model_kind = st.radio(
    "Model",
    options=["hgb", "logreg"],
    format_func=lambda x: "Gradient boosting (primary)" if x == "hgb" else "Logistic regression",
  )
  normalize = st.checkbox("Normalize win probabilities within race", value=True)
  return model_kind, normalize


def render_prediction_block(
  race_df: pd.DataFrame,
  meta: pd.Series,
  mode: str,
  model_kind: ModelKind,
  normalize: bool,
) -> pd.DataFrame:
  scored = weekend_card(race_df, race_mode=mode, kind=model_kind)
  if not normalize:
    scored["win_prob"] = scored["win_prob_raw"]
  pick = model_pick(scored)
  pole = safe_pole_pick(race_df)
  champ = champ_leader_pick(race_df)
  inference = scored["inference_mode"].iloc[0] if "inference_mode" in scored.columns else "pre_quali"

  st.subheader(meta["race_name"])
  st.caption(f"{meta['circuit_name']} · {meta['date']}")
  status_badge(mode)

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
    hero_card("Model mode", inference.replace("_", " ").title(), "Calibrated field probs")

  if mode == "pre_quali":
    st.info("Pre-qualifying forecast — uses form and standings only (no quali/grid).")
  elif mode == "post_quali":
    st.success("Post-qualifying forecast — quali and grid included.")

  col_a, col_b = st.columns([2, 1])
  with col_a:
    st.plotly_chart(win_prob_bar(scored), use_container_width=True)
  with col_b:
    st.plotly_chart(podium_strip(scored), use_container_width=True)

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
