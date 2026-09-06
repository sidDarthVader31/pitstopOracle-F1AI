from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CONSTRUCTOR_COLORS: dict[str, str] = {
  "mercedes": "#27F4D2",
  "ferrari": "#E80020",
  "red_bull": "#3671C6",
  "mclaren": "#FF8000",
  "aston_martin": "#229971",
  "alpine": "#FF87BC",
  "williams": "#64C4FF",
  "rb": "#6692FF",
  "alphatauri": "#6692FF",
  "haas": "#B6BABD",
  "sauber": "#52E252",
  "kick_sauber": "#52E252",
  "cadillac": "#B6BABD",
  "audi": "#F50537",
}

CHART_LAYOUT_BASE = dict(
  paper_bgcolor="rgba(0,0,0,0)",
  plot_bgcolor="rgba(0,0,0,0)",
  font=dict(family="Inter, system-ui, sans-serif", color="#374151", size=13),
  margin=dict(l=10, r=10, t=48, b=72),
  xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", zerolinecolor="#e5e7eb"),
  yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", zerolinecolor="#e5e7eb"),
)


def _legend_below() -> dict:
  """Horizontal legend below the plot — avoids overlapping Streamlit headings above."""
  return dict(orientation="h", yanchor="top", y=-0.22, x=0.5, xanchor="center")


def _layout(**overrides) -> dict:
  return {**CHART_LAYOUT_BASE, "height": 420, **overrides}


def _team_color(constructor_id: str) -> str:
  key = (constructor_id or "").lower().replace(" ", "_")
  for name, color in CONSTRUCTOR_COLORS.items():
    if name in key:
      return color
  return "#9ca3af"


def _driver_label(row: pd.Series) -> str:
  return f"{row['given_name']} {row['family_name']}"


def win_prob_bar(scored: pd.DataFrame, top_n: int = 10) -> go.Figure:
  data = scored.head(top_n).sort_values("win_prob", ascending=True).copy()
  data["driver"] = data.apply(_driver_label, axis=1)
  colors = data["constructor_id"].map(_team_color)
  fig = go.Figure(
    go.Bar(
      x=data["win_prob"] * 100,
      y=data["driver"],
      orientation="h",
      marker_color=colors,
      text=[f"{v:.1f}%" for v in data["win_prob"] * 100],
      textposition="outside",
      hovertemplate="%{y}<br>Win: %{x:.1f}%<extra></extra>",
    )
  )
  fig.update_layout(
    **_layout(
      title="Win probability (top drivers)",
      xaxis_title="Win %",
      yaxis=dict(autorange=True),
    )
  )
  return fig


def podium_strip(scored: pd.DataFrame) -> go.Figure:
  top3 = scored.head(3).copy()
  top3["driver"] = top3.apply(_driver_label, axis=1)
  positions = ["P1", "P2", "P3"]
  colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
  fig = go.Figure(
    go.Bar(
      x=positions[: len(top3)],
      y=top3["win_prob"] * 100,
      marker_color=colors[: len(top3)],
      text=[f"{r['driver']}<br>{r['win_prob']*100:.1f}%" for _, r in top3.iterrows()],
      textposition="outside",
    )
  )
  fig.update_layout(**_layout(title="Model top 3", yaxis_title="Win %", height=320))
  return fig


def compare_scenarios(baseline: pd.DataFrame, scenario: pd.DataFrame, top_n: int = 8) -> go.Figure:
  b = baseline.head(top_n).copy()
  b["driver"] = b.apply(_driver_label, axis=1)
  s = scenario.set_index("driver_id")
  b["scenario_prob"] = b["driver_id"].map(
    lambda d: s.loc[d, "win_prob"] * 100 if d in s.index else 0.0
  )
  b["baseline_prob"] = b["win_prob"] * 100
  fig = go.Figure()
  fig.add_trace(
    go.Bar(
      name="Baseline",
      x=b["driver"],
      y=b["baseline_prob"],
      marker_color="#6b7280",
    )
  )
  fig.add_trace(
    go.Bar(
      name="Scenario",
      x=b["driver"],
      y=b["scenario_prob"],
      marker_color="#e10600",
    )
  )
  fig.update_layout(
    **_layout(
      barmode="group",
      title="Baseline vs scenario",
      yaxis_title="Win %",
      legend=_legend_below(),
    )
  )
  return fig


def accuracy_bars(metrics: dict, mode: str = "post_quali") -> go.Figure:
  split = metrics.get(f"test_{mode}", {})
  if mode == "pre_quali":
    methods = ["oracle", "equal", "champ"]
    labels = ["Oracle", "Equal field", "Champ leader"]
    test = [
      split.get(f"test_{mode}_hit", 0) * 100,
      split.get(f"test_{mode}_equal_hit", 0) * 100,
      split.get(f"test_{mode}_champ_hit", 0) * 100,
    ]
    train_split = metrics.get(f"train_{mode}", {})
    train = [
      train_split.get(f"train_{mode}_hit", 0) * 100,
      0,
      0,
    ]
  else:
    methods = ["oracle", "pole", "champ"]
    labels = ["Oracle", "Pole", "Champ leader"]
    test = [
      split.get(f"test_{mode}_hit", 0) * 100,
      split.get(f"test_{mode}_pole_hit", 0) * 100,
      split.get(f"test_{mode}_champ_hit", 0) * 100,
    ]
    train_split = metrics.get(f"train_{mode}", {})
    train = [
      train_split.get(f"train_{mode}_hit", 0) * 100,
      train_split.get(f"train_{mode}_pole_hit", 0) * 100,
      0,
    ]
  fig = go.Figure()
  fig.add_trace(go.Bar(name="Train", x=labels, y=train, marker_color="#6b7280"))
  fig.add_trace(go.Bar(name="Test", x=labels, y=test, marker_color="#e10600"))
  fig.update_layout(
    **_layout(
      barmode="group",
      title=f"Winner hit rate — {mode.replace('_', ' ')} (%)",
      yaxis_title="Correct winner %",
      yaxis=dict(range=[0, 100]),
      legend=_legend_below(),
    )
  )
  return fig


def log_loss_bars(metrics: dict, mode: str = "post_quali") -> go.Figure:
  split = metrics.get(f"test_{mode}", {})
  if mode == "pre_quali":
    labels = ["Oracle", "Equal field", "Champ leader"]
    values = [
      split.get(f"test_{mode}_log_loss", 0),
      split.get(f"test_{mode}_equal_log_loss", 0),
      split.get(f"test_{mode}_champ_log_loss", 0),
    ]
  else:
    labels = ["Oracle", "Pole", "Champ leader"]
    values = [
      split.get(f"test_{mode}_log_loss", 0),
      split.get(f"test_{mode}_pole_log_loss", 0),
      split.get(f"test_{mode}_champ_log_loss", 0),
    ]
  fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#3671C6"))
  fig.update_layout(
    **_layout(
      title=f"Log-loss (lower is better) — {mode.replace('_', ' ')}",
      yaxis_title="Log-loss",
    )
  )
  return fig


def calibration_chart(bins: list[dict], title: str) -> go.Figure:
  if not bins:
    return go.Figure()
  pred = [b["mean_pred"] * 100 for b in bins]
  actual = [b["actual_rate"] * 100 for b in bins]
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=pred, y=actual, mode="markers+lines", name="Bins", marker=dict(size=10)))
  fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect", line=dict(dash="dash", color="#9ca3af")))
  fig.update_layout(
    **_layout(
      title=title,
      xaxis_title="Predicted win %",
      yaxis_title="Actual win %",
      height=360,
      legend=_legend_below(),
    )
  )
  return fig


def fantasy_pts_bar(scored: pd.DataFrame, top_n: int = 10) -> go.Figure:
  if "expected_fantasy_pts" not in scored.columns:
    return go.Figure()
  data = scored.nlargest(top_n, "expected_fantasy_pts").sort_values("expected_fantasy_pts", ascending=True)
  data["driver"] = data.apply(_driver_label, axis=1)
  colors = data["constructor_id"].map(_team_color)
  fig = go.Figure(
    go.Bar(
      x=data["expected_fantasy_pts"],
      y=data["driver"],
      orientation="h",
      marker_color=colors,
      text=[f"{v:.1f}" for v in data["expected_fantasy_pts"]],
      textposition="outside",
      hovertemplate="%{y}<br>Fantasy pts: %{x:.1f}<extra></extra>",
    )
  )
  fig.update_layout(
    **_layout(title="Expected fantasy points", xaxis_title="Points", height=360)
  )
  return fig


def market_comparison(scored: pd.DataFrame, top_n: int = 8, *, title: str | None = None) -> go.Figure:
  data = scored.head(top_n).copy()
  data["driver"] = data.apply(_driver_label, axis=1)
  fig = go.Figure()
  fig.add_trace(go.Bar(name="Win %", x=data["driver"], y=data["win_prob"] * 100, marker_color="#e10600"))
  if "podium_prob" in data.columns:
    fig.add_trace(go.Bar(name="Podium %", x=data["driver"], y=data["podium_prob"] * 100, marker_color="#3671C6"))
  if "dnf_prob" in data.columns:
    fig.add_trace(go.Bar(name="DNF %", x=data["driver"], y=data["dnf_prob"] * 100, marker_color="#6b7280"))
  layout = dict(
    barmode="group",
    yaxis_title="%",
    legend=_legend_below(),
  )
  if title:
    layout["title"] = title
  fig.update_layout(**_layout(**layout))
  return fig
