"""Evaluation metrics for race-level probabilistic forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss


def race_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(["season", "round"], sort=False)]


def winner_hit_rate(df: pd.DataFrame, scores: pd.Series) -> float:
    hits = n = 0
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        n += 1
        pick = scores.loc[g.index].idxmax()
        if pick in winners:
            hits += 1
    return hits / n if n else 0.0


def winner_hit_rate_probs(df: pd.DataFrame, probs: pd.Series) -> float:
    """Hit rate when multiple drivers can share the top probability (e.g. equal field)."""
    hits = n = 0
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        n += 1
        p = probs.loc[g.index]
        max_p = p.max()
        if p.loc[winners[0]] >= max_p - 1e-12:
            hits += 1
    return hits / n if n else 0.0


def topk_hit_rate(df: pd.DataFrame, scores: pd.Series, k: int = 3) -> float:
    hits = n = 0
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        n += 1
        top = scores.loc[g.index].nlargest(k)
        if winners[0] in top.index:
            hits += 1
    return hits / n if n else 0.0


def race_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert raw scores to race-level probabilities summing to 1."""
    if temperature <= 0:
        temperature = 1.0
    z = scores / temperature
    z = z - z.max()
    exp = np.exp(z)
    total = exp.sum()
    if total <= 0:
        return np.ones_like(exp) / len(exp)
    return exp / total


def race_log_loss(df: pd.DataFrame, probs: pd.Series, eps: float = 1e-15) -> float:
    """Multiclass log-loss: one winner per race."""
    losses = []
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        p = probs.loc[g.index].clip(eps, 1 - eps)
        p = p / p.sum()
        losses.append(-np.log(p.loc[winners[0]]))
    return float(np.mean(losses)) if losses else float("nan")


def race_brier(df: pd.DataFrame, probs: pd.Series) -> float:
    """Mean Brier score across drivers (winner one-hot)."""
    scores = []
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        y = (g.index == winners[0]).astype(float)
        p = probs.loc[g.index]
        scores.append(brier_score_loss(y, p))
    return float(np.mean(scores)) if scores else float("nan")


def rank_correlation(df: pd.DataFrame, scores: pd.Series) -> float:
    """Mean Spearman correlation between predicted rank and actual finish."""
    corrs = []
    for g in race_groups(df):
        if g["position"].isna().all():
            continue
        actual = g["position"].astype(float)
        pred = -scores.loc[g.index].astype(float)  # higher score = better predicted rank
        if actual.nunique() < 2:
            continue
        rho, _ = spearmanr(pred, actual)
        if not np.isnan(rho):
            corrs.append(rho)
    return float(np.mean(corrs)) if corrs else float("nan")


def finish_mae(df: pd.DataFrame, expected_finish: pd.Series) -> float:
  """Mean absolute error of expected finishing position."""
  errors = []
  for g in race_groups(df):
    if g["position"].isna().all():
      continue
    actual = g["position"].astype(float)
    pred = expected_finish.loc[g.index].astype(float)
    errors.append((pred - actual).abs().mean())
  return float(np.mean(errors)) if errors else float("nan")


def calibration_bins(
    df: pd.DataFrame, probs: pd.Series, n_bins: int = 10
) -> list[dict]:
    """Reliability bins for win probability calibration."""
    rows = []
    for g in race_groups(df):
        winners = g.index[g["won"] == 1]
        if len(winners) == 0:
            continue
        for idx in g.index:
            rows.append({"prob": probs.loc[idx], "won": int(idx == winners[0])})
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["bin"] = pd.cut(frame["prob"], bins=n_bins, labels=False)
    out = []
    for b, grp in frame.groupby("bin"):
        if grp.empty:
            continue
        out.append(
            {
                "bin": int(b),
                "mean_pred": float(grp["prob"].mean()),
                "actual_rate": float(grp["won"].mean()),
                "count": int(len(grp)),
            }
        )
    return out


def pole_probabilities(df: pd.DataFrame) -> pd.Series:
    """One-hot pole sitter as probability distribution per race."""
    probs = pd.Series(0.0, index=df.index)
    for g in race_groups(df):
        valid = g.dropna(subset=["quali_position"])
        if valid.empty:
            n = len(g)
            probs.loc[g.index] = 1.0 / n if n else 0.0
        else:
            pole_idx = valid["quali_position"].idxmin()
            probs.loc[pole_idx] = 1.0
    return probs


def equal_field_probabilities(df: pd.DataFrame) -> pd.Series:
    probs = pd.Series(0.0, index=df.index)
    for g in race_groups(df):
        n = len(g)
        if n:
            probs.loc[g.index] = 1.0 / n
    return probs


def champ_leader_probabilities(df: pd.DataFrame) -> pd.Series:
    probs = pd.Series(0.0, index=df.index)
    for g in race_groups(df):
        if g["champ_points_before"].fillna(0).sum() == 0:
            valid = g.dropna(subset=["quali_position"])
            if not valid.empty:
                pick = valid["quali_position"].idxmin()
            else:
                pick = g.index[0]
        else:
            pick = g["champ_position_before"].idxmin()
        probs.loc[pick] = 1.0
    return probs


def fit_temperature(
    df: pd.DataFrame, raw_scores: pd.Series, grid: np.ndarray | None = None
) -> float:
    """Find temperature minimizing race log-loss on validation data."""
    if grid is None:
        grid = np.linspace(0.3, 3.0, 28)
    best_t, best_ll = 1.0, float("inf")
    for t in grid:
        probs = pd.Series(index=df.index, dtype=float)
        for g in race_groups(df):
            s = raw_scores.loc[g.index].values
            p = race_softmax(s, temperature=float(t))
            probs.loc[g.index] = p
        ll = race_log_loss(df, probs)
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    return best_t


def slice_metrics(
    df: pd.DataFrame, probs: pd.Series, scores: pd.Series
) -> dict:
    """Metrics sliced by wet/dry."""
    out = {}
    for label, mask_col in [("wet", "is_wet"), ("dry", "is_wet")]:
        if mask_col not in df.columns:
            continue
        if label == "wet":
            mask = df["is_wet"] == 1
        else:
            mask = df["is_wet"] != 1
        sub = df[mask]
        if sub.empty:
            continue
        sub_probs = probs.loc[sub.index]
        sub_scores = scores.loc[sub.index]
        out[f"{label}_log_loss"] = race_log_loss(sub, sub_probs)
        out[f"{label}_hit"] = winner_hit_rate(sub, sub_scores)
    return out
