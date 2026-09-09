"""Model pipelines and race-level scoring (v3: delegates to ranking module)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from f1ml.eval import race_probs_from_binary, race_softmax
from f1ml.ranking import WeekendRanker, market_probs_from_scores, predict_scores
from f1ml.specs import WeekendMode, feature_columns

# Backward-compatible aliases.
WeekendModels = WeekendRanker


def apply_race_probs(
    df: pd.DataFrame, raw_scores: pd.Series, temperature: float = 1.0
) -> pd.Series:
    """Convert raw pace scores to race-level win probabilities via softmax."""
    probs = pd.Series(0.0, index=df.index)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        s = raw_scores.loc[g.index].values.astype(float)
        p = race_softmax(s, temperature=temperature)
        probs.loc[g.index] = p
    return probs


def race_probs_from_binary_scores(
    binary_probs: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    return race_probs_from_binary(binary_probs, temperature=temperature)


def raw_win_scores(model: Pipeline, df: pd.DataFrame, mode: WeekendMode) -> np.ndarray:
    return predict_scores(model, df, mode)


def feature_matrix(df: pd.DataFrame, mode: WeekendMode) -> pd.DataFrame:
    numeric, categorical = feature_columns(mode)
    cols = numeric + categorical
    work = df.copy()
    for c in cols:
        if c not in work.columns:
            work[c] = np.nan
    return work[cols]


def plackett_luce_win_probs(
    df: pd.DataFrame, expected_finish: pd.Series, temperature: float = 1.0
) -> pd.Series:
    """Derive win probabilities from expected finish via race-level softmax on inverse rank."""
    raw = pd.Series(0.0, index=df.index)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        ef = expected_finish.loc[g.index]
        scores = -ef.astype(float).values
        p = race_softmax(scores, temperature=temperature)
        raw.loc[g.index] = p
    return raw


def predict_binary_proba(model: Pipeline, df: pd.DataFrame, mode: WeekendMode) -> np.ndarray:
    """Legacy: approximate podium prob from pace scores (not used in v3 weekend_card)."""
    scores = predict_scores(model, df, mode)
    return 1.0 / (1.0 + np.exp(-scores))


def finish_position_probs(
    expected_finish: pd.Series, n_positions: int, sigma: float = 2.0
) -> pd.DataFrame:
    positions = list(range(1, n_positions + 1))
    rows = {}
    for idx, ef in expected_finish.items():
        weights = [max(0.01, np.exp(-abs(p - float(ef)) / sigma)) for p in positions]
        total = sum(weights)
        rows[idx] = [w / total for w in weights]
    return pd.DataFrame(rows, index=positions).T
