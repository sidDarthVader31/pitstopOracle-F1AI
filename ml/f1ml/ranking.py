"""Race ranking model: pairwise scorer + Plackett-Luce finish distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1ml.eval import race_softmax
from f1ml.specs import WeekendMode, feature_columns


def build_rank_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250,
        max_depth=4,
        learning_rate=0.06,
        min_samples_leaf=12,
        l2_regularization=0.2,
        random_state=42,
    )


def build_dnf_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.06,
        min_samples_leaf=10,
        l2_regularization=0.15,
        random_state=42,
    )


def make_rank_pipeline(numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ]
    )
    return Pipeline([("pre", pre), ("reg", build_rank_regressor())])


def make_dnf_pipeline(numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric,
            )
        ]
    )
    return Pipeline([("pre", pre), ("clf", build_dnf_classifier())])


def _feature_matrix(df: pd.DataFrame, mode: WeekendMode) -> pd.DataFrame:
    numeric, _ = feature_columns(mode)
    work = df.copy()
    for c in numeric:
        if c not in work.columns:
            work[c] = np.nan
    return work[numeric]


def sample_weight_for_row(season: int, eval_year: int = 2026) -> float:
    """Recency weights: 2026 >> 2025 > older seasons."""
    if season >= eval_year:
        return 6.0
    if season == eval_year - 1:
        return 2.5
    if season == eval_year - 2:
        return 1.5
    return 1.0


def race_sample_weights(df: pd.DataFrame, eval_year: int = 2026) -> np.ndarray:
    return df["season"].map(lambda s: sample_weight_for_row(int(s), eval_year)).values.astype(float)


def train_pairwise_ranker(
    train: pd.DataFrame,
    mode: WeekendMode,
    eval_year: int = 2026,
) -> Pipeline:
    """
    Train regressor on latent pace score = -finish_position.
    Higher predicted score => better expected finish.
    """
    numeric, _ = feature_columns(mode)
    pipe = make_rank_pipeline(numeric)
    X = _feature_matrix(train, mode)
    y = -train["position"].astype(float)
    w = race_sample_weights(train, eval_year)
    pipe.fit(X, y, reg__sample_weight=w)
    return pipe


def train_dnf_model(
    train: pd.DataFrame,
    mode: WeekendMode,
    eval_year: int = 2026,
) -> Pipeline:
    numeric, _ = feature_columns(mode)
    dnf_features = [
        c
        for c in numeric
        if c
        in (
            "car_pace_ewma",
            "driver_skill_ewma",
            "dnf_hazard_car",
            "dnf_hazard_driver",
            "dnf_hazard_combined",
            "constructor_id_dnf_rate_last_5",
            "driver_id_dnf_rate_last_5",
            "circuit_grid_to_finish_delta",
            "is_wet",
            "precipitation_mm",
            "round",
        )
    ]
    if not dnf_features:
        dnf_features = numeric[:8]
    pipe = make_dnf_pipeline(dnf_features)
    X = train[dnf_features].copy()
    for c in dnf_features:
        if c not in X.columns:
            X[c] = np.nan
    w = race_sample_weights(train, eval_year)
    pipe.fit(X, train["dnf"].astype(int), clf__sample_weight=w)
    return pipe, dnf_features


def predict_scores(ranker: Pipeline, df: pd.DataFrame, mode: WeekendMode) -> np.ndarray:
    X = _feature_matrix(df, mode)
    return ranker.predict(X).astype(float)


def predict_dnf_probs(
    dnf_model: Pipeline, df: pd.DataFrame, dnf_features: list[str]
) -> np.ndarray:
    X = df.copy()
    for c in dnf_features:
        if c not in X.columns:
            X[c] = np.nan
    return dnf_model.predict_proba(X[dnf_features])[:, 1]


def plackett_luce_sample_order(
    scores: np.ndarray, rng: np.random.Generator, n_samples: int = 8000
) -> np.ndarray:
    """Return array shape (n_samples, n) of finish positions 1..n per sample."""
    n = len(scores)
    if n == 0:
        return np.empty((0, 0), dtype=int)
    results = np.zeros((n_samples, n), dtype=int)
    for s in range(n_samples):
        remaining = list(range(n))
        order = []
        for pos in range(n):
            idxs = np.array(remaining)
            w = np.exp(scores[idxs] - scores[idxs].max())
            w = w / w.sum()
            pick = rng.choice(idxs, p=w)
            order.append(int(pick))
            remaining.remove(int(pick))
        ranks = np.empty(n, dtype=int)
        for rank, drv in enumerate(order, start=1):
            ranks[drv] = rank
        results[s] = ranks
    return results


def market_probs_from_scores(
    scores: np.ndarray,
    dnf_probs: np.ndarray | None = None,
    temperature: float = 1.0,
    n_samples: int = 6000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Monte Carlo Plackett-Luce finish distribution.
    DNF drivers get a heavy penalty in score space (expected P20+).
    """
    n = len(scores)
    adj = scores.astype(float) / max(temperature, 0.05)
    if dnf_probs is not None:
        adj = adj - dnf_probs.astype(float) * 3.0

    rng = np.random.default_rng(seed)
    samples = plackett_luce_sample_order(adj, rng, n_samples=n_samples)

    win = np.zeros(n)
    podium = np.zeros(n)
    points = np.zeros(n)
    exp_finish = np.zeros(n)
    for s in range(n_samples):
        ranks = samples[s]
        win[int(np.argmin(ranks))] += 1
        for i in range(n):
            r = ranks[i]
            exp_finish[i] += r
            if r <= 3:
                podium[i] += 1
            if r <= 10:
                points[i] += 1
    win /= n_samples
    podium /= n_samples
    points /= n_samples
    exp_finish /= n_samples
    return {
        "win_prob": win,
        "podium_prob": podium,
        "points_prob": points,
        "expected_finish": exp_finish,
    }


def race_level_win_probs(
    df: pd.DataFrame,
    scores: pd.Series,
    temperature: float = 1.0,
    dnf_probs: pd.Series | None = None,
) -> pd.Series:
    """Normalize win probs per race via softmax on scores (fast path)."""
    out = pd.Series(0.0, index=df.index)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        s = scores.loc[g.index].values.astype(float)
        dnf = dnf_probs.loc[g.index].values.astype(float) if dnf_probs is not None else None
        if dnf is not None:
            s = s - dnf * 3.0
        p = race_softmax(s, temperature=temperature)
        out.loc[g.index] = p
    return out


class WeekendRanker:
    """Production ranker bundle: pace scorer + DNF hazard + calibration."""

    def __init__(
        self,
        mode: WeekendMode,
        ranker: Pipeline,
        dnf_model: Pipeline,
        dnf_features: list[str],
        temperature: float = 1.0,
        mc_samples: int = 6000,
    ):
        self.mode = mode
        self.ranker = ranker
        self.dnf_model = dnf_model
        self.dnf_features = dnf_features
        self.temperature = temperature
        self.mc_samples = mc_samples

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        scores = predict_scores(self.ranker, df, self.mode)
        dnf_p = predict_dnf_probs(self.dnf_model, df, self.dnf_features)
        out["pace_score"] = scores
        out["dnf_prob"] = dnf_p

        win = pd.Series(0.0, index=df.index)
        podium = pd.Series(0.0, index=df.index)
        points = pd.Series(0.0, index=df.index)
        exp_finish = pd.Series(0.0, index=df.index)

        for (_, _), g in df.groupby(["season", "round"], sort=False):
            idx = g.index
            s = scores[df.index.get_indexer(idx)]
            d = dnf_p[df.index.get_indexer(idx)]
            markets = market_probs_from_scores(
                s,
                d,
                temperature=self.temperature,
                n_samples=self.mc_samples,
                seed=int(g["season"].iloc[0]) * 100 + int(g["round"].iloc[0]),
            )
            win.loc[idx] = markets["win_prob"]
            podium.loc[idx] = markets["podium_prob"]
            points.loc[idx] = markets["points_prob"]
            exp_finish.loc[idx] = markets["expected_finish"]

        out["win_prob"] = win
        out["podium_prob"] = podium
        out["points_prob"] = points
        out["expected_finish"] = exp_finish
        out["win_prob_softmax"] = race_level_win_probs(
            df, pd.Series(scores, index=df.index), self.temperature, pd.Series(dnf_p, index=df.index)
        )
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.ranker, directory / "ranker.joblib")
        joblib.dump(self.dnf_model, directory / "dnf_model.joblib")
        joblib.dump(
            {
                "mode": self.mode,
                "dnf_features": self.dnf_features,
                "temperature": self.temperature,
                "mc_samples": self.mc_samples,
            },
            directory / "meta.joblib",
        )
        numeric, categorical = feature_columns(self.mode)
        joblib.dump({"numeric": numeric, "categorical": categorical}, directory / "features.joblib")

    @classmethod
    def load(cls, directory: Path) -> "WeekendRanker":
        meta = joblib.load(directory / "meta.joblib")
        return cls(
            mode=meta["mode"],
            ranker=joblib.load(directory / "ranker.joblib"),
            dnf_model=joblib.load(directory / "dnf_model.joblib"),
            dnf_features=meta["dnf_features"],
            temperature=float(meta.get("temperature", 1.0)),
            mc_samples=int(meta.get("mc_samples", 6000)),
        )
