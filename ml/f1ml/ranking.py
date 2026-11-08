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
from f1ml.specs import (
    ERA_RESET_YEAR,
    MONOTONIC_DIRECTION,
    WeekendMode,
    all_numeric_columns,
    feature_columns,
)


class EmpiricalDNFModel:
    """DNF risk = team's retirement rate in the current regulation era, shrunk to the field rate.

    The earlier gradient-boosted DNF classifier scored ROC-AUC ~0.5 out of sample
    (no better than guessing), so a transparent rate is used instead.
    """

    def __init__(self, prior_strength: float = 20.0):
        self.prior_strength = prior_strength
        self.field_rate = 0.12
        self.team_rate: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "EmpiricalDNFModel":
        d = df[df["dnf"].notna()]
        if d.empty:
            return self
        era = (d["season"] >= ERA_RESET_YEAR).astype(int)
        latest = int(era.max())
        era_rows = d[era == latest]
        enough = era_rows.groupby(["season", "round"]).ngroups >= 5
        base = era_rows if enough else d
        self.field_rate = float(base["dnf"].mean())
        team = era_rows.groupby("constructor_id")["dnf"].agg(["sum", "count"])
        k = self.prior_strength
        self.team_rate = (
            (team["sum"] + k * self.field_rate) / (team["count"] + k)
        ).astype(float).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        teams = df.get("constructor_id", pd.Series(index=df.index, dtype=object))
        return teams.map(self.team_rate).fillna(self.field_rate).to_numpy(dtype=float)


def build_rank_regressor(monotonic_cst: list[int] | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        monotonic_cst=monotonic_cst,
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
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ]
    )
    import os

    use_mono = os.environ.get("F1ML_MONOTONIC", "1") != "0"
    directions = [MONOTONIC_DIRECTION.get(c, 0) if use_mono else 0 for c in numeric]
    return Pipeline([("pre", pre), ("reg", build_rank_regressor(directions))])


def make_dnf_pipeline(numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))]),
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
    rank_train = train[pd.to_numeric(train["position"], errors="coerce").notna()].copy()
    if rank_train.empty:
        raise ValueError("ranker training requires at least one numeric finishing position")
    pipe = make_rank_pipeline(numeric)
    X = _feature_matrix(rank_train, mode)
    y = -rank_train["position"].astype(float)
    w = race_sample_weights(rank_train, eval_year)
    pipe.fit(X, y, reg__sample_weight=w)
    return pipe


def train_dnf_model(
    train: pd.DataFrame,
    mode: WeekendMode,
    eval_year: int = 2026,
) -> tuple[EmpiricalDNFModel, list[str]]:
    """Fit the empirical team DNF-rate model (see EmpiricalDNFModel)."""
    import os

    if os.environ.get("F1ML_DNF") == "hgb":
        return train_dnf_classifier(train, mode, eval_year)
    dnf_train = train[train["dnf"].notna()]
    if dnf_train["dnf"].astype(int).nunique() < 2:
        raise ValueError("DNF training requires both positive and negative labels")
    return EmpiricalDNFModel().fit(dnf_train), ["constructor_id"]


def train_dnf_classifier(
    train: pd.DataFrame,
    mode: WeekendMode,
    eval_year: int = 2026,
) -> tuple[Pipeline, list[str]]:
    """Legacy gradient-boosted DNF classifier (kept for comparison)."""
    dnf_train = train[train["dnf"].notna()].copy()
    labels = dnf_train["dnf"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("DNF training requires both positive and negative labels")
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
    if mode == "post_sprint" and "sprint_position" in numeric:
        dnf_features.append("sprint_position")
    pipe = make_dnf_pipeline(dnf_features)
    X = dnf_train[dnf_features].copy()
    for c in dnf_features:
        if c not in X.columns:
            X[c] = np.nan
    w = race_sample_weights(dnf_train, eval_year)
    pipe.fit(X, labels, clf__sample_weight=w)
    return pipe, dnf_features


def predict_scores(ranker: Pipeline, df: pd.DataFrame, mode: WeekendMode) -> np.ndarray:
    X = _feature_matrix(df, mode)
    # Older saved artifacts can still declare features removed from the safe
    # allowlist. Keep their input schema loadable, but impute those columns so
    # stale race/weather/sprint observations cannot enter current forecasts.
    pre = ranker.named_steps.get("pre") if hasattr(ranker, "named_steps") else None
    if pre is not None:
        required = []
        for name, _transformer, columns in pre.transformers:
            if name == "num":
                required.extend(list(columns))
        for column in required:
            if column not in X.columns:
                X[column] = np.nan
        X = X[required]
    return ranker.predict(X).astype(float)


def predict_dnf_probs(
    dnf_model: Pipeline, df: pd.DataFrame, dnf_features: list[str]
) -> np.ndarray:
    if isinstance(dnf_model, EmpiricalDNFModel):
        return dnf_model.predict(df)
    # Legacy classifiers may declare inputs outside the cutoff-safe allowlist
    # (e.g. observed race-day weather); impute those rather than read them.
    safe = set(all_numeric_columns())
    X = df.copy()
    for c in dnf_features:
        if c not in safe or c not in X.columns:
            X[c] = np.nan
    return dnf_model.predict_proba(X[dnf_features])[:, 1]


def plackett_luce_sample_order(
    scores: np.ndarray, rng: np.random.Generator, n_samples: int = 8000
) -> np.ndarray:
    """Sample Plackett–Luce orders with the equivalent Gumbel-top-k method."""
    n = len(scores)
    if n == 0:
        return np.empty((0, 0), dtype=int)
    logits = np.asarray(scores, dtype=float)
    perturbed = logits[None, :] + rng.gumbel(size=(n_samples, n))
    order = np.argsort(-perturbed, axis=1)
    return np.argsort(order, axis=1) + 1


def market_probs_from_scores(
    scores: np.ndarray,
    dnf_probs: np.ndarray | None = None,
    temperature: float = 1.0,
    n_samples: int = 6000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate DNF separately, then sample a Plackett–Luce classified order.

    Position probabilities and expected finish are conditional on classification;
    win, podium, points and DNF probabilities are unconditional race outcomes.
    """
    n = len(scores)
    adj = scores.astype(float) / max(temperature, 0.05)
    rng = np.random.default_rng(seed)
    risk = (
        np.clip(np.asarray(dnf_probs, dtype=float), 0.0, 1.0)
        if dnf_probs is not None
        else np.zeros(n)
    )
    retired = rng.random((n_samples, n)) < risk[None, :]
    all_retired = retired.all(axis=1) if n else np.zeros(n_samples, dtype=bool)
    # A completed race has at least one classified driver. In the vanishingly
    # rare all-retired draw, keep the lowest-risk driver classified.
    if n and all_retired.any():
        retired[all_retired, int(np.argmin(risk))] = False

    # Exact Plackett-Luce win probability given each retirement draw: softmax over
    # classified drivers. Averaging these is smooth and never exactly zero.
    logits = np.where(retired, -np.inf, adj[None, :])
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    win_prob = (weights / weights.sum(axis=1, keepdims=True)).mean(axis=0)

    perturbed = adj[None, :] + rng.gumbel(size=(n_samples, n))
    perturbed[retired] = -np.inf
    order = np.argsort(-perturbed, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    ranks[retired] = 0

    classified = ranks > 0
    classified_n = classified.sum(axis=0)
    position_probs = np.zeros((n, n), dtype=float)
    for position in range(1, n + 1):
        position_probs[:, position - 1] = np.divide(
            (ranks == position).sum(axis=0),
            classified_n,
            out=np.full(n, np.nan, dtype=float),
            where=classified_n > 0,
        )
    finish_sum = np.where(classified, ranks, 0).sum(axis=0)
    podium_prob = np.maximum(((ranks >= 1) & (ranks <= 3)).mean(axis=0), win_prob)
    points_prob = np.maximum(((ranks >= 1) & (ranks <= 10)).mean(axis=0), podium_prob)
    return {
        "win_prob": win_prob,
        "podium_prob": podium_prob,
        "points_prob": points_prob,
        "expected_finish": np.divide(
            finish_sum,
            classified_n,
            out=np.full(n, np.nan, dtype=float),
            where=classified_n > 0,
        ),
        "finish_position_probs": position_probs,
        "dnf_prob": retired.mean(axis=0),
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


FIT_ONLY_RNG_ATTRS = ("_feature_subsample_rng",)


def strip_fit_only_state(model: Any) -> Any:
    """Drop numpy Generator objects that scikit-learn keeps only for fitting.

    A pickled numpy 2.x Generator cannot be unpickled under numpy 1.x ("PCG64 is
    not a known BitGenerator module"), which breaks the dashboard when Streamlit
    runs in a different Python environment. Prediction never uses them, and
    refitting recreates them.
    """
    steps = model.named_steps.values() if hasattr(model, "named_steps") else [model]
    for step in steps:
        for attr in FIT_ONLY_RNG_ATTRS:
            if hasattr(step, attr):
                setattr(step, attr, None)
    return model


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
        max_positions = 0

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
            max_positions = max(max_positions, len(g))
            for position in range(1, len(g) + 1):
                col = f"finish_p{position}_prob"
                if col not in out:
                    out[col] = np.nan
                out.loc[idx, col] = markets["finish_position_probs"][:, position - 1]

        out["win_prob"] = win
        out["podium_prob"] = podium
        out["points_prob"] = points
        out["expected_finish"] = exp_finish
        for position in range(1, max_positions + 1):
            col = f"finish_p{position}_prob"
            if col not in out:
                out[col] = np.nan
        out["win_prob_softmax"] = race_level_win_probs(
            df, pd.Series(scores, index=df.index), self.temperature, pd.Series(dnf_p, index=df.index)
        )
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(strip_fit_only_state(self.ranker), directory / "ranker.joblib")
        joblib.dump(strip_fit_only_state(self.dnf_model), directory / "dnf_model.joblib")
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
