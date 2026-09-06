"""Model pipelines and race-level scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1ml.eval import race_softmax
from f1ml.specs import WeekendMode, feature_columns


def build_classifier(kind: str = "hgb") -> Any:
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=8,
        l2_regularization=0.1,
        random_state=42,
    )


def build_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=8,
        l2_regularization=0.1,
        random_state=42,
    )


def make_pipeline(kind: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if kind == "logreg":
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)]
    )
    if kind == "finish":
        clf = build_regressor()
    else:
        clf = build_classifier(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def feature_matrix(df: pd.DataFrame, mode: WeekendMode) -> pd.DataFrame:
    numeric, categorical = feature_columns(mode)
    cols = numeric + categorical
    work = df.copy()
    for c in cols:
        if c not in work.columns:
            work[c] = np.nan
    return work[cols]


def raw_win_scores(model: Pipeline, df: pd.DataFrame, mode: WeekendMode) -> np.ndarray:
    X = feature_matrix(df, mode)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def apply_race_probs(
    df: pd.DataFrame, raw_scores: pd.Series, temperature: float = 1.0
) -> pd.Series:
    probs = pd.Series(0.0, index=df.index)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        s = raw_scores.loc[g.index].values.astype(float)
        p = race_softmax(s, temperature=temperature)
        probs.loc[g.index] = p
    return probs


def finish_position_probs(
    expected_finish: pd.Series, n_positions: int, sigma: float = 2.0
) -> pd.DataFrame:
    """Plackett-Luce style finish distribution from expected finish positions."""
    positions = list(range(1, n_positions + 1))
    rows = {}
    for idx, ef in expected_finish.items():
        weights = [max(0.01, np.exp(-abs(p - float(ef)) / sigma)) for p in positions]
        total = sum(weights)
        rows[idx] = [w / total for w in weights]
    return pd.DataFrame(rows, index=positions).T


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


def predict_binary_proba(
    model: Pipeline, df: pd.DataFrame, mode: WeekendMode
) -> np.ndarray:
    X = feature_matrix(df, mode)
    return model.predict_proba(X)[:, 1]


class WeekendModels:
    """Bundle of models for one weekend mode."""

    def __init__(
        self,
        mode: WeekendMode,
        win_model: Pipeline,
        podium_model: Pipeline,
        dnf_model: Pipeline,
        finish_model: Pipeline,
        win_logreg: Pipeline | None = None,
        temperature: float = 1.0,
    ):
        self.mode = mode
        self.win_model = win_model
        self.podium_model = podium_model
        self.dnf_model = dnf_model
        self.finish_model = finish_model
        self.win_logreg = win_logreg
        self.temperature = temperature

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.win_model, directory / "win_hgb.joblib")
        joblib.dump(self.podium_model, directory / "podium_hgb.joblib")
        joblib.dump(self.dnf_model, directory / "dnf_hgb.joblib")
        joblib.dump(self.finish_model, directory / "finish_hgb.joblib")
        if self.win_logreg is not None:
            joblib.dump(self.win_logreg, directory / "win_logreg.joblib")
        joblib.dump({"temperature": self.temperature}, directory / "meta.joblib")
        numeric, categorical = feature_columns(self.mode)
        joblib.dump(
            {"numeric": numeric, "categorical": categorical, "mode": self.mode},
            directory / "features.joblib",
        )

    @classmethod
    def load(cls, directory: Path) -> "WeekendModels":
        meta = joblib.load(directory / "meta.joblib")
        logreg_path = directory / "win_logreg.joblib"
        return cls(
            mode=joblib.load(directory / "features.joblib")["mode"],
            win_model=joblib.load(directory / "win_hgb.joblib"),
            podium_model=joblib.load(directory / "podium_hgb.joblib"),
            dnf_model=joblib.load(directory / "dnf_hgb.joblib"),
            finish_model=joblib.load(directory / "finish_hgb.joblib"),
            win_logreg=joblib.load(logreg_path) if logreg_path.exists() else None,
            temperature=float(meta.get("temperature", 1.0)),
        )
