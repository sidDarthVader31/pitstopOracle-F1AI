"""Race-level Safety Car/VSC labels and a chronological baseline model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_safety_car_labels(
    sessions: pd.DataFrame, control: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Derive race-level deployed SC/VSC labels from OpenF1 race-control text."""
    rows = []
    if sessions.empty:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "has_sc", "has_vsc", "source"]
        )
    for event in sessions.itertuples(index=False):
        event_control = control[control["session_key"] == event.session_key]
        event_control = event_control[event_control["category"].eq("SafetyCar")]
        messages = event_control["message"].fillna("").astype(str).str.upper()
        deployments = messages[~messages.str.contains("ENDING|ENDED", regex=True)]
        has_vsc = deployments.str.contains("VIRTUAL SAFETY CAR|VSC DEPLOYED", regex=True).any()
        has_sc = (
            deployments.str.contains("SAFETY CAR", regex=False)
            & ~deployments.str.contains("VIRTUAL", regex=False)
            & ~deployments.str.contains("VSC", regex=False)
        ).any()
        rows.append(
            {
                "season": int(event.season),
                "round": int(event.round),
                "session_key": int(event.session_key),
                "circuit_key": getattr(event, "circuit_key", np.nan),
                "has_sc": int(has_sc),
                "has_vsc": int(has_vsc),
                "sc_period_messages": int(
                    deployments.str.contains("SAFETY CAR", regex=False)
                    .where(~deployments.str.contains("VIRTUAL|VSC", regex=True), False)
                    .sum()
                ),
                "vsc_period_messages": int(
                    deployments.str.contains("VIRTUAL SAFETY CAR|VSC", regex=True).sum()
                ),
                "race_control_rows": int(len(control[control["session_key"] == event.session_key])),
                "source": "OpenF1 race_control",
            }
        )
    labels = pd.DataFrame(rows).merge(
        calendar[["season", "round", "circuit_id"]].drop_duplicates(["season", "round"]),
        on=["season", "round"],
        how="left",
        validate="one_to_one",
    )
    return labels


def add_prior_circuit_safety_features(
    frame: pd.DataFrame, event_labels: pd.DataFrame
) -> pd.DataFrame:
    """Attach shifted circuit SC/VSC rates, leaving first visits missing."""
    out = frame.drop(columns=["circuit_sc_rate_prior", "circuit_vsc_rate_prior"], errors="ignore")
    if event_labels.empty:
        out["circuit_sc_rate_prior"] = np.nan
        out["circuit_vsc_rate_prior"] = np.nan
        return out
    rates = event_labels.sort_values(["season", "round"]).copy()
    for label, feature in (("has_sc", "circuit_sc_rate_prior"), ("has_vsc", "circuit_vsc_rate_prior")):
        rates[feature] = rates.groupby("circuit_id", sort=False)[label].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
    return out.merge(
        rates[["season", "round", "circuit_id", "circuit_sc_rate_prior", "circuit_vsc_rate_prior"]],
        on=["season", "round", "circuit_id"],
        how="left",
        validate="many_to_one",
    )


def add_upcoming_circuit_safety_features(
    frame: pd.DataFrame, event_labels: pd.DataFrame, season: int, round_num: int
) -> pd.DataFrame:
    """Attach event-history rates to a future race using completed events only."""
    out = frame.copy()
    for feature, label in (
        ("circuit_sc_rate_prior", "has_sc"),
        ("circuit_vsc_rate_prior", "has_vsc"),
    ):
        values = []
        for circuit_id in out["circuit_id"]:
            prior = event_labels[
                (event_labels["circuit_id"] == circuit_id)
                & (
                    (event_labels["season"] < season)
                    | ((event_labels["season"] == season) & (event_labels["round"] < round_num))
                )
            ]
            values.append(float(prior[label].mean()) if not prior.empty else np.nan)
        out[feature] = values
    return out


SAFETY_FEATURES = [
    "circuit_sc_rate_prior",
    "circuit_vsc_rate_prior",
    "circuit_street",
    "round",
    "regs_era",
]


def build_safety_event_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def evaluate_safety_event_model(labels: pd.DataFrame, target: str, min_train_races: int = 30) -> dict:
    """Expanding evaluation of a single event label using prior events only."""
    view = labels.sort_values(["season", "round"]).reset_index(drop=True)
    predictions = []
    for i in range(min_train_races, len(view)):
        train, test = view.iloc[:i], view.iloc[[i]]
        if train[target].nunique() < 2:
            continue
        model = build_safety_event_model()
        medians = train[SAFETY_FEATURES].median().fillna(0.5)
        x_train = train[SAFETY_FEATURES].fillna(medians)
        x_test = test[SAFETY_FEATURES].fillna(medians)
        model.fit(x_train, train[target].astype(int))
        prob = float(model.predict_proba(x_test)[0, 1])
        baseline_prob = float(train[target].mean())
        predictions.append(
            {
                "season": int(test["season"].iloc[0]),
                "round": int(test["round"].iloc[0]),
                "actual": int(test[target].iloc[0]),
                "probability": prob,
                "baseline_probability": baseline_prob,
            }
        )
    if not predictions:
        return {"n_races": 0, "predictions": []}
    y = np.array([row["actual"] for row in predictions])
    p = np.array([row["probability"] for row in predictions])
    baseline = np.array([row["baseline_probability"] for row in predictions])
    result = {
        "n_races": len(y),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "baseline_log_loss": float(log_loss(y, baseline, labels=[0, 1])),
        "baseline_brier": float(brier_score_loss(y, baseline)),
        "predictions": predictions,
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    return result
