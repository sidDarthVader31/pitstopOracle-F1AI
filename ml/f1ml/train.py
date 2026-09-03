from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1ml.paths import MODELS, PROCESSED, REPORTS

FEATURE_NUM = [
    "quali_position",
    "grid",
    "grid_vs_quali",
    "best_quali_seconds",
    "champ_position_before",
    "champ_points_before",
    "driver_id_points_last_3",
    "driver_id_points_last_5",
    "driver_id_avg_finish_last_5",
    "driver_id_dnf_rate_last_5",
    "constructor_id_points_last_3",
    "constructor_id_points_last_5",
    "constructor_id_avg_finish_last_5",
    "constructor_id_dnf_rate_last_5",
    "quali_vs_teammate",
    "circuit_avg_finish_prior",
    "is_wet",
    "has_sprint",
    "sprint_position",
    "round",
    "precipitation_mm",
]
FEATURE_CAT = ["constructor_id"]
TRAIN_SEASONS = {2022, 2023, 2024}


def _race_key(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype(str) + "-" + df["round"].astype(str)


def pole_baseline(df: pd.DataFrame) -> pd.Series:
    """Predict the pole sitter (best qualifying position) in each race."""
    idx = df.groupby(["season", "round"])["quali_position"].idxmin()
    pred = pd.Series(False, index=df.index)
    pred.loc[idx] = True
    return pred


def champ_leader_baseline(df: pd.DataFrame) -> pd.Series:
    """Championship leader before the race; round 1 falls back to pole."""
    pred = pd.Series(False, index=df.index)
    for (_, _), g in df.groupby(["season", "round"]):
        if g["champ_points_before"].fillna(0).sum() == 0:
            pick = g["quali_position"].idxmin()
        else:
            pick = g["champ_position_before"].idxmin()
        pred.loc[pick] = True
    return pred


def winner_hit_rate(df: pd.DataFrame, picked: pd.Series) -> float:
    hits = 0
    n = 0
    for _, g in df.groupby(_race_key(df)):
        n += 1
        winner_idx = g.index[g["won"] == 1]
        if len(winner_idx) == 0:
            continue
        if bool(picked.loc[winner_idx].any()):
            hits += 1
    return hits / n if n else 0.0


def topk_hit_rate(df: pd.DataFrame, scores: pd.Series, k: int = 3) -> float:
    hits = 0
    n = 0
    for _, g in df.groupby(["season", "round"]):
        n += 1
        winner = g.index[g["won"] == 1]
        if len(winner) == 0:
            continue
        top = scores.loc[g.index].nlargest(k)
        if winner[0] in top.index:
            hits += 1
    return hits / n if n else 0.0


def _pipeline(kind: str) -> Pipeline:
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if kind == "logreg":
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(steps=num_steps)
    cat = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [("num", num, FEATURE_NUM), ("cat", cat, FEATURE_CAT)]
    )
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline([("pre", pre), ("clf", clf)])


def _eda(df: pd.DataFrame) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pole = pole_baseline(df)
    pole_rate = winner_hit_rate(df, pole)
    wet_races = df.groupby(["season", "round"])["is_wet"].max()
    dnf_by_team = (
        df.groupby("constructor_name")["dnf"].mean().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Pole wins"], [pole_rate])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of races")
    ax.set_title("How often the pole sitter wins (2022+)")
    fig.tight_layout()
    fig.savefig(REPORTS / "pole_win_rate.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    dnf_by_team.plot(kind="bar", ax=ax)
    ax.set_ylabel("DNF rate")
    ax.set_title("DNF rate by constructor")
    fig.tight_layout()
    fig.savefig(REPORTS / "dnf_by_constructor.png", dpi=120)
    plt.close(fig)

    return {
        "n_rows": int(len(df)),
        "n_races": int(df.groupby(["season", "round"]).ngroups),
        "pole_win_rate": pole_rate,
        "wet_race_share": float(wet_races.mean()),
        "overall_dnf_rate": float(df["dnf"].mean()),
    }


def train_and_eval() -> dict:
    df = pd.read_parquet(PROCESSED / "driver_race.parquet")
    df = df[df["position"].notna() | (df["won"] == 0)].copy()
    # Keep races that have a winner label.
    has_winner = df.groupby(["season", "round"])["won"].transform("max") == 1
    df = df[has_winner].copy()

    eda = _eda(df)
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[~df["season"].isin(TRAIN_SEASONS)].copy()

    metrics: dict = {"eda": eda, "train_races": int(train.groupby(["season", "round"]).ngroups),
                     "test_races": int(test.groupby(["season", "round"]).ngroups)}

    for split_name, split in [("train", train), ("test", test)]:
        pole = pole_baseline(split)
        champ = champ_leader_baseline(split)
        metrics[f"{split_name}_pole_hit"] = winner_hit_rate(split, pole)
        metrics[f"{split_name}_champ_hit"] = winner_hit_rate(split, champ)

    X_train = train[FEATURE_NUM + FEATURE_CAT]
    y_train = train["won"]
    X_test = test[FEATURE_NUM + FEATURE_CAT]
    models = {}
    for kind in ("logreg", "rf"):
        pipe = _pipeline(kind)
        pipe.fit(X_train, y_train)
        models[kind] = pipe
        for split_name, split, X in (("train", train, X_train), ("test", test, X_test)):
            proba = pipe.predict_proba(X)[:, 1]
            scores = pd.Series(proba, index=split.index)
            picked = pd.Series(False, index=split.index)
            for _, g in split.groupby(["season", "round"]):
                picked.loc[scores.loc[g.index].idxmax()] = True
            metrics[f"{split_name}_{kind}_hit"] = winner_hit_rate(split, picked)
            metrics[f"{split_name}_{kind}_top3"] = topk_hit_rate(split, scores, k=3)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["rf"], MODELS / "winner_rf.joblib")
    joblib.dump(models["logreg"], MODELS / "winner_logreg.joblib")
    joblib.dump({"numeric": FEATURE_NUM, "categorical": FEATURE_CAT}, MODELS / "features.joblib")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_eval_md(metrics, test, models["rf"])
    return metrics


def _write_eval_md(metrics: dict, test: pd.DataFrame, rf: Pipeline) -> None:
    beat = metrics["test_rf_hit"] > metrics["test_pole_hit"]
    lines = [
        "# F1 winner model — evaluation",
        "",
        "Time split: **train 2022–2024**, **test 2025–2026** (held-out later seasons).",
        "",
        "## Dataset",
        "",
        f"- Rows (driver-race): {metrics['eda']['n_rows']}",
        f"- Races: {metrics['eda']['n_races']} (train {metrics['train_races']}, test {metrics['test_races']})",
        f"- Pole sitter win rate (all data): **{metrics['eda']['pole_win_rate']:.1%}**",
        f"- Wet races (daily precipitation ≥ 1 mm): {metrics['eda']['wet_race_share']:.1%}",
        f"- Overall DNF rate: {metrics['eda']['overall_dnf_rate']:.1%}",
        "",
        "See `pole_win_rate.png` and `dnf_by_constructor.png`.",
        "",
        "## Winner hit rate (share of races where predicted winner is correct)",
        "",
        "| Method | Train | Test |",
        "|---|---:|---:|",
        f"| Baseline: pole | {metrics['train_pole_hit']:.1%} | {metrics['test_pole_hit']:.1%} |",
        f"| Baseline: championship leader | {metrics['train_champ_hit']:.1%} | {metrics['test_champ_hit']:.1%} |",
        f"| Logistic regression | {metrics['train_logreg_hit']:.1%} | {metrics['test_logreg_hit']:.1%} |",
        f"| Random forest | {metrics['train_rf_hit']:.1%} | {metrics['test_rf_hit']:.1%} |",
        "",
        "## Top-3 (actual winner in model's top 3 probabilities)",
        "",
        f"- Train RF: {metrics['train_rf_top3']:.1%}",
        f"- Test RF: {metrics['test_rf_top3']:.1%}",
        f"- Train logreg: {metrics['train_logreg_top3']:.1%}",
        f"- Test logreg: {metrics['test_logreg_top3']:.1%}",
        "",
        "## Did ML beat pole?",
        "",
        (
            f"Yes — test RF {metrics['test_rf_hit']:.1%} vs pole {metrics['test_pole_hit']:.1%}."
            if beat
            else f"Not on this split — test RF {metrics['test_rf_hit']:.1%} vs pole {metrics['test_pole_hit']:.1%}. "
            "That is expected with ~20 races/year, DNFs, and qualifying already explaining most winners."
        ),
        "",
        "Saved models: `ml/models/winner_rf.joblib`, `ml/models/winner_logreg.joblib`.",
        "",
    ]
    (REPORTS / "EVAL.md").write_text("\n".join(lines))
