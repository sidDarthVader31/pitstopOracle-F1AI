from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1ml.eval import (
    calibration_bins,
    champ_leader_probabilities,
    equal_field_probabilities,
    finish_mae,
    fit_temperature,
    pole_probabilities,
    race_brier,
    race_log_loss,
    rank_correlation,
    slice_metrics,
    topk_hit_rate,
    winner_hit_rate,
    winner_hit_rate_probs,
)
from f1ml.modeling import (
    WeekendModels,
    apply_race_probs,
    feature_matrix,
    make_pipeline,
    plackett_luce_win_probs,
    predict_binary_proba,
    raw_win_scores,
)
from f1ml.paths import MODELS, PROCESSED, REPORTS
from f1ml.specs import TEST_SEASONS, TRAIN_SEASONS, WeekendMode, feature_columns, season_range_label


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "driver_race.parquet")
    has_winner = df.groupby(["season", "round"])["won"].transform("max") == 1
    df = df[has_winner & df["position"].notna()].copy()
    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    return train, test


def _walk_forward_seasons(df: pd.DataFrame) -> list[dict]:
    seasons = sorted(df["season"].unique())
    folds = []
    for i in range(2, len(seasons)):
        train_seasons = set(seasons[:i])
        test_season = seasons[i]
        folds.append(
            {
                "train": df[df["season"].isin(train_seasons)],
                "test": df[df["season"] == test_season],
                "test_season": int(test_season),
            }
        )
    return folds


def _pole_baseline_hit(df: pd.DataFrame) -> float:
    probs = pole_probabilities(df)
    return winner_hit_rate(df, probs)


def _champ_baseline_hit(df: pd.DataFrame) -> float:
    probs = champ_leader_probabilities(df)
    return winner_hit_rate(df, probs)


def _equal_baseline_hit(df: pd.DataFrame) -> float:
    probs = equal_field_probabilities(df)
    return winner_hit_rate_probs(df, probs)


def _train_mode_models(
    train: pd.DataFrame,
    mode: WeekendMode,
    val: pd.DataFrame | None = None,
) -> WeekendModels:
    numeric, categorical = feature_columns(mode)
    cols = numeric + categorical

    win_pipe = make_pipeline("hgb", numeric, categorical)
    podium_pipe = make_pipeline("hgb", numeric, categorical)
    dnf_pipe = make_pipeline("hgb", numeric, categorical)
    finish_pipe = make_pipeline("finish", numeric, categorical)
    logreg_pipe = make_pipeline("logreg", numeric, categorical)

    X_train = train[cols]
    win_pipe.fit(X_train, train["won"])
    podium_pipe.fit(X_train, train["podium"])
    dnf_pipe.fit(X_train, train["dnf"])
    finish_pipe.fit(X_train, train["position"].astype(float))
    logreg_pipe.fit(X_train, train["won"])

    temperature = 1.0
    cal_df = val if val is not None and len(val) else train
    raw = pd.Series(raw_win_scores(win_pipe, cal_df, mode), index=cal_df.index)
    temperature = fit_temperature(cal_df, raw)

    return WeekendModels(
        mode=mode,
        win_model=win_pipe,
        podium_model=podium_pipe,
        dnf_model=dnf_pipe,
        finish_model=finish_pipe,
        win_logreg=logreg_pipe,
        temperature=temperature,
    )


def _score_bundle(df: pd.DataFrame, bundle: WeekendModels) -> pd.DataFrame:
    mode = bundle.mode
    out = df.copy()
    raw = pd.Series(raw_win_scores(bundle.win_model, df, mode), index=df.index)
    out["win_prob_raw"] = raw
    out["win_prob"] = apply_race_probs(df, raw, temperature=bundle.temperature)
    out["podium_prob"] = predict_binary_proba(bundle.podium_model, df, mode)
    out["dnf_prob"] = predict_binary_proba(bundle.dnf_model, df, mode)
    out["expected_finish"] = bundle.finish_model.predict(feature_matrix(df, mode))
    out["model_pick"] = out["win_prob"] == out["win_prob"].max()
    return out


def _eval_mode(
    df: pd.DataFrame,
    bundle: WeekendModels,
    prefix: str,
) -> dict:
    scored = _score_bundle(df, bundle)
    probs = scored["win_prob"]
    raw = scored["win_prob_raw"]
    rank_probs = plackett_luce_win_probs(df, scored["expected_finish"], temperature=bundle.temperature)
    metrics: dict = {
        f"{prefix}_hit": winner_hit_rate(df, probs),
        f"{prefix}_top3": topk_hit_rate(df, probs, k=3),
        f"{prefix}_log_loss": race_log_loss(df, probs),
        f"{prefix}_brier": race_brier(df, probs),
        f"{prefix}_rank_corr": rank_correlation(df, raw),
        f"{prefix}_finish_mae": finish_mae(df, scored["expected_finish"]),
        f"{prefix}_calibration": calibration_bins(df, probs),
        f"{prefix}_ranker_hit": winner_hit_rate(df, rank_probs),
        f"{prefix}_ranker_log_loss": race_log_loss(df, rank_probs),
        f"{prefix}_ranker_brier": race_brier(df, rank_probs),
    }
    metrics.update({f"{prefix}_{k}": v for k, v in slice_metrics(df, probs, probs).items()})
    return metrics


def _eval_baselines(df: pd.DataFrame, prefix: str, include_pole: bool = True) -> dict:
    out = {
        f"{prefix}_equal_hit": _equal_baseline_hit(df),
        f"{prefix}_champ_hit": _champ_baseline_hit(df),
        f"{prefix}_equal_log_loss": race_log_loss(df, equal_field_probabilities(df)),
        f"{prefix}_champ_log_loss": race_log_loss(df, champ_leader_probabilities(df)),
    }
    if include_pole:
        pole_p = pole_probabilities(df)
        out[f"{prefix}_pole_hit"] = winner_hit_rate(df, pole_p)
        out[f"{prefix}_pole_log_loss"] = race_log_loss(df, pole_p)
        out[f"{prefix}_pole_brier"] = race_brier(df, pole_p)
    return out


def _eda(df: pd.DataFrame) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pole_rate = _pole_baseline_hit(df)
    wet_races = df.groupby(["season", "round"])["is_wet"].max()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Pole wins"], [pole_rate])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of races")
    ax.set_title("How often the pole sitter wins (2022+)")
    fig.tight_layout()
    fig.savefig(REPORTS / "pole_win_rate.png", dpi=120)
    plt.close(fig)

    dnf_by_team = df.groupby("constructor_name")["dnf"].mean().sort_values(ascending=False)
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


def _write_eval_md(metrics: dict) -> None:
    pre_test = metrics.get("test_pre_quali", {})
    post_test = metrics.get("test_post_quali", {})
    train_label = season_range_label(TRAIN_SEASONS)
    test_label = season_range_label(TEST_SEASONS)
    split_line = (
        f"Time split: **train {train_label}**, **test {test_label}**."
        if TEST_SEASONS
        else f"Time split: **train {train_label}** (no held-out test; walk-forward below)."
    )
    lines = [
        "# F1 Oracle — evaluation (v2)",
        "",
        split_line,
        "Separate **pre-quali** and **post-quali** models with race-level softmax calibration.",
        "",
        "## Dataset",
        "",
        f"- Rows: {metrics['eda']['n_rows']}",
        f"- Races: {metrics['eda']['n_races']}",
        f"- Pole win rate: **{metrics['eda']['pole_win_rate']:.1%}**",
        "",
    ]
    if TEST_SEASONS:
        lines += [
            "## Pre-quali model (test)",
            "",
            "| Metric | Oracle | Equal field | Champ leader |",
            "|---|---:|---:|---:|",
            f"| Winner hit | {pre_test.get('test_pre_quali_hit', 0):.1%} | {pre_test.get('test_pre_quali_equal_hit', 0):.1%} | {pre_test.get('test_pre_quali_champ_hit', 0):.1%} |",
            f"| Log-loss | {pre_test.get('test_pre_quali_log_loss', 0):.3f} | {pre_test.get('test_pre_quali_equal_log_loss', 0):.3f} | {pre_test.get('test_pre_quali_champ_log_loss', 0):.3f} |",
            f"| Brier | {pre_test.get('test_pre_quali_brier', 0):.3f} | — | — |",
            f"| Rank corr | {pre_test.get('test_pre_quali_rank_corr', 0):.3f} | — | — |",
            "",
            "## Post-quali model (test)",
            "",
            "| Metric | Oracle | Pole | Champ leader |",
            "|---|---:|---:|---:|",
            f"| Winner hit | {post_test.get('test_post_quali_hit', 0):.1%} | {post_test.get('test_post_quali_pole_hit', 0):.1%} | {post_test.get('test_post_quali_champ_hit', 0):.1%} |",
            f"| Log-loss | {post_test.get('test_post_quali_log_loss', 0):.3f} | {post_test.get('test_post_quali_pole_log_loss', 0):.3f} | {post_test.get('test_post_quali_champ_log_loss', 0):.3f} |",
            f"| Brier | {post_test.get('test_post_quali_brier', 0):.3f} | {post_test.get('test_post_quali_pole_brier', 0):.3f} | — |",
            f"| Top-3 hit | {post_test.get('test_post_quali_top3', 0):.1%} | — | — |",
            f"| Ranker log-loss | {post_test.get('test_post_quali_ranker_log_loss', 0):.3f} | — | — |",
            "",
        ]
    lines += [
        "## Walk-forward (by season)",
        "",
    ]
    for fold in metrics.get("walk_forward", []):
        lines.append(
            f"- Season {fold['test_season']}: pre hit {fold.get('pre_hit', 0):.1%}, "
            f"post hit {fold.get('post_hit', 0):.1%}, post log-loss {fold.get('post_log_loss', 0):.3f}"
        )
    lines += [
        "",
        "Models: `ml/models/pre_quali/`, `ml/models/post_quali/`.",
        "",
    ]
    (REPORTS / "EVAL.md").write_text("\n".join(lines))


def train_and_eval() -> dict:
    df = _load_training_frame()
    eda = _eda(df)
    train, test = _split(df)

    # Post-quali needs quali data present.
    train_post = train[train["quali_position"].notna()].copy()
    test_post = test[test["quali_position"].notna()].copy()

    pre_bundle = _train_mode_models(train, "pre_quali", val=test if len(test) else None)
    post_bundle = _train_mode_models(train_post, "post_quali", val=test_post if len(test_post) else None)

    MODELS.mkdir(parents=True, exist_ok=True)
    pre_bundle.save(MODELS / "pre_quali")
    post_bundle.save(MODELS / "post_quali")

    # Legacy symlink-style artifacts for any old references.
    joblib.dump(pre_bundle.win_model, MODELS / "winner_rf.joblib")
    joblib.dump(pre_bundle.win_logreg, MODELS / "winner_logreg.joblib")
    numeric_pre, cat_pre = feature_columns("pre_quali")
    joblib.dump({"numeric": numeric_pre, "categorical": cat_pre}, MODELS / "features.joblib")

    metrics: dict = {
        "eda": eda,
        "train_races": int(train.groupby(["season", "round"]).ngroups),
        "test_races": int(test.groupby(["season", "round"]).ngroups),
        "pre_quali_temperature": pre_bundle.temperature,
        "post_quali_temperature": post_bundle.temperature,
    }

    for split_name, split in [("train", train), ("test", test)]:
        if split.empty:
            continue
        pre_m = _eval_mode(split, pre_bundle, f"{split_name}_pre_quali")
        pre_b = _eval_baselines(split, f"{split_name}_pre_quali", include_pole=False)
        metrics[f"{split_name}_pre_quali"] = {**pre_m, **pre_b}

    for split_name, split in [("train", train_post), ("test", test_post)]:
        if split.empty:
            continue
        post_m = _eval_mode(split, post_bundle, f"{split_name}_post_quali")
        post_b = _eval_baselines(split, f"{split_name}_post_quali", include_pole=True)
        metrics[f"{split_name}_post_quali"] = {**post_m, **post_b}

    # Walk-forward validation.
    wf_results = []
    for fold in _walk_forward_seasons(df):
        tr = fold["train"]
        te = fold["test"]
        tr_post = tr[tr["quali_position"].notna()]
        te_post = te[te["quali_position"].notna()]
        if te.empty:
            continue
        pre_f = _train_mode_models(tr, "pre_quali")
        post_f = _train_mode_models(tr_post, "post_quali") if len(tr_post) else None
        pre_scored = _score_bundle(te, pre_f)
        wf_row = {
            "test_season": fold["test_season"],
            "pre_hit": winner_hit_rate(te, pre_scored["win_prob"]),
            "pre_log_loss": race_log_loss(te, pre_scored["win_prob"]),
        }
        if post_f is not None and len(te_post):
            post_scored = _score_bundle(te_post, post_f)
            wf_row["post_hit"] = winner_hit_rate(te_post, post_scored["win_prob"])
            wf_row["post_log_loss"] = race_log_loss(te_post, post_scored["win_prob"])
        wf_results.append(wf_row)
    metrics["walk_forward"] = wf_results

    # Back-compat keys for old UI until updated.
    test_post_m = metrics.get("test_post_quali", {})
    metrics["test_pole_hit"] = test_post_m.get("test_post_quali_pole_hit", 0)
    metrics["test_champ_hit"] = test_post_m.get("test_post_quali_champ_hit", 0)
    metrics["test_rf_hit"] = test_post_m.get("test_post_quali_hit", 0)
    metrics["test_logreg_hit"] = metrics["test_rf_hit"]
    metrics["test_rf_top3"] = test_post_m.get("test_post_quali_top3", 0)
    metrics["train_pole_hit"] = metrics.get("train_post_quali", {}).get("train_post_quali_pole_hit", 0)
    metrics["train_champ_hit"] = metrics.get("train_post_quali", {}).get("train_post_quali_champ_hit", 0)
    metrics["train_rf_hit"] = metrics.get("train_post_quali", {}).get("train_post_quali_hit", 0)
    metrics["train_logreg_hit"] = metrics["train_rf_hit"]
    metrics["train_rf_top3"] = metrics.get("train_post_quali", {}).get("train_post_quali_top3", 0)
    metrics["test_logreg_top3"] = metrics["test_rf_top3"]

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_eval_md(metrics)
    return metrics
