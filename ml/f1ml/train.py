from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from f1ml.eval import (
    calibration_bins,
    champ_leader_probabilities,
    equal_field_probabilities,
    finish_mae,
    pole_probabilities,
    prior_form_probabilities,
    grid_probabilities,
    quali_gap_probabilities,
    race_brier,
    race_log_loss,
    rank_correlation,
    slice_disagree_with_pole,
    slice_metrics,
    topk_hit_rate,
    winner_hit_rate,
    winner_hit_rate_probs,
)
from f1ml.paths import MODELS, PROCESSED, REPORTS
from f1ml.ranking import (
    WeekendRanker,
    predict_dnf_probs,
    predict_scores,
    train_dnf_model,
    train_pairwise_ranker,
    market_probs_from_scores,
)
from f1ml.specs import ERA_RESET_YEAR, EVAL_YEAR, WeekendMode


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "driver_race.parquet")
    has_winner = df.groupby(["season", "round"])["won"].transform("max") == 1
    df = df[has_winner & df["dnf"].notna()].copy()
    return df


def _race_keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(df.groupby(["season", "round"]).groups.keys())


def _in_keys(df: pd.DataFrame, keys) -> pd.Series:
    """Boolean mask: row's (season, round) is in keys (vectorised)."""
    codes = df["season"].astype(int) * 100 + df["round"].astype(int)
    wanted = {int(s) * 100 + int(r) for s, r in keys}
    return codes.isin(wanted)


def _walk_forward_folds(df: pd.DataFrame, eval_year: int | None = None) -> list[dict]:
    """Expanding-window folds for all seasons or one requested season."""
    all_keys = _race_keys(df)
    keys = [key for key in all_keys if eval_year is None or key[0] == eval_year]
    folds = []
    for test_key in keys:
        prior = [k for k in all_keys if k < test_key]
        if not prior:
            continue
        train_mask = _in_keys(df, prior)
        test_mask = (df["season"] == test_key[0]) & (df["round"] == test_key[1])
        folds.append(
            {
                "train": df[train_mask].copy(),
                "test": df[test_mask].copy(),
                "test_season": int(test_key[0]),
                "test_round": int(test_key[1]),
                "eval_year": test_key[0],
            }
        )
    return folds


def _walk_forward_2026(df: pd.DataFrame) -> list[dict]:
    return _walk_forward_folds(df, eval_year=EVAL_YEAR)


def _train_mode_bundle(
    train: pd.DataFrame,
    mode: WeekendMode,
    val: pd.DataFrame | None = None,
    eval_year: int = EVAL_YEAR,
) -> WeekendRanker:
    ranker = train_pairwise_ranker(train, mode, eval_year=eval_year)
    dnf_model, dnf_features = train_dnf_model(train, mode, eval_year=eval_year)

    temperature = 1.0
    cal_df = val if val is not None and len(val) else train.tail(0)
    if len(cal_df):
        scores = predict_scores(ranker, cal_df, mode)
        dnf_probs = predict_dnf_probs(dnf_model, cal_df, dnf_features)
        temperature = _fit_plackett_luce_temperature(cal_df, scores, dnf_probs)

    return WeekendRanker(
        mode=mode,
        ranker=ranker,
        dnf_model=dnf_model,
        dnf_features=dnf_features,
        temperature=temperature,
        mc_samples=1200,
    )


def _score_bundle_fast(df: pd.DataFrame, bundle: WeekendRanker) -> pd.DataFrame:
    """Compatibility alias for the production Monte Carlo scoring path."""
    return bundle.predict_frame(df)


def _score_bundle(df: pd.DataFrame, bundle: WeekendRanker) -> pd.DataFrame:
    return bundle.predict_frame(df)


def _eval_mode(df: pd.DataFrame, bundle: WeekendRanker, prefix: str) -> dict:
    scored = _score_bundle(df, bundle)
    probs = scored["win_prob"]
    raw = scored["pace_score"]
    metrics: dict = {
        f"{prefix}_hit": winner_hit_rate(df, probs),
        f"{prefix}_top3": topk_hit_rate(df, probs, k=3),
        f"{prefix}_log_loss": race_log_loss(df, probs),
        f"{prefix}_brier": race_brier(df, probs),
        f"{prefix}_rank_corr": rank_correlation(df, raw),
        f"{prefix}_finish_mae": finish_mae(df, scored["expected_finish"]),
        f"{prefix}_calibration": calibration_bins(df, probs),
    }
    metrics.update({f"{prefix}_{k}": v for k, v in slice_metrics(df, probs, probs).items()})
    disagree = slice_disagree_with_pole(df, probs)
    metrics.update({f"{prefix}_{k}": v for k, v in disagree.items()})
    return metrics


def _eval_baselines(df: pd.DataFrame, prefix: str, include_pole: bool = True) -> dict:
    if "pre_quali" in prefix:
        prior = prior_form_probabilities(df)
        return {
            f"{prefix}_prior_form_hit": winner_hit_rate_probs(df, prior),
            f"{prefix}_prior_form_log_loss": race_log_loss(df, prior),
            f"{prefix}_prior_form_brier": race_brier(df, prior),
            f"{prefix}_equal_log_loss": race_log_loss(df, equal_field_probabilities(df)),
        }
    grid = grid_probabilities(df)
    quali = quali_gap_probabilities(df)
    out = {
        f"{prefix}_grid_hit": winner_hit_rate_probs(df, grid),
        f"{prefix}_grid_log_loss": race_log_loss(df, grid),
        f"{prefix}_grid_brier": race_brier(df, grid),
        f"{prefix}_quali_gap_hit": winner_hit_rate(df, quali),
        f"{prefix}_quali_gap_log_loss": race_log_loss(df, quali),
        f"{prefix}_quali_gap_brier": race_brier(df, quali),
    }
    if include_pole:
        pole_p = pole_probabilities(df)
        out[f"{prefix}_pole_hit"] = winner_hit_rate(df, pole_p)
        out[f"{prefix}_pole_log_loss"] = race_log_loss(df, pole_p)
        out[f"{prefix}_pole_brier"] = race_brier(df, pole_p)
    return out


def _fit_plackett_luce_temperature(
    df: pd.DataFrame,
    scores: np.ndarray,
    dnf_probs: np.ndarray,
    *,
    samples: int = 100,
) -> float:
    """Calibrate the same DNF-adjusted Plackett-Luce win distribution used at inference."""
    grid = np.geomspace(0.3, 10.0, 16)
    score_series = pd.Series(scores, index=df.index)
    dnf_series = pd.Series(dnf_probs, index=df.index)
    best_t, best_ll = 1.0, float("inf")
    for temperature in grid:
        probs = pd.Series(0.0, index=df.index)
        for (season, rnd), race in df.groupby(["season", "round"], sort=False):
            ix = race.index
            markets = market_probs_from_scores(
                score_series.loc[ix].to_numpy(),
                dnf_series.loc[ix].to_numpy(),
                temperature=float(temperature),
                n_samples=samples,
                seed=int(season) * 100 + int(rnd),
            )
            probs.loc[ix] = markets["win_prob"]
        loss = race_log_loss(df, probs)
        if loss < best_ll:
            best_t, best_ll = float(temperature), loss
    return best_t


def _eda(df: pd.DataFrame) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pole_rate = winner_hit_rate_probs(df, pole_probabilities(df))
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


def _oos_summary(wf_results: list[dict], eval_year: int | None = EVAL_YEAR) -> dict:
    """Aggregate chronological walk-forward results, optionally for one season."""
    wf = (
        [r for r in wf_results if r.get("test_season") == eval_year]
        if eval_year is not None
        else wf_results
    )
    if not wf:
        return {}
    summary = {"oos_n_races": len(wf)}
    for key in (
        "pre_hit", "pre_top3", "pre_log_loss", "pre_brier", "pre_rank_corr", "pre_finish_mae",
        "pre_prior_form_log_loss", "post_hit", "post_top3", "post_log_loss", "post_brier",
        "post_rank_corr", "post_finish_mae", "post_grid_log_loss", "post_quali_gap_log_loss",
        "post_pole_log_loss", "post_grid_hit", "post_pole_hit", "post_champ_hit",
        "sprint_hit", "sprint_top3", "sprint_log_loss", "sprint_brier", "sprint_rank_corr",
        "dnf_pr_auc", "dnf_roc_auc", "dnf_log_loss", "dnf_brier",
    ):
        values = [r[key] for r in wf if key in r and pd.notna(r[key])]
        if values:
            summary[f"oos_{key}"] = float(np.mean(values))
    dates = [r["test_date"] for r in wf if r.get("test_date")]
    if dates:
        summary["oos_start_date"] = min(dates)
        summary["oos_end_date"] = max(dates)
    summary["oos_seasons"] = sorted({int(r["test_season"]) for r in wf})
    summary["oos_pre_n_races"] = sum("pre_hit" in r for r in wf)
    summary["oos_post_n_races"] = sum("post_hit" in r for r in wf)
    return summary


def _transfer_eval_2026(df: pd.DataFrame) -> dict:
    """Stress test: train on seasons <= 2025, score 2026 only (not the live protocol)."""
    train = df[df["season"] < ERA_RESET_YEAR].copy()
    test = df[(df["season"] == ERA_RESET_YEAR) & df["quali_position"].notna()].copy()
    if train.empty or test.empty:
        return {"transfer_n_train_races": 0, "transfer_n_test_races": 0}
    train_post = train[train["quali_position"].notna()]
    keys = _race_keys(train)
    val_keys = set(keys[-10:]) if len(keys) >= 10 else set()
    val_mask = _in_keys(train, val_keys)
    val_df = train[val_mask]
    tr_fit = train[~val_mask]
    if tr_fit.empty:
        return {"transfer_n_train_races": len(keys), "transfer_n_test_races": 0}
    val_post = val_df[val_df["quali_position"].notna()]
    tr_post_fit = train_post[~train_post.index.isin(val_post.index)]

    pre_bundle = _train_mode_bundle(tr_fit, "pre_quali", val=val_df if len(val_df) else None)
    post_bundle = _train_mode_bundle(
        tr_post_fit if len(tr_post_fit) > 30 else train_post,
        "post_quali",
        val=val_post if len(val_post) else None,
    )
    pre_scored = _score_bundle_fast(test, pre_bundle)
    post_scored = _score_bundle_fast(test, post_bundle)
    qg_p = quali_gap_probabilities(test)
    disagree = slice_disagree_with_pole(test, post_scored["win_prob"])
    return {
        "transfer_n_train_races": int(train.groupby(["season", "round"]).ngroups),
        "transfer_n_test_races": int(test.groupby(["season", "round"]).ngroups),
        "transfer_pre_hit": winner_hit_rate(test, pre_scored["win_prob"]),
        "transfer_pre_log_loss": race_log_loss(test, pre_scored["win_prob"]),
        "transfer_post_hit": winner_hit_rate(test, post_scored["win_prob"]),
        "transfer_post_log_loss": race_log_loss(test, post_scored["win_prob"]),
        "transfer_quali_gap_log_loss": race_log_loss(test, qg_p),
        "transfer_disagree_n_races": disagree.get("disagree_n_races", 0),
        "transfer_disagree_hit": disagree.get("disagree_hit", 0.0),
    }


def _write_eval_md(metrics: dict) -> None:
    train_label = metrics.get("training_date_range", ["unknown", "unknown"])
    oos = metrics.get("oos", {})
    all_oos = metrics.get("oos_all", {})
    lines = [
        "# Pitstop Oracle — walk-forward evaluation",
        "",
        f"Training data range: **{train_label[0]} to {train_label[1]}**.",
        "Walk-forward folds train and calibrate only on races before the test Grand Prix.",
        "Primary scores use the same DNF-adjusted Plackett–Luce Monte Carlo path as inference.",
        "Pre-quali baseline: shifted car/driver form ratings. Post-quali baselines: official grid and qualifying gaps.",
        "No production model artifacts are written by this evaluation run.",
        "",
        "## Dataset",
        "",
        f"- Rows: {metrics['eda']['n_rows']}",
        f"- Races: {metrics['eda']['n_races']}",
        f"- Pole win rate: **{metrics['eda']['pole_win_rate']:.1%}**",
        "",
        "## 2026 walk-forward (new regulation era)",
        "",
    ]
    wf26 = metrics.get("walk_forward_2026", [])
    if wf26:
        hits = [f.get("post_hit", f.get("pre_hit", 0)) for f in wf26 if "post_hit" in f or "pre_hit" in f]
        ll = [f.get("post_log_loss", np.nan) for f in wf26 if "post_log_loss" in f]
        pole_ll = [f.get("post_pole_log_loss", np.nan) for f in wf26 if "post_pole_log_loss" in f]
        qg_ll = [f.get("post_quali_gap_log_loss", np.nan) for f in wf26 if "post_quali_gap_log_loss" in f]
        lines += [
            f"- Races scored: {len(wf26)}",
            f"- Date range: {oos.get('oos_start_date', 'unknown')} to {oos.get('oos_end_date', 'unknown')}",
            f"- Pre-quali hit / top-3 / log-loss / Brier: **{oos.get('oos_pre_hit', 0):.1%} / {oos.get('oos_pre_top3', 0):.1%} / {oos.get('oos_pre_log_loss', 0):.3f} / {oos.get('oos_pre_brier', 0):.3f}**",
            f"- Prior-form baseline log-loss: **{oos.get('oos_pre_prior_form_log_loss', 0):.3f}**",
            f"- Post-quali hit / top-3 / log-loss / Brier: **{oos.get('oos_post_hit', 0):.1%} / {oos.get('oos_post_top3', 0):.1%} / {oos.get('oos_post_log_loss', 0):.3f} / {oos.get('oos_post_brier', 0):.3f}**",
            f"- Post-quali rank correlation / finish MAE: **{oos.get('oos_post_rank_corr', 0):.3f} / {oos.get('oos_post_finish_mae', 0):.3f}**",
            f"- Grid / quali-gap log-loss: **{oos.get('oos_post_grid_log_loss', 0):.3f} / {oos.get('oos_post_quali_gap_log_loss', 0):.3f}**",
            f"- DNF log-loss / Brier / PR-AUC / ROC-AUC: **{oos.get('oos_dnf_log_loss', 0):.3f} / {oos.get('oos_dnf_brier', 0):.3f} / {oos.get('oos_dnf_pr_auc', 0):.3f} / {oos.get('oos_dnf_roc_auc', 0):.3f}**",
            f"- Post-sprint races / win hit / top-3 / log-loss / Brier: **{sum('sprint_hit' in fold for fold in wf26)} / {oos.get('oos_sprint_hit', float('nan')):.1%} / {oos.get('oos_sprint_top3', float('nan')):.1%} / {oos.get('oos_sprint_log_loss', float('nan')):.3f} / {oos.get('oos_sprint_brier', float('nan')):.3f}**",
            "",
        ]
        for fold in wf26:
            lines.append(
                f"- 2026 R{fold['test_round']:02d}: post hit {fold.get('post_hit', 0):.1%}, "
                f"post LL {fold.get('post_log_loss', 0):.3f}, "
                f"pole LL {fold.get('post_pole_log_loss', 0):.3f}, "
                f"quali-gap LL {fold.get('post_quali_gap_log_loss', 0):.3f}"
            )
    transfer = metrics.get("transfer_2026", {})
    if transfer.get("transfer_n_test_races", 0) > 0:
        lines += [
            "",
            "## Transfer diagnostic (train <2026, test 2026 only)",
            "",
            f"- Train races: {transfer['transfer_n_train_races']}",
            f"- Test races (2026): {transfer['transfer_n_test_races']}",
            f"- Post-quali hit: **{transfer.get('transfer_post_hit', 0):.1%}**",
            f"- Post-quali log-loss: **{transfer.get('transfer_post_log_loss', 0):.3f}**",
            f"- Quali-gap log-loss: **{transfer.get('transfer_quali_gap_log_loss', 0):.3f}**",
            "",
        ]
    disagree = metrics.get("train_post_quali", {})
    if disagree.get("train_post_quali_disagree_n_races", 0) > 0:
        lines += [
            "## When model pick ≠ pole (train set)",
            "",
            f"- Races: {disagree.get('train_post_quali_disagree_n_races', 0)}",
            f"- Model hit: **{disagree.get('train_post_quali_disagree_hit', 0):.1%}**",
            f"- Log-loss: **{disagree.get('train_post_quali_disagree_log_loss', 0):.3f}**",
            "",
        ]
    lines += [
        "",
        "## All-season chronological out-of-sample window",
        "",
        f"- Test races: {all_oos.get('oos_n_races', 0)} ({all_oos.get('oos_start_date', 'unknown')} to {all_oos.get('oos_end_date', 'unknown')})",
        "- Every eligible race is held out once; each fit and calibration set contains earlier races only.",
        f"- Seasons: {', '.join(str(s) for s in all_oos.get('oos_seasons', []))}",
        f"- Pre-quali races / hit / log-loss / prior-form log-loss: {all_oos.get('oos_pre_n_races', 0)} / {all_oos.get('oos_pre_hit', 0):.1%} / {all_oos.get('oos_pre_log_loss', 0):.3f} / {all_oos.get('oos_pre_prior_form_log_loss', 0):.3f}",
        f"- Pre-quali top-3 / Brier / rank correlation / finish MAE: {all_oos.get('oos_pre_top3', 0):.1%} / {all_oos.get('oos_pre_brier', 0):.3f} / {all_oos.get('oos_pre_rank_corr', 0):.3f} / {all_oos.get('oos_pre_finish_mae', 0):.3f}",
        f"- Post-quali races / hit / log-loss / grid log-loss / quali-gap log-loss: {all_oos.get('oos_post_n_races', 0)} / {all_oos.get('oos_post_hit', 0):.1%} / {all_oos.get('oos_post_log_loss', 0):.3f} / {all_oos.get('oos_post_grid_log_loss', 0):.3f} / {all_oos.get('oos_post_quali_gap_log_loss', 0):.3f}",
        f"- Post-quali top-3 / Brier / rank correlation / finish MAE: {all_oos.get('oos_post_top3', 0):.1%} / {all_oos.get('oos_post_brier', 0):.3f} / {all_oos.get('oos_post_rank_corr', 0):.3f} / {all_oos.get('oos_post_finish_mae', 0):.3f}",
        f"- DNF log-loss / Brier / PR-AUC / ROC-AUC: {all_oos.get('oos_dnf_log_loss', 0):.3f} / {all_oos.get('oos_dnf_brier', 0):.3f} / {all_oos.get('oos_dnf_pr_auc', 0):.3f} / {all_oos.get('oos_dnf_roc_auc', 0):.3f}",
        f"- Post-sprint races / win hit / top-3 / log-loss / Brier: {sum('sprint_hit' in fold for fold in metrics.get('walk_forward', []))} / {all_oos.get('oos_sprint_hit', float('nan')):.1%} / {all_oos.get('oos_sprint_top3', float('nan')):.1%} / {all_oos.get('oos_sprint_log_loss', float('nan')):.3f} / {all_oos.get('oos_sprint_brier', float('nan')):.3f}",
        "",
    ]
    for season in all_oos.get("oos_seasons", []):
        season_summary = _oos_summary(metrics.get("walk_forward", []), int(season))
        lines.append(
            f"- {season}: {season_summary.get('oos_n_races', 0)} races; "
            f"pre LL {season_summary.get('oos_pre_log_loss', float('nan')):.3f}; "
            f"post LL {season_summary.get('oos_post_log_loss', float('nan')):.3f}; "
            f"DNF PR-AUC {season_summary.get('oos_dnf_pr_auc', float('nan')):.3f}"
        )
    lines += [
        "",
        "Training-set diagnostics are stored separately under `train_*` keys in `metrics.json`; they are not performance claims.",
        "Win probability temperature is calibrated from preceding races in each fold; DNF probabilities are reported separately from conditional classified finishing distributions.",
        "",
        "Important limitation: historical practice and weather snapshots are unavailable at forecast cutoffs, so current-event FP and weather features are excluded from both model feature views.",
        "",
    ]
    (REPORTS / "EVAL.md").write_text("\n".join(lines))


def train_and_eval() -> dict:
    df = _load_training_frame()
    eda = _eda(df)

    # Train an in-memory diagnostic bundle only. Champion artifacts are not
    # overwritten by an evaluation run; promotion is a separate review step.
    train_post = df[df["quali_position"].notna()].copy()
    pre_bundle = _train_mode_bundle(df, "pre_quali", val=None)
    post_bundle = _train_mode_bundle(train_post, "post_quali", val=None)

    metrics: dict = {
        "eda": eda,
        "train_races": int(df.groupby(["season", "round"]).ngroups),
        "pre_quali_temperature": pre_bundle.temperature,
        "post_quali_temperature": post_bundle.temperature,
        "model_artifacts_written": False,
        "training_date_range": [str(df["date"].min()), str(df["date"].max())],
    }

    # In-sample diagnostics (not the shipping metric).
    pre_m = _eval_mode(df, pre_bundle, "train_pre_quali")
    pre_b = _eval_baselines(df, "train_pre_quali", include_pole=False)
    metrics["train_pre_quali"] = {**pre_m, **pre_b}

    post_m = _eval_mode(train_post, post_bundle, "train_post_quali")
    post_b = _eval_baselines(train_post, "train_post_quali", include_pole=True)
    metrics["train_post_quali"] = {**post_m, **post_b}

    train_sprint = df[pd.to_numeric(df["sprint_position"], errors="coerce") > 0].copy()
    if train_sprint.groupby(["season", "round"]).ngroups >= 12:
        sprint_bundle = _train_mode_bundle(train_sprint, "post_sprint", val=None)
        sprint_m = _eval_mode(train_sprint, sprint_bundle, "train_post_sprint")
        sprint_b = _eval_baselines(train_sprint, "train_post_sprint", include_pole=True)
        metrics["train_post_sprint"] = {**sprint_m, **sprint_b}
        metrics["post_sprint_temperature"] = sprint_bundle.temperature

    # Walk-forward: each race is tested only against earlier races.
    wf_results = []
    first_test_season = int(os.environ.get("F1ML_EVAL_FROM_SEASON", "0"))
    for fold in _walk_forward_folds(df):
        if fold["test_season"] < first_test_season:
            continue
        tr = fold["train"]
        te = fold["test"]
        if te.empty or len(tr) < 20:
            continue
        tr_post = tr[tr["quali_position"].notna()]
        te_post = te[te["quali_position"].notna()]
        # Temperature fit on last 10 races before test, not on test.
        prior_keys = _race_keys(tr)
        val_keys = set(prior_keys[-10:])
        fit_keys = set(prior_keys[:-10])
        if len(fit_keys) < 10:
            continue
        val_mask = _in_keys(tr, val_keys)
        val_df = tr[val_mask]
        tr_fit = tr[_in_keys(tr, fit_keys)]
        if tr_fit.empty:
            continue

        fold_year = int(fold["test_season"]) if os.environ.get("F1ML_FOLD_WEIGHTS", "1") != "0" else EVAL_YEAR
        pre_f = _train_mode_bundle(
            tr_fit, "pre_quali", val=val_df if len(val_df) else None, eval_year=fold_year
        )
        pre_scored = _score_bundle(te, pre_f)
        dnf_probs = predict_dnf_probs(pre_f.dnf_model, te, pre_f.dnf_features)
        dnf_y = te["dnf"].astype(int).to_numpy()
        pre_prob = pre_scored["win_prob"]
        prior_p = prior_form_probabilities(te)
        test_date = str(pd.to_datetime(te["date"]).min().date()) if "date" in te else None
        wf_row = {
            "test_season": fold["test_season"],
            "test_round": fold["test_round"],
            "test_date": test_date,
            "pre_hit": winner_hit_rate(te, pre_prob),
            "pre_top3": topk_hit_rate(te, pre_prob, k=3),
            "pre_log_loss": race_log_loss(te, pre_prob),
            "pre_brier": race_brier(te, pre_prob),
            "pre_rank_corr": rank_correlation(te, pre_scored["pace_score"]),
            "pre_finish_mae": finish_mae(te, pre_scored["expected_finish"]),
            "pre_prior_form_hit": winner_hit_rate_probs(te, prior_p),
            "pre_prior_form_log_loss": race_log_loss(te, prior_p),
            "pre_prior_form_brier": race_brier(te, prior_p),
            "dnf_log_loss": float(log_loss(dnf_y, dnf_probs, labels=[0, 1])),
            "dnf_brier": float(brier_score_loss(dnf_y, dnf_probs)),
        }
        if len(np.unique(dnf_y)) == 2:
            wf_row["dnf_pr_auc"] = float(average_precision_score(dnf_y, dnf_probs))
            wf_row["dnf_roc_auc"] = float(roc_auc_score(dnf_y, dnf_probs))
        if len(tr_post) and len(te_post):
            val_post = val_df[val_df["quali_position"].notna()]
            tr_post_fit = tr_post[~tr_post.index.isin(val_post.index)]
            post_f = _train_mode_bundle(
                tr_post_fit,
                "post_quali",
                val=val_post if len(val_post) else None,
                eval_year=fold_year,
            )
            post_scored = _score_bundle(te_post, post_f)
            pole_p = pole_probabilities(te_post)
            qg_p = quali_gap_probabilities(te_post)
            grid_p = grid_probabilities(te_post)
            champ_p = champ_leader_probabilities(te_post)
            post_prob = post_scored["win_prob"]
            wf_row["post_hit"] = winner_hit_rate(te_post, post_scored["win_prob"])
            wf_row["post_top3"] = topk_hit_rate(te_post, post_prob, k=3)
            wf_row["post_log_loss"] = race_log_loss(te_post, post_prob)
            wf_row["post_brier"] = race_brier(te_post, post_prob)
            wf_row["post_rank_corr"] = rank_correlation(te_post, post_scored["pace_score"])
            wf_row["post_finish_mae"] = finish_mae(te_post, post_scored["expected_finish"])
            wf_row["post_pole_log_loss"] = race_log_loss(te_post, pole_p)
            wf_row["post_quali_gap_log_loss"] = race_log_loss(te_post, qg_p)
            wf_row["post_grid_log_loss"] = race_log_loss(te_post, grid_p)
            wf_row["post_grid_hit"] = winner_hit_rate_probs(te_post, grid_p)
            wf_row["post_pole_hit"] = winner_hit_rate_probs(te_post, pole_p)
            wf_row["post_champ_hit"] = winner_hit_rate_probs(te_post, champ_p)

        tr_sprint = tr[pd.to_numeric(tr["sprint_position"], errors="coerce") > 0].copy()
        te_sprint = te[pd.to_numeric(te["sprint_position"], errors="coerce") > 0].copy()
        sprint_keys = _race_keys(tr_sprint) if not tr_sprint.empty else []
        if len(sprint_keys) >= 15 and not te_sprint.empty:
            val_sprint_keys = set(sprint_keys[-5:])
            fit_sprint_keys = set(sprint_keys[:-5])
            val_sprint = tr_sprint[
                _in_keys(tr_sprint, val_sprint_keys)
            ]
            fit_sprint = tr_sprint[
                _in_keys(tr_sprint, fit_sprint_keys)
            ]
            sprint_f = _train_mode_bundle(
                fit_sprint, "post_sprint", val=val_sprint, eval_year=fold_year
            )
            sprint_scored = _score_bundle(te_sprint, sprint_f)
            wf_row["sprint_hit"] = winner_hit_rate(te_sprint, sprint_scored["win_prob"])
            wf_row["sprint_top3"] = topk_hit_rate(te_sprint, sprint_scored["win_prob"], k=3)
            wf_row["sprint_log_loss"] = race_log_loss(te_sprint, sprint_scored["win_prob"])
            wf_row["sprint_brier"] = race_brier(te_sprint, sprint_scored["win_prob"])
            wf_row["sprint_rank_corr"] = rank_correlation(te_sprint, sprint_scored["pace_score"])
        wf_results.append(wf_row)
    metrics["walk_forward"] = wf_results
    metrics["walk_forward_2026"] = [r for r in wf_results if r["test_season"] == EVAL_YEAR]
    metrics["oos"] = _oos_summary(wf_results)
    metrics["oos_all"] = _oos_summary(wf_results, eval_year=None)
    metrics["transfer_2026"] = _transfer_eval_2026(df)
    oos = metrics["oos"]
    metrics["test_pre_quali"] = {
        "test_pre_quali_hit": oos.get("oos_pre_hit", 0),
        "test_pre_quali_log_loss": oos.get("oos_pre_log_loss", 0),
    }
    metrics["test_post_quali"] = {
        "test_post_quali_hit": oos.get("oos_post_hit", 0),
        "test_post_quali_log_loss": oos.get("oos_post_log_loss", 0),
        "test_post_quali_quali_gap_log_loss": oos.get("oos_post_quali_gap_log_loss", 0),
    }

    # Compatibility aliases are out-of-sample only. Training diagnostics use
    # the explicit train_* keys above.
    wf26_post = [r for r in wf_results if r.get("test_season") == EVAL_YEAR and "post_hit" in r]
    metrics["test_pole_hit"] = float(np.mean([r["post_pole_hit"] for r in wf26_post])) if wf26_post else 0.0
    metrics["test_champ_hit"] = float(np.mean([r.get("post_champ_hit", 0.0) for r in wf26_post])) if wf26_post else 0.0
    metrics["test_rf_hit"] = float(np.mean([r["post_hit"] for r in wf26_post])) if wf26_post else 0
    metrics["test_rf_top3"] = float(np.mean([r["post_top3"] for r in wf26_post])) if wf26_post else 0
    metrics["train_pole_hit"] = metrics["train_post_quali"].get("train_post_quali_pole_hit", 0)
    metrics["train_champ_hit"] = metrics["train_post_quali"].get("train_post_quali_champ_hit", 0)
    metrics["train_rf_hit"] = metrics["train_post_quali"].get("train_post_quali_hit", 0)
    metrics["train_rf_top3"] = metrics["train_post_quali"].get("train_post_quali_top3", 0)
    metrics["walk_forward_protocol"] = "One race per test fold; train/calibration data end before test race; production Monte Carlo Plackett-Luce scorer."
    metrics["walk_forward_mc_samples"] = pre_bundle.mc_samples

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_eval_md(metrics)
    return metrics


def recalibrate_temperatures() -> dict:
    """Refit temperature on trailing validation races without retraining rankers."""
    df = _load_training_frame()
    MODELS.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for mode in ("pre_quali", "post_quali", "post_sprint"):
        if not (MODELS / mode / "meta.joblib").exists():
            continue
        bundle = WeekendRanker.load(MODELS / mode)
        if mode == "pre_quali":
            sub = df
        elif mode == "post_quali":
            sub = df[df["quali_position"].notna()].copy()
        else:
            sub = df[pd.to_numeric(df["sprint_position"], errors="coerce") > 0].copy()
        keys = _race_keys(sub)
        if len(keys) < 12:
            continue
        val_keys = set(keys[-12:])
        val_df = sub[_in_keys(sub, val_keys)]
        scores = predict_scores(bundle.ranker, val_df, mode)
        dnf_probs = predict_dnf_probs(bundle.dnf_model, val_df, bundle.dnf_features)
        bundle.temperature = _fit_plackett_luce_temperature(val_df, scores, dnf_probs)
        bundle.save(MODELS / mode)
        results[f"{mode}_temperature"] = bundle.temperature
    return results
