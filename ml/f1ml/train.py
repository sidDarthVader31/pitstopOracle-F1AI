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
    race_level_win_probs,
    train_dnf_model,
    train_pairwise_ranker,
)
from f1ml.specs import ERA_RESET_YEAR, EVAL_YEAR, TRAIN_SEASONS, WeekendMode, season_range_label


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "driver_race.parquet")
    has_winner = df.groupby(["season", "round"])["won"].transform("max") == 1
    df = df[has_winner & df["position"].notna()].copy()
    return df


def _race_keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(df.groupby(["season", "round"]).groups.keys())


def _walk_forward_folds(df: pd.DataFrame, eval_year: int = EVAL_YEAR) -> list[dict]:
    """Expanding-window folds at season boundaries + each eval_year race."""
    keys = _race_keys(df)
    fold_keys: list[tuple[int, int]] = []
    seen_seasons: set[int] = set()
    for key in keys:
        if key[0] not in seen_seasons:
            fold_keys.append(key)
            seen_seasons.add(key[0])
        if key[0] == eval_year:
            fold_keys.append(key)
    # Deduplicate while preserving order.
    ordered: list[tuple[int, int]] = []
    for k in fold_keys:
        if k not in ordered:
            ordered.append(k)

    folds = []
    for test_key in ordered:
        prior = [k for k in keys if k < test_key]
        if not prior:
            continue
        train_keys = set(prior)
        train_mask = df.apply(lambda r: (r["season"], r["round"]) in train_keys, axis=1)
        test_mask = (df["season"] == test_key[0]) & (df["round"] == test_key[1])
        folds.append(
            {
                "train": df[train_mask].copy(),
                "test": df[test_mask].copy(),
                "test_season": int(test_key[0]),
                "test_round": int(test_key[1]),
            }
        )
    return folds


def _walk_forward_2026(df: pd.DataFrame) -> list[dict]:
    keys = _race_keys(df)
    folds = []
    for test_key in keys:
        if test_key[0] != EVAL_YEAR:
            continue
        prior = [k for k in keys if k < test_key]
        if not prior:
            continue
        train_keys = set(prior)
        train_mask = df.apply(lambda r: (r["season"], r["round"]) in train_keys, axis=1)
        test_mask = (df["season"] == test_key[0]) & (df["round"] == test_key[1])
        folds.append(
            {
                "train": df[train_mask].copy(),
                "test": df[test_mask].copy(),
                "test_season": int(test_key[0]),
                "test_round": int(test_key[1]),
            }
        )
    return folds


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
        raw = pd.Series(scores, index=cal_df.index)
        temperature = fit_temperature(
            cal_df, raw, from_binary_probs=False
        )

    return WeekendRanker(
        mode=mode,
        ranker=ranker,
        dnf_model=dnf_model,
        dnf_features=dnf_features,
        temperature=temperature,
        mc_samples=4000,
    )


def _score_bundle_fast(df: pd.DataFrame, bundle: WeekendRanker) -> pd.DataFrame:
    """Fast scoring for walk-forward eval (softmax win probs, analytic podium approx)."""
    out = df.copy()
    scores = predict_scores(bundle.ranker, df, bundle.mode)
    dnf_p = predict_dnf_probs(bundle.dnf_model, df, bundle.dnf_features)
    out["pace_score"] = scores
    out["dnf_prob"] = dnf_p
    raw = pd.Series(scores, index=df.index)
    dnf_s = pd.Series(dnf_p, index=df.index)
    out["win_prob"] = race_level_win_probs(df, raw, bundle.temperature, dnf_s)
    # Podium approx: top-3 softmax mass + boost from score rank.
    out["podium_prob"] = out["win_prob"] * 2.8 + (1.0 - out["dnf_prob"]) * 0.08
    out["podium_prob"] = out["podium_prob"].clip(0, 1)
    out["expected_finish"] = -raw  # higher score => lower finish number
    out["expected_finish"] = out.groupby(["season", "round"])["expected_finish"].transform(
        lambda s: (-s).rank(method="average")
    )
    return out


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
    out = {
        f"{prefix}_equal_hit": winner_hit_rate_probs(df, equal_field_probabilities(df)),
        f"{prefix}_champ_hit": winner_hit_rate(df, champ_leader_probabilities(df)),
        f"{prefix}_equal_log_loss": race_log_loss(df, equal_field_probabilities(df)),
        f"{prefix}_champ_log_loss": race_log_loss(df, champ_leader_probabilities(df)),
        f"{prefix}_quali_gap_hit": winner_hit_rate(df, quali_gap_probabilities(df)),
        f"{prefix}_quali_gap_log_loss": race_log_loss(df, quali_gap_probabilities(df)),
        f"{prefix}_quali_gap_brier": race_brier(df, quali_gap_probabilities(df)),
    }
    if include_pole:
        pole_p = pole_probabilities(df)
        out[f"{prefix}_pole_hit"] = winner_hit_rate(df, pole_p)
        out[f"{prefix}_pole_log_loss"] = race_log_loss(df, pole_p)
        out[f"{prefix}_pole_brier"] = race_brier(df, pole_p)
    return out


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


def _oos_summary(wf_results: list[dict], eval_year: int = EVAL_YEAR) -> dict:
    """Aggregate walk-forward 2026 (or eval_year) into out-of-sample summary keys."""
    wf = [r for r in wf_results if r.get("test_season") == eval_year]
    if not wf:
        return {}
    post_hits = [r["post_hit"] for r in wf if "post_hit" in r]
    post_ll = [r["post_log_loss"] for r in wf if "post_log_loss" in r]
    qg_ll = [r["post_quali_gap_log_loss"] for r in wf if "post_quali_gap_log_loss" in r]
    pre_hits = [r["pre_hit"] for r in wf if "pre_hit" in r]
    pre_ll = [r["pre_log_loss"] for r in wf if "pre_log_loss" in r]
    return {
        "oos_n_races": len(wf),
        "oos_post_quali_hit": float(np.mean(post_hits)) if post_hits else 0.0,
        "oos_post_quali_log_loss": float(np.mean(post_ll)) if post_ll else 0.0,
        "oos_post_quali_quali_gap_log_loss": float(np.mean(qg_ll)) if qg_ll else 0.0,
        "oos_post_quali_n_races": len(wf),
        "oos_pre_quali_hit": float(np.mean(pre_hits)) if pre_hits else 0.0,
        "oos_pre_quali_log_loss": float(np.mean(pre_ll)) if pre_ll else 0.0,
    }


def _transfer_eval_2026(df: pd.DataFrame) -> dict:
    """Stress test: train on seasons <= 2025, score 2026 only (not the live protocol)."""
    train = df[df["season"] < ERA_RESET_YEAR].copy()
    test = df[(df["season"] == ERA_RESET_YEAR) & df["quali_position"].notna()].copy()
    if train.empty or test.empty:
        return {"transfer_n_train_races": 0, "transfer_n_test_races": 0}
    train_post = train[train["quali_position"].notna()]
    keys = _race_keys(train)
    val_keys = set(keys[-10:]) if len(keys) >= 10 else set()
    val_mask = train.apply(lambda r: (r["season"], r["round"]) in val_keys, axis=1)
    val_df = train[val_mask]
    tr_fit = train[~val_mask] if len(train) > len(val_df) + 30 else train
    val_post = val_df[val_df["quali_position"].notna()]
    tr_post_fit = train_post[~train_post.index.isin(val_post.index)] if len(val_post) else train_post

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


def _remove_legacy_model_artifacts() -> None:
    """Drop unused v2 classifier heads; v3 uses ranker.joblib + dnf_model.joblib only."""
    legacy_names = (
        "win_hgb.joblib",
        "podium_hgb.joblib",
        "finish_hgb.joblib",
        "dnf_hgb.joblib",
        "win_logreg.joblib",
    )
    for mode in ("pre_quali", "post_quali"):
        mode_dir = MODELS / mode
        if not mode_dir.exists():
            continue
        for name in legacy_names:
            path = mode_dir / name
            if path.exists():
                path.unlink()
    for name in ("winner_rf.joblib", "winner_logreg.joblib"):
        path = MODELS / name
        if path.exists():
            path.unlink()


def _write_eval_md(metrics: dict) -> None:
    train_label = season_range_label(TRAIN_SEASONS)
    lines = [
        "# F1 Oracle — evaluation (v3 ranker)",
        "",
        f"Training pool: **{train_label}** with era-reset constructor ratings and race-by-race walk-forward.",
        "Primary shipping metric: **2026 walk-forward** log-loss vs pole and quali-gap softmax.",
        "",
        "## Dataset",
        "",
        f"- Rows: {metrics['eda']['n_rows']}",
        f"- Races: {metrics['eda']['n_races']}",
        f"- Pole win rate: **{metrics['eda']['pole_win_rate']:.1%}**",
        "",
        "## 2026 walk-forward (primary)",
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
            f"- Post-quali hit rate (mean): **{np.nanmean(hits):.1%}**" if hits else "",
            f"- Post-quali log-loss (mean): **{np.nanmean(ll):.3f}**" if ll else "",
            f"- Pole log-loss (mean): **{np.nanmean(pole_ll):.3f}**" if pole_ll else "",
            f"- Quali-gap log-loss (mean): **{np.nanmean(qg_ll):.3f}**" if qg_ll else "",
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
        "## Full walk-forward (by season)",
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

    # Final production models: train on all completed races.
    train_post = df[df["quali_position"].notna()].copy()
    pre_bundle = _train_mode_bundle(df, "pre_quali", val=None)
    post_bundle = _train_mode_bundle(train_post, "post_quali", val=None)

    MODELS.mkdir(parents=True, exist_ok=True)
    pre_bundle.save(MODELS / "pre_quali")
    post_bundle.save(MODELS / "post_quali")
    _remove_legacy_model_artifacts()

    from f1ml.specs import feature_columns

    numeric_pre, cat_pre = feature_columns("pre_quali")
    joblib.dump({"numeric": numeric_pre, "categorical": cat_pre}, MODELS / "features.joblib")

    metrics: dict = {
        "eda": eda,
        "train_races": int(df.groupby(["season", "round"]).ngroups),
        "pre_quali_temperature": pre_bundle.temperature,
        "post_quali_temperature": post_bundle.temperature,
    }

    # In-sample diagnostics (not the shipping metric).
    pre_m = _eval_mode(df, pre_bundle, "train_pre_quali")
    pre_b = _eval_baselines(df, "train_pre_quali", include_pole=False)
    metrics["train_pre_quali"] = {**pre_m, **pre_b}

    post_m = _eval_mode(train_post, post_bundle, "train_post_quali")
    post_b = _eval_baselines(train_post, "train_post_quali", include_pole=True)
    metrics["train_post_quali"] = {**post_m, **post_b}

    # Walk-forward: all seasons.
    wf_results = []
    for fold in _walk_forward_folds(df):
        tr = fold["train"]
        te = fold["test"]
        if te.empty or len(tr) < 50:
            continue
        tr_post = tr[tr["quali_position"].notna()]
        te_post = te[te["quali_position"].notna()]
        # Temperature fit on last 10 races before test, not on test.
        val_keys = _race_keys(tr)[-10:]
        val_mask = tr.apply(lambda r: (r["season"], r["round"]) in set(val_keys), axis=1)
        val_df = tr[val_mask]
        tr_fit = tr[~val_mask] if len(tr) > len(val_df) + 30 else tr

        pre_f = _train_mode_bundle(tr_fit, "pre_quali", val=val_df if len(val_df) else None)
        pre_scored = _score_bundle_fast(te, pre_f)
        wf_row = {
            "test_season": fold["test_season"],
            "test_round": fold["test_round"],
            "pre_hit": winner_hit_rate(te, pre_scored["win_prob"]),
            "pre_log_loss": race_log_loss(te, pre_scored["win_prob"]),
        }
        if len(tr_post) and len(te_post):
            val_post = val_df[val_df["quali_position"].notna()]
            tr_post_fit = tr_post[~tr_post.index.isin(val_post.index)] if len(val_post) else tr_post
            post_f = _train_mode_bundle(
                tr_post_fit if len(tr_post_fit) > 30 else tr_post,
                "post_quali",
                val=val_post if len(val_post) else None,
            )
            post_scored = _score_bundle_fast(te_post, post_f)
            pole_p = pole_probabilities(te_post)
            qg_p = quali_gap_probabilities(te_post)
            wf_row["post_hit"] = winner_hit_rate(te_post, post_scored["win_prob"])
            wf_row["post_log_loss"] = race_log_loss(te_post, post_scored["win_prob"])
            wf_row["post_pole_log_loss"] = race_log_loss(te_post, pole_p)
            wf_row["post_quali_gap_log_loss"] = race_log_loss(te_post, qg_p)
        wf_results.append(wf_row)
    metrics["walk_forward"] = wf_results
    metrics["walk_forward_2026"] = [r for r in wf_results if r["test_season"] == EVAL_YEAR]
    metrics["oos"] = _oos_summary(wf_results)
    metrics["transfer_2026"] = _transfer_eval_2026(df)
    oos = metrics["oos"]
    metrics["test_pre_quali"] = {
        "test_pre_quali_hit": oos.get("oos_pre_quali_hit", 0),
        "test_pre_quali_log_loss": oos.get("oos_pre_quali_log_loss", 0),
    }
    metrics["test_post_quali"] = {
        "test_post_quali_hit": oos.get("oos_post_quali_hit", 0),
        "test_post_quali_log_loss": oos.get("oos_post_quali_log_loss", 0),
        "test_post_quali_quali_gap_log_loss": oos.get("oos_post_quali_quali_gap_log_loss", 0),
    }

    # Back-compat keys for UI.
    wf26_post = [r for r in wf_results if r.get("test_season") == EVAL_YEAR and "post_hit" in r]
    metrics["test_pole_hit"] = metrics["train_post_quali"].get("train_post_quali_pole_hit", 0)
    metrics["test_champ_hit"] = metrics["train_post_quali"].get("train_post_quali_champ_hit", 0)
    metrics["test_rf_hit"] = float(np.mean([r["post_hit"] for r in wf26_post])) if wf26_post else 0
    metrics["test_logreg_hit"] = metrics["test_rf_hit"]
    metrics["test_rf_top3"] = metrics["train_post_quali"].get("train_post_quali_top3", 0)
    metrics["train_pole_hit"] = metrics["train_post_quali"].get("train_post_quali_pole_hit", 0)
    metrics["train_champ_hit"] = metrics["train_post_quali"].get("train_post_quali_champ_hit", 0)
    metrics["train_rf_hit"] = metrics["train_post_quali"].get("train_post_quali_hit", 0)
    metrics["train_logreg_hit"] = metrics["train_rf_hit"]
    metrics["train_rf_top3"] = metrics["train_post_quali"].get("train_post_quali_top3", 0)
    metrics["test_logreg_top3"] = metrics["test_rf_top3"]

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_eval_md(metrics)
    return metrics


def recalibrate_temperatures() -> dict:
    """Refit temperature on trailing validation races without retraining rankers."""
    df = _load_training_frame()
    MODELS.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for mode in ("pre_quali", "post_quali"):
        bundle = WeekendRanker.load(MODELS / mode)
        sub = df if mode == "pre_quali" else df[df["quali_position"].notna()].copy()
        keys = _race_keys(sub)
        if len(keys) < 12:
            continue
        val_keys = set(keys[-12:])
        val_df = sub[sub.apply(lambda r: (r["season"], r["round"]) in val_keys, axis=1)]
        scores = predict_scores(bundle.ranker, val_df, mode)
        bundle.temperature = fit_temperature(
            val_df, pd.Series(scores, index=val_df.index), from_binary_probs=False
        )
        bundle.save(MODELS / mode)
        results[f"{mode}_temperature"] = bundle.temperature
    return results
