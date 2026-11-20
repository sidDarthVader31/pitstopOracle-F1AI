#!/usr/bin/env python3
"""EXP-007: how useful are the model's full finishing distributions for F1 Fantasy?

    python scripts/wf_dump.py baseline       # once: cache walk-forward outputs
    python scripts/exp_fantasy.py baseline

Scores, on walk-forward races only, pre-quali (when Fantasy teams lock) and post-quali:
- points finish (top 10) and podium (top 3) probabilities: log-loss and Brier;
- expected finishing position: MAE on classified finishers, and rank correlation;
- expected Fantasy points (race part): the current flat formula in ``fantasy.py``
  vs one that uses the simulated position distribution, plus "pick 5" points.

Baselines are fitted only on races before each test race:
- ``rank_rate``: P(event | baseline rank) estimated from earlier races, with the rank
  taken from the prior-form rating (pre-quali) or the official grid (post-quali);
- ``base_rate``: the same probability for everyone (10/n for points, 3/n for podium).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1ml import fantasy
from f1ml.paths import CACHE, PROCESSED, REPORTS

EXP = CACHE / "experiments"
EPS = 1e-6


def _binary_ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _boot(d: np.ndarray, n: int = 5000, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def _rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="first", ascending=ascending, na_option="bottom")


def baseline_ranks(df: pd.DataFrame, stage: str) -> pd.Series:
    """Rank 1 = best, per race, from information available at the stage cutoff."""
    key = "grid" if stage == "post_quali" else "car_driver_combo"
    return df.groupby(["season", "round"], group_keys=False)[key].apply(
        lambda s: _rank(s.where(s > 0) if key == "grid" else s))


def rank_rate_probs(history: pd.DataFrame, test: pd.DataFrame, stage: str, target: str) -> np.ndarray:
    """P(target | baseline rank) from earlier races, smoothed toward the overall rate."""
    hr = baseline_ranks(history, stage).clip(upper=22)
    table = history.assign(r=hr).groupby("r")[target].agg(["sum", "count"])
    base = history[target].mean()
    k = 5.0
    rate = (table["sum"] + k * base) / (table["count"] + k)
    tr = baseline_ranks(test, stage).clip(upper=22)
    return tr.map(rate).fillna(base).to_numpy(float)


def rank_mean_finish(history: pd.DataFrame, test: pd.DataFrame, stage: str) -> np.ndarray:
    """Average classified finishing position for each baseline rank, from earlier races."""
    fin = history[(history["dnf"] == 0) & history["position"].notna()]
    hr = baseline_ranks(fin, stage).clip(upper=22)
    table = fin.assign(r=hr).groupby("r")["position"].mean()
    tr = baseline_ranks(test, stage).clip(upper=22)
    return tr.map(table).fillna(fin["position"].mean()).to_numpy(float)


def actual_race_points(df: pd.DataFrame, with_quali: bool) -> np.ndarray:
    out = []
    for r in df.itertuples(index=False):
        dnf = bool(r.dnf == 1) or pd.isna(r.position)
        quali = int(r.quali_position) if with_quali and pd.notna(r.quali_position) else None
        grid = int(r.grid) if with_quali and pd.notna(r.grid) and r.grid > 0 else quali
        finish = int(r.position) if pd.notna(r.position) else 0
        out.append(fantasy.driver_fantasy_points_row(finish, quali, grid, dnf))
    return np.array(out, dtype=float)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preds = pd.read_parquet(EXP / f"{tag}.parquet")
    hist_all = pd.read_parquet(PROCESSED / "driver_race.parquet")
    hist_all = hist_all[hist_all["dnf"].notna() & (hist_all["is_upcoming"].fillna(0) != 1)]
    extra = hist_all[["season", "round", "driver_id", "car_driver_combo"]]
    report: dict = {"tag": tag, "stages": {}}
    for stage in ("pre_quali", "post_quali"):
        df = preds[preds["stage"] == stage].merge(extra, on=["season", "round", "driver_id"], how="left")
        # Disqualified / non-starters have no DNF label; they score no points.
        df["points_finish"] = df["points_finish"].fillna(0).astype(int)
        df["podium"] = df["podium"].fillna(0).astype(int)
        rows = []
        for (s, rnd), g in df.groupby(["season", "round"], sort=True):
            history = hist_all[(hist_all["season"] * 100 + hist_all["round"]) < s * 100 + rnd]
            if stage == "post_quali":
                history = history[history["grid"].notna()]
            g = g.copy()
            n = len(g)
            g["rr_points"] = rank_rate_probs(history, g, stage, "points_finish")
            g["rr_podium"] = rank_rate_probs(history, g, stage, "podium")
            g["base_points"] = min(10, n) / n
            g["base_podium"] = 3 / n
            g["b_rank"] = baseline_ranks(g, stage).to_numpy()
            g["b_mean_finish"] = rank_mean_finish(history, g, stage)
            # Expected Fantasy race points: current formula vs the simulated distribution.
            g["xp_current"] = fantasy.expected_driver_fantasy_points(g.assign(has_sprint=0), use_distribution=False).to_numpy()
            g["xp_dist"] = fantasy.expected_driver_fantasy_points(g.assign(has_sprint=0), use_distribution=True).to_numpy()
            with_q = stage == "post_quali"
            g["actual_pts"] = actual_race_points(g, with_quali=with_q)
            for event, y in (("points", g["points_finish"].to_numpy()), ("podium", g["podium"].to_numpy())):
                for name, col in (("model", f"{event}_prob" if event == "points" else "podium_prob"),
                                  ("rank_rate", f"rr_{event}"), ("base_rate", f"base_{event}")):
                    p = g[col].to_numpy(float)
                    rows.append({"season": s, "round": rnd, "metric": f"{event}_ll", "who": name, "v": _binary_ll(p, y)})
                    rows.append({"season": s, "round": rnd, "metric": f"{event}_brier", "who": name,
                                 "v": float(np.mean((p - y) ** 2))})
            fin = g[(g["dnf"] == 0) & g["position"].notna()]
            for name, pred in (("model", fin["expected_finish"]), ("baseline_rank", fin["b_rank"]),
                               ("rank_mean", fin["b_mean_finish"])):
                rows.append({"season": s, "round": rnd, "metric": "finish_mae", "who": name,
                             "v": float((pred - fin["position"]).abs().mean())})
            pos = g["position"].astype(float)
            for name, sc in (("model", -g["pace_score"]), ("baseline_rank", g["b_rank"])):
                rows.append({"season": s, "round": rnd, "metric": "rank_corr", "who": name,
                             "v": float(spearmanr(sc, pos).statistic)})
            for name, col in (("xp_current", "xp_current"), ("xp_dist", "xp_dist")):
                rows.append({"season": s, "round": rnd, "metric": "xp_mae", "who": name,
                             "v": float((g[col] - g["actual_pts"]).abs().mean())})
                rows.append({"season": s, "round": rnd, "metric": "xp_rank_corr", "who": name,
                             "v": float(spearmanr(g[col], g["actual_pts"]).statistic)})
                rows.append({"season": s, "round": rnd, "metric": "pick5_pts", "who": name,
                             "v": float(g.nlargest(5, col)["actual_pts"].sum())})
            rows.append({"season": s, "round": rnd, "metric": "pick5_pts", "who": "baseline_rank",
                         "v": float(g.nsmallest(5, "b_rank")["actual_pts"].sum())})
            rows.append({"season": s, "round": rnd, "metric": "pick5_pts", "who": "hindsight_best",
                         "v": float(g.nlargest(5, "actual_pts")["actual_pts"].sum())})
            rows.append({"season": s, "round": rnd, "metric": "xp_rank_corr", "who": "baseline_rank",
                         "v": float(spearmanr(-g["b_rank"], g["actual_pts"]).statistic)})
        long = pd.DataFrame(rows)
        stage_out: dict = {}
        for window, mask in (("all", long["season"] > 0), ("2025-26", long["season"] >= 2025),
                             ("2026", long["season"] == 2026)):
            sub = long[mask]
            table = sub.groupby(["metric", "who"])["v"].mean().unstack()
            stage_out[window] = {"races": int(sub[["season", "round"]].drop_duplicates().shape[0]),
                                 "table": table.round(4).to_dict(orient="index")}
            # paired CIs: model vs the best simple baseline
            cis = {}
            for metric, a, b in (("points_ll", "model", "rank_rate"), ("podium_ll", "model", "rank_rate"),
                                 ("points_brier", "model", "rank_rate"), ("finish_mae", "model", "rank_mean"),
                                 ("rank_corr", "model", "baseline_rank"), ("pick5_pts", "xp_dist", "baseline_rank"), ("pick5_pts", "xp_dist", "xp_current"),
                                 ("xp_rank_corr", "xp_dist", "xp_current"), ("xp_mae", "xp_dist", "xp_current")):
                piv = sub[sub["metric"] == metric].pivot_table(index=["season", "round"], columns="who", values="v")
                if a in piv and b in piv:
                    d = (piv[a] - piv[b]).dropna().to_numpy()
                    cis[f"{metric}: {a} - {b}"] = _boot(d)
            stage_out[window]["paired_ci"] = cis
        report["stages"][stage] = stage_out
        print(f"\n===== {stage} =====")
        for window, v in stage_out.items():
            print(f"-- {window} ({v['races']} races)")
            print(pd.DataFrame(v["table"]).T.to_string())
            for k, ci in v["paired_ci"].items():
                print(f"   {k}: {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"exp007_fantasy_{tag}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
