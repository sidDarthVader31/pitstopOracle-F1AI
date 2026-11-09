#!/usr/bin/env python3
"""EXP-005: calibrate the model's win probabilities, especially the favourites.

    python scripts/wf_dump.py baseline          # once: cache walk-forward outputs
    python scripts/exp_calibration.py baseline  # re-score calibration methods offline

Every calibrator for test race k is fitted only on the walk-forward (out-of-sample)
predictions of races *before* k, so nothing about race k or later is used. The
methods re-map the same pace scores and DNF probabilities that the walk-forward
model produced; the pace model itself is unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from f1ml import calibration as cal
from f1ml.eval import grid_probabilities, prior_form_probabilities
from f1ml.paths import CACHE, PROCESSED, REPORTS

EXP = CACHE / "experiments"
BUCKETS = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0001]
T_GRID = cal.T_GRID
MIX_GRID = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3])
MIN_HISTORY = 10
EPS = 1e-15


def _load_market_module():
    spec = importlib.util.spec_from_file_location("market_benchmark", ROOT / "scripts" / "market_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- core maths
def pl_win_probs(scores, dnf, temperature, retired):  # noqa: ARG001 - dnf kept for call sites
    return cal.pl_win_probs(scores, retired, temperature)


def race_ll(p: np.ndarray, won: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    return float(-np.log(p[won == 1][0]))


def race_brier(p: np.ndarray, won: np.ndarray) -> float:
    return float(np.mean((p - (won == 1)) ** 2))


# ----------------------------------------------------------------------------- calibrators
class Races:
    """Per-race arrays for one stage, in chronological order."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
        self.df = df
        self.keys = list(df.groupby(["season", "round"], sort=True).groups.keys())
        self.idx = [df.index[(df["season"] == s) & (df["round"] == r)].to_numpy() for s, r in self.keys]
        self.won = [df.loc[i, "won"].to_numpy(int) for i in self.idx]
        self.scores = [df.loc[i, "pace_score"].to_numpy(float) for i in self.idx]
        self.dnf = [df.loc[i, "dnf_prob"].to_numpy(float) for i in self.idx]
        self.base = [df.loc[i, "win_prob"].to_numpy(float) for i in self.idx]
        if "grid" in df and df["grid"].notna().any() and (df["stage"].iloc[0] != "pre_quali"):
            gp = grid_probabilities(df).to_numpy()
        else:  # pre-quali: no grid yet, blend toward an equal field
            gp = pd.Series(np.nan, index=df.index).to_numpy()
            gp[np.concatenate(self.idx)] = np.concatenate([np.full(len(i), 1.0 / len(i)) for i in self.idx])
        self.grid = [gp[i] for i in self.idx]
        self.retired = [cal.dnf_draws(d, int(s) * 100 + int(r)) for d, (s, r) in zip(self.dnf, self.keys)]
        # p_T[t][k] = win probs of race k at temperature T_GRID[t]
        self.p_T = [[pl_win_probs(s, d, t, ret) for s, d, ret in zip(self.scores, self.dnf, self.retired)]
                    for t in T_GRID]

    def __len__(self) -> int:
        return len(self.keys)


def _mix(p: np.ndarray, a: float) -> np.ndarray:
    return (1 - a) * p + a / len(p)


def calibrate(r: Races) -> dict[str, list[np.ndarray]]:
    """Return {method: [probs per race]} using only earlier races to fit each one."""
    n = len(r)
    # loss tables over the grid, per race
    ll_T = np.array([[race_ll(r.p_T[t][k], r.won[k]) for k in range(n)] for t in range(len(T_GRID))])
    ll_TA = np.array([[[race_ll(_mix(r.p_T[t][k], a), r.won[k]) for k in range(n)] for a in MIX_GRID]
                      for t in range(len(T_GRID))])
    ll_TB = cal.loss_table([cal.RaceInputs(s, ret, g, w)
                            for s, ret, g, w in zip(r.scores, r.retired, r.grid, r.won)])
    methods = ("current", "rolling_T", "rolling_T_last20", "rolling_T_mix", "rolling_T_grid_blend",
               "rolling_T_floor1", "platt", "isotonic")
    out: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    out["fitted"] = []
    for k in range(n):
        out["current"].append(r.base[k])
        if k < MIN_HISTORY:
            for m in methods[1:]:
                out[m].append(r.base[k])
            out["fitted"].append({})
            continue
        hist = slice(0, k)
        t1 = int(np.argmin(ll_T[:, hist].mean(axis=1)))
        out["rolling_T"].append(r.p_T[t1][k])
        t20 = int(np.argmin(ll_T[:, max(0, k - 20):k].mean(axis=1)))
        out["rolling_T_last20"].append(r.p_T[t20][k])
        tb, bb = cal.best_params(ll_TB[:, :, hist])
        out["rolling_T_grid_blend"].append(cal.blend(cal.pl_win_probs(r.scores[k], r.retired[k], tb), r.grid[k], bb))
        floor = T_GRID >= 1.0
        tf = int(np.flatnonzero(floor)[np.argmin(ll_T[floor][:, hist].mean(axis=1))])
        out["rolling_T_floor1"].append(r.p_T[tf][k])
        m = ll_TA[:, :, hist].mean(axis=2)
        tm, am = np.unravel_index(int(np.argmin(m)), m.shape)
        out["rolling_T_mix"].append(_mix(r.p_T[tm][k], MIX_GRID[am]))
        # Platt / isotonic on the fold's own calibrated win probability
        xp = np.concatenate(r.base[:k])
        yp = np.concatenate([w == 1 for w in r.won[:k]]).astype(int)
        lg = np.log(np.clip(xp, 1e-6, 1 - 1e-6) / (1 - np.clip(xp, 1e-6, 1 - 1e-6)))
        lr = LogisticRegression(C=1e6).fit(lg[:, None], yp)
        xk = np.clip(r.base[k], 1e-6, 1 - 1e-6)
        pk = lr.predict_proba(np.log(xk / (1 - xk))[:, None])[:, 1]
        out["platt"].append(pk / pk.sum())
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0).fit(xp, yp)
        pi = np.maximum(iso.predict(r.base[k]), 1e-4)
        out["isotonic"].append(pi / pi.sum())
        out["fitted"].append({"T": float(T_GRID[t1]), "T_last20": float(T_GRID[t20]),
                              "T_mix": float(T_GRID[tm]), "mix": float(MIX_GRID[am]),
                              "T_blend": tb, "grid_weight": bb,
                              "platt_slope": float(lr.coef_[0][0]), "platt_icept": float(lr.intercept_[0])})
    return out


# ----------------------------------------------------------------------------- scoring
def _frame(r: Races, probs: list[np.ndarray], col: str = "win_prob") -> pd.DataFrame:
    df = r.df.copy()
    df[col] = np.nan
    for i, p in zip(r.idx, probs):
        df.loc[i, col] = p
    return df


def calibration_table(df: pd.DataFrame, col: str) -> list[dict]:
    b = pd.cut(df[col], BUCKETS, right=False)
    out = []
    for iv, g in df.groupby(b, observed=True):
        out.append({"bucket": f"{iv.left:.0%}-{min(iv.right, 1):.0%}", "n": int(len(g)),
                    "said": float(g[col].mean()), "happened": float(g["won"].mean())})
    return out


def score(df: pd.DataFrame, col: str) -> dict:
    lls, brs, hits = [], [], []
    for _, g in df.groupby(["season", "round"], sort=True):
        p, w = g[col].to_numpy(float), g["won"].to_numpy(int)
        lls.append(race_ll(p, w))
        brs.append(race_brier(p, w))
        hits.append(int(w[np.argmax(p)] == 1))
    return {"races": len(lls), "log_loss": float(np.mean(lls)), "brier": float(np.mean(brs)),
            "hit": float(np.mean(hits)), "per_race": lls}


def boot_diff(a: list[float], b: list[float], n: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


WINDOWS = {
    "all": lambda d: d,
    "2025-26": lambda d: d[d["season"] >= 2025],
    "2026": lambda d: d[d["season"] == 2026],
    "market_window": lambda d: d[(d["season"] * 100 + d["round"]) >= 202416],
}


def evaluate(r: Races, methods: dict[str, list[np.ndarray]], stage: str) -> dict:
    drivers = pd.read_parquet(PROCESSED / "driver_race.parquet")[["season", "round", "driver_id", "car_driver_combo"]]
    res: dict = {"stage": stage, "methods": {}}
    base_df = r.df.merge(drivers, on=["season", "round", "driver_id"], how="left")
    base_df["baseline"] = (grid_probabilities(base_df) if stage != "pre_quali"
                           else prior_form_probabilities(base_df)).to_numpy()
    for name, probs in methods.items():
        if name == "fitted":
            continue
        df = _frame(r, probs)
        df["baseline"] = base_df["baseline"].to_numpy()
        res["methods"][name] = {}
        for w, f in WINDOWS.items():
            sub = f(df)
            s, b = score(sub, "win_prob"), score(sub, "baseline")
            res["methods"][name][w] = {
                "races": s["races"], "log_loss": s["log_loss"], "brier": s["brier"], "hit": s["hit"],
                "baseline_log_loss": b["log_loss"], "baseline_brier": b["brier"],
                "minus_baseline": boot_diff(s["per_race"], b["per_race"]),
                "calibration": calibration_table(sub, "win_prob"),
            }
    res["baseline_name"] = "prior_form" if stage == "pre_quali" else "grid"
    return res


def market_scores(r: Races, probs: list[np.ndarray], stage: str, mb) -> dict | None:
    snaps_path = mb.OUT / "polymarket_snapshots.parquet"
    if not snaps_path.exists():
        return None
    snaps = pd.read_parquet(snaps_path)
    drivers = pd.read_parquet(PROCESSED / "driver_race.parquet")[["season", "round", "driver_id", "car_driver_combo"]]
    preds = _frame(r, probs).merge(drivers, on=["season", "round", "driver_id"], how="left")
    preds["stage"] = stage
    field = preds[["season", "round", "driver_id"]].drop_duplicates()
    snap = "post_quali" if stage == "post_quali" else "pre_weekend"
    mkt = mb.market_frame(snaps, field, "win", snap)
    res = mb.compare(preds, mkt, stage, "win", f"{stage} vs market {snap}")
    if not res.get("races"):
        return None
    return {"races": res["races"], "scores": res["scores"], "model_minus_market": res["model_minus_market_log_loss"],
            "model_beats_market_races": res["model_beats_market_races"], "calibration": res["calibration"]["model"],
            "market_calibration": res["calibration"]["market"], "hit_rate": res["hit_rate"]}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preds = pd.read_parquet(EXP / f"{tag}.parquet")
    mb = _load_market_module()
    report: dict = {"tag": tag, "stages": {}}
    for stage in ("post_quali", "pre_quali"):
        r = Races(preds[preds["stage"] == stage])
        methods = calibrate(r)
        ev = evaluate(r, methods, stage)
        ev["fold_temperatures"] = r.df.groupby(["season", "round"])["temperature"].first().describe().to_dict()
        ev["fitted_last"] = methods["fitted"][-1]
        ev["fitted_by_race"] = [{"season": int(k[0]), "round": int(k[1]), **f} for k, f in zip(r.keys, methods["fitted"])]
        ev["market"] = {m: market_scores(r, methods[m], stage, mb) for m in methods if m != "fitted"}
        report["stages"][stage] = ev
        print(f"\n== {stage} (baseline: {ev['baseline_name']}) ==")
        for m, v in ev["methods"].items():
            a, b = v["2025-26"], v["2026"]
            top = [c for c in v["all"]["calibration"] if c["bucket"].startswith("50%")]
            top_s = f"{top[0]['n']}/{top[0]['said']:.1%}/{top[0]['happened']:.1%}" if top else "-"
            mk = ev["market"].get(m)
            mk_s = f"{mk['scores']['model']['log_loss']:.3f} vs mkt {mk['scores']['market']['log_loss']:.3f}" if mk else "-"
            print(f"{m:18s} all {v['all']['log_loss']:.3f} (base {v['all']['baseline_log_loss']:.3f}) | "
                  f"25-26 {a['log_loss']:.3f} (base {a['baseline_log_loss']:.3f}, d {a['minus_baseline'][0]:+.3f} "
                  f"[{a['minus_baseline'][1]:+.3f},{a['minus_baseline'][2]:+.3f}]) | 2026 {b['log_loss']:.3f} "
                  f"(base {b['baseline_log_loss']:.3f}) | top50+ all {top_s} | market {mk_s}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"exp005_calibration_{tag}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
