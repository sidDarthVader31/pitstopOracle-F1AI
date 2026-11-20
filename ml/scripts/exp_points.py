#!/usr/bin/env python3
"""EXP-008: calibrate the chance of finishing in the points (top 10).

    python scripts/wf_dump.py adopted      # once: cache walk-forward outputs (~8 min)
    python scripts/exp_points.py adopted   # ~1-2 min

Every variant for test race k is fitted only on the walk-forward (out-of-sample)
forecasts of races *before* k. The pace scores, DNF probabilities and winner
temperature are exactly what the walk-forward model produced; only the step that
turns them into points / podium chances changes.

Variants:
- ``adopted``: today's recipe (one temperature for the whole finishing order).
- ``step``: winner at the fitted temperature, P2 onwards at temperature x m.
- ``ramp``: multiplier rising from 1 at P1 to m at P10.
- ``rank_blend``: logit blend of today's points chance with the historical points
  rate for the driver's baseline rank (grid slot / prior-form rank), weight fitted.
- ``platt``: logistic recalibration of today's points chance.
Baseline: ``rank_rate`` (historical points rate per grid slot / form rank).
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
from sklearn.linear_model import LogisticRegression

from f1ml import calibration as cal
from f1ml.paths import CACHE, PROCESSED, REPORTS

EXP = CACHE / "experiments"
DRAWS = 2000
MIN_HISTORY = cal.MIN_HISTORY_RACES
BLEND_W = np.round(np.arange(0.0, 1.01, 0.1), 2)
P_BUCKETS = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0001]
WINDOWS = {"all": lambda s: s > 0, "2025-26": lambda s: s >= 2025, "2026": lambda s: s == 2026}


def _fantasy_module():
    spec = importlib.util.spec_from_file_location("exp_fantasy", ROOT / "scripts" / "exp_fantasy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _boot(d: np.ndarray, n: int = 5000, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def calibration_table(p: np.ndarray, y: np.ndarray) -> list[dict]:
    b = pd.cut(pd.Series(p), P_BUCKETS, right=False)
    out = []
    for iv, g in pd.DataFrame({"p": p, "y": y}).groupby(b.values, observed=True):
        out.append({"bucket": f"{iv.left:.0%}-{min(iv.right, 1):.0%}", "n": int(len(g)),
                    "said": float(g["p"].mean()), "happened": float(g["y"].mean())})
    return out


def run_stage(preds: pd.DataFrame, hist_all: pd.DataFrame, stage: str, fx) -> dict:
    df = preds[preds["stage"] == stage].merge(
        hist_all[["season", "round", "driver_id", "car_driver_combo"]], on=["season", "round", "driver_id"], how="left")
    df["points_finish"] = df["points_finish"].fillna(0).astype(int)
    df["podium"] = df["podium"].fillna(0).astype(int)
    races = [g.sort_values("driver_id").reset_index(drop=True) for _, g in df.groupby(["season", "round"], sort=True)]
    keys = [(int(g["season"].iloc[0]), int(g["round"].iloc[0])) for g in races]
    n_m = len(cal.REST_MULTIPLIERS)
    sched = {"step": [], "ramp": []}  # per race: dict m-index -> (points, podium, exp_finish)
    rr = []
    for g, (s, r) in zip(races, keys):
        history = hist_all[(hist_all["season"] * 100 + hist_all["round"]) < s * 100 + r]
        if stage == "post_quali":
            history = history[history["grid"].notna()]
        rr.append(fx.rank_rate_probs(history, g, stage, "points_finish"))
        retired = cal.dnf_draws(g["dnf_prob"].to_numpy(float), s * 100 + r, draws=DRAWS)
        temp = float(g["temperature"].iloc[0])
        for name in sched:
            outs = []
            for m in cal.REST_MULTIPLIERS:
                if name == "ramp" and m == 1.0 and sched["step"]:
                    outs.append(sched["step"][-1][0])
                    continue
                ranks = cal.sequential_pl_ranks(g["pace_score"].to_numpy(float), retired, temp, m,
                                                schedule=name, seed=s * 100 + r + 1)
                outs.append(cal.outcome_probs_from_ranks(ranks))
            sched[name].append(outs)
    y_pts = [g["points_finish"].to_numpy() for g in races]
    y_pod = [g["podium"].to_numpy() for g in races]
    fin = [((g["dnf"] == 0) & g["position"].notna()).to_numpy() for g in races]
    pos = [pd.to_numeric(g["position"], errors="coerce").to_numpy(float) for g in races]
    # per race losses for every multiplier
    ll = {name: np.array([[cal.binary_log_loss(sched[name][k][j]["points_prob"], y_pts[k]) for k in range(len(races))]
                          for j in range(n_m)]) for name in sched}
    adopted = [sched["step"][k][0] for k in range(len(races))]
    variants: dict[str, list[dict]] = {"adopted": [], "step": [], "ramp": [], "rank_blend": [], "platt": [],
                                       "rank_rate": []}
    fitted = []
    for k in range(len(races)):
        base = adopted[k]
        variants["adopted"].append(base)
        variants["rank_rate"].append({"points_prob": rr[k], "podium_prob": base["podium_prob"]})
        info = {"season": keys[k][0], "round": keys[k][1]}
        for name in ("step", "ramp"):
            m = cal.fit_rest_multiplier(ll[name][:, :k])
            j = 0 if m is None else int(np.flatnonzero(cal.REST_MULTIPLIERS == m)[0])
            variants[name].append(sched[name][k][j])
            info[f"{name}_m"] = 1.0 if m is None else m
        if k < MIN_HISTORY:
            variants["rank_blend"].append(base)
            variants["platt"].append(base)
            fitted.append(info)
            continue
        hp = np.concatenate([adopted[i]["points_prob"] for i in range(k)])
        hr = np.concatenate(rr[:k])
        hy = np.concatenate(y_pts[:k])
        losses = [cal.binary_log_loss(_sigmoid((1 - w) * _logit(hp) + w * _logit(hr)), hy) for w in BLEND_W]
        w = float(BLEND_W[int(np.argmin(losses))])
        pb = _sigmoid((1 - w) * _logit(base["points_prob"]) + w * _logit(rr[k]))
        variants["rank_blend"].append({**base, "points_prob": pb})
        lr = LogisticRegression(C=1e6).fit(_logit(hp)[:, None], hy)
        pp = lr.predict_proba(_logit(base["points_prob"])[:, None])[:, 1]
        variants["platt"].append({**base, "points_prob": pp})
        info.update({"blend_w": w, "platt_a": float(lr.coef_[0][0]), "platt_b": float(lr.intercept_[0])})
        fitted.append(info)

    seasons = np.array([k[0] for k in keys])
    out: dict = {"races": len(races), "methods": {}, "fitted_last": fitted[-1], "fitted": fitted}
    per_race = {}
    for name, vals in variants.items():
        per_race[name] = {
            "points_ll": np.array([cal.binary_log_loss(v["points_prob"], y) for v, y in zip(vals, y_pts)]),
            "points_brier": np.array([np.mean((v["points_prob"] - y) ** 2) for v, y in zip(vals, y_pts)]),
            "podium_ll": np.array([cal.binary_log_loss(v["podium_prob"], y) for v, y in zip(vals, y_pod)]),
            "points_sum": np.array([v["points_prob"].sum() for v in vals]),
            "finish_mae": np.array([np.nanmean(np.abs(v["expected_finish"][f] - p[f])) if "expected_finish" in v
                                    else np.nan for v, f, p in zip(vals, fin, pos)]),
        }
    for name, vals in variants.items():
        out["methods"][name] = {}
        for wname, f in WINDOWS.items():
            mask = f(seasons)
            p = np.concatenate([v["points_prob"] for v, keep in zip(vals, mask) if keep])
            y = np.concatenate([yy for yy, keep in zip(y_pts, mask) if keep])
            pr = per_race[name]
            res = {"races": int(mask.sum()),
                   "points_ll": float(pr["points_ll"][mask].mean()),
                   "points_brier": float(pr["points_brier"][mask].mean()),
                   "podium_ll": float(pr["podium_ll"][mask].mean()),
                   "points_sum_mean": float(pr["points_sum"][mask].mean()),
                   "finish_mae": float(np.nanmean(pr["finish_mae"][mask])),
                   "under_1pct": {"n": int((p < 0.01).sum()), "scored": int(y[p < 0.01].sum())},
                   "calibration": calibration_table(p, y)}
            for ref in ("rank_rate", "adopted"):
                if ref != name:
                    res[f"minus_{ref}_points_ll"] = _boot(pr["points_ll"][mask] - per_race[ref]["points_ll"][mask])
            if name in ("step", "ramp"):
                res["minus_adopted_finish_mae"] = _boot(pr["finish_mae"][mask] - per_race["adopted"]["finish_mae"][mask])
            res["minus_adopted_podium_ll"] = _boot(pr["podium_ll"][mask] - per_race["adopted"]["podium_ll"][mask]) \
                if name != "adopted" else None
            out["methods"][name][wname] = res
    return out


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "adopted"
    preds = pd.read_parquet(EXP / f"{tag}.parquet")
    hist_all = pd.read_parquet(PROCESSED / "driver_race.parquet")
    hist_all = hist_all[hist_all["dnf"].notna() & (hist_all["is_upcoming"].fillna(0) != 1)]
    fx = _fantasy_module()
    report = {"tag": tag, "draws": DRAWS, "multipliers": cal.REST_MULTIPLIERS.tolist(), "stages": {}}
    for stage in ("pre_quali", "post_quali"):
        res = run_stage(preds, hist_all, stage, fx)
        report["stages"][stage] = res
        print(f"\n===== {stage} ({res['races']} races) last fit: {res['fitted_last']}")
        for name, v in res["methods"].items():
            line = []
            for w in WINDOWS:
                x = v[w]
                d = x.get("minus_rank_rate_points_ll")
                ds = f" d_rr {d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]" if d else ""
                line.append(f"{w}: pts {x['points_ll']:.3f}{ds} pod {x['podium_ll']:.3f} "
                            f"<1% {x['under_1pct']['n']}/{x['under_1pct']['scored']}")
            print(f"{name:11s} " + " | ".join(line))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"exp008_points_{tag}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
