#!/usr/bin/env python3
"""EXP-009: are the pre-qualifying favourites over-confident, and does a fix help?

    python scripts/wf_dump.py adopted        # once (~8 min)
    python scripts/exp_prequali.py adopted   # ~1 min

Pre-quali win chances from the adopted recipe (EXP-005 rolling temperature +
EXP-006 pace model) are re-calibrated offline. Every calibrator for test race k is
fitted only on the walk-forward forecasts of races before k.

Variants:
- ``adopted``: today's recipe (rolling temperature).
- ``equal_blend``: rolling temperature + log-space blend toward an equal field.
- ``form_blend``: rolling temperature + log-space blend toward the prior-form
  baseline (``eval.prior_form_probabilities``), both fitted together.
Baseline: prior form. The internal market check (never a model input) uses the
Polymarket pre-weekend snapshot from EXP-004.
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

from f1ml import calibration as cal
from f1ml.eval import prior_form_probabilities
from f1ml.paths import CACHE, PROCESSED, REPORTS

EXP = CACHE / "experiments"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "adopted"
    ex = _module("exp_calibration")
    mb = ex._load_market_module()
    preds = pd.read_parquet(EXP / f"{tag}.parquet")
    r = ex.Races(preds[preds["stage"] == "pre_quali"])
    drivers = pd.read_parquet(PROCESSED / "driver_race.parquet")[["season", "round", "driver_id", "car_driver_combo"]]
    merged = r.df.merge(drivers, on=["season", "round", "driver_id"], how="left")
    form = prior_form_probabilities(merged).to_numpy()
    form_by_race = [form[i] for i in r.idx]

    eq_table = cal.loss_table([cal.RaceInputs(s, ret, g, w) for s, ret, g, w in zip(r.scores, r.retired, r.grid, r.won)])
    fm_table = cal.loss_table([cal.RaceInputs(s, ret, f, w) for s, ret, f, w in zip(r.scores, r.retired, form_by_race, r.won)])
    ll_T = np.array([[ex.race_ll(r.p_T[t][k], r.won[k]) for k in range(len(r))] for t in range(len(cal.T_GRID))])
    recency = {"last24": ("window", 24), "decay_hl12": ("decay", 12), "decay_hl24": ("decay", 24)}
    methods: dict[str, list] = {"adopted": [], "equal_blend": [], "form_blend": [], **{m: [] for m in recency}}
    fitted = []
    for k in range(len(r)):
        methods["adopted"].append(r.base[k])
        if k < ex.MIN_HISTORY:
            methods["equal_blend"].append(r.base[k])
            methods["form_blend"].append(r.base[k])
            for m in recency:
                methods[m].append(r.base[k])
            fitted.append({})
            continue
        te, we = cal.best_params(eq_table[:, :, :k])
        tf, wf = cal.best_params(fm_table[:, :, :k])
        methods["equal_blend"].append(cal.blend(cal.pl_win_probs(r.scores[k], r.retired[k], te), r.grid[k], we))
        methods["form_blend"].append(cal.blend(cal.pl_win_probs(r.scores[k], r.retired[k], tf), form_by_race[k], wf))
        rec_t = {}
        for m, (kind, n) in recency.items():
            ages = np.arange(k)[::-1]  # 0 = most recent earlier race
            w = (ages < n).astype(float) if kind == "window" else 0.5 ** (ages / n)
            t = int(np.argmin((ll_T[:, :k] * w).sum(axis=1) / w.sum()))
            methods[m].append(r.p_T[t][k])
            rec_t[f"{m}_T"] = float(cal.T_GRID[t])
        fitted.append({**rec_t, "season": int(r.keys[k][0]), "round": int(r.keys[k][1]), "equal_T": te, "equal_w": we,
                       "form_T": tf, "form_w": wf, "adopted_T": float(r.df.loc[r.idx[k], "temperature"].iloc[0])})
    ev = ex.evaluate(r, methods, "pre_quali")
    ev["market"] = {m: ex.market_scores(r, methods[m], "pre_quali", mb) for m in methods}
    ev["fitted_last"] = fitted[-1]
    ev["fitted_by_race"] = fitted
    # favourites: each race's top pick, said vs won, per window
    fav = {}
    for m, probs in methods.items():
        rows = [{"season": k[0], "p": float(p.max()), "won": int(w[np.argmax(p)] == 1)}
                for k, p, w in zip(r.keys, probs, r.won)]
        d = pd.DataFrame(rows)
        fav[m] = {w: {"races": int(len(s)), "said": float(s["p"].mean()), "won": float(s["won"].mean())}
                  for w, s in (("all", d), ("2025-26", d[d["season"] >= 2025]), ("2026", d[d["season"] == 2026]))}
    ev["favourite"] = fav
    print("== pre_quali (baseline: prior form) ==")
    for m, v in ev["methods"].items():
        parts = []
        for w in ("all", "2025-26", "2026"):
            x = v[w]
            top = [c for c in x["calibration"] if c["bucket"].startswith("50%")]
            ts = f"{top[0]['n']}/{top[0]['said']:.0%}/{top[0]['happened']:.0%}" if top else "-"
            d = x["minus_baseline"]
            parts.append(f"{w} {x['log_loss']:.3f} (form {x['baseline_log_loss']:.3f}, d {d[0]:+.3f}"
                         f"[{d[1]:+.3f},{d[2]:+.3f}]) top {ts} fav {fav[m][w]['said']:.0%}/{fav[m][w]['won']:.0%}")
        mk = ev["market"].get(m)
        ms = f"{mk['scores']['model']['log_loss']:.3f} vs {mk['scores']['market']['log_loss']:.3f}" if mk else "-"
        print(f"{m:12s} " + " | ".join(parts) + f" | market {ms}")
    print("last fit:", fitted[-1])
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"exp009_prequali_{tag}.json").write_text(json.dumps(ev, indent=2, default=float))


if __name__ == "__main__":
    main()
