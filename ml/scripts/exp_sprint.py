#!/usr/bin/env python3
"""EXP-010: the post-sprint stage (forecast after Saturday's sprint and GP qualifying).

    python scripts/wf_dump.py adopted       # once (~8 min): post_quali / post_sprint outputs
    python scripts/exp_sprint.py adopted    # ~2 min: refits the post-sprint variants

Same walk-forward folds as ``reports/EVAL.md``: each sprint weekend is forecast with
models trained on earlier races only. All variants are scored on the same drivers
(those with a sprint result) of the same sprint weekends.

Variants:
- ``post_sprint``: today's post-sprint stage (from the walk-forward dump).
- ``post_quali``: ignore the sprint; use the adopted post-quali forecast.
- ``sprint_classified``: post-sprint pace model trained on classified finishers
  (EXP-006) with the post-quali rolling temperature (EXP-005), fitted on earlier
  post-quali out-of-sample forecasts only.
- ``mix``: a fixed 50/50 geometric mix of ``post_quali`` and ``post_sprint``.
Baseline: the official grid.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from f1ml import calibration as cal
from f1ml.eval import grid_probabilities, race_brier, race_log_loss, winner_hit_rate_probs
from f1ml.paths import CACHE, REPORTS
from f1ml.ranking import WeekendRanker, train_dnf_model, train_pairwise_ranker
from f1ml.train import (
    MODE_RULES,
    WALK_FORWARD_MC_SAMPLES,
    _load_eval_field,
    _load_training_frame,
    _walk_forward_folds,
)

EXP = CACHE / "experiments"
KEY = ["season", "round", "driver_id"]


def _boot(d: np.ndarray, n: int = 5000, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def sprint_classified_bundle(train: pd.DataFrame, year: int, pq_history: pd.DataFrame) -> WeekendRanker | None:
    pick, _, n_min = MODE_RULES["post_sprint"]
    eligible = pick(train)
    if eligible.groupby(["season", "round"]).ngroups < n_min:
        return None
    temperature = cal.fit_rolling_temperature(pq_history)
    if temperature is None:
        return None
    old = os.environ.get("F1ML_PACE_TRAIN")
    os.environ["F1ML_PACE_TRAIN"] = "classified"
    try:
        ranker = train_pairwise_ranker(eligible, "post_sprint", eval_year=year)
    finally:
        if old is None:
            os.environ.pop("F1ML_PACE_TRAIN", None)
        else:
            os.environ["F1ML_PACE_TRAIN"] = old
    dnf_model, dnf_features = train_dnf_model(eligible, "post_sprint", eval_year=year)
    return WeekendRanker("post_sprint", ranker, dnf_model, dnf_features, temperature, WALK_FORWARD_MC_SAMPLES)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "adopted"
    preds = pd.read_parquet(EXP / f"{tag}.parquet")
    ps = preds[preds["stage"] == "post_sprint"]
    pq = preds[preds["stage"] == "post_quali"]
    sprint_keys = sorted(set(zip(ps["season"], ps["round"])))
    df = _load_training_frame()
    folds = {(f["test_season"], f["test_round"]): f for f in _walk_forward_folds(df, test_field=_load_eval_field())}
    frames = []
    for s, r in sprint_keys:
        fold = folds[(s, r)]
        te = fold["test"]
        te = te[(pd.to_numeric(te["sprint_position"], errors="coerce") > 0) & te["quali_position"].notna()].copy()
        code = s * 100 + r
        hist = pq[(pq["season"] * 100 + pq["round"]) < code].reindex(columns=cal.HISTORY_COLUMNS)
        bundle = sprint_classified_bundle(fold["train"], int(s), hist)
        out = te[KEY + ["won", "grid", "quali_position", "sprint_position"]].copy()
        out = out.merge(ps[KEY + ["win_prob"]].rename(columns={"win_prob": "post_sprint"}), on=KEY, how="left")
        out = out.merge(pq[KEY + ["win_prob"]].rename(columns={"win_prob": "post_quali"}), on=KEY, how="left")
        if bundle is not None:
            out["sprint_classified"] = bundle.predict_frame(te)["win_prob"].to_numpy()
            out["sc_temperature"] = bundle.temperature
        frames.append(out)
    d = pd.concat(frames, ignore_index=True)
    d["won"] = d["won"].fillna(0).astype(int)
    d["grid_base"] = grid_probabilities(d).to_numpy()
    mix = np.exp(0.5 * np.log(d["post_quali"].clip(1e-9)) + 0.5 * np.log(d["post_sprint"].clip(1e-9)))
    d["mix"] = mix / d.assign(m=mix).groupby(["season", "round"])["m"].transform("sum")
    for c in ("post_sprint", "post_quali", "sprint_classified"):
        if c in d:
            d[c] = d[c] / d.groupby(["season", "round"])[c].transform("sum")

    methods = [c for c in ("post_sprint", "post_quali", "sprint_classified", "mix", "grid_base") if c in d]
    report: dict = {"tag": tag, "races": [list(map(int, k)) for k in sprint_keys], "windows": {}}
    for wname, mask in (("all", d["season"] > 0), ("2025-26", d["season"] >= 2025), ("2026", d["season"] == 2026)):
        sub = d[mask]
        per_race = {m: np.array([race_log_loss(g, g[m]) for _, g in sub.groupby(["season", "round"], sort=True)])
                    for m in methods}
        w = {"races": int(sub[["season", "round"]].drop_duplicates().shape[0]), "methods": {}}
        for m in methods:
            w["methods"][m] = {
                "log_loss": float(per_race[m].mean()),
                "brier": race_brier(sub, sub[m]),
                "hit": winner_hit_rate_probs(sub, sub[m]),
                "minus_grid": _boot(per_race[m] - per_race["grid_base"]) if m != "grid_base" else None,
                "minus_post_quali": _boot(per_race[m] - per_race["post_quali"]) if m != "post_quali" else None,
                "favourite": {"said": float(sub.groupby(["season", "round"])[m].max().mean()),
                              "won": float(sub.loc[sub.groupby(["season", "round"])[m].idxmax(), "won"].mean())},
            }
        report["windows"][wname] = w
        print(f"-- {wname} ({w['races']} sprint weekends)")
        for m, v in w["methods"].items():
            g = v["minus_grid"]
            q = v["minus_post_quali"]
            print(f"  {m:18s} LL {v['log_loss']:.3f} brier {v['brier']:.4f} hit {v['hit']:.0%} "
                  f"fav {v['favourite']['said']:.0%}/{v['favourite']['won']:.0%}"
                  + (f" | -grid {g[0]:+.3f}[{g[1]:+.3f},{g[2]:+.3f}]" if g else "")
                  + (f" | -post_quali {q[0]:+.3f}[{q[1]:+.3f},{q[2]:+.3f}]" if q else ""))
    per = d.groupby(["season", "round"]).apply(
        lambda g: {m: float(-np.log(max(g.loc[g["won"] == 1, m].iloc[0], 1e-15))) if (g["won"] == 1).any() else None
                   for m in methods}, include_groups=False)
    report["per_race"] = [{"season": int(k[0]), "round": int(k[1]), **v} for k, v in per.items()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"exp010_sprint_{tag}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
