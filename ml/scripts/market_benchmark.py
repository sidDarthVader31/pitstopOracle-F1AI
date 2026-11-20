#!/usr/bin/env python3
"""EXP-004: compare walk-forward model probabilities with the betting market.

Three steps, each cached so the slow ones run once:

  fetch    OpenF1 session times + Polymarket F1 race-winner / podium price history
           -> data/cache/market/{sessions,polymarket_snapshots}.parquet
  predict  walk-forward model probabilities (the same folds as reports/EVAL.md)
           -> data/cache/market/model_wf_predictions.parquet
  report   join on the same races and score market vs model vs grid
           -> reports/market_benchmark.json + reports/MARKET_BENCHMARK.md

Usage (from ml/):
  python scripts/market_benchmark.py all          # fetch + predict + report
  python scripts/market_benchmark.py report       # re-score from the caches

Market data: Polymarket's public, documented market-data API (gamma-api / clob,
no account or key). Raw prices stay in the git-ignored cache and are not committed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1ml.eval import grid_probabilities, prior_form_probabilities  # noqa: E402
from f1ml.paths import CACHE, PROCESSED, REPORTS  # noqa: E402

OUT = CACHE / "market"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
OPENF1 = "https://api.openf1.org/v1"
SEASONS = (2024, 2025, 2026)  # Polymarket per-race F1 markets start at the 2024 Italian GP
UA = {"User-Agent": "pitstop-oracle-research/0.1"}
EPS = 1e-6

# Market snapshot -> (session whose start is the cutoff, offset in minutes)
SNAPSHOTS = {
    "pre_weekend": ("Practice 1", 0),  # before any running: same information as the pre-quali model
    "pre_quali": ("Qualifying", 0),  # after practice (and, on 2024+ sprint weekends, the sprint)
    "post_quali": ("Race", -5),  # after qualifying and grid penalties, 5 min before lights out
}


# --------------------------------------------------------------------------- fetch
def _get(url: str, params: dict | None = None, retries: int = 7):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=40)
            if r.status_code == 429:
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(1 + 2 * attempt)
    return None


def fetch_sessions() -> pd.DataFrame:
    """Scheduled start of FP1, qualifying and race for every GP weekend (OpenF1)."""
    races = pd.read_parquet(PROCESSED / "driver_race.parquet")[["season", "round", "date"]].drop_duplicates()
    races["date"] = pd.to_datetime(races["date"]).dt.date
    rows = []
    for year in SEASONS:
        sessions = pd.DataFrame(_get(f"{OPENF1}/sessions", {"year": year}) or [])
        if sessions.empty:
            continue
        sessions["start"] = pd.to_datetime(sessions["date_start"], utc=True)
        for meeting, g in sessions.groupby("meeting_key"):
            race = g[g["session_name"] == "Race"]
            if race.empty:
                continue
            race_day = race["start"].iloc[0].date()
            match = races[(races["season"] == year) & (races["date"].map(lambda d: abs((d - race_day).days) <= 1))]
            if match.empty:
                continue
            row = {"season": year, "round": int(match["round"].iloc[0]), "meeting_key": int(meeting),
                   "sprint_weekend": bool((g["session_name"] == "Sprint").any())}
            for name in ("Practice 1", "Qualifying", "Race"):
                s = g[g["session_name"] == name]
                row[name] = s["start"].iloc[0] if len(s) else pd.NaT
            rows.append(row)
    return pd.DataFrame(rows)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.lower().replace("-", " ").strip()


def _market_kind(title: str) -> str | None:
    t = title.lower()
    if "grand prix" not in t or any(k in t for k in ("sprint", "pole", "constructor", "h2h", "qualifying")):
        return None
    if "podium" in t:
        return "podium"
    if "winner" in t:
        return "win"
    return None


def fetch_polymarket_events() -> list[dict]:
    """All events with the F1 tag (pages of 100; the API resets connections now and then)."""
    events, offset = {}, 0
    while True:
        batch = _get(f"{GAMMA}/events", {"tag_slug": "f1", "limit": 100, "offset": offset}) or []
        for e in batch:
            events[e["id"]] = e
        if len(batch) < 100:
            break
        offset += 100
    return list(events.values())


def _match_driver(title: str, field: pd.DataFrame) -> str | None:
    t = _norm(title)
    for row in field.itertuples(index=False):
        fam, given = _norm(row.family_name), _norm(row.given_name)
        if t == fam or t.endswith(" " + fam) or t == f"{given} {fam}" or fam in t.split():
            return row.driver_id
    for row in field.itertuples(index=False):  # typos such as "George Russel"
        fam, given = _norm(row.family_name), _norm(row.given_name)
        last = t.split()[-1] if t.split() else ""
        if len(fam) >= 5 and len(last) >= 5 and last[:5] == fam[:5] and (len(t.split()) == 1 or t.startswith(given[:3])):
            return row.driver_id
    return None


ALIASES = {"catalunya": "barcelona", "brazlian": "brazil", "mexican": "mexico"}
GENERIC = {"f1", "grand", "prix", "winner", "driver", "podium", "finish", "the", "formula"}


def _name_keys(text: str) -> set[str]:
    words = [ALIASES.get(w, w) for w in _norm(text).replace(":", " ").split() if w not in GENERIC]
    return {w[:4] for w in words if len(w) >= 4}


def _match_race(event: dict, sessions: pd.DataFrame, race_names: dict) -> pd.Series | None:
    """Map a Polymarket event to (season, round).

    The event's endDate is only a nominal resolve-by date (in 2026 often a week after the
    race), so match on the Grand Prix name among races held while the market was open.
    """
    start = pd.Timestamp(event.get("startDate") or event["endDate"]).tz_convert("UTC") - pd.Timedelta(days=1)
    end = pd.Timestamp(event["endDate"]).tz_convert("UTC") + pd.Timedelta(days=1)
    cand = sessions[(sessions["Race"] >= start) & (sessions["Race"] <= end)].copy()
    if cand.empty:
        return None
    keys = _name_keys(event.get("title", ""))
    cand["name_match"] = [bool(keys & _name_keys(race_names.get((r.season, r.round), "")))
                          for r in cand.itertuples()]
    named = cand[cand["name_match"]]
    if len(named):
        return named.iloc[(named["Race"] - end).abs().argmin()]
    if len(cand) == 1 and abs(cand["Race"].iloc[0] - end) <= pd.Timedelta(days=3):
        return cand.iloc[0]
    return None


def fetch_snapshots(sessions: pd.DataFrame) -> pd.DataFrame:
    drivers = pd.read_parquet(PROCESSED / "driver_race.parquet")
    calendar = pd.read_parquet(ROOT / "data" / "raw" / "races.parquet")
    race_names = {(int(r.season), int(r.round)): r.race_name for r in calendar.itertuples()}
    rows = []
    for event in fetch_polymarket_events():
        kind = _market_kind(event.get("title", ""))
        if kind is None or not event.get("endDate"):
            continue
        s = _match_race(event, sessions, race_names)
        if s is None:
            continue
        field = drivers[(drivers["season"] == s["season"]) & (drivers["round"] == s["round"])]
        start = s["Practice 1"] - pd.Timedelta(hours=6) if pd.notna(s["Practice 1"]) else s["Race"] - pd.Timedelta(days=4)
        stop = s["Race"] + pd.Timedelta(minutes=5)
        for m in event.get("markets", []):
            try:
                yes_token = json.loads(m["clobTokenIds"])[0]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            label = m.get("groupItemTitle") or m.get("question", "")
            hist = (_get(f"{CLOB}/prices-history", {
                "market": yes_token, "startTs": int(start.timestamp()), "endTs": int(stop.timestamp()),
                "fidelity": 10,
            }) or {}).get("history", [])
            h = pd.DataFrame(hist)
            if h.empty:
                continue
            h["t"] = pd.to_datetime(h["t"], unit="s", utc=True)
            for snap, (session, offset_min) in SNAPSHOTS.items():
                cutoff = s[session]
                if pd.isna(cutoff):
                    continue
                cutoff = cutoff + pd.Timedelta(minutes=offset_min)
                before = h[(h["t"] <= cutoff) & (h["t"] >= cutoff - pd.Timedelta(hours=12))]
                if before.empty:
                    continue
                last = before.iloc[-1]
                rows.append({
                    "season": int(s["season"]), "round": int(s["round"]), "kind": kind, "snapshot": snap,
                    "label": label, "driver_id": _match_driver(label, field), "price": float(last["p"]),
                    "price_ts": last["t"], "cutoff": cutoff, "event_slug": event.get("slug"),
                    "event_volume": float(event.get("volume") or 0.0), "sprint_weekend": bool(s["sprint_weekend"]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- predict
def predict() -> pd.DataFrame:
    """Walk-forward probabilities from f1ml.train (no reports or artifacts written)."""
    from f1ml.train import train_and_eval

    os.environ.setdefault("F1ML_EVAL_FROM_SEASON", str(min(SEASONS)))
    rows: list[dict] = []
    train_and_eval(write_reports=False, predictions_out=rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- de-margin
def devig(prices: pd.Series, kind: str = "win", method: str = "proportional") -> pd.Series:
    """Remove the margin so the probabilities add to 1 (win) or 3 (podium).

    proportional: divide every price by the book total (the standard method).
    power: p_i = price_i ** k with k chosen so the total is right; shrinks longshots more
    (a common correction for the favourite-longshot bias).
    """
    target = 1.0 if kind == "win" else 3.0
    p = prices.clip(lower=EPS).astype(float)
    if method == "proportional":
        out = p * target / p.sum()
        return out.clip(upper=1.0)
    lo, hi = 0.2, 5.0
    for _ in range(80):
        k = (lo + hi) / 2
        total = (p ** k).sum()
        lo, hi = (lo, k) if total < target else (k, hi)
    return (p ** ((lo + hi) / 2)).clip(upper=1.0)


def market_frame(snaps: pd.DataFrame, field: pd.DataFrame, kind: str, snapshot: str,
                 method: str = "proportional") -> pd.DataFrame:
    """Per-driver de-margined market probability for every race with a usable book."""
    out = []
    sub = snaps[(snaps["kind"] == kind) & (snaps["snapshot"] == snapshot)]
    for (season, rnd), g in sub.groupby(["season", "round"]):
        g = g.sort_values("price_ts").drop_duplicates("label", keep="last")
        # Outcomes naming a driver who is not in this race (e.g. a replaced driver) are void
        # and often carry stale prices; only a genuine "Other" outcome stands for the rest.
        is_other = g["label"].astype(str).str.lower().str.contains("other|another|field")
        g = g[g["driver_id"].notna() | is_other]
        book = float(g["price"].sum())
        target = 1.0 if kind == "win" else 3.0
        if not (0.85 * target <= book <= 1.35 * target):
            continue  # stale, empty or broken book
        race_field = field[(field["season"] == season) & (field["round"] == rnd)]
        if race_field.empty:
            continue
        matched = g[g["driver_id"].notna()].groupby("driver_id")["price"].max()
        unmatched_mass = float(g.loc[g["driver_id"].isna(), "price"].sum())  # "Other" etc.
        missing = [d for d in race_field["driver_id"] if d not in matched.index]
        fill = (unmatched_mass / len(missing)) if (missing and unmatched_mass > 0) else 0.001
        raw = pd.Series({d: matched.get(d, fill) for d in race_field["driver_id"]})
        prob = devig(raw, kind, method)
        for d in race_field["driver_id"]:
            out.append({"season": season, "round": rnd, "driver_id": d, "price": raw[d], "mkt_prob": prob[d],
                        "book": book, "sprint_weekend": bool(g["sprint_weekend"].iloc[0]),
                        "event_volume": float(g["event_volume"].max())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- scoring
def _race_scores(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    rows = []
    for (season, rnd), g in df.groupby(["season", "round"]):
        y = g[target].astype(float).to_numpy()
        p = g[col].astype(float).clip(EPS, 1 - EPS).to_numpy()
        if target == "won":
            p_norm = p / p.sum()
            ll = -np.log(p_norm[y == 1]).sum() if y.sum() == 1 else np.nan
        else:  # podium: mean binary log-loss per driver
            ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        rows.append({"season": season, "round": rnd, "log_loss": float(ll),
                     "brier": float(((p - y) ** 2).mean())})
    return pd.DataFrame(rows)


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def calibration_table(df: pd.DataFrame, col: str, target: str = "won") -> list[dict]:
    edges = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0001]
    out = []
    bins = pd.cut(df[col], edges, right=False)
    for b, g in df.groupby(bins, observed=True):
        out.append({"bin": f"{b.left:.0%}–{min(b.right, 1):.0%}", "n": int(len(g)),
                    "mean_pred": float(g[col].mean()), "actual": float(g[target].mean())})
    return out


def betting_sim(df: pd.DataFrame, model_col: str, target: str, edges=(0.0, 0.10, 0.25, 0.50),
                fee: float = 0.0) -> list[dict]:
    """Flat 1-unit back bet on every driver where model prob > market prob * (1 + edge).

    You pay the raw (margin-included) market price; a winning unit returns 1/price.
    `fee` is a commission on net winnings (Betfair UK: 2-5%).
    """
    out = []
    for edge in edges:
        bets = df[(df[model_col] > df["mkt_prob"] * (1 + edge)) & (df["price"] > 0.005)].copy()
        if bets.empty:
            out.append({"edge": edge, "bets": 0})
            continue
        win = bets[target].astype(float)
        profit = win * (1.0 / bets["price"] - 1.0) * (1 - fee) - (1 - win)
        bets["profit"] = profit
        per_race = bets.groupby(["season", "round"])["profit"].agg(["sum", "count"])
        rng = np.random.default_rng(11)
        idx = rng.integers(0, len(per_race), size=(5000, len(per_race)))
        s, c = per_race["sum"].to_numpy(), per_race["count"].to_numpy()
        roi_boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        out.append({
            "edge": edge, "bets": int(len(bets)), "races": int(len(per_race)), "winners": int(win.sum()),
            "staked": float(len(bets)), "profit": float(profit.sum()), "roi": float(profit.sum() / len(bets)),
            "roi_ci": [float(np.quantile(roi_boot, 0.025)), float(np.quantile(roi_boot, 0.975))],
            "avg_price": float(bets["price"].mean()),
        })
    return out


def compare(model: pd.DataFrame, market: pd.DataFrame, stage: str, kind: str, label: str) -> dict:
    target = "won" if kind == "win" else "podium"
    mcol = "win_prob" if kind == "win" else "podium_prob"
    m = model[model["stage"] == stage]
    j = m.merge(market, on=["season", "round", "driver_id"], how="inner")
    # keep races where both sides cover the whole field the model scored
    full = j.groupby(["season", "round"]).size() == m.groupby(["season", "round"]).size().reindex(
        j.groupby(["season", "round"]).size().index)
    j = j.set_index(["season", "round"]).loc[full[full].index].reset_index()
    if j.empty:
        return {"label": label, "races": 0}
    j[target] = j[target].astype(float)
    if kind == "win" and stage == "post_quali":
        j["base_prob"] = grid_probabilities(j).to_numpy()
        base_name = "grid"
    elif kind == "win":
        j["base_prob"] = prior_form_probabilities(j).to_numpy()
        base_name = "prior_form"
    else:
        base_name = None
    # 50/50 blend in log space: does the model add information on top of the market?
    if kind == "win":
        lb = 0.5 * np.log(j[mcol].clip(EPS)) + 0.5 * np.log(j["mkt_prob"].clip(EPS))
        j["blend_prob"] = np.exp(lb)
        j["blend_prob"] = j["blend_prob"] / j.groupby(["season", "round"])["blend_prob"].transform("sum")
    cols = {"model": mcol, "market": "mkt_prob"}
    if base_name:
        cols[base_name] = "base_prob"
    if kind == "win":
        cols["blend_50_50"] = "blend_prob"
    scores = {name: _race_scores(j, col, target) for name, col in cols.items()}
    res = {
        "label": label, "stage": stage, "kind": kind, "races": int(j.groupby(["season", "round"]).ngroups),
        "seasons": sorted(int(s) for s in j["season"].unique()),
        "sprint_races": int(j.groupby(["season", "round"])["sprint_weekend"].first().sum()),
        "mean_book": float(j.groupby(["season", "round"])["book"].first().mean()),
        "scores": {n: {"log_loss": float(s["log_loss"].mean()), "brier": float(s["brier"].mean())}
                   for n, s in scores.items()},
        "model_minus_market_log_loss": _bootstrap_diff(scores["model"]["log_loss"].to_numpy(),
                                                       scores["market"]["log_loss"].to_numpy()),
        "model_beats_market_races": int((scores["model"]["log_loss"] < scores["market"]["log_loss"]).sum()),
        "calibration": {"model": calibration_table(j, mcol, target), "market": calibration_table(j, "mkt_prob", target)},
        "hit_rate": {},
        "by_season": {},
        "betting": betting_sim(j, mcol, target),
        "betting_5pct_commission": betting_sim(j, mcol, target, fee=0.05),
    }
    if kind == "win":
        for name, col in cols.items():
            picks = j.loc[j.groupby(["season", "round"])[col].idxmax()]
            res["hit_rate"][name] = float(picks["won"].mean())
        res["blend_minus_market_log_loss"] = _bootstrap_diff(scores["blend_50_50"]["log_loss"].to_numpy(),
                                                             scores["market"]["log_loss"].to_numpy())
    for season in res["seasons"]:
        res["by_season"][season] = {
            n: float(s.loc[s["season"] == season, "log_loss"].mean()) for n, s in scores.items()}
        res["by_season"][season]["races"] = int((scores["model"]["season"] == season).sum())
    return res


def report() -> dict:
    snaps = pd.read_parquet(OUT / "polymarket_snapshots.parquet")
    preds = pd.read_parquet(OUT / "model_wf_predictions.parquet")
    drivers = pd.read_parquet(PROCESSED / "driver_race.parquet")
    keep = ["season", "round", "driver_id", "car_driver_combo"]
    preds = preds.merge(drivers[keep], on=["season", "round", "driver_id"], how="left")
    field = preds[["season", "round", "driver_id"]].drop_duplicates()
    results = {"generated": pd.Timestamp.now(tz="UTC").isoformat(), "comparisons": []}
    plan = [
        ("post_quali", "win", "post_quali", "Winner, after qualifying (model post-quali vs market 5 min before the race)"),
        ("pre_quali", "win", "pre_weekend", "Winner, before the weekend (model pre-quali vs market before FP1)"),
        ("pre_quali", "win", "pre_quali", "Winner, model pre-quali vs market just before qualifying (market has practice)"),
        ("post_quali", "podium", "post_quali", "Podium, after qualifying"),
        ("pre_quali", "podium", "pre_weekend", "Podium, before the weekend"),
    ]
    for stage, kind, snap, label in plan:
        mkt = market_frame(snaps, field, kind, snap)
        res = compare(preds, mkt, stage, kind, label)
        res["snapshot"] = snap
        if res.get("races"):
            mkt_pow = market_frame(snaps, field, kind, snap, method="power")
            alt = compare(preds, mkt_pow, stage, kind, label)
            res["market_power_devig"] = alt["scores"]["market"]
            nonsprint = mkt[~mkt["sprint_weekend"]]
            ns = compare(preds, nonsprint, stage, kind, label)
            res["non_sprint"] = {"races": ns.get("races", 0), "scores": ns.get("scores", {})}
        results["comparisons"].append(res)
    coverage = (snaps.groupby(["kind", "snapshot"])[["season", "round"]]
                .apply(lambda g: g.drop_duplicates().shape[0]).to_dict())
    results["coverage_races_with_any_price"] = {f"{k[0]}/{k[1]}": int(v) for k, v in coverage.items()}
    results["unmatched_labels"] = sorted(set(snaps.loc[snaps["driver_id"].isna(), "label"].astype(str)))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "market_benchmark.json").write_text(json.dumps(results, indent=2, default=str))
    (REPORTS / "MARKET_BENCHMARK.md").write_text(_markdown(results))
    return results


def _markdown(results: dict) -> str:
    lines = ["# Model vs market (generated by `scripts/market_benchmark.py`)", "",
             f"Generated {results['generated'][:16]} UTC. Market = Polymarket, margin removed proportionally. "
             "Lower log-loss / Brier is better. See docs/experiments/EXP-004-market-benchmark.md.", ""]
    for r in results["comparisons"]:
        lines += [f"## {r['label']}", ""]
        if not r.get("races"):
            lines += ["No overlapping races.", ""]
            continue
        lines += [f"Races: {r['races']} (seasons {', '.join(map(str, r['seasons']))}; sprint weekends {r['sprint_races']}). "
                  f"Average book total before margin removal: {r['mean_book']:.3f}.", "",
                  "| Forecast | Log-loss | Brier | Top-pick hit |", "|---|---:|---:|---:|"]
        for name, s in r["scores"].items():
            hit = r["hit_rate"].get(name)
            lines.append(f"| {name} | {s['log_loss']:.3f} | {s['brier']:.4f} | {'' if hit is None else f'{hit:.1%}'} |")
        if "market_power_devig" in r:
            lines.append(f"| market (power de-margin) | {r['market_power_devig']['log_loss']:.3f} | {r['market_power_devig']['brier']:.4f} | |")
        d = r["model_minus_market_log_loss"]
        lines += ["", f"Model − market log-loss: {d[0]:+.3f} (95% bootstrap CI {d[1]:+.3f} to {d[2]:+.3f}); "
                  f"model better in {r['model_beats_market_races']}/{r['races']} races."]
        if "blend_minus_market_log_loss" in r:
            b = r["blend_minus_market_log_loss"]
            lines.append(f"50/50 blend − market: {b[0]:+.3f} (CI {b[1]:+.3f} to {b[2]:+.3f}).")
        if r.get("non_sprint", {}).get("races"):
            ns = r["non_sprint"]
            lines.append(f"Non-sprint weekends only ({ns['races']} races): model {ns['scores']['model']['log_loss']:.3f} vs "
                         f"market {ns['scores']['market']['log_loss']:.3f}.")
        lines += ["", "By season (log-loss):", "", "| Season | Races | " + " | ".join(r["scores"]) + " |",
                  "|---|---:|" + "---:|" * len(r["scores"])]
        for season, v in r["by_season"].items():
            lines.append(f"| {season} | {v['races']} | " + " | ".join(f"{v[n]:.3f}" for n in r["scores"]) + " |")
        lines += ["", "Calibration (predicted vs actual frequency):", "",
                  "| Bin | Model n | Model pred | Model actual | Market n | Market pred | Market actual |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        mk = {c["bin"]: c for c in r["calibration"]["market"]}
        bins = [c["bin"] for c in r["calibration"]["model"]] + [b for b in mk if b not in
                                                                 {c["bin"] for c in r["calibration"]["model"]}]
        mo = {c["bin"]: c for c in r["calibration"]["model"]}
        for b in bins:
            a, m = mo.get(b), mk.get(b)
            fa = f"{a['n']} | {a['mean_pred']:.3f} | {a['actual']:.3f}" if a else "0 | | "
            fm = f"{m['n']} | {m['mean_pred']:.3f} | {m['actual']:.3f}" if m else "0 | | "
            lines.append(f"| {b} | {fa} | {fm} |")
        lines += ["", "Flat 1-unit bets where model > market × (1 + edge), paying the raw market price:", "",
                  "| Edge | Bets | Races | Winners | Avg price | Profit | ROI | ROI 95% CI | ROI after 5% commission |",
                  "|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
        for b, bc in zip(r["betting"], r["betting_5pct_commission"]):
            if not b.get("bets"):
                lines.append(f"| {b['edge']:.0%} | 0 | | | | | | | |")
                continue
            lines.append(f"| {b['edge']:.0%} | {b['bets']} | {b['races']} | {b['winners']} | {b['avg_price']:.3f} | "
                         f"{b['profit']:+.1f} | {b['roi']:+.1%} | {b['roi_ci'][0]:+.0%} to {b['roi_ci'][1]:+.0%} | {bc['roi']:+.1%} |")
        lines.append("")
    lines += ["## Coverage", "", "Races with any Polymarket price per market/snapshot: " +
              ", ".join(f"{k} {v}" for k, v in results["coverage_races_with_any_price"].items()), "",
              "Unmatched outcome labels (treated as 'rest of field'): " + ", ".join(results["unmatched_labels"]), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "predict", "report", "all"])
    step = ap.parse_args().step
    OUT.mkdir(parents=True, exist_ok=True)
    if step in ("fetch", "all"):
        sessions = fetch_sessions()
        sessions.to_parquet(OUT / "sessions.parquet", index=False)
        snaps = fetch_snapshots(sessions)
        snaps.to_parquet(OUT / "polymarket_snapshots.parquet", index=False)
        print(f"sessions: {len(sessions)} weekends; snapshots: {len(snaps)} rows, "
              f"{snaps[['season', 'round']].drop_duplicates().shape[0]} races")
    if step in ("predict", "all"):
        preds = predict()
        preds.to_parquet(OUT / "model_wf_predictions.parquet", index=False)
        print(f"walk-forward predictions: {len(preds)} rows")
    if step in ("report", "all"):
        res = report()
        for r in res["comparisons"]:
            if r.get("races"):
                print(r["label"], r["races"], {k: round(v["log_loss"], 3) for k, v in r["scores"].items()})


if __name__ == "__main__":
    main()
