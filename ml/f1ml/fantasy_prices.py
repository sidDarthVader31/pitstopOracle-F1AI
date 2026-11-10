"""Hand-entered F1 Fantasy prices (``data/manual/fantasy_prices.csv``) with validation.

The owner enters prices weekly (MVP_SPEC §9.5; no scraping). Missing or invalid prices
block every budget feature; there is no silent default price.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from f1ml.fantasy_rules import ScoringRules
from f1ml.paths import MANUAL

PRICES_PATH = MANUAL / "fantasy_prices.csv"
COLUMNS = ["season", "round", "entity_type", "entity_id", "name", "cost_m"]


@dataclass
class PriceCheck:
    prices: pd.DataFrame
    problems: list[str] = field(default_factory=list)
    version: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems and not self.prices.empty

    @property
    def status(self) -> str:
        if self.prices.empty:
            return "missing"
        return "ok" if not self.problems else "invalid"

    def driver_price(self) -> dict[str, float]:
        p = self.prices[self.prices["entity_type"] == "driver"]
        return dict(zip(p["entity_id"], p["cost_m"].astype(float)))

    def constructor_price(self) -> dict[str, float]:
        p = self.prices[self.prices["entity_type"] == "constructor"]
        return dict(zip(p["entity_id"], p["cost_m"].astype(float)))


def read_prices(season: int, round_num: int, path=None) -> pd.DataFrame:
    path = path or PRICES_PATH
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df[(df["season"] == season) & (df["round"] == round_num)].reset_index(drop=True)


def validate_prices(
    prices: pd.DataFrame,
    driver_ids: list[str],
    constructor_ids: list[str],
    rules: ScoringRules,
) -> PriceCheck:
    """Every entrant priced exactly once, in range, nothing unknown."""
    problems: list[str] = []
    if prices.empty:
        return PriceCheck(prices, ["no prices entered for this round"], None)
    missing_cols = [c for c in COLUMNS if c not in prices.columns]
    if missing_cols:
        return PriceCheck(prices.iloc[0:0], [f"missing columns: {missing_cols}"], None)
    bad_type = prices[~prices["entity_type"].isin(["driver", "constructor"])]
    if not bad_type.empty:
        problems.append(f"entity_type must be driver or constructor: {sorted(bad_type['entity_type'].astype(str).unique())}")
    dup = prices[prices.duplicated(["entity_type", "entity_id"], keep=False)]
    if not dup.empty:
        problems.append(f"priced more than once: {sorted(dup['entity_id'].astype(str).unique())}")
    cost = pd.to_numeric(prices["cost_m"], errors="coerce")
    lo, hi = float(rules["team"].get("min_price_m", 3.0)), float(rules["team"].get("max_price_m", 40.0))
    out_of_range = prices[cost.isna() | (cost < lo) | (cost > hi)]
    if not out_of_range.empty:
        problems.append(f"cost_m must be a number between {lo} and {hi}: {out_of_range['entity_id'].tolist()}")
    not_tenths = prices[cost.notna() & ((cost * 10 - (cost * 10).round()).abs() > 1e-6)]
    if not not_tenths.empty:
        problems.append(f"cost_m must have at most one decimal ($0.1M steps): {not_tenths['entity_id'].tolist()}")
    for kind, expected in (("driver", driver_ids), ("constructor", constructor_ids)):
        have = set(prices.loc[prices["entity_type"] == kind, "entity_id"].astype(str))
        missing = sorted(set(expected) - have)
        unknown = sorted(have - set(expected))
        if missing:
            problems.append(f"{kind}s without a price: {missing}")
        if unknown:
            problems.append(f"{kind}s not in this race's entry list: {unknown}")
    canonical = prices.sort_values(["entity_type", "entity_id"])[["entity_type", "entity_id", "cost_m"]].to_csv(index=False)
    version = "p-" + hashlib.sha256(canonical.encode()).hexdigest()[:10]
    return PriceCheck(prices.assign(cost_m=cost), problems, version)


def price_template(season: int, round_num: int, field_df: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows to paste into the CSV for a new round: last round's price if known, else blank."""
    prev = {} if previous is None or previous.empty else dict(
        zip(zip(previous["entity_type"], previous["entity_id"]), previous["cost_m"])
    )
    rows = []
    for _, r in field_df.drop_duplicates("driver_id").iterrows():
        rows.append({"season": season, "round": round_num, "entity_type": "driver", "entity_id": r["driver_id"],
                     "name": f"{r.get('given_name', '')} {r.get('family_name', '')}".strip(),
                     "cost_m": prev.get(("driver", r["driver_id"]))})
    for _, r in field_df.drop_duplicates("constructor_id").iterrows():
        rows.append({"season": season, "round": round_num, "entity_type": "constructor", "entity_id": r["constructor_id"],
                     "name": r.get("constructor_name", r["constructor_id"]),
                     "cost_m": prev.get(("constructor", r["constructor_id"]))})
    return pd.DataFrame(rows, columns=COLUMNS)
