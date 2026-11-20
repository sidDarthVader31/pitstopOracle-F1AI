#!/usr/bin/env python3
"""Weekly F1 Fantasy price entry (about 15 minutes, before the pre-lock run).

    python scripts/check_prices.py --season 2026 --round 21 --template   # print rows to fill in
    python scripts/check_prices.py --season 2026 --round 21              # validate what you entered

1. `--template` prints one row per driver and constructor in this race's entry list, with
   last round's price pre-filled when known. Paste them into data/manual/fantasy_prices.csv.
2. Type each price from the official game ($M, one decimal).
3. Run without `--template`: it checks every entrant is priced once, in range, and nothing
   unknown is listed. Exit code 1 if anything is wrong. Missing prices never default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from f1ml.fantasy import check_prices  # noqa: E402
from f1ml.fantasy_prices import PRICES_PATH, price_template, read_prices  # noqa: E402
from f1ml.predict import resolve_race_features  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--template", action="store_true")
    a = p.parse_args()
    field, _ = resolve_race_features(a.season, a.round)
    if a.template:
        prev = read_prices(a.season, a.round - 1)
        print(price_template(a.season, a.round, field, prev).to_csv(index=False, header=False), end="")
        return 0
    check = check_prices(a.season, a.round, field)
    if check.ok:
        n = len(check.prices)
        print(f"OK: {n} prices for {a.season} R{a.round} in {PRICES_PATH.name} (prices version {check.version}).")
        return 0
    print(f"Prices for {a.season} R{a.round} are {check.status}:")
    for problem in check.problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
