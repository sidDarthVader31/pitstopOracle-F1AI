"""Versioned F1 Fantasy scoring rules (``data/manual/fantasy_rules_<season>.toml``)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from f1ml.paths import MANUAL


class RulesError(ValueError):
    pass


@dataclass(frozen=True)
class ScoringRules:
    version: str
    season: int
    status: str
    raw: dict = field(repr=False)

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]

    @property
    def is_placeholder(self) -> bool:
        return self.status != "active"

    def quali_points(self, pos: int | None) -> int:
        pts = self.raw["qualifying"]["points"]
        return int(pts[pos - 1]) if pos and 1 <= pos <= len(pts) else 0

    def race_points(self, pos: int | None) -> int:
        pts = self.raw["race"]["points"]
        return int(pts[pos - 1]) if pos and 1 <= pos <= len(pts) else 0

    def sprint_points(self, pos: int | None) -> int:
        pts = self.raw["sprint"]["points"]
        return int(pts[pos - 1]) if pos and 1 <= pos <= len(pts) else 0


REQUIRED = {
    "team": ["budget_m", "drivers", "constructors", "free_transfers", "extra_transfer_penalty", "boost_multiplier"],
    "qualifying": ["points", "q2_cutoff", "q3_cutoff"],
    "race": ["points", "position_gained", "position_lost", "overtake", "fastest_lap", "dnf"],
    "sprint": ["points", "position_gained", "position_lost", "overtake", "fastest_lap", "dnf"],
    "constructor": ["quali_both_q3", "quali_one_q3", "quali_both_q2", "quali_one_q2", "quali_neither_q2"],
    "estimates": ["pit_stop_points_flat", "quali_race_correlation", "sprint_dnf_scale"],
}


def rules_path(season: int) -> Path:
    return MANUAL / f"fantasy_rules_{season}.toml"


def parse_rules(raw: dict) -> ScoringRules:
    for key in ("version", "season", "status"):
        if key not in raw:
            raise RulesError(f"rules file is missing top-level '{key}'")
    for section, keys in REQUIRED.items():
        missing = [k for k in keys if k not in raw.get(section, {})]
        if missing:
            raise RulesError(f"[{section}] is missing {missing}")
    for section in ("qualifying", "race", "sprint"):
        pts = raw[section]["points"]
        if pts != sorted(pts, reverse=True) or any(p < 0 for p in pts):
            raise RulesError(f"[{section}].points must be non-negative and non-increasing")
    if raw["qualifying"]["q3_cutoff"] >= raw["qualifying"]["q2_cutoff"]:
        raise RulesError("q3_cutoff must be smaller than q2_cutoff")
    rho = raw["estimates"]["quali_race_correlation"]
    if not 0.0 <= rho <= 1.0:
        raise RulesError("quali_race_correlation must be in [0, 1]")
    return ScoringRules(version=str(raw["version"]), season=int(raw["season"]), status=str(raw["status"]), raw=raw)


@lru_cache(maxsize=8)
def load_rules(season: int) -> ScoringRules:
    """Rules for a season; falls back to the latest earlier season's file (flagged as placeholder)."""
    for year in range(season, season - 5, -1):
        path = rules_path(year)
        if path.exists():
            rules = parse_rules(tomllib.loads(path.read_text()))
            if year != season:
                return ScoringRules(f"{rules.version}-as-{season}", season, "placeholder", rules.raw)
            return rules
    raise RulesError(f"no fantasy rules file for {season} in {MANUAL}")
