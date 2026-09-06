"""Feature specifications for pre-quali vs post-quali models."""

from __future__ import annotations

from typing import Literal

WeekendMode = Literal["pre_quali", "post_quali"]

# Shared form / context features (no quali leakage).
FEATURE_PRE_QUALI_NUM = [
    "champ_position_before",
    "champ_points_before",
    "champ_wins_before",
    "driver_id_points_last_3",
    "driver_id_points_last_5",
    "driver_id_avg_finish_last_5",
    "driver_id_dnf_rate_last_5",
    "constructor_id_points_last_3",
    "constructor_id_points_last_5",
    "constructor_id_avg_finish_last_5",
    "constructor_id_dnf_rate_last_5",
    "circuit_avg_finish_prior",
    "circuit_grid_to_finish_delta",
    "teammate_quali_h2h_rate",
    "is_wet",
    "has_sprint",
    "round",
    "precipitation_mm",
]

FEATURE_POST_QUALI_EXTRA = [
    "quali_position",
    "grid",
    "grid_vs_quali",
    "best_quali_seconds",
    "quali_vs_teammate",
    "sprint_position",
]

FEATURE_CAT = ["constructor_id"]

TRAIN_SEASONS = {2022, 2023, 2024, 2025, 2026}
TEST_SEASONS: set[int] = set()


def season_range_label(seasons: set[int]) -> str:
    """Human-readable season list, e.g. 2022–2026."""
    if not seasons:
        return "none"
    ordered = sorted(seasons)
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}–{ordered[-1]}"
    return ", ".join(str(s) for s in ordered)


def feature_columns(mode: WeekendMode) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) column lists for a weekend mode."""
    numeric = list(FEATURE_PRE_QUALI_NUM)
    if mode == "post_quali":
        numeric = numeric + FEATURE_POST_QUALI_EXTRA
    return numeric, list(FEATURE_CAT)


def all_numeric_columns() -> list[str]:
    return FEATURE_PRE_QUALI_NUM + FEATURE_POST_QUALI_EXTRA
