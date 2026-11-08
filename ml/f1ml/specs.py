"""Feature specifications for fixed pre- and post-qualifying cutoffs.

Current-weekend practice, weather and sprint inputs stay out of the model until
their historical availability at the forecast cutoff can be reproduced.
"""

from __future__ import annotations

from typing import Literal

WeekendMode = Literal["pre_quali", "post_quali", "post_sprint"]

ERA_RESET_YEAR = 2026

# Shared form / context features (no quali leakage for pre-quali).
FEATURE_PRE_QUALI_NUM = [
    "car_pace_ewma",
    "driver_skill_ewma",
    "car_driver_combo",
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
    "dnf_hazard_car",
    "dnf_hazard_driver",
    "dnf_hazard_combined",
    "circuit_avg_finish_prior",
    "circuit_grid_to_finish_delta",
    "teammate_quali_h2h_rate",
    "round",
    "pit_stops_avg",
    "pit_time_avg_s",
    "regs_era",
    "circuit_street",
    "car_street_interaction",
    "circuit_sc_rate_prior",
    "circuit_vsc_rate_prior",
]

FEATURE_POST_QUALI_EXTRA = [
    "quali_position",
    "quali_gap_s",
    "quali_gap_to_median_s",
    "grid",
    "grid_vs_quali",
    "best_quali_seconds",
    "quali_vs_teammate",
    "teammate_quali_gap_s",
]

FEATURE_POST_SPRINT_EXTRA = ["sprint_position"]

FEATURE_CAT: list[str] = []

TRAIN_SEASONS = {2022, 2023, 2024, 2025, 2026}
TEST_SEASONS: set[int] = set()
EVAL_YEAR = 2026


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
    if mode in ("post_quali", "post_sprint"):
        numeric = numeric + FEATURE_POST_QUALI_EXTRA
    if mode == "post_sprint":
        numeric = numeric + FEATURE_POST_SPRINT_EXTRA
    return numeric, list(FEATURE_CAT)


def all_numeric_columns() -> list[str]:
    return FEATURE_PRE_QUALI_NUM + FEATURE_POST_QUALI_EXTRA + FEATURE_POST_SPRINT_EXTRA
