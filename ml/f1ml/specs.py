"""Feature specifications for fixed pre- and post-qualifying cutoffs.

Current-weekend practice, weather and sprint inputs stay out of the model until
their historical availability at the forecast cutoff can be reproduced.
"""

from __future__ import annotations

import os
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
    "car_race_pace_ewma",
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

# EXP-003 track-aware features. Set F1ML_TRACK_FEATURES=0 to evaluate without them.
FEATURE_TRACK_PRE = [
    "track_high_speed",
    "track_low_speed",
    "circuit_grid_importance",
    "car_quali_class_ewma",
    "car_class_delta",
    "driver_race_vs_teammate_ewma",
]
FEATURE_TRACK_POST = [
    "quali_gap_pct",
    "grid_x_grid_importance",
    "quali_gap_x_grid_importance",
]


# F1 common sense the pace model must respect (+1: higher value -> better result,
# -1: higher value -> worse result, unlisted: free). Stops the model learning
# backwards patterns from a small dataset, e.g. "more points -> slower".
MONOTONIC_DIRECTION = {
    "car_pace_ewma": -1,
    "driver_skill_ewma": -1,
    "car_driver_combo": -1,
    "car_race_pace_ewma": -1,
    "car_pace_legacy_ewma": -1,
    "champ_position_before": -1,
    "champ_points_before": 1,
    "champ_wins_before": 1,
    "driver_id_points_last_3": 1,
    "driver_id_points_last_5": 1,
    "driver_id_avg_finish_last_5": -1,
    "constructor_id_points_last_3": 1,
    "constructor_id_points_last_5": 1,
    "constructor_id_avg_finish_last_5": -1,
    "circuit_avg_finish_prior": -1,
    "teammate_quali_h2h_rate": 1,
    "car_quali_class_ewma": -1,
    "car_class_delta": -1,
    "driver_race_vs_teammate_ewma": -1,
    "quali_position": -1,
    "quali_gap_s": -1,
    "quali_gap_to_median_s": -1,
    "quali_gap_pct": -1,
    "grid": -1,
    "quali_vs_teammate": -1,
    "teammate_quali_gap_s": -1,
    "grid_x_grid_importance": -1,
    "quali_gap_x_grid_importance": -1,
    "sprint_position": -1,
}


def track_features_enabled() -> bool:
    return os.environ.get("F1ML_TRACK_FEATURES", "1") != "0"

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
    if os.environ.get("F1ML_CAR_PACE") == "legacy":
        numeric = ["car_pace_legacy_ewma" if c == "car_pace_ewma" else c for c in numeric]
    track = track_features_enabled()
    if track:
        numeric = numeric + FEATURE_TRACK_PRE
    if mode in ("post_quali", "post_sprint"):
        numeric = numeric + FEATURE_POST_QUALI_EXTRA
        if track:
            numeric = numeric + FEATURE_TRACK_POST
    if mode == "post_sprint":
        numeric = numeric + FEATURE_POST_SPRINT_EXTRA
    dropped = {c for c in os.environ.get("F1ML_DROP_FEATURES", "").split(",") if c}
    numeric = [c for c in numeric if c not in dropped]
    return numeric, list(FEATURE_CAT)


def all_numeric_columns() -> list[str]:
    return (
        FEATURE_PRE_QUALI_NUM
        + FEATURE_TRACK_PRE
        + FEATURE_POST_QUALI_EXTRA
        + FEATURE_TRACK_POST
        + FEATURE_POST_SPRINT_EXTRA
        + ["car_pace_legacy_ewma"]
    )
