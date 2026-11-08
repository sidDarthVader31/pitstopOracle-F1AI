"""Reproducible inventory and quality checks for the local race dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1ml.features import _is_classified

TABLES = {
    "raw/races.parquet": ("Jolpica calendar", "Published calendar; event-level, one row per race"),
    "raw/results.parquet": ("Jolpica results", "Final result/status; known only after the race"),
    "raw/qualifying.parquet": ("Jolpica qualifying", "Qualifying result; known after qualifying"),
    "raw/sprints.parquet": ("Jolpica sprint", "Sprint result; known only after sprint"),
    "raw/standings.parquet": ("Jolpica standings", "Championship after the named round"),
    "raw/pitstops.parquet": ("Jolpica pit stops", "Observed during/after the race; history only for future events"),
    "raw/weather.parquet": ("Open-Meteo archive", "Observed weather; not a historical forecast snapshot"),
    "canonical/fp_pace.parquet": ("OpenF1 adapter", "Session lap summaries; local table has no source timestamp"),
    "canonical/starting_grid.parquet": ("OpenF1/manual", "Grid snapshot; source and as_of may be available"),
    "canonical/weather_forecast.parquet": ("Open-Meteo forecast", "Current forecast snapshot; not a historical archive"),
    "raw/openf1_race_sessions.parquet": ("OpenF1 sessions/meetings", "Historical race session times and circuit metadata"),
    "raw/openf1_race_control.parquet": ("OpenF1 race_control", "Race-control messages with UTC timestamps"),
    "raw/openf1_safety_car_labels.parquet": ("Derived OpenF1 labels", "Race-level deployed SC/VSC indicators"),
    "processed/driver_race.parquet": ("Feature builder", "One row per driver × completed race; labels and features"),
}


def audit_dataset(data_root: Path) -> dict:
    """Return table coverage, row-grain checks and result-label counts."""
    tables = []
    frames: dict[str, pd.DataFrame] = {}
    for relative, (source, timing) in TABLES.items():
        path = data_root / relative
        if not path.exists():
            tables.append(
                {"path": relative, "source": source, "timing": timing, "exists": False}
            )
            continue
        frame = pd.read_parquet(path)
        frames[relative] = frame
        tables.append(
            {
                "path": relative,
                "source": source,
                "timing": timing,
                "exists": True,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_rate": float(frame.isna().to_numpy().mean()) if frame.size else 0.0,
            }
        )

    races = frames.get("raw/races.parquet", pd.DataFrame())
    results = frames.get("raw/results.parquet", pd.DataFrame())
    processed = frames.get("processed/driver_race.parquet", pd.DataFrame())
    status_counts = (
        results["status"].fillna("<missing>").value_counts().sort_index().to_dict()
        if "status" in results
        else {}
    )
    classified = results["status"].map(_is_classified) if "status" in results else pd.Series(dtype=bool)
    dnf_positive = int((processed["dnf"] == 1).sum()) if "dnf" in processed else None
    dnf_excluded = int(processed["dnf"].isna().sum()) if "dnf" in processed else None
    return {
        "tables": tables,
        "raw_races": len(races),
        "completed_races": int(results[["season", "round"]].drop_duplicates().shape[0])
        if {"season", "round"}.issubset(results.columns)
        else 0,
        "result_rows": len(results),
        "processed_rows": len(processed),
        "processed_races": int(processed[["season", "round"]].drop_duplicates().shape[0])
        if {"season", "round"}.issubset(processed.columns)
        else 0,
        "processed_duplicate_driver_race_keys": int(
            processed.duplicated(["season", "round", "driver_id"]).sum()
        )
        if {"season", "round", "driver_id"}.issubset(processed.columns)
        else None,
        "result_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "classified_rows": int(classified.sum()) if len(classified) else 0,
        "nonclassified_rows": int((~classified).sum()) if len(classified) else 0,
        "dnf_positive_rows": dnf_positive,
        "dnf_excluded_rows": dnf_excluded,
        "circuit_coordinate_missing_rows": int(races[["lat", "lng"]].isna().any(axis=1).sum())
        if {"lat", "lng"}.issubset(races.columns)
        else None,
    }


def render_audit_markdown(audit: dict) -> str:
    lines = [
        "# Local data inventory and quality audit",
        "",
        "Generated from the local parquet files. This records current coverage; source availability timing is not inferred from result values.",
        "",
        "## Tables",
        "",
        "| File | Source | Availability meaning | Rows | Columns | Missing cells | Missing rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for table in audit["tables"]:
        if not table["exists"]:
            lines.append(f"| `{table['path']}` | {table['source']} | {table['timing']} | missing | — | — | — |")
        else:
            lines.append(
                f"| `{table['path']}` | {table['source']} | {table['timing']} | {table['rows']} | {table['columns']} | {table['missing_cells']} | {table['missing_rate']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Row grain and labels",
            "",
            f"- Calendar races: {audit['raw_races']}; completed races represented in results: {audit['completed_races']}; processed races: {audit['processed_races']}.",
            f"- Result rows: {audit['result_rows']}; processed driver-race rows: {audit['processed_rows']}; duplicate processed `(season, round, driver_id)` keys: {audit['processed_duplicate_driver_race_keys']}.",
            f"- Classified rows under the model rule (`Finished`, `Lapped`, or `+N Lap`): {audit['classified_rows']}; non-classified proxy rows: {audit['nonclassified_rows']}.",
            f"- Retirement DNF labels: {audit['dnf_positive_rows']} positive; {audit['dnf_excluded_rows']} excluded DNS/DSQ/withdrawal labels.",
            f"- Race-calendar rows missing latitude or longitude: {audit['circuit_coordinate_missing_rows']}.",
            "",
            "Raw status is retained. The retirement DNF label treats classified finishers as 0 and retirements/non-classifications as 1; DNS, DSQ, and withdrawals are excluded as unknown/other outcomes. It is not a cause-specific mechanical failure label.",
            "",
            "## Provenance limits",
            "",
            "- The race calendar has circuit IDs, names, country, latitude and longitude, but no corner map, speed mix, elevation or pit-lane time loss.",
            "- Local OpenF1 pace contains only per-driver/session best-lap summaries and no source timestamp, sector splits or speed-trap samples.",
            "- Archived weather is observed weather. The one-row current forecast file is not a historical forecast archive.",
            "- OpenF1 meeting metadata provides a coarse circuit type; it does not expose corner-speed mix, elevation, pit-lane loss, or verified track geometry in the local table.",
            "- OpenF1 race-control records provide deployed SC/VSC event labels from 2023 onward; the event labels are useful for a cautious race-level baseline, not a deployment-time model.",
            "- Archived weather is observed weather, not a historical forecast archive; local lap summaries are not timestamped and lack sectors/speed traps.",
            "- Therefore high/low-speed car traits and weather-scenario training remain deferred; source-backed circuit-type and prior SC/VSC frequency can be tested.",
            "",
        ]
    )
    return "\n".join(lines)
