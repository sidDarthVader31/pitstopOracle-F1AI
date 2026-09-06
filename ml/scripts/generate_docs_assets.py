#!/usr/bin/env python3
"""Export Plotly charts and report images for docs/README screenshots."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DOCS_ASSETS = REPO / "docs" / "assets"
SCREENSHOTS = DOCS_ASSETS / "screenshots"

sys.path.insert(0, str(ROOT))

from f1ml.predict import (  # noqa: E402
  apply_what_if,
  load_metrics,
  resolve_race_features,
  score_race,
)
from f1ml.ui.charts import accuracy_bars, compare_scenarios, podium_strip, win_prob_bar  # noqa: E402


def _save(fig, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.write_image(str(path), scale=2, width=1000, height=520)
  print(f"wrote {path.relative_to(REPO)}")


def export_this_weekend() -> None:
  race_df, _ = resolve_race_features(2026, 13)
  scored = score_race(race_df)
  fig = win_prob_bar(scored, top_n=10)
  fig.update_layout(title="This Weekend — 2026 Italian GP (Monza)")
  _save(fig, SCREENSHOTS / "this-weekend.png")


def export_race_explorer() -> None:
  race_df, _ = resolve_race_features(2025, 16)
  scored = score_race(race_df)
  fig = win_prob_bar(scored, top_n=10)
  fig.update_layout(title="Race Explorer — 2025 Italian GP (actual results)")
  _save(fig, SCREENSHOTS / "race-explorer.png")


def export_fantasy_lab() -> None:
  race_df, _ = resolve_race_features(2026, 13)
  baseline = score_race(race_df)
  wet = apply_what_if(race_df, is_wet=True)
  scenario = score_race(wet)
  fig = compare_scenarios(baseline, scenario, top_n=8)
  fig.update_layout(title="Fantasy Lab — dry vs wet forecast")
  _save(fig, SCREENSHOTS / "fantasy-lab.png")


def export_model_performance() -> None:
  metrics = load_metrics()
  fig = accuracy_bars(metrics, mode="post_quali")
  _save(fig, SCREENSHOTS / "model-performance.png")


def copy_report_charts() -> None:
  src = ROOT / "reports" / "pole_win_rate.png"
  if src.exists():
    dest = DOCS_ASSETS / "pole-win-rate.png"
    shutil.copy2(src, dest)
    print(f"wrote {dest.relative_to(REPO)}")


def main() -> None:
  export_this_weekend()
  export_race_explorer()
  export_fantasy_lab()
  export_model_performance()
  copy_report_charts()
  print("Done — docs assets updated.")


if __name__ == "__main__":
  main()
