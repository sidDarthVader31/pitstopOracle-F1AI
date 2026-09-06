"""Manual CSV overrides for starting grid and fantasy prices."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1ml.adapters.base import SourceAdapter
from f1ml.paths import MANUAL
from f1ml.schema.entities import CanonicalBundle, StartingGrid


class ManualAdapter(SourceAdapter):
    name = "manual"

    def __init__(self, manual_dir: Path | None = None) -> None:
        self.manual_dir = manual_dir or MANUAL

    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        bundle = CanonicalBundle()
        grid_path = self.manual_dir / "starting_grid.csv"
        if not grid_path.exists():
            return bundle
        df = pd.read_csv(grid_path, comment="#")
        df = df[df["season"] == season]
        if round_num is not None:
            df = df[df["round"] == round_num]
        for row in df.itertuples(index=False):
            bundle.starting_grids.append(
                StartingGrid(
                    season=int(row.season),
                    round=int(row.round),
                    driver_id=str(row.driver_id),
                    grid_position=int(row.grid_position),
                    quali_position=int(row.quali_position) if pd.notna(getattr(row, "quali_position", None)) else None,
                    source=str(getattr(row, "source", "manual")),
                    as_of=str(getattr(row, "as_of", datetime.now(timezone.utc).isoformat())),
                )
            )
        return bundle

    def load_fantasy_prices(self, season: int, round_num: int) -> pd.DataFrame:
        path = self.manual_dir / "fantasy_prices.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, comment="#")
        mask = (df["season"] == season) & (df["round"] == round_num)
        return df[mask].copy()
