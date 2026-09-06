"""Jolpica/Ergast adapter — maps Jolpica API responses to canonical raw parquets.

The existing ingest pipeline in f1ml.ingest writes vendor-shaped tables to
data/raw/*.parquet. This adapter documents the mapping and can be extended to
emit CanonicalBundle directly in a future refactor.
"""

from __future__ import annotations

from f1ml.adapters.base import SourceAdapter
from f1ml.schema.entities import CanonicalBundle


class JolpicaAdapter(SourceAdapter):
    name = "jolpica"

    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        """Delegate to ingest.run(); canonical raw parquets are the v1 contract."""
        from f1ml.ingest import run as ingest_run

        ingest_run(season, season, refresh=False)
        return CanonicalBundle(extra={"note": "Jolpica data written to data/raw/ via ingest.run()"})
