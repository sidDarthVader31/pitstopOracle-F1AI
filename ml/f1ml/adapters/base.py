"""Adapter protocol for canonical data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from f1ml.schema.entities import CanonicalBundle

AdapterName = Literal["jolpica", "openf1", "open_meteo", "manual"]


class SourceAdapter(ABC):
    """Pull vendor data and return a CanonicalBundle."""

    name: AdapterName

    @abstractmethod
    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        """Fetch data for a season (all rounds) or a single round."""

    def write_canonical(self, bundle: CanonicalBundle) -> None:
        """Optional hook to persist canonical tables. Override in subclasses."""
