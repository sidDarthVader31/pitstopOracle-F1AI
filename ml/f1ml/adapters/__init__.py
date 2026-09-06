"""Pluggable data-source adapters."""

from f1ml.adapters.base import SourceAdapter
from f1ml.adapters.manual import ManualAdapter
from f1ml.adapters.openf1 import OpenF1Adapter
from f1ml.adapters.openmeteo import OpenMeteoForecastAdapter

__all__ = [
    "ManualAdapter",
    "OpenF1Adapter",
    "OpenMeteoForecastAdapter",
    "SourceAdapter",
]
