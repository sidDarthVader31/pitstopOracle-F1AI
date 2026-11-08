"""Rebuild event-level SC/VSC targets from already downloaded OpenF1 data."""

from f1ml.paths import RAW
from f1ml.safety_car import build_safety_car_labels
import pandas as pd


def main() -> None:
    sessions = pd.read_parquet(RAW / "openf1_race_sessions.parquet")
    control = pd.read_parquet(RAW / "openf1_race_control.parquet")
    calendar = pd.read_parquet(RAW / "races.parquet")
    labels = build_safety_car_labels(sessions, control, calendar)
    labels.to_parquet(RAW / "openf1_safety_car_labels.parquet", index=False)
    print("races", len(labels), "SC", int(labels.has_sc.sum()), "VSC", int(labels.has_vsc.sum()))


if __name__ == "__main__":
    main()
