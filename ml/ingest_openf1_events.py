"""Fetch historical OpenF1 event metadata and race-control messages."""

import argparse

from f1ml.adapters.openf1_events import save_openf1_event_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    counts = save_openf1_event_data(list(range(args.start_year, args.end_year + 1)))
    print(counts)


if __name__ == "__main__":
    main()
