"""Promote an evaluated candidate only when saved evidence passes the gate."""

import argparse
from pathlib import Path

from f1ml.lifecycle import promote_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    record = promote_candidate(args.version, args.evidence)
    print(record)


if __name__ == "__main__":
    main()
