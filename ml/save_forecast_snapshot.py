"""Generate and persist one immutable race forecast snapshot."""

import argparse
from pathlib import Path

from f1ml.paths import MODELS
from f1ml.predict import resolve_race_features
from f1ml.ranking import WeekendRanker
from f1ml.lifecycle import persist_forecast_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=["pre_quali", "post_quali_pre_sprint", "post_sprint"],
        required=True,
    )
    parser.add_argument("--model-version", default="champion")
    parser.add_argument("--model-dir", default=str(MODELS))
    args = parser.parse_args()

    features, _race_mode = resolve_race_features(args.season, args.round)
    if features.get("won", 0).max() == 1:
        raise ValueError("snapshots are only for races without a final result")
    if args.stage == "pre_quali" and features["quali_position"].notna().any():
        raise ValueError("pre_quali snapshot requested after qualifying data is present")
    if args.stage == "post_quali_pre_sprint":
        if features["quali_position"].isna().all():
            raise ValueError("post-quali snapshot requires qualifying results")
        grid = features["grid"] if "grid" in features else features.get("grid_position")
        if grid is None or grid.notna().sum() == 0:
            raise ValueError("post-quali snapshot requires the official starting grid")
        if features["sprint_position"].fillna(-1).gt(0).any():
            raise ValueError("use post_sprint after sprint results are available")
    if args.stage == "post_sprint" and not features["sprint_position"].fillna(-1).gt(0).any():
        raise ValueError("post-sprint snapshot requires completed sprint results")

    model_mode = {
        "pre_quali": "pre_quali",
        "post_quali_pre_sprint": "post_quali",
        "post_sprint": "post_sprint",
    }[args.stage]
    bundle = WeekendRanker.load(Path(args.model_dir) / model_mode)
    scored = bundle.predict_frame(features)
    output = persist_forecast_snapshot(
        features,
        scored,
        args.stage,
        args.model_version,
    )
    print(output)


if __name__ == "__main__":
    main()
