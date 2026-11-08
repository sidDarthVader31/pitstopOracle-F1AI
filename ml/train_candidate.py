"""Train a versioned candidate without replacing champion model files."""

import argparse

from f1ml.lifecycle import train_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="New candidate version label")
    args = parser.parse_args()
    print(train_candidate(args.version))


if __name__ == "__main__":
    main()
