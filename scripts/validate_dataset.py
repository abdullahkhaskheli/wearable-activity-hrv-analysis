"""Validate installed recordings; returns exit code 2 when unavailable/invalid."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_hrv.config import load_config
from wearable_hrv.validation import print_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        return 0 if print_validation(load_config(args.config).path("dataset")) else 2
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
