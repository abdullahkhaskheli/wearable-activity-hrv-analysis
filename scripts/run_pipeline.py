"""Run physiological analysis only on installed official recordings."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_hrv.config import load_config
from wearable_hrv.validation import print_validation
from wearable_hrv.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-subjects", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_config(args.config)
        if not print_validation(config.path("dataset")):
            print(
                "Dataset unavailable or invalid. Install PPG-DaLiA first. See data/README.md."
            )
            return 2
        run_pipeline(config, args.max_subjects)
        return 0
    except (ValueError, OSError) as exc:
        logging.error("Analysis not completed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
