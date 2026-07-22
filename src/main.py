"""Single entry point that dispatches to the per-stage scripts.

Each stage lives in its own folder (gather_data/, call_models/,
analyze_results/, ground_truth/) and can also be run directly from there.
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for stage in ("gather_data", "call_models", "analyze_results", "ground_truth"):
    sys.path.insert(0, os.path.join(SRC_DIR, stage))

from gather_data import gather_data
from call_models import call_models
from analyze_results import analyze_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gather-data", action="store_true", help="Load and summarize data"
    )
    parser.add_argument(
        "--call-models", action="store_true", help="Run cloud model API calls"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze saved model results"
    )
    args = parser.parse_args()

    if args.gather_data:
        try:
            gather_data()
        except Exception as e:
            print(f"Error occurred while gathering data: {e}")

    if args.call_models:
        call_models()

    if args.analyze:
        try:
            analyze_results()
        except Exception as e:
            print(f"Error occurred while analyzing results: {e}")


if __name__ == "__main__":
    main()
