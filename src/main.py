"""Single entry point that dispatches to the per-stage scripts.

Each stage lives in its own folder (call_models/, hitl/) and can also be run
directly from there.
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for stage in ("call_models", "hitl"):
    sys.path.insert(0, os.path.join(SRC_DIR, stage))

from call_models import call_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--call-models", action="store_true", help="Run single-turn cloud model API calls"
    )
    parser.add_argument(
        "--hitl", action="store_true", help="Run the multi-turn HITL sycophancy experiment (cloud models)"
    )
    parser.add_argument(
        "--analyze-hitl", action="store_true", help="Analyze saved HITL results"
    )
    args = parser.parse_args()

    if args.call_models:
        call_models()

    if args.hitl:
        from run_hitl import main as run_hitl_main

        sys.argv = [sys.argv[0]]
        run_hitl_main()

    if args.analyze_hitl:
        from analyze_hitl import main as analyze_hitl_main

        try:
            analyze_hitl_main()
        except Exception as e:
            print(f"Error occurred while analyzing HITL results: {e}")


if __name__ == "__main__":
    main()
