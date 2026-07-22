"""
Statistical bias tests on the raw loan-level HMDA data.

Tests whether the application decision (denied vs approved) depends on
race, sex, and ethnicity:

1. Chi-square tests of independence: decision vs each attribute marginally
   (no controls).
2. Logistic regression with financial controls (same model as the ground
   truth labeling in run_regression.py):
       denied ~ race + sex + ethnicity + dti + ltv + income
                + loan_amount + property_value

Reuses load_and_clean / run_regression / extract_ground_truth_labels from
run_regression.py so the model definition stays in one place.
"""

import sys
import os
import argparse
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA, PATH_TO_GROUND_TRUTH
from ground_truth.run_regression import (
    load_and_clean,
    run_regression,
    extract_ground_truth_labels,
)


def run_chi_square_tests(df):
    """Marginal chi-square test of decision vs each sensitive attribute."""
    results = []
    for attr in ["race", "sex", "ethnicity"]:
        table = pd.crosstab(df[attr], df["denied"])
        chi2, p, dof, _ = chi2_contingency(table.values)
        results.append(
            {
                "attribute": attr,
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "significant": p < 0.05,
            }
        )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(PATH_TO_DATA, "preprocessed_data.csv"),
        help="Path to loan-level data (default: data/gather_data/preprocessed_data.csv)",
    )
    args = parser.parse_args()

    os.makedirs(PATH_TO_GROUND_TRUTH, exist_ok=True)

    df = load_and_clean(args.input)
    print(f"Rows after cleaning: {len(df):,}")

    chi_df = run_chi_square_tests(df)
    print("\nChi-square tests (decision vs attribute):")
    print(chi_df.to_string(index=False))
    chi_df.to_csv(
        os.path.join(PATH_TO_GROUND_TRUTH, "chi_square_tests.csv"), index=False
    )

    model = run_regression(df)
    print()
    print(model.summary())

    labels = extract_ground_truth_labels(model)

    with open(
        os.path.join(PATH_TO_GROUND_TRUTH, "statistical_tests_summary.txt"), "w"
    ) as f:
        f.write(f"Input: {args.input}\n")
        f.write(f"Rows after cleaning: {len(df):,}\n\n")
        f.write("Chi-square tests (decision vs attribute):\n")
        f.write(chi_df.to_string(index=False))
        f.write("\n\n")
        f.write(model.summary().as_text())
        f.write("\n\nBias labels (sensitive attributes):\n")
        f.write(labels.to_string(index=False))

    labels_path = os.path.join(PATH_TO_GROUND_TRUTH, "bias_test_labels.csv")
    labels.to_csv(labels_path, index=False)
    print(f"\nBias labels written to {labels_path}")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
