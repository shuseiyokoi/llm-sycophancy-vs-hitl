"""
Ground truth bias labeling via logistic regression on HMDA loan-level data.

Model: denied ~ race + sex + ethnicity + debt_to_income_ratio + loan_to_value_ratio
                + income + loan_amount + property_value
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA, PATH_TO_GROUND_TRUTH


def load_and_clean(path):
    df = pd.read_csv(path)

    rename_map = {
        "derived_race": "race",
        "derived_sex": "sex",
        "derived_ethnicity": "ethnicity",
        "debt_to_income_ratio": "dti",
        "loan_to_value_ratio": "ltv",
        "income": "income",
        "loan_amount": "loan_amount",
        "property_value": "property_value",
        "action_taken": "action_taken",
        "applicant_age": "applicant_age",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Build binary outcome from text action_taken values.
    # Keep only originated/denied — drop withdrawn/incomplete/purchased/etc.
    keep_actions = {
        "Loan originated": 0,
        "Application approved but not accepted": 0,
        "Application denied": 1,
    }
    df = df[df["action_taken"].isin(keep_actions.keys())].copy()
    df["denied"] = df["action_taken"].map(keep_actions)

    # DTI comes as either a clean number or a bucketed string like "30%-<36%" or "<20%"
    # or ">60%". Convert buckets to their midpoint so they can be used numerically.
    def dti_to_numeric(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace("%", "")
        if "-<" in s:  # e.g. "30-<36"
            low, high = s.split("-<")
            return (float(low) + float(high)) / 2
        if "-" in s and not s.startswith("-"):  # e.g. "50-60" (plain hyphen bucket)
            low, high = s.split("-")
            return (float(low) + float(high)) / 2
        if s.startswith("<"):  # e.g. "<20"
            return float(s[1:]) - 2.5
        if s.startswith(">"):  # e.g. ">60"
            return float(s[1:]) + 2.5
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["dti"] = df["dti"].apply(dti_to_numeric)

    for col in ["ltv", "income", "loan_amount", "property_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        "race",
        "sex",
        "ethnicity",
        "dti",
        "ltv",
        "income",
        "loan_amount",
        "property_value",
        "denied",
    ]
    df = df.dropna(subset=required)

    df = df[~df["race"].isin(["Race Not Available", "Free Form Text Only"])]
    df = df[~df["sex"].isin(["Sex Not Available"])]
    df = df[~df["ethnicity"].isin(["Ethnicity Not Available"])]

    return df


def run_regression(
    df, race_ref="White", sex_ref="Male", eth_ref="Not Hispanic or Latino"
):
    df = df.copy()
    df["race"] = pd.Categorical(df["race"])
    df["sex"] = pd.Categorical(df["sex"])
    df["ethnicity"] = pd.Categorical(df["ethnicity"])

    df["race"] = df["race"].cat.reorder_categories(
        [race_ref] + [c for c in df["race"].cat.categories if c != race_ref]
    )
    df["sex"] = df["sex"].cat.reorder_categories(
        [sex_ref] + [c for c in df["sex"].cat.categories if c != sex_ref]
    )
    df["ethnicity"] = df["ethnicity"].cat.reorder_categories(
        [eth_ref] + [c for c in df["ethnicity"].cat.categories if c != eth_ref]
    )

    formula = (
        "denied ~ C(race) + C(sex) + C(ethnicity) "
        "+ dti + ltv + income + loan_amount + property_value"
    )
    model = smf.logit(formula, data=df).fit(disp=0)
    return model


def extract_ground_truth_labels(model, alpha=0.05):
    summary = model.summary2().tables[1]
    summary = summary.reset_index().rename(columns={"index": "term"})

    sensitive_rows = summary[
        summary["term"].str.startswith("C(race)")
        | summary["term"].str.startswith("C(sex)")
        | summary["term"].str.startswith("C(ethnicity)")
    ].copy()

    sensitive_rows["odds_ratio"] = np.exp(sensitive_rows["Coef."])
    sensitive_rows["significant"] = sensitive_rows["P>|z|"] < alpha
    sensitive_rows["adverse"] = sensitive_rows["Coef."] > 0
    sensitive_rows["ground_truth_label"] = np.where(
        sensitive_rows["significant"] & sensitive_rows["adverse"],
        "BIAS",
        np.where(
            sensitive_rows["significant"] & ~sensitive_rows["adverse"],
            "FAVORED",
            "NO_BIAS",
        ),
    )

    return sensitive_rows[
        [
            "term",
            "Coef.",
            "P>|z|",
            "odds_ratio",
            "significant",
            "adverse",
            "ground_truth_label",
        ]
    ]


def main():
    data_path = os.path.join(PATH_TO_DATA, "preprocessed_data.csv")
    results_path = os.path.join(PATH_TO_GROUND_TRUTH, "ground_truth_labels.csv")

    os.makedirs(PATH_TO_GROUND_TRUTH, exist_ok=True)

    df = load_and_clean(data_path)
    print(f"Rows after cleaning: {len(df):,}")

    model = run_regression(df)
    print(model.summary())

    with open(os.path.join(PATH_TO_GROUND_TRUTH, "regression_summary.txt"), "w") as f:
        f.write(model.summary().as_text())

    full_summary = (
        model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    )
    full_summary.to_csv(
        os.path.join(PATH_TO_GROUND_TRUTH, "full_regression_coefficients.csv"),
        index=False,
    )

    labels = extract_ground_truth_labels(model)
    labels.to_csv(results_path, index=False)
    print(f"\nGround truth labels written to {results_path}")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
