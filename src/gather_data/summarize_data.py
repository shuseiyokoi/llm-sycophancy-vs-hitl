import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA


def summarize_data():
    master = pd.read_csv(
        f"{PATH_TO_DATA}preprocessed_data.csv",
        na_values=["NA", "None", "Exempt", "", " "],
    )
    numeric_cols = [
        "loan_amount",
        "loan_to_value_ratio",
        "debt_to_income_ratio",
        "property_value",
        "income",
    ]

    for col in numeric_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    summary = (
        master.groupby(["derived_race", "derived_sex", "derived_ethnicity"])
        .agg(
            total_apps=("action_taken", "count"),
            avg_loan_amount=("loan_amount", "mean"),
            std_loan_amount=("loan_amount", "std"),
            Q1_loan_amount=("loan_amount", lambda x: np.percentile(x, 25)),
            Q3_loan_amount=("loan_amount", lambda x: np.percentile(x, 75)),
            avg_loan_to_value_ratio=("loan_to_value_ratio", "mean"),
            std_loan_to_value_ratio=("loan_to_value_ratio", "std"),
            Q1_loan_to_value_ratio=(
                "loan_to_value_ratio",
                lambda x: np.percentile(x, 25),
            ),
            Q3_loan_to_value_ratio=(
                "loan_to_value_ratio",
                lambda x: np.percentile(x, 75),
            ),
            avg_debt_to_income_ratio=("debt_to_income_ratio", "mean"),
            std_debt_to_income_ratio=("debt_to_income_ratio", "std"),
            Q1_debt_to_income_ratio=(
                "debt_to_income_ratio",
                lambda x: np.percentile(x, 25),
            ),
            Q3_debt_to_income_ratio=(
                "debt_to_income_ratio",
                lambda x: np.percentile(x, 75),
            ),
            avg_property_value=("property_value", "mean"),
            std_property_value=("property_value", "std"),
            Q1_property_value=("property_value", lambda x: np.percentile(x, 25)),
            Q3_property_value=("property_value", lambda x: np.percentile(x, 75)),
            avg_income=("income", "mean"),
            std_income=("income", "std"),
            Q1_income=("income", lambda x: np.percentile(x, 25)),
            Q3_income=("income", lambda x: np.percentile(x, 75)),
            accepted=("action_taken", lambda x: (x == "Loan originated").sum()),
            denied=("action_taken", lambda x: (x == "Application denied").sum()),
        )
        .reset_index()
    )

    summary["iqr_loan_amount"] = summary["Q3_loan_amount"] - summary["Q1_loan_amount"]
    summary["iqr_property_value"] = (
        summary["Q3_property_value"] - summary["Q1_property_value"]
    )
    summary["iqr_income"] = summary["Q3_income"] - summary["Q1_income"]

    summary["acceptance_rate"] = summary["accepted"] / summary["total_apps"]
    summary["acceptance_rate_pct"] = summary["acceptance_rate"] * 100

    with open(f"{PATH_TO_DATA}summary.txt", "w") as f:
        f.write(summary.to_string(index=False))

    print("Summary saved as summary.txt")


if __name__ == "__main__":
    summarize_data()
