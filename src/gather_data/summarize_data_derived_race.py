import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA


def summarize_data():
    columns_to_keep = [
        "lei",
        "state_code",
        "county_code",
        "derived_ethnicity",
        "derived_race",
        "derived_sex",
        "action_taken",
        "loan_amount",
        "loan_term",
        "property_value",
        "income",
        "applicant_age",
    ]

    master = pd.read_csv(
        f"{PATH_TO_DATA}hmda_CA_2024.csv",
        na_values=["NA", "None", "Exempt", "", " "],
    )

    master = master[columns_to_keep]

    numeric_cols = [
        "loan_amount",
        "loan_term",
        "property_value",
        "income",
    ]

    for col in numeric_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    master = master.dropna(subset=columns_to_keep)

    action_map = {
        1: "Loan originated",
        2: "Application approved but not accepted",
        3: "Application denied",
    }

    master["action_taken"] = master["action_taken"].map(action_map)

    # Keep only the race groups you want
    race_groups = [
        "American Indian or Alaska Native",
        "Asian",
        "Black or African American",
        "Native Hawaiian or Other Pacific Islander",
        "White",
        "2 or more minority races",
        "Joint",
        "Free Form Text Only",
        "Race Not Available",
    ]
    master = master[master["derived_race"].isin(race_groups)]

    summary = (
        master.groupby(["derived_race", "derived_sex"])
        .agg(
            total_apps=("action_taken", "count"),
            avg_loan_amount=("loan_amount", "mean"),
            std_loan_amount=("loan_amount", "std"),
            Q1_loan_amount=("loan_amount", lambda x: np.percentile(x, 25)),
            Q3_loan_amount=("loan_amount", lambda x: np.percentile(x, 75)),
            avg_loan_term=("loan_term", "mean"),
            std_loan_term=("loan_term", "std"),
            Q1_loan_term=("loan_term", lambda x: np.percentile(x, 25)),
            Q3_loan_term=("loan_term", lambda x: np.percentile(x, 75)),
            avg_property_value=("property_value", "mean"),
            std_property_value=("property_value", "std"),
            Q1_property_value=("property_value", lambda x: np.percentile(x, 25)),
            Q3_property_value=("property_value", lambda x: np.percentile(x, 75)),
            avg_income=("income", "mean"),
            std_income=("income", "std"),
            Q1_income=("income", lambda x: np.percentile(x, 25)),
            Q3_income=("income", lambda x: np.percentile(x, 75)),
            accepted=("action_taken", lambda x: (x == "Loan originated").sum()),
            approved_not_accepted=(
                "action_taken",
                lambda x: (x == "Application approved but not accepted").sum(),
            ),
            denied=("action_taken", lambda x: (x == "Application denied").sum()),
        )
        .reset_index()
        .rename(columns={"derived_race": "derived_race"})
    )

    summary["iqr_loan_amount"] = summary["Q3_loan_amount"] - summary["Q1_loan_amount"]
    summary["iqr_loan_term"] = summary["Q3_loan_term"] - summary["Q1_loan_term"]
    summary["iqr_property_value"] = (
        summary["Q3_property_value"] - summary["Q1_property_value"]
    )
    summary["iqr_income"] = summary["Q3_income"] - summary["Q1_income"]

    summary["acceptance_rate"] = summary["accepted"] / summary["total_apps"]
    summary["acceptance_rate_pct"] = summary["acceptance_rate"] * 100

    with open(f"{PATH_TO_DATA}summary.txt", "w") as f:
        f.write(summary.to_string(index=False))

    # Optional CSV output
    # summary.to_csv(f"{PATH_TO_DATA}summary.csv", index=False)

    print("Summary saved as summary.txt")


if __name__ == "__main__":
    summarize_data()
