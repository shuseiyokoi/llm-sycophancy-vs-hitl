import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA


def preprocess_data():
    summary_cols = [
        "loan_purpose",
        "loan_amount",
        "debt_to_income_ratio",
        "property_value",  # There is outliers in this column, but we will keep it for now 999995000.0
        "income",
        "occupancy_type",
        "derived_ethnicity",
        "derived_race",
        "derived_sex",
        "applicant_age",
        "loan_to_value_ratio",
    ]

    filter_cols = [
        "derived_loan_product_type",
        "conforming_loan_limit",
        "business_or_commercial_purpose",
        "initially_payable_to_institution",
        "action_taken",
        "loan_purpose",
    ]

    columns_to_read = summary_cols + filter_cols

    master = pd.read_csv(
        f"{PATH_TO_DATA}hmda_CA_2024.csv",
        usecols=columns_to_read,
        na_values=["NA", "None", "Exempt", "", " "],
    )

    master = master[
        (master["derived_loan_product_type"] == "Conventional:First Lien")
        & (master["conforming_loan_limit"] == "C")
        & (master["business_or_commercial_purpose"] == 2)
        & (master["initially_payable_to_institution"] == 1)
        & (master["occupancy_type"] == 1)
        & (master["action_taken"].isin([1, 2, 3]))
        & (~master["applicant_age"].isin(["8888", "9999"]))
        & (
            ~master["derived_ethnicity"].isin(
                ["Joint", "Ethnicity Not Available", "Free Form Text Only"]
            )
        )
        & (
            ~master["derived_race"].isin(
                [
                    "Joint",
                    "Free Form Text Only",
                    "Race Not Available",
                ]
            )
        )
        & (~master["derived_sex"].isin(["Joint", "Sex Not Available"]))
        & (master["loan_purpose"] == 1)
        & (master["property_value"] < 999995000.0)
    ]

    master = master.dropna()

    # Keep only the columns you want after filtering
    master = master[summary_cols + ["action_taken"]]

    action_map = {
        1: "Loan originated",
        2: "Application approved but not accepted",
        3: "Application denied",
    }
    master["action_taken"] = master["action_taken"].apply(
        lambda x: 1 if x in [1, 2] else 3
    )
    master["action_taken"] = master["action_taken"].map(action_map)

    master.to_csv(f"{PATH_TO_DATA}preprocessed_data.csv", index=False)

    print("Preprocessed data saved as preprocessed_data.csv")


if __name__ == "__main__":
    preprocess_data()
