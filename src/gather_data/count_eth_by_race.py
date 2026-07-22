import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA


def count_eth_by_race():
    master = pd.read_csv(
            f"{PATH_TO_DATA}hmda_CA_2024.csv",
            na_values=["NA", "None", "Exempt", "", " "],
        )
    combo_counts = (
    master
    .groupby(["derived_ethnicity", "derived_race"])
    .size()
    .reset_index(name="count")
    .sort_values("count")
    )

    combo_counts["percent"] = (
        combo_counts["count"] / combo_counts["count"].sum() * 100
    ).round(2)

    combo_counts.to_csv(
        f"{PATH_TO_DATA}ethnicity_race_counts.csv",
        index=False
    )

    print(combo_counts)
if __name__ == "__main__":
    count_eth_by_race()