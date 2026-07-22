import os
import sys

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DATA


def load_data():
    os.makedirs(PATH_TO_DATA, exist_ok=True)
    file_path = f"{PATH_TO_DATA}/hmda_CA_2024.csv"

    if os.path.exists(file_path):
        print("hmda_CA_2024.csv already exists")
        return

    url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    params = {"states": "CA", "years": "2024", "action_taken": "1,2,3"}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/octet-stream,*/*",
        "Referer": "https://ffiec.cfpb.gov/data-browser/",
    }

    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print("Saved as hmda_CA_2024.csv")


if __name__ == "__main__":
    load_data()
