import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from load_data import load_data
from preprocess_data import preprocess_data
from summarize_data import summarize_data


def gather_data():
    load_data()
    preprocess_data()
    summarize_data()


if __name__ == "__main__":
    try:
        gather_data()
    except Exception as e:
        print(f"Error occurred while loading or summarizing data: {e}")
