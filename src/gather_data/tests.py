import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from load_data import load_data
from summarize_data_derived_ethnicity import summarize_data

try:
    load_data()
    summarize_data()
except Exception as e:
    print(f"Error occurred while loading or summarizing data: {e}")
