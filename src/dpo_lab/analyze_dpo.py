# analyze_dpo.py
#
# Stage 4 of the DPO virtual lab: the dose-response picture.
#
# Reads every data/dpo_lab/eval/eval_{tag}.jsonl (tags: "base" or
# "{arm}-{pct}pct") and produces, in results/dpo_lab/:
#
#   dpo_flip_rates.csv         accuracy + flip rate per checkpoint
#   flip-rate-dose-response.png  flip rate vs. preference-data fraction,
#                                one line per arm, base model at x=0
#   accuracy-dose-response.png   capability check: did training hurt accuracy?
#   dpo_stats.csv              Fisher exact test of each checkpoint vs base

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DPO_DATA, PATH_TO_DPO_RESULTS

EVAL_DIR = os.path.join(PATH_TO_DPO_DATA, "eval")
TAG_PATTERN = re.compile(r"^(?P<arm>human|ai)-(?P<pct>\d+)pct$")


def parse_tag(tag):
    match = TAG_PATTERN.match(tag)
    if match:
        return match.group("arm"), int(match.group("pct")) / 100
    return "base", 0.0


def load_evals():
    rows = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "eval_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df):
    rows = []
    for tag, group in df.groupby("tag"):
        arm, fraction = parse_tag(tag)
        pushed = group[group["first_correct"]]
        rows.append(
            {
                "tag": tag,
                "arm": arm,
                "fraction": fraction,
                "n_questions": len(group),
                "n_correct": len(pushed),
                "accuracy": len(pushed) / len(group) if len(group) else float("nan"),
                "n_flipped": int(pushed["flipped"].sum()) if len(pushed) else 0,
                "flip_rate": pushed["flipped"].mean() if len(pushed) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "fraction"]).reset_index(drop=True)


def stats_vs_base(summary):
    base = summary[summary["arm"] == "base"]
    if base.empty:
        return pd.DataFrame()
    base = base.iloc[0]

    rows = []
    for _, row in summary[summary["arm"] != "base"].iterrows():
        table = [
            [row["n_flipped"], row["n_correct"] - row["n_flipped"]],
            [base["n_flipped"], base["n_correct"] - base["n_flipped"]],
        ]
        odds_ratio, p_value = fisher_exact(table)
        rows.append({"tag": row["tag"], "vs": "base", "odds_ratio": odds_ratio, "p_value": p_value})
    return pd.DataFrame(rows)


def plot_dose_response(summary, value_col, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = summary[summary["arm"] == "base"]

    for arm, group in summary[summary["arm"] != "base"].groupby("arm"):
        group = group.sort_values("fraction")
        x = list(group["fraction"])
        y = list(group[value_col])
        if not base.empty:  # base model = zero preference data, shared origin
            x = [0.0] + x
            y = [base.iloc[0][value_col]] + y
        ax.plot(x, y, marker="o", label=f"{arm} feedback")

    if not base.empty and summary[summary["arm"] != "base"].empty:
        ax.scatter([0.0], [base.iloc[0][value_col]], label="base")

    ax.set_xlabel("Fraction of preference data used for DPO")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. amount of preference training")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PATH_TO_DPO_RESULTS, filename), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PATH_TO_DPO_RESULTS, exist_ok=True)

    df = load_evals()
    if df.empty:
        print(f"No eval files in {EVAL_DIR} — run eval_sycophancy.py first.")
        return

    summary = summarize(df)
    print(summary.to_string(index=False), "\n")
    summary.to_csv(os.path.join(PATH_TO_DPO_RESULTS, "dpo_flip_rates.csv"), index=False)

    stats = stats_vs_base(summary)
    if not stats.empty:
        print("Fisher exact tests (flips, each checkpoint vs base):")
        print(stats.to_string(index=False), "\n")
        stats.to_csv(os.path.join(PATH_TO_DPO_RESULTS, "dpo_stats.csv"), index=False)

    plot_dose_response(summary, "flip_rate", "Sycophantic flip rate", "flip-rate-dose-response.png")
    plot_dose_response(summary, "accuracy", "GSM8K accuracy", "accuracy-dose-response.png")

    print(f"Saved plots and CSVs to {PATH_TO_DPO_RESULTS}")


if __name__ == "__main__":
    main()
