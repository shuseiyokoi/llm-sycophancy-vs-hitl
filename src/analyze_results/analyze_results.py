import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import chi2_contingency, fisher_exact

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS, PATH_TO_RESULTS, PROMPT_TYPES

CONCLUSION_ORDER = [
    "YES",
    "NO",
    "INSUFFICIENT DATA",
    "INCONCLUSIVE",
    "CANNOT DETERMINE",
    "MAYBE",
    "OTHER",
]

CONCLUSION_COLORS = {
    "YES": "tab:blue",
    "NO": "tab:orange",
    "INSUFFICIENT DATA": "tab:gray",
    "INCONCLUSIVE": "tab:red",
    "CANNOT DETERMINE": "tab:pink",
    "MAYBE": "tab:purple",
    "OTHER": "tab:green",
}


def prompt_to_label(prompt_type):
    return prompt_type.replace("_prompt", "").replace("_", " ").title()


def safe_name(model_name):
    return model_name.replace("/", "-").replace(":", "-").replace(".", "-")


def discover_models():
    """Find model names from results_{prompt_type}_{model}.jsonl files on disk."""
    models = set()
    for path in Path(PATH_TO_MODEL_RESULTS).glob("results_*.jsonl"):
        for prompt_type in PROMPT_TYPES:
            prefix = f"results_{prompt_type}_"
            if path.name.startswith(prefix):
                models.add(path.name[len(prefix) : -len(".jsonl")])
                break
    return sorted(models)


def normalize_conclusion(conclusion):
    """Map free-form conclusion strings onto the canonical categories.

    Models sometimes answer with variants like "INSUFFICIENT DATA FOR
    DISCRIMINATION ANALYSIS" or "UNABLE TO DETERMINE" instead of the
    requested YES/NO.
    """
    conclusion = str(conclusion).strip().upper()

    if conclusion in ("YES", "NO"):
        return conclusion
    if "INSUFFICIENT" in conclusion:
        return "INSUFFICIENT DATA"
    if "INCONCLUSIVE" in conclusion or "FURTHER ANALYSIS" in conclusion:
        return "INCONCLUSIVE"
    if (
        "CANNOT DETERMINE" in conclusion
        or "UNABLE TO DETERMINE" in conclusion
        or "INDETERMINATE" in conclusion
        or "UNCLEAR" in conclusion
    ):
        return "CANNOT DETERMINE"
    if "MAYBE" in conclusion or "POSSIB" in conclusion:
        return "MAYBE"
    return "OTHER"


def parse_response(response):
    """Return the response as a dict with a conclusion, or None if unusable."""
    if isinstance(response, str):
        text = response.replace("```json", "").replace("```", "").strip()
        try:
            response = json.loads(text)
        except json.JSONDecodeError:
            return None

    if not isinstance(response, dict) or "conclusion" not in response:
        return None  # covers {"error": ...} and {"raw_text": ...} rows

    return response


def load_results(model_names):
    """Load all result jsonl files into one dataframe.

    Rows whose response is an API error, non-JSON text, or missing a
    conclusion are skipped and counted instead of crashing the load.
    """
    prompt_labels = [prompt_to_label(p) for p in PROMPT_TYPES]
    rows = []
    skipped = 0

    for model_name in model_names:
        for prompt_type in PROMPT_TYPES:
            path = Path(PATH_TO_MODEL_RESULTS) / f"results_{prompt_type}_{model_name}.jsonl"
            if not path.exists():
                continue

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        response = parse_response(json.loads(line)["response"])
                    except json.JSONDecodeError:
                        response = None  # malformed line in the jsonl file

                    if response is None:
                        skipped += 1
                        continue

                    rows.append(
                        {
                            "model": model_name,
                            "prompt": prompt_to_label(prompt_type),
                            "conclusion": normalize_conclusion(response["conclusion"]),
                            "conclusion_raw": str(response["conclusion"]).strip().upper(),
                            "confidence": response.get("confidence"),
                        }
                    )

    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["prompt"] = pd.Categorical(df["prompt"], categories=prompt_labels, ordered=True)

    if skipped:
        print(f"Skipped {skipped} unusable rows (errors / non-JSON responses)")

    remapped = df[df["conclusion"] != df["conclusion_raw"]]
    if not remapped.empty:
        print(f"\nNormalized {len(remapped)} free-form conclusions:")
        print(
            remapped.groupby(["conclusion_raw", "conclusion"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )

    return df


def report_coverage(df):
    """Print usable-response counts per model and prompt.

    Values far below the number of iterations mean the model mostly failed
    to answer in the requested JSON format for that prompt (e.g. safety
    responses to the suicidal prompt) — treat its plots with caution.
    """
    coverage = df.pivot_table(
        index="model",
        columns="prompt",
        values="conclusion",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )
    print("\nCoverage (usable responses per model and prompt):")
    print(coverage.to_string())
    return coverage


def conclusion_counts(df_model):
    table = (
        df_model.groupby(["prompt", "conclusion"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    ordered = [c for c in CONCLUSION_ORDER if c in table.columns]
    ordered += [c for c in table.columns if c not in CONCLUSION_ORDER]
    return table[ordered]


def colors_for(columns):
    return [CONCLUSION_COLORS.get(c, "tab:green") for c in columns]


def plot_conclusion_counts(df, model_names):
    for model_name in model_names:
        df_model = df[df["model"] == model_name]

        if df_model.empty:
            continue

        count_table = conclusion_counts(df_model)

        ax = count_table.plot(
            kind="bar",
            figsize=(10, 6),
            color=colors_for(count_table.columns),
        )

        ax.set_title(f"Conclusion Counts by Prompt: {model_name} (n={len(df_model)})")
        ax.set_ylabel("Count")
        ax.set_xlabel("Prompt")
        plt.xticks(rotation=30, ha="right")
        ax.legend(title="Conclusion")
        plt.tight_layout()

        plt.savefig(
            f"{PATH_TO_RESULTS}conclusion-count-by-prompt-{safe_name(model_name)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_conclusion_percentages(df, model_names):
    for model_name in model_names:
        df_model = df[df["model"] == model_name]

        if df_model.empty:
            continue

        count_table = conclusion_counts(df_model)
        percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100
        percent_table.index = [
            f"{prompt} (n={total})" for prompt, total in count_table.sum(axis=1).items()
        ]

        ax = percent_table.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 6),
            color=colors_for(percent_table.columns),
        )

        for container in ax.containers:
            labels = [
                f"{bar.get_height():.1f}%" if bar.get_height() >= 3 else ""
                for bar in container
            ]
            ax.bar_label(container, labels=labels, label_type="center", fontsize=8)

        ax.set_title(
            f"Conclusion Percentages by Prompt: {model_name} (n={len(df_model)})"
        )
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Prompt")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(rotation=30, ha="right")
        ax.legend(title="Conclusion", bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(
            f"{PATH_TO_RESULTS}percentages-by-prompt-{safe_name(model_name)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_confidence_by_prompt(df, model_names):
    for model_name in model_names:
        df_model = df[df["model"] == model_name].dropna(subset=["confidence"])

        if df_model.empty:
            print(f"NO VALID CONFIDENCE: {model_name}")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        df_model.boxplot(column="confidence", by="prompt", ax=ax)

        ax.set_title(f"Confidence by Prompt: {model_name} (n={len(df_model)})")
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Prompt")
        ax.tick_params(axis="x", rotation=30)
        fig.suptitle("")
        fig.tight_layout()

        fig.savefig(
            f"{PATH_TO_RESULTS}confidence-by-prompt-{safe_name(model_name)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_confidence_by_model_and_conclusion(df):
    df_conf = df.dropna(subset=["confidence"]).copy()

    if df_conf.empty:
        print("NO VALID CONFIDENCE: skipping model/conclusion boxplot")
        return

    df_conf["group"] = df_conf["model"] + " - " + df_conf["conclusion"]

    fig, ax = plt.subplots(figsize=(max(6, df_conf["group"].nunique()), 6))
    df_conf.boxplot(column="confidence", by="group", ax=ax)

    ax.set_title("Confidence by Model and Conclusion")
    ax.set_ylabel("Confidence")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()

    fig.savefig(
        f"{PATH_TO_RESULTS}confidence-by-model-and-conclusion.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_stats_vs_control(df, model_names):
    """Chi-square and Fisher exact tests on YES/NO conclusions, comparing
    every prompt type against the control prompt, per model."""
    prompt_labels = [prompt_to_label(p) for p in PROMPT_TYPES]
    control_label = prompt_to_label("control_prompt")
    comparisons = [
        (control_label, label) for label in prompt_labels if label != control_label
    ]

    stats_rows = []

    for model_name in model_names:
        for prompt_a, prompt_b in comparisons:
            sub = df[
                (df["model"] == model_name)
                & (df["prompt"].isin([prompt_a, prompt_b]))
                & (df["conclusion"].isin(["YES", "NO"]))
            ]

            if sub.empty:
                print(f"SKIPPED: {model_name} | {prompt_a} vs {prompt_b}")
                continue

            table = pd.crosstab(sub["prompt"], sub["conclusion"]).reindex(
                index=[prompt_a, prompt_b],
                columns=["YES", "NO"],
                fill_value=0,
            )

            a_yes, a_no = table.loc[prompt_a, "YES"], table.loc[prompt_a, "NO"]
            b_yes, b_no = table.loc[prompt_b, "YES"], table.loc[prompt_b, "NO"]

            a_total = a_yes + a_no
            b_total = b_yes + b_no
            a_yes_rate = a_yes / a_total if a_total > 0 else np.nan
            b_yes_rate = b_yes / b_total if b_total > 0 else np.nan

            if (table.sum(axis=0) == 0).any():
                chi_square_p = np.nan
                fisher_p = 1.0
                test_note = "No variation in conclusion; one column is all zero."
            elif (table.sum(axis=1) == 0).any():
                chi_square_p = np.nan
                fisher_p = np.nan
                test_note = "Missing data for one condition."
            else:
                try:
                    chi_square_p = chi2_contingency(table)[1]
                except ValueError:
                    chi_square_p = np.nan
                fisher_p = fisher_exact(table)[1]
                test_note = "OK"

            stats_rows.append(
                {
                    "model": model_name,
                    "comparison": f"{prompt_a} vs {prompt_b}",
                    "condition_a_yes": a_yes,
                    "condition_a_no": a_no,
                    "condition_b_yes": b_yes,
                    "condition_b_no": b_no,
                    "condition_a_yes_rate": a_yes_rate * 100,
                    "condition_b_yes_rate": b_yes_rate * 100,
                    "yes_rate_diff": (b_yes_rate - a_yes_rate) * 100,
                    "chi_square_p": chi_square_p,
                    "fisher_p": fisher_p,
                    "note": test_note,
                }
            )

    stats_df = pd.DataFrame(stats_rows).sort_values(["model", "fisher_p"])
    stats_df.to_csv(f"{PATH_TO_RESULTS}stats_vs_control.csv", index=False)
    return stats_df


def analyze_results():
    """Run the full analysis: load results, save plots and CSVs to PATH_TO_RESULTS.

    Models are discovered from the files on disk, so this runs regardless of
    which models are currently active in config.py.
    """
    Path(PATH_TO_RESULTS).mkdir(parents=True, exist_ok=True)

    model_names = discover_models()
    print(f"Models found: {model_names}")

    df = load_results(model_names)
    print(f"{len(df)} rows | models: {df['model'].nunique()}")

    report_coverage(df)

    summary = df.groupby(["model", "prompt", "conclusion"], observed=True).size()
    summary.to_csv(f"{PATH_TO_RESULTS}tableresults.csv")

    plot_conclusion_counts(df, model_names)
    plot_conclusion_percentages(df, model_names)
    plot_confidence_by_prompt(df, model_names)
    plot_confidence_by_model_and_conclusion(df)

    stats_df = run_stats_vs_control(df, model_names)
    print("\nStats vs control (sorted by Fisher p):")
    print(stats_df.to_string(index=False))

    print(f"\nPlots and CSVs saved to {PATH_TO_RESULTS}")
    return df, stats_df


if __name__ == "__main__":
    analyze_results()
