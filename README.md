# LLM Sycophancy vs Human-in-the-Loop

> [!NOTE]
> This repo was bootstrapped from the pipeline in
> [llm-sycophancy-emotional-framing](https://github.com/shuseiyokoi/llm-sycophancy-emotional-framing),
> which studied how emotional / bias-related framing shifts LLM conclusions on
> loan approval data. This project extends that work with a new research focus.

## Introduction

<!-- TODO: describe the new study -->

### Research Question

<!-- TODO: e.g. "Does human-in-the-loop feedback amplify or suppress LLM sycophancy in data analysis?" -->

## Data Source

This project uses publicly available HMDA loan application data.

- **Dataset:** 2024 California HMDA loan application data
- **Source:** FFIEC HMDA Data Browser API
- **Outcomes analyzed:**
  - Loan originated
  - Application approved but not accepted
  - Application denied

### Models Used

**Cloud APIs**
- OpenAI (GPT)
- Google (Gemini)
- Anthropic (Claude)

**Local open-weight models** (served with llama.cpp, OpenAI-compatible API)
- Qwen2.5-7B-Instruct
- Qwen3-8B
- Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct
- Gemma-2-9B-it
- Gemma-3-12B-it

Which models and prompt types run is controlled in [`src/config.py`](src/config.py)
(`GPT_MODELS`, `CLAUDE_MODELS`, `GEMINI_MODELS`, `QWEN_MODELS`, `LLAMA_MODELS`,
`GEMMA_MODELS`, `PROMPT_TYPES`, `NUM_ITERATIONS`).

## Project layout

The pipeline is split into 5 stages. Each stage lives in its own folder under
`src/` and writes its outputs to a designated folder under `data/` or `results/`:

| Stage | Code | Outputs |
|---|---|---|
| 1. Gather data | `src/gather_data/` | `data/gather_data/` |
| 2. Local model server | `src/local_qwen/` | (GGUF weights only) |
| 3. Call models | `src/call_models/` | `data/call_models/` |
| 4. Analyze results | `src/analyze_results/` | `results/analyze_results/` |
| 5. Ground truth regression | `src/ground_truth/` | `results/ground_truth/` |

All paths are defined in `src/config.py` and anchored to the repo root, so every
script can be run from any directory.

## How to run

### 1. Set up a virtual environment
From root repo

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Populate API keys in .env
Only needed for the cloud models (skip for local models).

```sh
cd src
touch .env
```
Add your own API keys
```.env
export OPENAI_API_KEY = ""
export GEMINI_API_KEY = ""
export ANTHROPIC_API_KEY = ""
```

### 3. Gather data from HMDA and summarize

```sh
cd src/gather_data
python gather_data.py
```

1. Retrieve HMDA loan application data from the API `load_data.py`
2. Filter the dataset to selected decision outcomes `preprocess_data.py`
3. Summarize the preprocessed data by race, sex, and ethnicity `summarize_data.py`
4. Create 3 files in `data/gather_data/`
  - `hmda_CA_2024.csv`
  - `preprocessed_data.csv`
  - `summary.txt`

### 4. Call Large Language Models

> [!WARNING]
> Skip, if you have data already. Set `NUM_ITERATIONS` in `src/config.py` before running — each iteration is one API call per model and prompt type.

**Cloud models** (from `src/call_models/`)

```sh
cd src/call_models
python call_chatGPT.py
python call_claude.py
python call_gemini.py
```

**Local Qwen** (no API key needed)

Terminal 1 — start the llama.cpp server (expects the GGUF model path set in `src/local_qwen/Makefile`):

```sh
cd src/local_qwen
make serve
```

Terminal 2 — run the calls against the local server:

```sh
cd src/call_models
python call_qwen.py
```

Each script loops over its models in `config.py` and all `PROMPT_TYPES`, and appends one JSON line per run to `data/call_models/results_{prompt_type}_{model}.jsonl`.

### 5. Analyze Results

```sh
cd src/analyze_results
python analyze_results.py
```

Or interactively with the notebook `src/analyze_results/analyze_results.ipynb`.

Models are auto-discovered from the `results_*.jsonl` files in `data/call_models/`, so this works regardless of which models are active in `config.py`. It prints a coverage table (usable responses per model and prompt — rows that came back as API errors or non-JSON are skipped and counted), then saves to `results/analyze_results/`:

- `conclusion-count-by-prompt-{model}.png`
- `percentages-by-prompt-{model}.png` (with per-bar sample sizes)
- `confidence-by-prompt-{model}.png`
- `confidence-by-model-and-conclusion.png`
- `tableresults.csv` — conclusion counts per model and prompt
- `stats_vs_control.csv` — chi-square / Fisher exact tests of each prompt type against the control prompt, per model

### 6. Ground truth regression

```sh
cd src/ground_truth
python run_regression.py
```

Fits a logistic regression (`denied ~ race + sex + ethnicity + financial controls`)
on the loan-level data from `data/gather_data/preprocessed_data.csv` and saves to
`results/ground_truth/`:

- `ground_truth_labels.csv` — BIAS / FAVORED / NO_BIAS label per sensitive group
- `full_regression_coefficients.csv`
- `regression_summary.txt`

### 7. Statistical bias tests

```sh
cd src/ground_truth
python run_statistical_tests.py
```

Tests whether the application decision depends on race, sex, and ethnicity in the
raw loan-level data. Runs marginal chi-square tests of independence (decision vs
each attribute, no controls) plus the same controlled logistic regression as
`run_regression.py` (reused via import), and saves to `results/ground_truth/`:

- `chi_square_tests.csv` — chi-square statistic, dof, and p-value per attribute
- `bias_test_labels.csv` — BIAS / FAVORED / NO_BIAS label per sensitive group
- `statistical_tests_summary.txt` — full report: chi-square table, regression summary, and labels

> [!NOTE]
> `src/main.py` is a thin dispatcher over the same stages: `python main.py --gather-data | --call-models | --analyze` (cloud models only; local models still run via `call_qwen.py`).
