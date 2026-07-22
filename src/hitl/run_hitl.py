# run_hitl.py
#
# Human-in-the-loop sycophancy experiment.
#
# Hypothesis: as an LLM receives (pushback) feedback from a human, it becomes
# more sycophantic — it drifts toward the human's stated opinion even though
# the underlying data has not changed.
#
# One iteration = one multi-turn conversation with a single model:
#   turn 0: control analyst prompt + HMDA summary -> initial conclusion (JSON)
#   turn 1..HITL_NUM_TURNS: simulated human feedback pushing AGAINST the
#     model's initial conclusion (or a neutral "are you sure?" control),
#     model re-answers in the same JSON format each time.
#
# Output: one JSON line per completed iteration in
#   data/hitl/hitl_{feedback_type}_{model}.jsonl
# with the full trajectory, so no information is lost for analysis.
#
# Usage:
#   python run_hitl.py                     # all cloud models in config.py
#   python run_hitl.py --local             # all local models (starts llama.cpp per model)
#   python run_hitl.py --models gpt-4o-mini claude-haiku-4-5-20251001
#   python run_hitl.py --smoke             # 1 iteration, 1 turn, reject_flat only -> smoke_*.jsonl

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "call_models"))

from config import (
    PATH_TO_HITL,
    GPT_MODELS,
    CLAUDE_MODELS,
    GEMINI_MODELS,
    LOCAL_MODELS,
    HITL_FEEDBACK_TYPES,
    HITL_NUM_TURNS,
    HITL_NUM_ITERATIONS,
)
from call_qwen import parse_json_response, start_server, stop_server
from hitl_prompts import get_initial_prompt, get_target_conclusion, get_feedback
from providers import get_chat_fn

CLOUD_MODELS = GPT_MODELS + CLAUDE_MODELS + GEMINI_MODELS

SLEEP_BETWEEN_CALLS = 1


def normalize_conclusion(parsed):
    """Return 'YES' / 'NO' from a parsed response, or None if unusable."""
    conclusion = str(parsed.get("conclusion", "")).strip().upper()
    return conclusion if conclusion in ("YES", "NO") else None


def count_existing_runs(output_file):
    try:
        with open(output_file, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run_conversation(chat_fn, model_name, feedback_type, run_id, num_turns):
    """Run one multi-turn conversation and return the result record."""
    messages = [{"role": "user", "content": get_initial_prompt()}]

    try:
        raw_text = chat_fn(messages)
        initial = parse_json_response(raw_text)
    except Exception as e:
        initial = {"error": str(e)}

    record = {
        "run": run_id,
        "model": model_name,
        "feedback_type": feedback_type,
        "num_turns": num_turns,
        "initial": initial,
        "target_conclusion": None,
        "turns": [],
    }

    initial_conclusion = normalize_conclusion(initial) if isinstance(initial, dict) else None
    if initial_conclusion is None:
        # no usable initial answer -> nothing to push against; record and stop
        return record

    messages.append({"role": "assistant", "content": json.dumps(initial)})
    target = get_target_conclusion(feedback_type, initial_conclusion)
    record["target_conclusion"] = target

    for turn in range(1, num_turns + 1):
        feedback = get_feedback(feedback_type, target, turn)
        messages.append({"role": "user", "content": feedback})

        time.sleep(SLEEP_BETWEEN_CALLS)
        try:
            raw_text = chat_fn(messages)
            parsed = parse_json_response(raw_text)
        except Exception as e:
            parsed = {"error": str(e)}

        record["turns"].append({"turn": turn, "feedback": feedback, "response": parsed})

        # keep the conversation going with whatever the model actually said
        content = json.dumps(parsed) if "error" not in parsed else "(no answer)"
        messages.append({"role": "assistant", "content": content})

    return record


def run_model(model_name, feedback_types, num_iterations, num_turns, output_prefix):
    chat_fn = get_chat_fn(model_name)

    for feedback_type in feedback_types:
        output_file = f"{PATH_TO_HITL}{output_prefix}_{feedback_type}_{model_name}.jsonl"

        done = count_existing_runs(output_file)
        if done >= num_iterations:
            print(f"Skipping {model_name} | {feedback_type}: {done} runs already recorded")
            continue

        print(f"\nStarting: {model_name} | {feedback_type}" + (f" (resuming from run {done + 1})" if done else ""))

        for i in range(done, num_iterations):
            record = run_conversation(chat_fn, model_name, feedback_type, i + 1, num_turns)

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            final = record["turns"][-1]["response"] if record["turns"] else record["initial"]
            print(
                f"{model_name} | {feedback_type} | Run {i + 1}: "
                f"initial={normalize_conclusion(record['initial']) if isinstance(record['initial'], dict) else None} "
                f"target={record['target_conclusion']} "
                f"final={normalize_conclusion(final) if isinstance(final, dict) else None}"
            )

            time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"Finished: {model_name} | {feedback_type}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="specific model names (default: all cloud models)")
    parser.add_argument("--local", action="store_true", help="run the local models (manages the llama.cpp server)")
    parser.add_argument("--smoke", action="store_true", help="quick test: reject_flat only, 1 iteration, 1 turn, writes to smoke_*.jsonl")
    args = parser.parse_args()

    os.makedirs(PATH_TO_HITL, exist_ok=True)

    if args.models:
        models = args.models
    elif args.local:
        models = LOCAL_MODELS
    else:
        models = CLOUD_MODELS

    if args.smoke:
        feedback_types, num_iterations, num_turns, prefix = ["reject_flat"], 1, 1, "smoke"
    else:
        feedback_types, num_iterations, num_turns, prefix = (
            HITL_FEEDBACK_TYPES,
            HITL_NUM_ITERATIONS,
            HITL_NUM_TURNS,
            "hitl",
        )

    total_calls = len(models) * len(feedback_types) * num_iterations * (1 + num_turns)
    print(f"Models: {models}")
    print(f"Feedback types: {feedback_types}")
    print(f"{num_iterations} iterations x {1 + num_turns} calls each -> up to {total_calls} API calls total")

    for model_name in models:
        proc = None
        if model_name in LOCAL_MODELS:
            proc = start_server(model_name)
        try:
            run_model(model_name, feedback_types, num_iterations, num_turns, prefix)
        except Exception as e:
            print(f"{model_name} failed: {e}")
        finally:
            if proc is not None:
                stop_server(proc)


if __name__ == "__main__":
    main()
