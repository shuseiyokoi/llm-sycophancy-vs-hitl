# prepare_data.py
#
# Stage 1 of the DPO virtual lab: build the two preference-data arms.
#
#   human arm: real human preference judgments from Anthropic/hh-rlhf,
#     converted to DPO format (prompt / chosen / rejected).
#   ai arm: the SAME response pairs, but the chosen/rejected label is assigned
#     by an AI judge instead of the human annotator (RLAIF-style control).
#     This isolates "human-sourced feedback" as the experimental variable.
#
# Outputs (data/dpo_lab/):
#   human_prefs.jsonl   {"prompt", "chosen", "rejected"}
#   ai_prefs.jsonl      same pairs, judge-assigned labels, plus judge metadata
#
# The AI judge is any OpenAI-compatible endpoint: a cloud model (needs
# OPENAI_API_KEY in src/.env) or the local llama.cpp server (--judge-local,
# start it first with `make serve` in src/local_qwen).
#
# Usage:
#   python prepare_data.py                  # human arm only (no judge needed)
#   python prepare_data.py --ai-arm         # also build the AI arm (resumable)
#   python prepare_data.py --ai-arm --judge-local

import argparse
import json
import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DPO_DATA, DPO_NUM_PAIRS, DPO_JUDGE_MODEL

ASSISTANT_MARKER = "\n\nAssistant:"

JUDGE_TEMPLATE = """You are evaluating two assistant responses to the same conversation.

Conversation:
{prompt}

Response A:
{a}

Response B:
{b}

Which response is better (more helpful and more harmless)? Answer with exactly one letter: A or B."""


def split_transcript(text):
    """hh-rlhf stores full transcripts; split into (prompt, final reply)."""
    idx = text.rfind(ASSISTANT_MARKER)
    if idx == -1:
        return None, None
    return text[: idx + len(ASSISTANT_MARKER)], text[idx + len(ASSISTANT_MARKER) :].strip()


def build_human_arm(num_pairs, output_file):
    from datasets import load_dataset

    print(f"Loading Anthropic/hh-rlhf (keeping {num_pairs} pairs)...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    kept = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            prompt, chosen = split_transcript(row["chosen"])
            prompt_r, rejected = split_transcript(row["rejected"])
            # keep only pairs where both responses share the identical prompt
            if not prompt or prompt != prompt_r or not chosen or not rejected or chosen == rejected:
                continue
            f.write(
                json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if kept >= num_pairs:
                break

    print(f"human arm: wrote {kept} pairs to {output_file}")


def get_judge_client(local):
    from dotenv import load_dotenv
    from openai import OpenAI

    if local:
        port = int(os.getenv("QWEN_PORT", "8080"))
        return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="local")
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    return OpenAI()


def judge_pair(client, judge_model, prompt, chosen, rejected, rng):
    """Ask the judge which response is better. Position-randomized."""
    swap = rng.random() < 0.5  # A/B order randomized to cancel position bias
    a, b = (rejected, chosen) if swap else (chosen, rejected)

    completion = client.chat.completions.create(
        model=judge_model,
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, a=a, b=b)}],
    )
    verdict = completion.choices[0].message.content.strip().upper()

    if not verdict.startswith(("A", "B")):
        return None, swap
    picked_first = verdict.startswith("A")
    # map back through the swap: did the judge pick the human-chosen response?
    judge_prefers_human_choice = picked_first != swap
    return judge_prefers_human_choice, swap


def count_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def build_ai_arm(human_file, output_file, judge_model, local):
    client = get_judge_client(local)
    rng = random.Random(0)

    done = count_lines(output_file)
    pairs = [json.loads(line) for line in open(human_file, encoding="utf-8")]
    # burn the RNG for already-judged pairs so resume stays deterministic
    for _ in range(done):
        rng.random()
    if done:
        print(f"ai arm: resuming from pair {done + 1}")

    agree = 0
    for i, pair in enumerate(pairs[done:], start=done + 1):
        try:
            prefers_human, swap = judge_pair(
                client, judge_model, pair["prompt"], pair["chosen"], pair["rejected"], rng
            )
        except Exception as e:
            print(f"pair {i}: judge error, skipping ({e})")
            prefers_human, swap = None, False

        if prefers_human is None:
            record = None  # unusable verdict: drop the pair from this arm
        elif prefers_human:
            record = {"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]}
            agree += 1
        else:
            record = {"prompt": pair["prompt"], "chosen": pair["rejected"], "rejected": pair["chosen"]}

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        **(record or {"prompt": pair["prompt"], "skip": True}),
                        "judge": judge_model,
                        "judge_agrees_with_human": prefers_human,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if i % 100 == 0:
            print(f"ai arm: {i}/{len(pairs)} judged")
        if not local:
            time.sleep(0.2)

    print(f"ai arm: done -> {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=DPO_NUM_PAIRS)
    parser.add_argument("--ai-arm", action="store_true", help="also build the AI-judged arm")
    parser.add_argument("--judge-local", action="store_true", help="judge with the local llama.cpp server")
    parser.add_argument("--judge-model", default=DPO_JUDGE_MODEL)
    args = parser.parse_args()

    os.makedirs(PATH_TO_DPO_DATA, exist_ok=True)
    human_file = os.path.join(PATH_TO_DPO_DATA, "human_prefs.jsonl")
    ai_file = os.path.join(PATH_TO_DPO_DATA, "ai_prefs.jsonl")

    if count_lines(human_file) >= args.num_pairs:
        print(f"human arm already complete: {human_file}")
    else:
        build_human_arm(args.num_pairs, human_file)

    if args.ai_arm:
        build_ai_arm(human_file, ai_file, args.judge_model, args.judge_local)


if __name__ == "__main__":
    main()
