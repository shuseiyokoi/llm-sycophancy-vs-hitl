import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PATH_TO_MODEL_RESULTS,
    LOCAL_QWEN_DIR,
    LOCAL_MODELS,
    NUM_ITERATIONS,
    PROMPT_TYPES,
)
from prompts import get_embedded_prompt

LLAMA_SERVER = os.getenv("LLAMA_SERVER", "llama-server")
PORT = int(os.getenv("QWEN_PORT", "8080"))
BASE_URL = f"http://localhost:{PORT}/v1"

MODELS_DIR = os.path.join(LOCAL_QWEN_DIR, "models")

# GGUF file and extra llama-server flags for each model in config.LOCAL_MODELS
LOCAL_SERVER_CONFIG = {
    "qwen2.5-7b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "qwen3-8b": {
        "gguf": os.path.join(MODELS_DIR, "Qwen3-8B-Q4_K_M.gguf"),
        # disable thinking so the output is only the strict JSON answer
        "extra_args": ["--reasoning-budget", "0"],
    },
    "llama-3.1-8b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "llama-3.2-3b-instruct": {
        "gguf": os.path.join(MODELS_DIR, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "gemma-2-9b-it": {
        "gguf": os.path.join(MODELS_DIR, "gemma-2-9b-it-Q4_K_M.gguf"),
        "extra_args": [],
    },
    "gemma-3-12b-it": {
        "gguf": os.path.join(MODELS_DIR, "google_gemma-3-12b-it-Q4_K_M.gguf"),
        "extra_args": [],
    },
}


def parse_json_response(raw_text):
    raw_text = raw_text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[len("```json") :].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:].strip()

    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # some models (e.g. gemma-2) append prose after the JSON object;
        # fall back to parsing the first JSON object in the text
        try:
            start = raw_text.index("{")
            obj, _ = json.JSONDecoder().raw_decode(raw_text[start:])
            return obj
        except (ValueError, json.JSONDecodeError):
            return {"raw_text": raw_text}


def server_is_up():
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(model_name):
    if server_is_up():
        raise RuntimeError(
            f"A server is already running on port {PORT}. Stop it first so "
            "results are labeled with the model this script loads."
        )

    server_config = LOCAL_SERVER_CONFIG[model_name]
    if not os.path.exists(server_config["gguf"]):
        raise FileNotFoundError(
            f"{server_config['gguf']} not found. Run `make download` in local_qwen first."
        )

    proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "-m", server_config["gguf"],
            "--port", str(PORT),
            "-c", "8192",
            "-ngl", "99",
            *server_config["extra_args"],
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):  # model load can take a few minutes
        if server_is_up():
            print(f"llama.cpp server is up with {model_name}")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited while loading {model_name} "
                f"(exit code {proc.returncode})"
            )
        time.sleep(5)

    proc.terminate()
    raise RuntimeError(f"Server for {model_name} was not healthy after 5 minutes")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def count_existing_runs(output_file):
    try:
        with open(output_file, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run_prompt_set(client, model_name, prompt_text, output_file, prompt_name, num_iterations=NUM_ITERATIONS, start_run=0):
    for i in range(start_run, num_iterations):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )

            raw_text = completion.choices[0].message.content.strip()
            parsed_response = parse_json_response(raw_text)

        except Exception as e:
            parsed_response = {"error": str(e)}

        result = {
            "run": i + 1,
            "model": model_name,
            "prompt_type": prompt_name,
            "response": parsed_response,
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"{model_name} | {prompt_name} | Run {i + 1}: {parsed_response}")


def call_qwen(prompt_types=PROMPT_TYPES, num_iterations=NUM_ITERATIONS, output_prefix="results"):
    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = OpenAI(base_url=BASE_URL, api_key="local")

    for model_name in LOCAL_MODELS:
        # resume support: skip (model, prompt) pairs whose output file is
        # already complete, and continue partial ones where they left off
        pending = []
        for prompt_type in prompt_types:
            output_file = f"{PATH_TO_MODEL_RESULTS}{output_prefix}_{prompt_type}_{model_name}.jsonl"
            done = count_existing_runs(output_file)
            if done >= num_iterations:
                print(f"Skipping {model_name} | {prompt_type}: {done} runs already recorded")
            else:
                pending.append((prompt_type, output_file, done))

        if not pending:
            print(f"Skipping {model_name}: all prompt types complete")
            continue

        proc = start_server(model_name)

        try:
            for prompt_type, output_file, done in pending:
                print(f"\nStarting: {model_name} | {prompt_type}" + (f" (resuming from run {done + 1})" if done else ""))

                run_prompt_set(
                    client=client,
                    model_name=model_name,
                    prompt_text=get_embedded_prompt(prompt_type),
                    output_file=output_file,
                    prompt_name=prompt_type,
                    num_iterations=num_iterations,
                    start_run=done,
                )

                print(f"Finished: {model_name} | {prompt_type}")
        finally:
            stop_server(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="quick test: control_prompt only, 1 run per model, writes to smoke_*.jsonl",
    )
    args = parser.parse_args()

    if args.smoke:
        call_qwen(prompt_types=["control_prompt"], num_iterations=1, output_prefix="smoke")
    else:
        call_qwen()
