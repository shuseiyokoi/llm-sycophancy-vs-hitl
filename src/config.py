import os

# Anchor all paths to the repo root so scripts work from any directory
# (src/, a stage folder like src/call_models/, or the repo root).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# One designated output folder per pipeline stage:
# 1. src/local_qwen/   -> (model weights only, no pipeline outputs)
# 2. src/call_models/  -> data/call_models/   (single-turn, from the original study)
# 3. src/hitl/         -> data/hitl/ and results/analyze_hitl/
# data/gather_data/ holds the inputs carried over from the original study
# (summary.txt is the LLM input, preprocessed_data.csv the ground-truth data).
PATH_TO_DATA = os.path.join(ROOT_DIR, "data", "gather_data") + os.sep
PATH_TO_MODEL_RESULTS = os.path.join(ROOT_DIR, "data", "call_models") + os.sep
PATH_TO_HITL = os.path.join(ROOT_DIR, "data", "hitl") + os.sep
PATH_TO_HITL_RESULTS = os.path.join(ROOT_DIR, "results", "analyze_hitl") + os.sep

# llama.cpp server + GGUF weights for the local models
LOCAL_QWEN_DIR = os.path.join(SRC_DIR, "local_qwen")

NUM_ITERATIONS = 300

GPT_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]

CLAUDE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]  # claude-sonnet-4-20250514 gives non json responses
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-pro"]

QWEN_MODELS = [
    "qwen2.5-7b-instruct",
    "qwen3-8b",
]

LLAMA_MODELS = [
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
]

GEMMA_MODELS = [
    "gemma-2-9b-it",
    "gemma-3-12b-it",
]

# All models served locally with llama.cpp (see call_qwen.py / local_qwen)
LOCAL_MODELS = QWEN_MODELS + LLAMA_MODELS + GEMMA_MODELS

# prompt_jobs_config.py

PROMPT_TYPES = [
    "control_prompt",
    "emotional_identity_prompt",
    "emotional_prompt",
    "emotional_extreme_prompt",
    "emotional_suicidal_prompt",
    "identity_prompt",
    "identity_hypothetical_prompt",
]

# --- HITL sycophancy experiment (src/hitl/) ---
# Each iteration is one multi-turn conversation:
#   turn 0: control analyst prompt -> initial conclusion
#   turns 1..HITL_NUM_TURNS: simulated human feedback -> re-answer
# Total API calls = models x feedback types x iterations x (1 + HITL_NUM_TURNS).

HITL_FEEDBACK_TYPES = [
    "neutral_check",     # "are you sure?" — control for mere re-asking
    "reject_flat",       # plain disagreement with the model's initial conclusion
    "reject_emotional",  # emotional pushback
    "reject_expert",     # authority pushback (domain-expert persona)
]

HITL_NUM_TURNS = 3
HITL_NUM_ITERATIONS = 30

# --- DPO virtual lab (src/dpo_lab/) ---
# Same base model, two preference-data arms (human-labeled vs AI-labeled),
# trained at increasing data fractions; sycophancy measured per checkpoint.

PATH_TO_DPO_DATA = os.path.join(ROOT_DIR, "data", "dpo_lab") + os.sep
PATH_TO_DPO_CHECKPOINTS = os.path.join(ROOT_DIR, "models", "dpo_lab") + os.sep
PATH_TO_DPO_RESULTS = os.path.join(ROOT_DIR, "results", "dpo_lab") + os.sep

DPO_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DPO_NUM_PAIRS = 10000          # preference pairs drawn from Anthropic/hh-rlhf
DPO_FRACTIONS = [0.1, 0.25, 0.5, 1.0]  # dose-response checkpoints per arm
DPO_JUDGE_MODEL = "gpt-4o-mini"        # AI-arm judge (any OpenAI-compatible model)
DPO_EVAL_NUM_QUESTIONS = 200           # GSM8K questions per checkpoint eval
