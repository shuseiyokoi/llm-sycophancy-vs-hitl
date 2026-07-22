# providers.py
#
# One uniform multi-turn chat interface over every provider in config.py.
# Messages use the OpenAI shape: [{"role": "user"|"assistant", "content": str}].
# get_chat_fn(model_name) routes by the model lists in config.py and returns a
# chat(messages) -> raw_text callable.

import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import GPT_MODELS, CLAUDE_MODELS, GEMINI_MODELS, LOCAL_MODELS

MAX_TOKENS = 1024

LOCAL_PORT = int(os.getenv("QWEN_PORT", "8080"))
LOCAL_BASE_URL = f"http://localhost:{LOCAL_PORT}/v1"

_clients = {}


def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI

        load_dotenv()
        _clients["openai"] = OpenAI()
    return _clients["openai"]


def _local_client():
    if "local" not in _clients:
        from openai import OpenAI

        _clients["local"] = OpenAI(base_url=LOCAL_BASE_URL, api_key="local")
    return _clients["local"]


def _anthropic_client():
    if "anthropic" not in _clients:
        import anthropic

        load_dotenv()
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def _gemini_client():
    if "gemini" not in _clients:
        from google import genai

        load_dotenv()
        _clients["gemini"] = genai.Client()
    return _clients["gemini"]


def _chat_openai_style(client, model_name, messages):
    completion = client.chat.completions.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def _chat_anthropic(model_name, messages):
    message = _anthropic_client().messages.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return message.content[0].text.strip()


def _chat_gemini(model_name, messages):
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]
    response = _gemini_client().models.generate_content(
        model=model_name,
        contents=contents,
    )
    return response.text.strip()


def get_chat_fn(model_name):
    if model_name in GPT_MODELS:
        return lambda messages: _chat_openai_style(_openai_client(), model_name, messages)
    if model_name in CLAUDE_MODELS:
        return lambda messages: _chat_anthropic(model_name, messages)
    if model_name in GEMINI_MODELS:
        return lambda messages: _chat_gemini(model_name, messages)
    if model_name in LOCAL_MODELS:
        return lambda messages: _chat_openai_style(_local_client(), model_name, messages)
    raise ValueError(f"Unknown model: {model_name} (not in any config.py model list)")
