from dotenv import load_dotenv
import os
import sys
from openai import OpenAI
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PATH_TO_DATA,
    PATH_TO_MODEL_RESULTS,
    GPT_MODELS,
    NUM_ITERATIONS,
    PROMPT_TYPES,
)
from prompts import get_prompt


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
        # some models append prose after the JSON object;
        # fall back to parsing the first JSON object in the text
        try:
            start = raw_text.index("{")
            obj, _ = json.JSONDecoder().raw_decode(raw_text[start:])
            return obj
        except (ValueError, json.JSONDecodeError):
            return {"raw_text": raw_text}


def run_prompt_set(client, model_name, file_id, prompt_text, output_file, prompt_name):
    for i in range(NUM_ITERATIONS):
        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                            {
                                "type": "input_text",
                                "text": prompt_text,
                            },
                        ],
                    }
                ],
            )

            parsed_response = parse_json_response(response.output_text)

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

        time.sleep(1)


def call_chatGPT():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = OpenAI(api_key=api_key)

    uploaded_file = client.files.create(
        file=open(f"{PATH_TO_DATA}summary.txt", "rb"),
        purpose="user_data",
    )

    for model_name in GPT_MODELS:
        for prompt_type in PROMPT_TYPES:
            output_file = f"{PATH_TO_MODEL_RESULTS}results_{prompt_type}_{model_name}.jsonl"

            print(f"\nStarting: {model_name} | {prompt_type}")

            run_prompt_set(
                client=client,
                model_name=model_name,
                file_id=uploaded_file.id,
                prompt_text=get_prompt(prompt_type),
                output_file=output_file,
                prompt_name=prompt_type,
            )

            print(f"Finished: {model_name} | {prompt_type}")


if __name__ == "__main__":
    call_chatGPT()
