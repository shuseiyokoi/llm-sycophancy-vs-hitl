from dotenv import load_dotenv
from google import genai
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PATH_TO_DATA,
    PATH_TO_MODEL_RESULTS,
    GEMINI_MODELS,
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


def run_prompt_set(
    client, model_name, uploaded_file, prompt_text, output_file, prompt_name
):
    for i in range(NUM_ITERATIONS):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    uploaded_file,
                    prompt_text,
                ],
            )

            raw_text = response.text.strip()
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

        time.sleep(1)


def call_gemini():
    load_dotenv()

    os.makedirs(PATH_TO_MODEL_RESULTS, exist_ok=True)

    client = genai.Client()

    uploaded_file = client.files.upload(file=f"{PATH_TO_DATA}summary.txt")

    for model_name in GEMINI_MODELS:
        for prompt_type in PROMPT_TYPES:
            output_file = f"{PATH_TO_MODEL_RESULTS}results_{prompt_type}_{model_name}.jsonl"

            print(f"\nStarting: {model_name} | {prompt_type}")

            run_prompt_set(
                client=client,
                model_name=model_name,
                uploaded_file=uploaded_file,
                prompt_text=get_prompt(prompt_type),
                output_file=output_file,
                prompt_name=prompt_type,
            )

            print(f"Finished: {model_name} | {prompt_type}")


if __name__ == "__main__":
    call_gemini()
