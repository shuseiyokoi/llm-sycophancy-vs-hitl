"""One-off repair: recover structured fields from response.raw_text rows.

Some models (mainly llama-3.2-3b) emit almost-JSON: a missing closing
brace, or Python-style single-quoted fields. The verdict fields are
present in the text, so re-parse them instead of re-running generation.

Originals are copied to data/call_models/_backup_pre_repair/ before rewriting.
"""

import json
import glob
import os
import re
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_MODEL_RESULTS

DATA_DIR = PATH_TO_MODEL_RESULTS
BACKUP_DIR = os.path.join(DATA_DIR, "_backup_pre_repair")


def repair(raw):
    """Return a parsed dict, or None if the fields can't be recovered."""
    start = raw.find("{")
    if start == -1:
        return None
    fragment = raw[start:]

    # tier 1: JSON that just lost its closing brace / quote
    for suffix in ("", "}", '"}', '"\n}'):
        try:
            obj, _ = json.JSONDecoder().raw_decode(fragment + suffix)
            if "conclusion" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # tier 2: regex field extraction (handles single-quoted keys/values)
    conclusion = re.search(r"['\"]conclusion['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw)
    confidence = re.search(r"['\"]confidence['\"]\s*:\s*(\d+)", raw)
    evidence = re.search(
        r"['\"]evidence['\"]\s*:\s*['\"](.+?)['\"]?\s*}?\s*$", raw, re.S
    )
    if conclusion and confidence:
        return {
            "conclusion": conclusion.group(1).strip(),
            "confidence": int(confidence.group(1)),
            "evidence": evidence.group(1).strip() if evidence else None,
        }
    return None


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    grand_fixed = grand_left = 0

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "results_*.jsonl"))):
        rows = []
        fixed = left = 0
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # corrupt line from an old writer: keep it verbatim
                    rows.append(line.rstrip("\n"))
                    continue
                response = row.get("response")
                if isinstance(response, dict) and "raw_text" in response:
                    repaired = repair(response["raw_text"])
                    if repaired is not None:
                        row["response"] = repaired
                        fixed += 1
                    else:
                        left += 1
                rows.append(row)

        if fixed:
            backup = os.path.join(BACKUP_DIR, os.path.basename(path))
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    if isinstance(row, str):
                        f.write(row + "\n")
                    else:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{os.path.basename(path)}: repaired {fixed}, unrecoverable {left}")

        grand_fixed += fixed
        grand_left += left

    print(f"\nTotal repaired: {grand_fixed}, still raw_text: {grand_left}")


if __name__ == "__main__":
    main()
