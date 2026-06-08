"""Score a VLM run produced by `vlm/vlm_inference.py`.

`vlm_inference.py` only records raw model generations -- it does not judge correctness.
This script is the separate evaluation step: it reads a run file
(`<dataset>_<model>_<time>.json`), extracts each answer, compares it to ground truth,
and writes a sibling scores file (`<dataset>_<model>_<time>_scores.json`) plus prints
the aggregate metrics.

Keeping scoring separate means the (expensive) generations can be re-scored anytime --
e.g. after fixing the answer extractor -- without re-running the model.

Usage:
    python vlm/score_results.py vlm/result/countbench_Qwen3-VL-2B-Instruct_20260607-223800.json
"""

from pprint import pprint
from typing import Optional
import argparse
import json
import os
import re
import sys

# Reuse the boxed-answer helper + matching logic from the normal LLM pipeline.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
from answer_extractors import get_final_box_match  # noqa: E402
from oracle_verifiers import exact_match  # noqa: E402


# CountBenchQA answers are small integer counts. Models express them inconsistently:
# Gemma/Qwen put a digit in \boxed{}, while LLaVA-1.5 answers in prose ("There are six
# chairs"). This extractor reads a count from either form so every model is scored fairly.
EXTRACTOR_NAME = "count_boxed_or_prose"
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def extract_count(text: str) -> Optional[float]:
    """Extract an integer count from a model response.

    Prefers the content of the last \\boxed{...} (how Gemma/Qwen answer); if there is no
    box, falls back to the last number in the whole text -- either a digit (e.g. "12") or
    an English number-word (e.g. "six") -- which is how LLaVA-1.5 answers.
    """
    boxed = get_final_box_match(text)
    src = boxed if boxed is not None else text

    found = []  # (position, value)
    for m in re.finditer(r"\d+", src):
        found.append((m.start(), float(int(m.group()))))
    for word, val in _WORD_TO_NUM.items():
        for m in re.finditer(rf"\b{word}\b", src, re.IGNORECASE):
            found.append((m.start(), float(val)))

    if not found:
        return None
    found.sort()
    return found[-1][1]  # last number stated = the model's final count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str,
                        help="Run JSON produced by vlm/vlm_inference.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write the scores file (default: alongside results_file)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.results_file, "r", encoding="utf-8") as f:
        run = json.load(f)
    metadata = run.get("metadata", {})
    records = run["records"]

    # The scores file mirrors the run file exactly (same metadata + every record field),
    # and just appends `solver_extracted_answer` and `solver_correct` to each record.
    scored_records = []
    total = len(records)
    correct_count, bad_count = 0, 0
    for rec in records:
        extracted = extract_count(rec["solver_full_output"])
        if extracted is None:
            bad_count += 1
        is_correct = exact_match(data_row={"answer": rec["answer"]}, solver_extracted_answer=extracted)
        correct_count += is_correct
        scored_records.append({
            **rec,  # all original fields, unchanged
            "solver_extracted_answer": extracted,
            "solver_correct": is_correct,
        })

    metrics = {
        "solver": {
            "total": total,
            "bad_count": bad_count,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "accuracy": correct_count / total if total > 0 else 0.0,
        },
    }

    # Record how scoring was done so the file stays self-describing.
    scoring_metadata = {
        **metadata,
        "scored_from": os.path.abspath(args.results_file),
        "extractor": EXTRACTOR_NAME,
    }

    # Name the scores file after the run file: <run_id>_scores.json
    base = os.path.splitext(os.path.basename(args.results_file))[0]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base}_scores.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": scoring_metadata,
            "metrics": metrics,
            "records": scored_records,
        }, f, indent=4)

    print("============================ Metrics ============================")
    pprint(metrics)
    print(f"\nScored {total} records from {args.results_file}")
    print(f"Saved scores to {out_path}")


if __name__ == "__main__":
    main()
