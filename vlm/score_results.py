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
import argparse
import json
import os
import sys

# Reuse the exact answer-extraction and matching logic from the normal LLM pipeline.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
from answer_extractors import extract_float_answer  # noqa: E402
from oracle_verifiers import exact_match  # noqa: E402


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

    scores = []
    total = len(records)
    correct_count, bad_count = 0, 0
    for rec in records:
        extracted = extract_float_answer(rec["solver_full_output"])
        if extracted is None:
            bad_count += 1
        is_correct = exact_match(data_row={"answer": rec["answer"]}, solver_extracted_answer=extracted)
        correct_count += is_correct
        scores.append({
            "id": rec["id"],
            "answer": rec["answer"],
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

    # Name the scores file after the run file: <run_id>_scores.json
    base = os.path.splitext(os.path.basename(args.results_file))[0]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base}_scores.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "scored_from": os.path.abspath(args.results_file),
            "dataset": metadata.get("dataset"),
            "model": metadata.get("model"),
            "metrics": metrics,
            "scores": scores,
        }, f, indent=4)

    print("============================ Metrics ============================")
    pprint(metrics)
    print(f"\nScored {total} records from {args.results_file}")
    print(f"Saved scores to {out_path}")


if __name__ == "__main__":
    main()
