"""Score a CharXiv VLM run produced by `vlm/vlm_inference.py`.

CharXiv reasoning answers are short free-text (model names, axis labels, numbers, short
phrases like "Joint-CNN", "lambda_L = 0.13", "(b) OPT", "94"), NOT integer counts. So the
CountBench scorer (`vlm/score_results.py`, which extracts a single integer and does exact
match) does not apply here, and this is its free-text sibling.

The *reference* CharXiv benchmark grades with a GPT-4 judge. To keep this pipeline fully
local, deterministic, and reproducible, we instead use a **relaxed normalized match**:

  1. Take the model's answer (the last \\boxed{...} if present, else the whole response).
  2. Normalize both gold and prediction: NFKC, lowercase, strip punctuation to spaces,
     collapse whitespace, drop a few stop words.
  3. Mark correct if ANY of:
       - normalized gold is a substring of the (boxed-or-full) prediction, or
       - every significant gold token appears somewhere in the full response, or
       - gold is numeric and the same number(s) appear in the response.

This is an approximate automatic metric -- it is lenient about phrasing/extra words but can
still miss paraphrases a human/LLM judge would accept. It is intentionally simple and
deterministic so the (expensive) generations can be re-scored anytime. The chosen extractor
is recorded in the output file's metadata so results stay self-describing.

Usage:
    python vlm/score_charxiv.py vlm/result/charxiv/charxiv_<model>_<time>.json
"""

from pprint import pprint
from typing import Optional, Set
import argparse
import json
import os
import re
import sys
import unicodedata

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
from answer_extractors import get_final_box_match  # noqa: E402

EXTRACTOR_NAME = "charxiv_relaxed_normalized_match"

_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "at", "and", "or",
         "approximately", "about", "around", "value", "answer", "it", "as", "for"}


def normalize(s: str) -> str:
    """Lowercase, fold unicode, reduce punctuation to spaces, collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("’", "'").replace("−", "-")
    s = re.sub(r"[^a-z0-9.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sig_tokens(s: str) -> Set[str]:
    return {t for t in normalize(s).split() if t and t not in _STOP}


def numbers(s: str) -> Set[str]:
    # Strip trailing dots/dashes so "0.13." -> "0.13"; keep signed decimals.
    return {n.strip(".-") for n in re.findall(r"-?\d+\.?\d*", normalize(s)) if n.strip(".-")}


def extract_answer(text: str) -> str:
    """The string we display as the model's answer: boxed content if any, else a tail snippet."""
    boxed = get_final_box_match(text)
    if boxed is not None:
        return boxed.strip()
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    return tail[-120:]


def is_correct(gold: str, response: str) -> bool:
    """Relaxed match of a free-text gold answer against a full model response."""
    g = normalize(gold)
    if not g:
        return False
    boxed = get_final_box_match(response)
    cand = normalize(boxed) if boxed is not None else normalize(response)
    full = normalize(response)

    if g in cand or g in full:
        return True
    gt = sig_tokens(gold)
    if gt and gt.issubset(sig_tokens(response)):
        return True
    gnums = numbers(gold)
    if gnums and gnums.issubset(numbers(response)):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str,
                        help="Run JSON produced by vlm/vlm_inference.py (CharXiv solver run)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write the scores file (default: alongside results_file)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.results_file, "r", encoding="utf-8") as f:
        run = json.load(f)
    metadata = run.get("metadata", {})
    records = run["records"]

    scored_records = []
    total = len(records)
    correct_count = 0
    for rec in records:
        extracted = extract_answer(rec["solver_full_output"])
        ok = is_correct(rec["answer"], rec["solver_full_output"])
        correct_count += ok
        scored_records.append({
            **rec,
            "solver_extracted_answer": extracted,
            "solver_correct": bool(ok),
        })

    metrics = {
        "solver": {
            "total": total,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "accuracy": correct_count / total if total > 0 else 0.0,
        },
    }
    scoring_metadata = {
        **metadata,
        "scored_from": os.path.abspath(args.results_file),
        "extractor": EXTRACTOR_NAME,
    }

    base = os.path.splitext(os.path.basename(args.results_file))[0]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base}_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": scoring_metadata, "metrics": metrics,
                   "records": scored_records}, f, indent=4)

    print("============================ Metrics ============================")
    pprint(metrics)
    print(f"\nScored {total} records from {args.results_file}")
    print(f"Saved scores to {out_path}")


if __name__ == "__main__":
    main()
