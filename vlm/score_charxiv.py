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
from typing import List, Set
import argparse
import json
import os
import re
import sys
import unicodedata

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
from answer_extractors import get_final_box_match  # noqa: E402

EXTRACTOR_NAME = "charxiv_finalanswer_normalized_match_v3"


def normalize(s: str) -> str:
    """Lowercase, fold unicode, reduce punctuation to spaces, collapse whitespace.

    Hyphens and decimal points are kept inside tokens so "round-robin-combo" and "0.13"
    survive as single tokens.
    """
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("’", "'").replace("−", "-")
    s = re.sub(r"[^a-z0-9.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> List[str]:
    return [t for t in normalize(s).split() if t]


def numbers(s: str) -> Set[str]:
    # Whole-number tokens only, so "3" never matches inside "30" or "2017".
    return {n.strip(".-") for n in re.findall(r"-?\d+\.?\d*", normalize(s)) if n.strip(".-")}


def extract_answer(text: str) -> str:
    """The model's FINAL answer only -- the last \\boxed{...} if present, else the last
    non-empty line. We deliberately do NOT scan the whole response: the gold answer often
    appears inside the chain-of-thought (e.g. a model enumerates every option) even when the
    model's actual answer is different, which caused rampant false positives."""
    boxed = get_final_box_match(text)
    if boxed is not None:
        return boxed.strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _contiguous(sub: List[str], seq: List[str]) -> bool:
    """True if token list `sub` appears as a contiguous run inside `seq`."""
    if not sub or len(sub) > len(seq):
        return False
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def is_correct(gold: str, response: str) -> bool:
    """Match the gold answer against the model's FINAL answer only (not the reasoning).

    Correct if any of: exact normalized equality; the gold tokens appear verbatim as a
    contiguous run in the answer (handles "(b) OPT" vs gold "OPT"); the gold and answer have
    the same token *set* (order-independent, e.g. "case, det" vs "det, case"); or both reduce
    to the same set of whole numbers (e.g. gold "0.13" vs answer "lambda_L = 0.13").
    """
    pred = extract_answer(response)
    g, p = normalize(gold), normalize(pred)
    if not g:
        return False
    if g == p:
        return True
    gt, pt = tokens(gold), tokens(pred)
    if gt and _contiguous(gt, pt):
        return True
    if gt and set(gt) == set(pt):
        return True
    gn, pn = numbers(gold), numbers(pred)
    # Only trust a bare-number match when the gold answer is itself purely numeric
    # (e.g. "0.13", "94"). Otherwise a stray number in an alphanumeric gold like
    # "ETKI (J=50)" or "fc1" matches on the digits alone -> false positive.
    if gn and gn == pn and not any(c.isalpha() for c in g):
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
