"""Re-parse existing verifier run files in place (no GPU / no re-generation).

Two post-hoc fixes are applied to each `verify_*.json`:
  1. Re-extract `verifier_verdict` with the prose-aware extractor (`extract_verdict`), so
     verdicts stated in plain text -- not just `\\boxed{...}` -- are counted. The strict
     boxed-only extractor wrongly marked many parseable LLaVA responses as `bad`.
  2. Add the exact prompt fed to the verifier to every record (`verifier_prompt` and
     `verifier_rendered_prompt`), reconstructed deterministically from the stored template,
     question, and (truncated) solver output. The solver's own prompt fields are dropped.

Because the model outputs (`verifier_response`) are untouched, this is a pure, reproducible
re-scoring; only parsing-derived fields and metrics change.

Usage:
    python vlm/rescore_verify.py vlm/result/verify_*.json
"""

import argparse
import glob
import json
import os
import sys

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
from vlm_verify import truncate_response, build_verifier_messages, extract_verdict, VERDICT_EXTRACTOR  # noqa: E402

from transformers import AutoProcessor  # noqa: E402


def recompute_metrics(records):
    tp = tn = fp = fn = bad = 0
    for r in records:
        v = r["verifier_verdict"]
        sc = bool(r["solver_correct"])
        if v is None:
            bad += 1
        elif sc and v:
            tp += 1
        elif (not sc) and (not v):
            tn += 1
        elif (not sc) and v:
            fp += 1
        else:
            fn += 1
    total = len(records)
    correct = tp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    sc_count = sum(bool(r["solver_correct"]) for r in records)
    return {
        "solver_correct_count": sc_count,
        "solver_accuracy": sc_count / total if total else 0.0,
        "verifier": {
            "total": total, "bad_count": bad, "correct_count": correct,
            "accuracy": correct / total if total else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="verify_*.json files (globs ok)")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)))

    processors = {}  # verifier model name -> processor (loaded once)
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        md = d["metadata"]
        verifier_model = md["verifier_model"]["name"]
        template = md["prompt"]["template"]
        max_chars = md["verification_params"]["max_response_chars"]

        if verifier_model not in processors:
            processors[verifier_model] = AutoProcessor.from_pretrained(verifier_model, trust_remote_code=True)
        processor = processors[verifier_model]

        new_records = []
        for rec in d["records"]:
            response = truncate_response(rec["solver_full_output"], max_chars)
            vtext = template.format(question=rec["question"], response=response)
            rendered = processor.apply_chat_template(
                build_verifier_messages(vtext), tokenize=False, add_generation_prompt=True
            )
            new_records.append({
                "id": rec["id"],
                "image": rec["image"],
                "question": rec["question"],
                "answer": rec["answer"],
                "solver_full_output": rec["solver_full_output"],
                "solver_extracted_answer": rec.get("solver_extracted_answer"),
                "solver_correct": rec["solver_correct"],
                "verifier_prompt": vtext,
                "verifier_rendered_prompt": rendered,
                "verifier_response": rec["verifier_response"],
                "verifier_verdict": extract_verdict(rec["verifier_response"]),
            })

        # Validate the reconstruction matches what was originally sent (record 0).
        sample = md.get("prompt", {}).get("sample_rendered_prompt")
        if sample is not None and new_records and new_records[0]["verifier_rendered_prompt"] != sample:
            print(f"  [WARN] reconstructed prompt != stored sample for {os.path.basename(path)}")

        old_bad = sum(1 for r in d["records"] if r.get("verifier_verdict") is None)
        d["records"] = new_records
        md["verdict_extractor"] = VERDICT_EXTRACTOR
        d["metrics"] = recompute_metrics(new_records)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=4)

        m = d["metrics"]["verifier"]
        print(f"{os.path.basename(path)}\n"
              f"   bad {old_bad}->{m['bad_count']} | acc={m['accuracy']:.2f} f1={m['f1']:.2f} "
              f"(tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']})")


if __name__ == "__main__":
    main()
