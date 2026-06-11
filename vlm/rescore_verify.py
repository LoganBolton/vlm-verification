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
from vlm_inference import load_chat_renderer  # noqa: E402


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


def load_solver_scores(score_files):
    """Map solver_model_name -> {example_id: (solver_correct, solver_extracted_answer)}.

    Lets us refresh a verify file's labels after the solver run was *re-scored* (e.g. with a
    fixed answer-matcher), so verifier metrics are recomputed against the corrected oracle
    without re-running any model.
    """
    by_model = {}
    for f in score_files:
        for g in sorted(glob.glob(f)):
            d = json.load(open(g))
            model = d.get("metadata", {}).get("model", {}).get("name")
            if not model:
                continue
            m = by_model.setdefault(model, {})
            for r in d["records"]:
                m[r["id"]] = (bool(r["solver_correct"]), r.get("solver_extracted_answer"))
    return by_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="verify_*.json files (globs ok)")
    ap.add_argument("--solver_scores", nargs="*", default=[],
                    help="Re-scored solver *_scores.json files; refreshes solver_correct/"
                         "solver_extracted_answer (joined by solver model + example id) "
                         "before recomputing verifier metrics.")
    args = ap.parse_args()

    solver_scores = load_solver_scores(args.solver_scores)

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
            processors[verifier_model] = load_chat_renderer(verifier_model)
        render_chat = processors[verifier_model]

        # Optionally refresh oracle labels from a re-scored solver run for this solver model.
        label_map = solver_scores.get(md.get("solver_model"), {})

        new_records = []
        for rec in d["records"]:
            response = truncate_response(rec["solver_full_output"], max_chars)
            vtext = template.format(question=rec["question"], response=response)
            rendered = render_chat(build_verifier_messages(vtext))
            sc, extracted = label_map.get(
                rec["id"], (rec["solver_correct"], rec.get("solver_extracted_answer")))
            new_records.append({
                "id": rec["id"],
                "image": rec["image"],
                "question": rec["question"],
                "answer": rec["answer"],
                "solver_full_output": rec["solver_full_output"],
                "solver_extracted_answer": extracted,
                "solver_correct": sc,
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
