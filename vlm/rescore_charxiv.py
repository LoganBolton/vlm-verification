#!/usr/bin/env python3
"""Re-derive CharXiv correctness OFFLINE with the current score_charxiv, no GPU / no re-running.

The verifier grid (verify_*.json) and agentic-zoom (records.json) stored their solver_correct
labels at run time with an OLDER scorer (pre-v3 lenient matching + the option-label bug). Every
raw model output is stored, so we recompute solver_correct with the current scorer and rewrite
the derived metrics:

  * GRID:    recompute solver_correct per record, then tp/tn/fp/fn from the STORED verifier_verdict
             (the judge's decision is unchanged), plus solver_accuracy. Mirrors vlm_verify.py.
  * AGENTIC: recompute solver_correct from solver_answer_text, then the run accuracy.

NOT handled here: rejection §5.1. Its records keep only the FINAL answer per problem, so the
attempt-0 "base" accuracy (hence realized_gain) cannot be recovered offline -- those cells must be
re-run to change. New queue cells already use the fixed scorer.

Backs up each file to <path>.prev once (never overwrites an existing .prev). Then re-run:
    .venv/bin/python vlm/verifier_gain.py --dataset charxiv
    .venv/bin/python vlm/build_charxiv_report.py
"""
import glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
import score_charxiv as sc

RES = "vlm/result"


def backup(path):
    if not os.path.exists(path + ".prev"):
        shutil.copyfile(path, path + ".prev")


def rescore_grid():
    files = glob.glob(f"{RES}/verifier_grid/charxiv/verify_*.json")
    n_files = flips = 0
    for f in sorted(files):
        d = json.load(open(f))
        recs = d.get("records") or []
        if not recs:
            continue
        tp = tn = fp = fn = bad = 0
        ff = 0
        for r in recs:
            verdict = r.get("verifier_verdict")           # True / False / None (stored judge call)
            new = sc.is_correct(r["answer"], r["solver_full_output"])
            if bool(r.get("solver_correct")) != bool(new):
                ff += 1
            r["solver_correct"] = bool(new)
            if verdict is None:
                bad += 1
            elif new and verdict:
                tp += 1
            elif (not new) and (not verdict):
                tn += 1
            elif (not new) and verdict:
                fp += 1
            else:
                fn += 1
        total = len(recs)
        correct_cnt = sum(bool(r["solver_correct"]) for r in recs)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) else 0.0
        backup(f)
        d["metrics"]["solver_correct_count"] = correct_cnt
        d["metrics"]["solver_accuracy"] = correct_cnt / total if total else 0.0
        d["metrics"]["verifier"].update(
            total=total, bad_count=bad, correct_count=tp + tn,
            accuracy=(tp + tn) / total if total else 0.0,
            tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec_, f1=f1)
        json.dump(d, open(f, "w"), indent=4)
        n_files += 1
        flips += ff
    print(f"GRID:    rewrote {n_files} verify files, {flips} record gradings changed")


def rescore_agentic():
    files = glob.glob(f"{RES}/agentic_vision/charxiv_c*/*/records.json")
    n_files = flips = 0
    for rf in sorted(files):
        raw = json.load(open(rf))
        recs = raw if isinstance(raw, list) else raw.get("records")
        if not recs:
            continue
        ff = 0
        for r in recs:
            text = r.get("solver_answer_text") or r.get("solver_full_output") or ""
            new = sc.is_correct(r["answer"], text)
            if bool(r.get("solver_correct")) != bool(new):
                ff += 1
            r["solver_correct"] = bool(new)
        acc = sum(bool(r["solver_correct"]) for r in recs) / len(recs)
        backup(rf)
        json.dump(raw, open(rf, "w"), indent=4)
        mp = os.path.join(os.path.dirname(rf), "metrics.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            backup(mp)
            m["accuracy"] = acc
            json.dump(m, open(mp, "w"), indent=4)
        n_files += 1
        flips += ff
    print(f"AGENTIC: rewrote {n_files} runs, {flips} record gradings changed")


if __name__ == "__main__":
    rescore_grid()
    rescore_agentic()
    print("done. now run: verifier_gain.py --dataset charxiv, then build_charxiv_report.py")
