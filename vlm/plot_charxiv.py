#!/usr/bin/env python3
"""CharXiv plots, mirroring plot_countbench.py, but re-scored with the fixed
(v3) scorer that only trusts a bare-number match when gold is purely numeric.

  (1) base solver accuracy per model   (re-scored from standalone single-pass runs)
  (2) verifier quality                 (re-scored from rejection records, final state)

Also rewrites each CharXiv rejection records.json with corrected solver_correct /
solver_extracted_answer, and prints old-vs-new accuracy for every run.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_charxiv as sc  # fixed v3 scorer

RES  = os.path.join(os.path.dirname(__file__), "result")
REJ  = os.path.join(RES, "rejection", "charxiv")
OUT  = os.path.join(RES, "plots")
os.makedirs(OUT, exist_ok=True)

def rescore_records(recs):
    """Return (new_correct_list, accuracy) recomputed with the fixed scorer."""
    out = []
    for r in recs:
        out.append(bool(sc.is_correct(r["answer"], r["solver_full_output"])))
    return out, sum(out)/len(out) if recs else 0.0

# ---------- Plot 1: base solver accuracy (standalone single-pass runs) ----------
def latest(pattern):
    fs = [f for f in glob.glob(pattern) if "_scores" not in f]
    return sorted(fs)[-1] if fs else None

SOLVERS = [
    ("InternVL3.5-1B", latest(f"{RES}/charxiv_intern/charxiv_InternVL3-5-1B_*.json")),
    ("InternVL3.5-2B", latest(f"{RES}/charxiv_intern/charxiv_InternVL3-5-2B_*.json")),
    ("InternVL3.5-4B", latest(f"{RES}/charxiv_intern/charxiv_InternVL3-5-4B_*.json")),
    ("InternVL3.5-8B", latest(f"{RES}/charxiv_intern/charxiv_InternVL3-5-8B_*.json")),
    ("Qwen3-VL-8B",    latest(f"{RES}/charxiv_tier3/charxiv_Qwen3-VL-8B-Instruct_*.json")),
    ("gemma-4-12B",    latest(f"{RES}/charxiv_tier3/charxiv_gemma-4-12B-it_*.json")),
]
print("=== Solver base accuracy (CharXiv, 1000, single attempt) ===")
print(f"  {'model':16} {'old':>6} {'new':>6} {'delta':>7}")
slabels, saccs = [], []
for lab, path in SOLVERS:
    run = json.load(open(path))
    _, new = rescore_records(run["records"])
    sp = path.replace(".json", "_scores.json")
    old = json.load(open(sp))["metrics"]["solver"]["accuracy"] if os.path.exists(sp) else float("nan")
    print(f"  {lab:16} {old:6.3f} {new:6.3f} {new-old:+7.3f}")
    slabels.append(lab); saccs.append(new)

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#4C72B0"]*4 + ["#DD8452", "#55A868"]
bars = ax.bar(slabels, saccs, color=colors)
ax.set_ylabel("CharXiv accuracy (single attempt)")
ax.set_title("How good is each solver?  (CharXiv, re-scored v3, 1000 problems)")
ax.set_ylim(0, 1)
for b, a in zip(bars, saccs):
    ax.text(b.get_x()+b.get_width()/2, a+0.015, f"{a:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
p1 = os.path.join(OUT, "charxiv_solver_accuracy.png")
plt.savefig(p1, dpi=150); plt.close()

# ---------- Re-score rejection runs (persist) + verifier quality ----------
VERIFIERS = [
    ("InternVL-2B\n(solver 8B)",  "InternVL3-5-8B__InternVL3-5-2B"),
    ("InternVL-8B\n(solver 8B)",  "InternVL3-5-8B__InternVL3-5-8B"),
    ("InternVL-14B\n(solver 8B)", "InternVL3-5-8B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 1B)", "InternVL3-5-1B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 2B)", "InternVL3-5-2B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 4B)", "InternVL3-5-4B__InternVL3-5-14B"),
    ("gemma-12B\n(solver Qwen)",  "Qwen3-VL-8B-Instruct__gemma-4-12B-it"),
]

print("\n=== Re-scored rejection runs (final-state accuracy) ===")
print(f"  {'run':38} {'old':>6} {'new':>6}  TP/FP/FN/TN  prec  rec   acc")
stats = []
for lab, run in VERIFIERS:
    p = os.path.join(REJ, run, "records.json")
    blob = json.load(open(p))
    recs = blob["records"]
    old_acc = sum(bool(r["solver_correct"]) for r in recs)/len(recs)
    newc, new_acc = rescore_records(recs)
    # persist corrected flags
    for r, c in zip(recs, newc):
        r["solver_extracted_answer"] = sc.extract_answer(r["solver_full_output"])
        r["solver_correct"] = c
    blob.setdefault("metadata", {})["rescored_extractor"] = sc.EXTRACTOR_NAME
    json.dump(blob, open(p, "w"), indent=4)
    # confusion: accept-decision vs corrected correctness
    TP=FP=FN=TN=0
    for r, c in zip(recs, newc):
        acc = r.get("accepted_attempt") is not None
        if acc and c: TP+=1
        elif acc and not c: FP+=1
        elif not acc and c: FN+=1
        else: TN+=1
    n=TP+FP+FN+TN
    prec=TP/(TP+FP) if TP+FP else float("nan")
    rec =TP/(TP+FN) if TP+FN else float("nan")
    vacc=(TP+TN)/n; base=(TP+FN)/n; arate=(TP+FP)/n
    stats.append((lab, dict(prec=prec, rec=rec, acc=vacc, base=base, arate=arate,
                            TP=TP, FP=FP, FN=FN, TN=TN)))
    print(f"  {run:38} {old_acc:6.3f} {new_acc:6.3f}  {TP}/{FP}/{FN}/{TN}  {prec:.2f} {rec:.2f} {vacc:.2f}")

labels=[l for l,_ in stats]
prec=[s["prec"] for _,s in stats]; rec=[s["rec"] for _,s in stats]
vacc=[s["acc"] for _,s in stats]; base=[s["base"] for _,s in stats]
x=np.arange(len(labels)); w=0.25
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(x-w, prec, w, label="Precision  P(correct | accepted)", color="#C44E52")
ax.bar(x,   rec,  w, label="Recall  P(accepted | correct)",    color="#4C72B0")
ax.bar(x+w, vacc, w, label="Accuracy of accept/reject",        color="#55A868")
for xi, b in zip(x, base):
    ax.plot([xi-w-w/2, xi-w+w/2], [b, b], color="black", lw=1.8)
ax.plot([], [], color="black", lw=1.8, label="base rate = precision if it accepted everything")
ax.set_ylabel("score")
ax.set_title("How well does each verifier verify?  (CharXiv, re-scored v3, accept vs ground truth)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.05); ax.legend(loc="upper right", fontsize=8)
ax.axhline(1.0, ls=":", c="gray", lw=0.8)
plt.tight_layout()
p2 = os.path.join(OUT, "charxiv_verifier_quality.png")
plt.savefig(p2, dpi=150); plt.close()
print(f"\nwrote:\n  {p1}\n  {p2}")
