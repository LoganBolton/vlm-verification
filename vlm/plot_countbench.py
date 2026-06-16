#!/usr/bin/env python3
"""CountBench plots: (1) base solver accuracy per model,
(2) verifier quality (accept-decision vs ground-truth correctness).

Verifier confusion matrix per run, over the final state of each problem:
  accepted = accepted_attempt is not None   (verifier ever said "yes")
  correct  = solver_correct
  TP accepted&correct  FP accepted&~correct  FN ~accepted&correct  TN ~accepted&~correct
  precision = P(correct | accepted)   recall = P(accepted | correct)
  accuracy  = (TP+TN)/N   base_rate = P(correct)  (= precision of a rubber stamp)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "result", "rejection", "countbench")
OUT  = os.path.join(os.path.dirname(__file__), "result", "plots")
os.makedirs(OUT, exist_ok=True)

def metrics(run):
    return json.load(open(os.path.join(ROOT, run, "metrics.json")))
def records(run):
    return json.load(open(os.path.join(ROOT, run, "records.json")))["records"]
def base_acc(run):  # round-0 solver accuracy
    return metrics(run)["iterations"]["0"]["solver"]["accuracy"]

# ---------- Plot 1: base solver accuracy ----------
SOLVERS = [  # label, run to pull round-0 accuracy from
    ("InternVL3.5-1B",  "InternVL3-5-1B__InternVL3-5-14B"),
    ("InternVL3.5-2B",  "InternVL3-5-2B__InternVL3-5-14B"),
    ("InternVL3.5-4B",  "InternVL3-5-4B__InternVL3-5-14B"),
    ("InternVL3.5-8B",  "InternVL3-5-8B__InternVL3-5-14B"),
    ("Qwen3-VL-8B",     "Qwen3-VL-8B-Instruct__oracle"),
    ("gemma-4-12B",     "gemma-4-12B-it__oracle"),
]
slabels = [l for l, _ in SOLVERS]
saccs   = [base_acc(r) for _, r in SOLVERS]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#4C72B0"]*4 + ["#DD8452", "#55A868"]
bars = ax.bar(slabels, saccs, color=colors)
ax.set_ylabel("CountBench accuracy (single attempt)")
ax.set_title("How good is each solver?  (base accuracy, 491 problems)")
ax.set_ylim(0, 1)
ax.axhline(0.5, ls=":", c="gray", lw=1)
for b, a in zip(bars, saccs):
    ax.text(b.get_x()+b.get_width()/2, a+0.015, f"{a:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
p1 = os.path.join(OUT, "countbench_solver_accuracy.png")
plt.savefig(p1, dpi=150); plt.close()

# ---------- Plot 2: verifier quality ----------
VERIFIERS = [  # label, run  (verifier judging that solver)
    ("InternVL-2B\n(solver 8B)",   "InternVL3-5-8B__InternVL3-5-2B"),
    ("InternVL-8B\n(solver 8B)",   "InternVL3-5-8B__InternVL3-5-8B"),
    ("InternVL-14B\n(solver 8B)",  "InternVL3-5-8B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 1B)",  "InternVL3-5-1B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 2B)",  "InternVL3-5-2B__InternVL3-5-14B"),
    ("InternVL-14B\n(solver 4B)",  "InternVL3-5-4B__InternVL3-5-14B"),
    ("gemma-12B\n(self, gemma)",   "gemma-4-12B-it__gemma-4-12B-it"),
    ("gemma-12B\n(solver Qwen)",   "Qwen3-VL-8B-Instruct__gemma-4-12B-it"),
]
def confusion(run):
    recs = records(run)
    TP=FP=FN=TN=0
    for r in recs:
        acc = r.get("accepted_attempt") is not None
        cor = bool(r.get("solver_correct"))
        if acc and cor: TP+=1
        elif acc and not cor: FP+=1
        elif not acc and cor: FN+=1
        else: TN+=1
    n = TP+FP+FN+TN
    prec = TP/(TP+FP) if TP+FP else float("nan")
    rec  = TP/(TP+FN) if TP+FN else float("nan")
    acc  = (TP+TN)/n
    base = (TP+FN)/n
    accept_rate = (TP+FP)/n
    return dict(prec=prec, rec=rec, acc=acc, base=base, accept_rate=accept_rate,
                TP=TP, FP=FP, FN=FN, TN=TN)

stats = [(lab, confusion(run)) for lab, run in VERIFIERS]
labels = [l for l, _ in stats]
prec = [s["prec"] for _, s in stats]
rec  = [s["rec"]  for _, s in stats]
vacc = [s["acc"]  for _, s in stats]
base = [s["base"] for _, s in stats]

x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.bar(x-w, prec, w, label="Precision  P(correct | accepted)", color="#C44E52")
ax.bar(x,   rec,  w, label="Recall  P(accepted | correct)",    color="#4C72B0")
ax.bar(x+w, vacc, w, label="Accuracy of accept/reject",        color="#55A868")
# base-rate markers: a rubber-stamp verifier (accept all) scores precision == base
for xi, b in zip(x, base):
    ax.plot([xi-w-w/2, xi-w+w/2], [b, b], color="black", lw=1.8)
ax.plot([], [], color="black", lw=1.8, label="base rate = precision if it accepted everything")
ax.set_ylabel("score")
ax.set_title("How well does each verifier verify?  (CountBench, accept-decision vs ground truth)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.05); ax.legend(loc="lower right", fontsize=8)
ax.axhline(1.0, ls=":", c="gray", lw=0.8)
plt.tight_layout()
p2 = os.path.join(OUT, "countbench_verifier_quality.png")
plt.savefig(p2, dpi=150); plt.close()

# ---------- print numeric tables ----------
print("\n=== Solver base accuracy (CountBench, 491) ===")
for l, a in zip(slabels, saccs):
    print(f"  {l:16} {a:.3f}")
print("\n=== Verifier quality ===")
print(f"  {'verifier (solver)':28} {'prec':>5} {'recall':>6} {'acc':>5} {'base':>5} {'accept%':>7}  TP/FP/FN/TN")
for lab, s in stats:
    one=lab.replace('\n',' ')
    print(f"  {one:28} {s['prec']:5.2f} {s['rec']:6.2f} {s['acc']:5.2f} {s['base']:5.2f} {s['accept_rate']*100:6.1f}%  {s['TP']}/{s['FP']}/{s['FN']}/{s['TN']}")
print(f"\nwrote:\n  {p1}\n  {p2}")
