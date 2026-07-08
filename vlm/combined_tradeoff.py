#!/usr/bin/env python3
"""Combined compute-vs-accuracy plot: base solo vs maj@5 vs best judge, all on ONE canonical axis.

For each model, three points priced on the SAME single-pass anchor C[m] (self-consistency
solver_gflops / n_samples), so the strategies are directly comparable:
  o  base solo   — one pass                 (compute = C[m],        acc = maj@1)
  s  maj@5       — 5-sample majority vote    (compute = 5*C[m],      acc = maj@5)
  D  best judge  — solver + best judge (k=5) (compute = C_s*sp+C_v*vp, acc = final)
The three are joined per model; the red line is the Pareto frontier over everything.

Writes report/figures/tradeoff/{ds}_combined_tradeoff.png (+ an 'avg'). Run:
  .venv/bin/python vlm/combined_tradeoff.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import build_charxiv_report as R
import compute_tradeoff as T     # canonical_cost, collect, best_judge, pareto, name
import maj_tradeoff as MJ        # collect -> {model: [(compute,acc) k=1..5]}

DATASETS = ["charxiv", "countbench"]
FIGDIR = "report/figures/tradeoff"
name = T.name


def collect(ds):
    """model -> dict(base=(g,a), maj5=(g,a) or None, judge=(g,a) or None)."""
    _, pipes = T.collect(ds)                 # canonical-priced judge pipelines
    best = T.best_judge(pipes)               # solver -> (gflops, acc, verifier)
    maj = MJ.collect(ds)                      # model -> [(compute, acc), ...] k=1..N
    out = {}
    for m, curve in maj.items():
        base = curve[0]                      # (C[m], maj@1)
        maj5 = curve[4] if len(curve) >= 5 else curve[-1]
        rec = dict(base=base, maj5=maj5, judge=None)
        if m in best:
            rec["judge"] = (best[m][0], best[m][1])
        out[m] = rec
    return out


STRAT = [   # key,     marker, size, frontier color, style, legend label
    # frontier lines kept neutral (dark grey) & distinguished by dash style, so they don't
    # clash with the fixed family colors (Qwen purple / gemma blue / InternVL orange / llava red)
    ("base",  "o", 95, "0.15", "-",  "base solo (1 pass)"),
    ("maj5",  "s", 70, "0.15", "--", "maj@5 vote"),
    ("judge", "D", 60, "0.15", "-.", "+ best judge (k=5)"),
]
FADED, FULL = 0.25, 1.0


def render(ds, data, title=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    cmap = plt.get_cmap("tab10")
    order = sorted(data, key=lambda m: (R.FAM_ORDER.get(R.family(m), 4), R.size(m)))

    # which models sit on each strategy's Pareto frontier
    frontier = {}
    for key, *_ in STRAT:
        pts = [(*data[m][key], m) for m in order if data[m][key] is not None]
        frontier[key] = [p[-1] for p in T.pareto(pts)]

    # scatter every point: on-frontier -> full opacity, else faded into the background
    for key, mk, sz, *_ in STRAT:
        onfr = set(frontier[key])
        for m in order:
            v = data[m][key]
            if v is None:
                continue
            on = m in onfr
            ax.scatter(v[0], v[1], s=sz, marker=mk, alpha=FULL if on else FADED,
                       zorder=5 if on else 3,
                       color=cmap(R.FAM_ORDER.get(R.family(m), 4)),
                       edgecolor="k" if on else "none")
            if on:                        # label only the models on this frontier
                ax.annotate(name(m), v, fontsize=7, xytext=(4, 3),
                            textcoords="offset points", zorder=6)

    # draw the three frontier lines
    handles = []
    for key, mk, sz, col, ls, lab in STRAT:
        line = [data[m][key] for m in frontier[key]]
        ax.plot([p[0] for p in line], [p[1] for p in line], ls, color=col, lw=2.2, zorder=4)
        handles.append(Line2D([], [], marker=mk, color=col, ls=ls, mec="k", ms=8, lw=2.2, label=lab))
    ax.legend(handles=handles, fontsize=8, loc="lower right", title="Pareto frontier per strategy")
    ax.set_xscale("log")
    ax.set_xlabel("total inference compute  (GFLOPs, whole dataset, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title or f"Compute vs accuracy — base vs maj@5 vs judge — {ds}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = f"{FIGDIR}/{ds}_combined_tradeoff.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def average(per_ds):
    dss = list(per_ds)
    common = set.intersection(*[set(per_ds[d]) for d in dss])
    avg = {}
    for m in common:
        rec = {}
        for k in ("base", "maj5", "judge"):
            vals = [per_ds[d][m][k] for d in dss]
            if any(v is None for v in vals):
                rec[k] = None
            else:
                rec[k] = (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))
        avg[m] = rec
    return avg


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    per_ds = {}
    for ds in DATASETS:
        data = collect(ds)
        if not data:
            print(f"[{ds}] no data"); continue
        per_ds[ds] = data
        print(f"  -> wrote {render(ds, data)}")
    if len(per_ds) == len(DATASETS):
        avg = average(per_ds)
        out = render("avg", avg, title="Compute vs accuracy — base vs maj@5 vs judge — avg of both datasets")
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
