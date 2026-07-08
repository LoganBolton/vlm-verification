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
import os, sys, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from adjustText import adjust_text          # optional: intelligent non-overlapping labels
except ImportError:
    adjust_text = None

sys.path.insert(0, os.path.dirname(__file__))
import build_charxiv_report as R
import compute_tradeoff as T     # canonical_cost, collect, best_judge, pareto, name
import maj_tradeoff as MJ        # collect -> {model: [(compute,acc) k=1..5]}
import vision_flops as VF        # vision-encoder GFLOPs

ZOOM_BUDGETS = [2, 4, 8]         # zoom crop budgets swept; per model we keep the best-accuracy one

DATASETS = ["charxiv", "countbench"]
FIGDIR = "report/figures/tradeoff"
name = T.name


def n_queries(ds):
    """number of problems (queries) in the dataset, from any run's metadata."""
    for f in glob.glob(f"vlm/result/self_consistency/{ds}/*/metrics.json"):
        md = json.load(open(f)).get("metadata", {})
        if md.get("n_problems"):
            return md["n_problems"]
    return None


def collect_zoom(ds, budgets=ZOOM_BUDGETS):
    """model -> (whole-dataset GFLOPs, accuracy) for the agentic zoom-tool run, keeping for each
    model the budget (c=2/4/8) that gave the BEST accuracy -- the budgets trade off differently
    per model, so this is an oracle-over-budgets pick of the highest-accuracy run.

    LLM side is the chosen run's real measured solver_gflops (a single multi-turn rollout, so all
    crop image-tokens are already counted). Vision side adds the ViT encode of the original image
    PLUS one encode per zoom crop (avg_crops of them), approximating each crop's patch count by the
    original image's -- the extra vision cost of *re-looking*, which is the whole point of zoom."""
    best = {}   # model -> (acc, compute)
    for c in budgets:
        for f in glob.glob(f"vlm/result/agentic_vision/{ds}_c{c}/*/metrics.json"):
            d = json.load(open(f)); md = d.get("metadata", {})
            m, g, npr = md.get("solver_model"), md.get("solver_gflops"), md.get("n_problems")
            acc = d.get("accuracy")
            if not (m and g and npr) or acc is None:
                continue
            comp = float(g)
            if T.INCLUDE_VISION:
                vt = VF.vision_gflops_total(m, ds, npr)          # one encode of the original image
                if vt:
                    comp += vt * (1 + (d.get("avg_crops") or 0))  # + one encode per zoom crop
            if m not in best or acc > best[m][0]:
                best[m] = (acc, comp)
    return {m: (comp, acc) for m, (acc, comp) in best.items()}


def collect(ds):
    """model -> dict(base, maj5, judge, zoom), each (g,a) or None.
    Compute is normalized to per-query GFLOPs (whole-dataset GFLOPs / n_problems)."""
    _, pipes = T.collect(ds)                 # canonical-priced judge pipelines
    best = T.best_judge(pipes)               # solver -> (gflops, acc, verifier)
    maj = MJ.collect(ds)                      # model -> [(compute, acc), ...] k=1..N
    zoom = collect_zoom(ds)                   # model -> (gflops, acc) at c=8
    nq = n_queries(ds) or 1                   # -> per-query average compute
    out = {}
    for m, curve in maj.items():
        base = curve[0]                      # (C[m], maj@1)
        maj5 = curve[4] if len(curve) >= 5 else curve[-1]
        rec = dict(base=base, maj5=maj5, judge=None, zoom=zoom.get(m))
        if m in best:
            rec["judge"] = (best[m][0], best[m][1])
        for k, v in rec.items():             # whole-dataset GFLOPs -> per-query
            if v is not None:
                rec[k] = (v[0] / nq, v[1])
        out[m] = rec
    return out


STRAT = [   # key,     marker, size, frontier color, style, legend label
    # each strategy's Pareto frontier gets its own line color. These are deliberately chosen to
    # avoid the family marker palette (Qwen purple / gemma blue / InternVL orange / llava red /
    # grey) -- black, green, magenta don't collide with any of them. Dash style is kept too.
    ("base",  "o", 95, "#000000", "-",  "base solo (1 pass)"),
    ("maj5",  "s", 70, "#2ca02c", "--", "maj@5 vote"),
    ("judge", "D", 60, "#e377c2", "-.", "+ best judge (k=5)"),
    ("zoom",  "^", 80, "#17becf", ":",  "zoom tool (best budget)"),
]
FADED, FULL = 0.18, 1.0


def render(ds, data, title=None):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    order = sorted(data, key=lambda m: (R.FAM_ORDER.get(R.family(m), 4), R.size(m)))

    # which models sit on each strategy's Pareto frontier
    frontier = {}
    for key, *_ in STRAT:
        pts = [(*data[m][key], m) for m in order if data[m][key] is not None]
        frontier[key] = [p[-1] for p in T.pareto(pts)]

    # scatter every point: on-frontier -> full opacity, else faded into the background.
    # collect all marker coords (to repel labels away from every point) and the on-frontier
    # label texts (placed non-overlappingly by adjust_text at the end).
    pxs, pys, texts, txs, tys = [], [], [], [], []   # txs/tys = the marker each label anchors to
    for key, mk, sz, *_ in STRAT:
        onfr = set(frontier[key])
        for m in order:
            v = data[m][key]
            if v is None:
                continue
            on = m in onfr
            ax.scatter(v[0], v[1], s=sz, marker=mk, alpha=FULL if on else FADED,
                       zorder=5 if on else 3, color=T.fam_color(m),
                       edgecolor="k" if on else "none")
            pxs.append(v[0]); pys.append(v[1])
            if on:                        # label only the models on this frontier
                # judge labels start on the LEFT of their marker (that side is less crowded);
                # adjust_text refines from there, leader lines re-anchored to the marker below.
                if key == "judge":
                    txt = ax.text(v[0] / 1.15, v[1], name(m), fontsize=7,
                                  ha="right", va="center", zorder=6)
                else:
                    txt = ax.text(v[0], v[1], name(m), fontsize=7, zorder=6)
                texts.append(txt); txs.append(v[0]); tys.append(v[1])

    # draw the frontier lines, and densify each into points so the labels treat the LINES as
    # obstacles too (not just the markers) -- sampled in log-x space to match the axis.
    import numpy as np
    handles = []
    for key, mk, sz, col, ls, lab in STRAT:
        line = [data[m][key] for m in frontier[key]]
        ax.plot([p[0] for p in line], [p[1] for p in line], ls, color=col, lw=2.2, zorder=4)
        handles.append(Line2D([], [], marker=mk, color=col, ls=ls, mec="k", ms=8, lw=2.2, label=lab))
        for (x0, y0), (x1, y1) in zip(line, line[1:]):     # sample along each segment
            t = np.linspace(0, 1, 12)
            pxs.extend(10 ** (np.log10(x0) + t * (np.log10(x1) - np.log10(x0))))
            pys.extend(y0 + t * (y1 - y0))
    ax.legend(handles=handles, fontsize=8, loc="lower right", title="Pareto frontier per strategy")
    ax.set_xscale("log")
    ax.set_xlabel("Avg Inference Compute Per Query (GFLOPs, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title or "Compute vs Accuracy Tradeoff Across Strategies")
    ax.grid(alpha=0.3, which="both")

    # nudge the labels apart so nothing overlaps the markers or each other; thin leader lines
    # connect a moved label back to its point.
    if adjust_text and texts:
        adjust_text(texts, x=pxs, y=pys, target_x=txs, target_y=tys, ax=ax,
                    expand=(1.3, 1.5), force_text=(0.4, 0.6), force_static=(0.2, 0.3),
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))
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
        for k, *_ in STRAT:
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
        out = render("avg", avg, title="Compute vs Accuracy Tradeoff Across Strategies")
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
