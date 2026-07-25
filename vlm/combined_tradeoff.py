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
MM_KS = [2, 3, 5, 8, 13]         # ensemble sizes plotted for the cross-MODEL random-vote line

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


def collect_model_majority(ds):
    """Cross-MODEL random majority-vote line: a few (k, per-query GFLOPs, accuracy) points.
    Unlike the per-model strategies, this votes ONE base answer from each of k DIFFERENT models
    (see vlm/model_majority.py). Accuracy at k is maj_at_k_random[k-1] -- the expected vote
    accuracy over a random k-model subset. Compute is the matching expected per-query cost of k
    random models = k * (mean single-model whole-dataset GFLOPs) / n_problems."""
    f = f"vlm/result/model_majority/{ds}/metrics.json"
    if not os.path.exists(f):
        return None
    m = json.load(open(f)); md = m["metadata"]
    M, P = md.get("n_models"), md.get("n_problems")
    total = md.get("total_ensemble_gflops")
    rnd = m.get("maj_at_k_random")
    if not (M and P and total and rnd):
        return None
    per_model = total / M                              # mean single-model whole-dataset GFLOPs
    return [(k, k * per_model / P, rnd[k - 1]) for k in MM_KS if k <= M]


def average_mm(per_ds_mm):
    """Average the cross-model line across datasets, aligned by ensemble size k."""
    dss = [d for d in per_ds_mm if per_ds_mm[d]]
    by_k = {}
    for d in dss:
        for k, c, a in per_ds_mm[d]:
            by_k.setdefault(k, []).append((c, a))
    out = []
    for k in MM_KS:
        vals = by_k.get(k)
        if vals and len(vals) == len(dss):
            out.append((k, sum(c for c, _ in vals) / len(vals), sum(a for _, a in vals) / len(vals)))
    return out


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

# Collapse redundant, near-coincident per-model labels: for these models several strategy points
# land almost on top of each other, so we keep only ONE label (the first on-frontier strategy in
# STRAT order). Maps a model-id substring -> the set of strategy keys that should share one label.
COLLAPSE = {
    "gemma-4-E2B": {"maj5", "zoom"},           # zoom + maj@5 sit together -> single label
    "gemma-4-12B": {"maj5", "judge", "zoom"},  # zoom + judge + maj@5 sit together -> single label
}

# (strategy, model-substring) label to suppress: the marker/frontier point stays, just no text.
OMIT_LABELS = [("judge", "Qwen3-VL-4B"),       # declutter the crowded best-judge region
               ("judge", "InternVL3_5-4B"), ("judge", "gemma-4-E2B"),
               ("judge", "gemma-4-E4B"),
               ("zoom", "gemma-4-E4B"), ("zoom", "InternVL3_5-4B"),
               ("maj5", "InternVL3_5-4B")]

# (strategy, model-substring) -> which side of the marker the label sits on ("left"/"right"),
# overriding the defaults (judge -> left, everything else -> right of the point).
LABEL_SIDE = {
    ("judge", "gemma-4-E2B"): "right",
    ("zoom",  "gemma-4-E4B"): "left",
    ("maj5",  "Qwen3-VL-2B"): "left",
}
# (strategy, model-substring) -> (side, gap-factor): PIN the label at a fixed close offset on that
# side of the marker and keep it out of adjust_text, so it stays put instead of drifting far away.
# gap-factor is the log-x divisor/multiplier -> smaller = closer to the marker.
LABEL_PIN = {
    ("judge", "Qwen3-VL-2B"): ("left", 1.06),
    ("judge", "Qwen3-VL-8B"): ("left", 1.05),      # snug to the left of its diamond
    ("maj5", "Qwen3-VL-2B"): ("left", 1.05),       # snug to the left of its square
}
# extra vertical nudge (accuracy units, +up/-down) for specific labels, applied after side placement.
LABEL_DY = {
    ("zoom", "gemma-4-E4B"): -0.045,    # move down, staying left
}


def render(ds, data, title=None, ymin=None, mm=None):
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
    leader = []                          # (text, [points]) -> thin leader line(s) drawn at the end
    collapse_pts = {}                    # collapsed model -> its near-coincident points (one shared label)
    for key, mk, sz, *_ in STRAT:
        onfr = set(frontier[key])
        for m in order:
            v = data[m][key]
            if v is None:
                continue
            if m not in onfr:             # only draw on-frontier points; hide the rest entirely
                continue
            ax.scatter(v[0], v[1], s=sz, marker=mk, alpha=FULL,
                       zorder=5, color=T.fam_color(m), edgecolor="k")
            pxs.append(v[0]); pys.append(v[1])
            if True:                      # label only the models on this frontier
                # collapsed models: stash each near-coincident point; one shared label is built
                # after the loop at the centroid, with a leader line fanning out to every point.
                rule = next((keys for sub, keys in COLLAPSE.items() if sub in m), None)
                if rule is not None and key in rule:
                    collapse_pts.setdefault(m, []).append((v[0], v[1]))
                    continue
                # base labels are PINNED to the left of the far-left frontier line (kept out of
                # adjust_text, which otherwise bounces the edge ones back to the right). The two
                # left-edge models (gemma-4-E2B, Qwen3-VL-2B) have no room on the left, so they go
                # to the RIGHT of their markers (below / above-right) to stay on the plot.
                if key == "base":
                    if "gemma-4-E2B" in m:
                        txt = ax.text(v[0] * 1.1, v[1] - 0.02, name(m), fontsize=10,
                                      ha="center", va="top", zorder=6)
                    elif "Qwen3-VL-2B" in m:
                        txt = ax.text(v[0] / 1.02, v[1] + 0.015, name(m), fontsize=10,
                                      ha="right", va="bottom", zorder=6)
                    elif "gemma-4-12B" in m:
                        txt = ax.text(v[0] / 1.08, v[1], name(m), fontsize=10,
                                      ha="right", va="center", zorder=6)
                    else:
                        txt = ax.text(v[0] / 1.18, v[1], name(m), fontsize=10,
                                      ha="right", va="center", zorder=6)
                    leader.append((txt, [(v[0], v[1])]))
                    continue
                # suppress specific labels (marker stays) to declutter crowded frontiers.
                if any(key == k and sub in m for k, sub in OMIT_LABELS):
                    continue
                # pinned labels: fixed close offset, kept out of adjust_text so they don't drift.
                pin = next(((side, fac) for (k, sub), (side, fac) in LABEL_PIN.items()
                            if key == k and sub in m), None)
                if pin is not None:
                    side, fac = pin
                    pdy = next((d for (k, sub), d in LABEL_DY.items() if key == k and sub in m), 0.0)
                    if side == "down":                       # fac = vertical gap in accuracy units
                        txt = ax.text(v[0], v[1] - fac, name(m), fontsize=10,
                                      ha="center", va="top", zorder=6)
                        leader.append((txt, [(v[0], v[1])]))
                        pxs.append(v[0]); pys.append(v[1] - fac)
                        continue
                    x = v[0] / fac if side == "left" else v[0] * fac
                    txt = ax.text(x, v[1] + pdy, name(m), fontsize=10,
                                  ha="right" if side == "left" else "left", va="center", zorder=6)
                    leader.append((txt, [(v[0], v[1])]))
                    pxs.append(x); pys.append(v[1] + pdy)
                    continue
                # judge labels start on the LEFT of their marker (that side is less crowded);
                # adjust_text refines from there, leader lines re-anchored to the marker below.
                override = next((s for (k, sub), s in LABEL_SIDE.items() if key == k and sub in m), None)
                dy = next((d for (k, sub), d in LABEL_DY.items() if key == k and sub in m), 0.0)
                if override == "right":
                    txt = ax.text(v[0] * 1.15, v[1] + dy, name(m), fontsize=10,
                                  ha="left", va="center", zorder=6)
                elif override == "left":
                    txt = ax.text(v[0] / 1.15, v[1] + dy, name(m), fontsize=10,
                                  ha="right", va="center", zorder=6)
                elif key == "judge":
                    txt = ax.text(v[0] / 1.15, v[1] + dy, name(m), fontsize=10,
                                  ha="right", va="center", zorder=6)
                else:
                    txt = ax.text(v[0], v[1] + dy, name(m), fontsize=10, zorder=6)
                texts.append(txt); txs.append(v[0]); tys.append(v[1])
                leader.append((txt, [(v[0], v[1])]))

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
    # cross-MODEL random majority-vote line: a separate k-sweep (2..13 random models voting), not a
    # per-model strategy, so it gets its own brown star line + one "k models" tag per point.
    if mm:
        mxs = [c for _, c, _ in mm]; mys = [a for _, _, a in mm]
        ax.plot(mxs, mys, "-", color="#4b0082", lw=2.2, zorder=4)
        ax.scatter(mxs, mys, s=150, marker="*", color="#4b0082", edgecolor="k", zorder=5)
        handles.append(Line2D([], [], marker="*", color="#4b0082", ls="-", mec="k", ms=12, lw=2.2,
                              label="cross-model maj@k (random)"))
        for k, c, a in mm:
            if k == 3:
                dx, ha = 0.91, "left"      # 3-models: slight left of its old spot
            elif k == 5:
                dx, ha = 1.02, "left"      # 5-models: half a hair right
            else:
                dx, ha = 1.0, "center"
            ax.text(c * dx, a - 0.018, f"{k} models", fontsize=9, ha=ha, va="top",
                    color="#4b0082", zorder=6)
        pxs.extend(mxs); pys.extend(mys)          # register as obstacles for label placement

    ax.legend(handles=handles, fontsize=8, loc="lower right", title="Pareto frontier per strategy")
    ax.set_xscale("log")
    ax.set_xlabel("Average Inference Compute Per Query (higher is worse)")
    ax.set_ylabel("accuracy")
    ax.set_title(title or "Compute vs Accuracy Tradeoff Across Strategies")
    ax.grid(alpha=0.3, which="both")
    right = 1.1e5                    # cut the x-axis off around 10^5 (all frontier points sit well left)
    if mm:                           # ...but leave room for the cross-model line's high-k points
        right = max(right, max(c for _, c, _ in mm) * 1.6)
    ax.set_xlim(right=right)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    # one merged label per collapsed model: since its points are near-coincident there's no room
    # for adjust_text to repel a centered label off them, so place it deterministically just BELOW
    # the cluster -- horizontally at the centroid (mean in log-x, x being log-scaled), vertically a
    # fixed gap under the lowest point -- then fan a leader line up to each shape. It's registered
    # as an obstacle (pxs/pys) so the other, adjust_text-managed labels steer around it.
    for m, pts in collapse_pts.items():
        if "gemma-4-12B" in m:                       # this cluster sits at the top -> label to its RIGHT
            xlab = 10 ** (max(np.log10(p[0]) for p in pts)) * 1.15
            ylab = sum(p[1] for p in pts) / len(pts) + 0.012   # nudged up a bit
            txt = ax.text(xlab, ylab, name(m), fontsize=10, ha="left", va="center", zorder=6)
        else:                                        # default: just below the cluster
            xlab = 10 ** (sum(np.log10(p[0]) for p in pts) / len(pts))
            ylab = min(p[1] for p in pts) - 0.017
            txt = ax.text(xlab, ylab, name(m), fontsize=10, ha="center", va="top", zorder=6)
        leader.append((txt, pts))
        pxs.append(xlab); pys.append(ylab)

    # nudge the labels apart so nothing overlaps the markers or each other (positioning only).
    if adjust_text and texts:
        adjust_text(texts, x=pxs, y=pys, target_x=txs, target_y=tys, ax=ax,
                    expand=(1.3, 1.5), force_text=(0.4, 0.6), force_static=(0.2, 0.3))
    # thin leader lines: connect each label to its point(s). Anchor the line at the spot on the
    # label's bounding box CLOSEST to the marker (a corner/edge of the word) instead of the text
    # center, so it stops next to the word. Measured after layout is final -> valid bboxes.
    fig.tight_layout()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for txt, pts in leader:
        bb = txt.get_window_extent(rend)                 # label rect in display (pixel) coords
        for px, py in pts:
            dx, dy = ax.transData.transform((px, py))    # marker in display coords
            cx = min(max(dx, bb.x0), bb.x1)              # nearest point on the label rect...
            cy = min(max(dy, bb.y0), bb.y1)              # ...to the marker (corner or edge)
            ex, ey = inv.transform((cx, cy))             # back to data coords
            ax.annotate("", xy=(px, py), xytext=(ex, ey), zorder=3,
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, shrinkA=2, shrinkB=4))
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
    per_ds, per_ds_mm = {}, {}
    for ds in DATASETS:
        data = collect(ds)
        if not data:
            print(f"[{ds}] no data"); continue
        per_ds[ds] = data
        per_ds_mm[ds] = collect_model_majority(ds)
        print(f"  -> wrote {render(ds, data, mm=per_ds_mm[ds])}")
    if len(per_ds) == len(DATASETS):
        avg = average(per_ds)
        out = render("avg", avg, title="Compute vs Accuracy Tradeoff Across Strategies", ymin=0.35,
                     mm=average_mm(per_ds_mm))
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
