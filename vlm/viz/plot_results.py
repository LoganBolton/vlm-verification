"""Visualize the VLM solver--verifier results.

Reads the verifier run files (`verify_*.json`) in a result directory, builds the
solver x verifier grid, writes a set of PNG charts to `<result_dir>/figures/`, and emits
`vlm/RESULTS_VISUAL.md` embedding them.

Usage:
    python vlm/plot_results.py --result_dir vlm/result --out_md vlm/RESULTS_VISUAL.md
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHORT = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "google/gemma-4-E2B-it": "Gemma-4-E2B",
    "google/gemma-4-E4B-it": "Gemma-4-E4B",
    "google/gemma-4-12B-it": "Gemma-4-12B",
    "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-hf/llava-1.5-13b-hf": "LLaVA-1.5-13B",
    "OpenGVLab/InternVL3_5-1B": "InternVL3.5-1B",
    "OpenGVLab/InternVL3_5-2B": "InternVL3.5-2B",
    "OpenGVLab/InternVL3_5-4B": "InternVL3.5-4B",
    "OpenGVLab/InternVL3_5-8B": "InternVL3.5-8B",
    "OpenGVLab/InternVL3_5-14B": "InternVL3.5-14B",
}
FAMILY_ORDER = ["Qwen3-VL", "Gemma", "LLaVA", "InternVL"]
FAMILY_COLORS = {"Qwen3-VL": "#4C72B0", "Gemma": "#55A868", "LLaVA": "#C44E52",
                 "InternVL": "#8172B2"}


def _family(short_name):
    return next((f for f in FAMILY_ORDER if short_name.startswith(f)), short_name)


def _size(short_name):
    """Approximate parameter count for within-family ordering (E2B/2B -> 2, 14B -> 14)."""
    m = re.search(r"(\d+)[Bb]", short_name)
    return int(m.group(1)) if m else 0


# Derived from the loaded grid in main() -- model sets differ per tier.
ORDER = []
COLORS = {}

# Set in main() so titles/labels reflect the actual dataset + size being plotted.
DATASET_LABEL = "CountBenchQA"
N = 0


def load_grid(result_dir):
    """Return (cells, solver_acc, n) where cells[(solver, verifier)] = verifier metrics dict."""
    cells, solver_acc, n = {}, {}, 0
    for f in glob.glob(os.path.join(result_dir, "verify_*.json")):
        d = json.load(open(f))
        md, m = d["metadata"], d["metrics"]
        s = SHORT.get(md["solver_model"], md["solver_model"])
        v = SHORT.get(md["verifier_model"]["name"], md["verifier_model"]["name"])
        cells[(s, v)] = m["verifier"]
        solver_acc[s] = m["solver_accuracy"]
        n = max(n, m["verifier"]["total"])
    return cells, solver_acc, n


def fig_solver_accuracy(solver_acc, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(ORDER))
    accs = [solver_acc[s] for s in ORDER]
    triv = [max(solver_acc[s], 1 - solver_acc[s]) for s in ORDER]
    ax.bar(xs, accs, color=[COLORS[s] for s in ORDER], width=0.6)
    ax.plot(xs, triv, "k--o", label="trivial baseline\n(majority class)")
    for x, a in zip(xs, accs):
        ax.text(x, a + 0.01, f"{a:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(ORDER)
    ax.set_ylabel("solver accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Solver accuracy on {DATASET_LABEL} (n={N})")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_heatmap(cells, solver_acc, path):
    grid = np.array([[cells[(s, v)]["accuracy"] for v in ORDER] for s in ORDER])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.2, vmax=0.95)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
    ax.set_xlabel("VERIFIER"); ax.set_ylabel("SOLVER (answers being judged)")
    for i, s in enumerate(ORDER):
        for j, v in enumerate(ORDER):
            acc = grid[i, j]
            base = max(solver_acc[s], 1 - solver_acc[s])
            mark = " *" if acc > base + 1e-9 else ""
            ax.text(j, i, f"{acc:.2f}{mark}", ha="center", va="center",
                    color="black", fontweight="bold")
    ax.set_title("Verifier accuracy grid\n(* = beats the solver's trivial baseline)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_verifier_vs_baseline(cells, solver_acc, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(ORDER)); w = 0.25
    for k, v in enumerate(ORDER):
        vals = [cells[(s, v)]["accuracy"] for s in ORDER]
        ax.bar(xs + (k - 1) * w, vals, width=w, label=f"verifier: {v}", color=COLORS[v])
    triv = [max(solver_acc[s], 1 - solver_acc[s]) for s in ORDER]
    for x, t in zip(xs, triv):
        ax.plot([x - 1.5 * w, x + 1.5 * w], [t, t], "k--", lw=1.5)
    ax.plot([], [], "k--", label="trivial baseline")
    ax.set_xticks(xs); ax.set_xticklabels([f"judging\n{s}" for s in ORDER])
    ax.set_ylabel("verifier accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Verifier accuracy vs. trivial 'accept-all' baseline")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_confusion(cells, path):
    labels = [f"{v}\nverifying\n{s}" for v in ORDER for s in ORDER]
    tp = [cells[(s, v)]["tp"] for v in ORDER for s in ORDER]
    tn = [cells[(s, v)]["tn"] for v in ORDER for s in ORDER]
    fp = [cells[(s, v)]["fp"] for v in ORDER for s in ORDER]
    fn = [cells[(s, v)]["fn"] for v in ORDER for s in ORDER]
    bad = [cells[(s, v)]["bad_count"] for v in ORDER for s in ORDER]
    tp, tn, fp, fn, bad = map(np.array, (tp, tn, fp, fn, bad))
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    b = np.zeros(len(labels))
    for arr, c, lab in [(tp, "#2ca02c", "TP correct→correct"),
                        (tn, "#1f77b4", "TN wrong→wrong (caught)"),
                        (fp, "#ff7f0e", "FP wrong→correct (missed)"),
                        (fn, "#d62728", "FN correct→wrong (harsh)"),
                        (bad, "#999999", "bad (no verdict)")]:
        ax.bar(xs, arr, bottom=b, color=c, label=lab); b = b + arr
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(f"examples (of {N})")
    ax.set_title("Verdict breakdown per (verifier, solver) — big orange = lenient 'yes-man'")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_leniency(cells, path):
    """Recall (accept-the-correct) vs specificity (catch-the-wrong), averaged per verifier."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = np.arange(len(ORDER)); w = 0.35
    recall, spec = [], []
    for v in ORDER:
        tp = sum(cells[(s, v)]["tp"] for s in ORDER)
        fn = sum(cells[(s, v)]["fn"] for s in ORDER)
        tn = sum(cells[(s, v)]["tn"] for s in ORDER)
        fp = sum(cells[(s, v)]["fp"] for s in ORDER)
        recall.append(tp / (tp + fn) if (tp + fn) else 0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0)
    ax.bar(xs - w / 2, recall, width=w, label="recall (accepts true-correct)", color="#2ca02c")
    ax.bar(xs + w / 2, spec, width=w, label="specificity (catches true-wrong)", color="#1f77b4")
    for x, r, s in zip(xs, recall, spec):
        ax.text(x - w / 2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(x + w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(ORDER)
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate (pooled over all solvers)")
    ax.set_title("Verifier leniency: high recall + low specificity = 'yes-man'")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    global DATASET_LABEL, N
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="vlm/result",
                    help="Directory holding the verify_*.json files (may be git-ignored)")
    ap.add_argument("--fig_dir", default="vlm/viz/figures",
                    help="Where to write PNGs (kept OUT of the git-ignored result dir)")
    ap.add_argument("--out_md", default="vlm/viz/RESULTS_VISUAL.md")
    ap.add_argument("--dataset_label", default="CountBenchQA",
                    help="Human-readable dataset name used in figure titles/markdown")
    args = ap.parse_args()

    cells, solver_acc, n = load_grid(args.result_dir)
    global ORDER, COLORS
    ORDER = sorted(solver_acc, key=lambda s: (FAMILY_ORDER.index(_family(s))
                                              if _family(s) in FAMILY_ORDER else 99,
                                              _size(s), s))
    COLORS = {s: FAMILY_COLORS.get(_family(s), "#888888") for s in ORDER}
    assert len(cells) == len(ORDER) ** 2, \
        f"expected {len(ORDER) ** 2} verifier cells for {ORDER}, found {len(cells)}"
    DATASET_LABEL, N = args.dataset_label, n
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)

    figs = {
        "solver_accuracy.png": fig_solver_accuracy,
        "verifier_accuracy_heatmap.png": fig_heatmap,
        "verifier_vs_baseline.png": fig_verifier_vs_baseline,
        "verdict_breakdown.png": fig_confusion,
        "leniency.png": fig_leniency,
    }
    for name, fn in figs.items():
        p = os.path.join(fig_dir, name)
        if fn is fig_solver_accuracy:
            fn(solver_acc, p)
        elif fn in (fig_heatmap, fig_verifier_vs_baseline):
            fn(cells, solver_acc, p)
        else:
            fn(cells, p)
        print("wrote", p)

    rel = os.path.relpath(fig_dir, os.path.dirname(os.path.abspath(args.out_md)))
    md = f"""# VLM Solver–Verifier Results — Visualized

{DATASET_LABEL} (n={N}). Three VLMs as both solver and verifier; every model verifies every
model. Raw data: the `verify_*.json` files in `{args.result_dir}`.

## Solver accuracy
![solver accuracy]({rel}/solver_accuracy.png)

## Verifier accuracy grid
Rows = solver whose answers are being judged; columns = verifier. `*` marks cells that beat
the solver's trivial "accept-all" baseline.
![verifier heatmap]({rel}/verifier_accuracy_heatmap.png)

## Verifier vs. trivial baseline
A verifier only adds value if it clears the dashed line (the majority-class baseline).
![verifier vs baseline]({rel}/verifier_vs_baseline.png)

## Verdict breakdown (TP / TN / FP / FN / bad)
Positive class = "solver was correct". Large **orange (FP)** = the verifier rubber-stamps
wrong answers; **blue (TN)** = it actually caught wrong answers; **grey** = no parseable verdict.
![verdict breakdown]({rel}/verdict_breakdown.png)

## Leniency: recall vs. specificity
High recall + low specificity = a lenient "yes-man" that accepts almost everything;
high specificity means the verifier actually catches wrong answers.
![leniency]({rel}/leniency.png)
"""
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote", args.out_md)


if __name__ == "__main__":
    main()
