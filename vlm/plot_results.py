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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHORT = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "google/gemma-4-E2B-it": "Gemma-4-E2B",
    "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
}
ORDER = ["Qwen3-VL-2B", "Gemma-4-E2B", "LLaVA-1.5-7B"]
COLORS = {"Qwen3-VL-2B": "#4C72B0", "Gemma-4-E2B": "#55A868", "LLaVA-1.5-7B": "#C44E52"}


def load_grid(result_dir):
    """Return (cells, solver_acc) where cells[(solver, verifier)] = verifier metrics dict."""
    cells, solver_acc = {}, {}
    for f in glob.glob(os.path.join(result_dir, "verify_*.json")):
        d = json.load(open(f))
        md, m = d["metadata"], d["metrics"]
        s = SHORT.get(md["solver_model"], md["solver_model"])
        v = SHORT.get(md["verifier_model"]["name"], md["verifier_model"]["name"])
        cells[(s, v)] = m["verifier"]
        solver_acc[s] = m["solver_accuracy"]
    return cells, solver_acc


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
    ax.set_title("Solver accuracy on CountBenchQA (n=100)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_heatmap(cells, solver_acc, path):
    grid = np.array([[cells[(s, v)]["accuracy"] for v in ORDER] for s in ORDER])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.2, vmax=0.95)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER)
    ax.set_yticks(range(3)); ax.set_yticklabels(ORDER)
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
    ax.set_ylabel("examples (of 100)")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="vlm/result",
                    help="Directory holding the verify_*.json files (may be git-ignored)")
    ap.add_argument("--fig_dir", default="vlm/figures",
                    help="Where to write PNGs (kept OUT of the git-ignored result dir)")
    ap.add_argument("--out_md", default="vlm/RESULTS_VISUAL.md")
    args = ap.parse_args()

    cells, solver_acc = load_grid(args.result_dir)
    assert len(cells) == 9, f"expected 9 verifier cells, found {len(cells)}"
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

    # figures live next to the md (vlm/result/figures); md is at vlm/RESULTS_VISUAL.md
    rel = os.path.relpath(fig_dir, os.path.dirname(os.path.abspath(args.out_md)))
    md = f"""# VLM Solver–Verifier Results — Visualized

CountBenchQA (n=100). Three VLMs as both solver and verifier; every model verifies every
model. See `vlm/REPORT.md` for the full writeup and `vlm/result/verify_*.json` for raw data.

## Solver accuracy
![solver accuracy]({rel}/solver_accuracy.png)

## Verifier accuracy grid
Rows = solver whose answers are being judged; columns = verifier. `*` marks cells that beat
the solver's trivial "accept-all" baseline.
![verifier heatmap]({rel}/verifier_accuracy_heatmap.png)

## Verifier vs. trivial baseline
A verifier only adds value if it clears the dashed line (the majority-class baseline).
Only Qwen3-VL-2B does so consistently.
![verifier vs baseline]({rel}/verifier_vs_baseline.png)

## Verdict breakdown (TP / TN / FP / FN / bad)
Positive class = "solver was correct". Large **orange (FP)** = the verifier rubber-stamps
wrong answers; **blue (TN)** = it actually caught wrong answers. LLaVA shows large grey
(no parseable verdict).
![verdict breakdown]({rel}/verdict_breakdown.png)

## Leniency: recall vs. specificity
High recall + low specificity = a lenient "yes-man" that accepts almost everything. Only
Qwen3-VL-2B has meaningful specificity (catches wrong answers).
![leniency]({rel}/leniency.png)
"""
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote", args.out_md)


if __name__ == "__main__":
    main()
