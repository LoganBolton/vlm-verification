"""Build one self-contained HTML report across all datasets of the VLM solver/verifier runs.

For each dataset directory (holding the `verify_*.json` grid produced by `vlm/vlm_verify.py`)
this emits:
  * a solver-accuracy table,
  * the 3x3 verifier-accuracy grid (with the trivial "accept-all" baseline marked),
  * a verifier F1 grid and a pooled leniency table (recall / specificity / bad-verdict rate),
  * the five PNG figures from `plot_results.py`, embedded as base64 so the report is portable,
  * links to the per-example HTML viewers from `view_examples.py`.

It does NOT run any model -- it only reads the result JSON + the already-rendered figures.
Run `plot_results.py` and `view_examples.py` first (the pipeline / `make_report` driver does).

Usage:
    python vlm/viz/make_report.py \
        --dataset CountBenchQA:vlm/result/countbench:vlm/viz/figures/countbench:views/countbench \
        --dataset CharXiv:vlm/result/charxiv:vlm/viz/figures/charxiv:views/charxiv \
        --out vlm/viz/REPORT.html
"""

import argparse
import base64
import glob
import html
import json
import os

SHORT = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "google/gemma-4-E2B-it": "Gemma-4-E2B",
    "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
}
ORDER = ["Qwen3-VL-2B", "Gemma-4-E2B", "LLaVA-1.5-7B"]
FIGS = ["solver_accuracy.png", "verifier_accuracy_heatmap.png", "verifier_vs_baseline.png",
        "verdict_breakdown.png", "leniency.png"]


def load_grid(result_dir):
    """cells[(solver,verifier)] = verifier metrics; solver_acc[solver] = acc; n = #examples."""
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


def heat(val, lo=0.2, hi=0.95):
    """Red->yellow->green background for a 0..1 metric (matches the heatmap figure)."""
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    r = int(220 if t < 0.5 else 220 - (t - 0.5) * 2 * 130)
    g = int(80 + t * 2 * 120) if t < 0.5 else 200
    return f"rgb({r},{g},90)"


def present_models(cells, solver_acc):
    """Only the models that actually have data (robust to a failed/missing model)."""
    solvers = [s for s in ORDER if s in solver_acc]
    verifiers = [v for v in ORDER if any((s, v) in cells for s in solvers)]
    return solvers, verifiers


def grid_table(cells, solver_acc, solvers, verifiers, metric):
    """An HTML solver x verifier table for a chosen verifier metric ('accuracy' or 'f1')."""
    head = "".join(f"<th>{v}</th>" for v in verifiers)
    rows = []
    for s in solvers:
        base = max(solver_acc[s], 1 - solver_acc[s])
        tds = []
        for v in verifiers:
            c = cells.get((s, v))
            if c is None:
                tds.append('<td class="na">—</td>'); continue
            val = c[metric]
            mark = " *" if (metric == "accuracy" and val > base + 1e-9) else ""
            tds.append(f'<td style="background:{heat(val)}">{val:.2f}{mark}</td>')
        rows.append(f"<tr><th class='rowh'>{s}<br><span class='sub'>solver acc "
                    f"{solver_acc[s]:.2f}</span></th>{''.join(tds)}</tr>")
    return (f"<table class='grid'><thead><tr><th class='corner'>solver \\ verifier</th>"
            f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def leniency_table(cells, solvers, verifiers):
    rows = []
    for v in verifiers:
        tp = sum(cells[(s, v)]["tp"] for s in solvers if (s, v) in cells)
        fn = sum(cells[(s, v)]["fn"] for s in solvers if (s, v) in cells)
        tn = sum(cells[(s, v)]["tn"] for s in solvers if (s, v) in cells)
        fp = sum(cells[(s, v)]["fp"] for s in solvers if (s, v) in cells)
        bad = sum(cells[(s, v)]["bad_count"] for s in solvers if (s, v) in cells)
        tot = tp + fn + tn + fp + bad
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(f"<tr><th class='rowh'>{v}</th><td>{recall:.2f}</td><td>{spec:.2f}</td>"
                    f"<td>{bad}</td><td>{tot}</td></tr>")
    return ("<table class='grid lean'><thead><tr><th class='corner'>verifier</th>"
            "<th>recall<br><span class='sub'>accepts true-correct</span></th>"
            "<th>specificity<br><span class='sub'>catches true-wrong</span></th>"
            "<th>bad<br><span class='sub'>no verdict</span></th>"
            f"<th>n (pooled)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def solver_table(solver_acc, cells, solvers, n):
    rows = []
    for s in solvers:
        # correct_count is identical across that solver's verifier rows; grab any.
        cc = next((cells[(s, v)] for v in ORDER if (s, v) in cells), None)
        nc = int(round(solver_acc[s] * n))
        base = max(solver_acc[s], 1 - solver_acc[s])
        rows.append(f"<tr><th class='rowh'>{s}</th><td>{solver_acc[s]:.2f}</td>"
                    f"<td>{nc}/{n}</td><td>{base:.2f}</td></tr>")
    return ("<table class='grid'><thead><tr><th class='corner'>solver</th><th>accuracy</th>"
            "<th>correct</th><th>trivial baseline<br><span class='sub'>majority class</span>"
            f"</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def embed_fig(path):
    if not os.path.exists(path):
        return f"<div class='missing'>[missing figure: {html.escape(os.path.basename(path))}]</div>"
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}' alt='{html.escape(os.path.basename(path))}'>"


def dataset_section(label, result_dir, fig_dir, views_rel, out_dir):
    cells, solver_acc, n = load_grid(result_dir)
    if not cells:
        return f"<section><h2>{html.escape(label)}</h2><p class='warn'>No verifier results found " \
               f"in <code>{html.escape(result_dir)}</code>.</p></section>"
    solvers, verifiers = present_models(cells, solver_acc)
    note = ""
    if "charxiv" in result_dir.lower():
        note = ("<p class='note'><b>Scoring note:</b> CharXiv answers are free-text. Correctness "
                "here uses a deterministic <i>relaxed normalized match</i> (see "
                "<code>vlm/score_charxiv.py</code>), not the reference GPT-4 judge — read absolute "
                "accuracies as approximate; the verifier comparisons are internally consistent.</p>")

    # view links (relative to the report's directory)
    links = []
    for s in solvers:
        href = os.path.join(views_rel, f"view_solver-{s.replace('/', '_')}.html")
        if os.path.exists(os.path.join(out_dir, href)):
            links.append(f"<a href='{html.escape(href)}'>{s}</a>")
    links_html = (" · ".join(links)) if links else "<span class='sub'>(run view_examples.py)</span>"

    figs_html = "".join(f"<figure>{embed_fig(os.path.join(fig_dir, fn))}</figure>" for fn in FIGS)

    return f"""
<section>
  <h2>{html.escape(label)} <span class='n'>n = {n}</span></h2>
  {note}
  <h3>Solver accuracy</h3>
  {solver_table(solver_acc, cells, solvers, n)}

  <h3>Verifier accuracy grid <span class='sub'>rows = solver judged · cols = verifier · * beats trivial baseline</span></h3>
  {grid_table(cells, solver_acc, solvers, verifiers, 'accuracy')}

  <h3>Verifier F1 grid <span class='sub'>positive class = "solver was correct"</span></h3>
  {grid_table(cells, solver_acc, solvers, verifiers, 'f1')}

  <h3>Leniency <span class='sub'>pooled over all solvers — high recall + low specificity = "yes-man"</span></h3>
  {leniency_table(cells, solvers, verifiers)}

  <h3>Figures</h3>
  <div class='figs'>{figs_html}</div>

  <h3>Per-example browsers</h3>
  <p class='links'>{links_html}</p>
</section>"""


def agentic_section(label, result_dir, views_rel, out_dir):
    """Summary of the agentic-vision (zoom-tool) runs + links to the rollout viewers.

    `result_dir` is an agentic_vision/<dataset> dir holding one <model>/metrics.json per
    model. We show accuracy, average zooms used, and the zoom-count distribution, then
    link each model's per-example rollout page (from view_agentic.py).
    """
    rows, links = [], []
    for mpath in sorted(glob.glob(os.path.join(result_dir, "*", "metrics.json"))):
        m = json.load(open(mpath))
        md = m.get("metadata", {})
        model = md.get("solver_model", os.path.basename(os.path.dirname(mpath)))
        s = SHORT.get(model, model.split("/")[-1])
        hist = m.get("crop_count_hist", {})
        hist_txt = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))
        rows.append(
            f"<tr><th class='rowh'>{html.escape(s)}</th>"
            f"<td style='background:{heat(m.get('accuracy', 0))}'>{m.get('accuracy', 0):.2f}</td>"
            f"<td>{md.get('n_problems', '—')}</td>"
            f"<td>{m.get('avg_crops', 0):.2f}</td>"
            f"<td>{m.get('frac_used_zoom', 0):.2f}</td>"
            f"<td class='hist'>{html.escape(hist_txt)}</td></tr>")
        href = os.path.join(views_rel, f"view_agentic-{s.replace('/', '_')}.html")
        if os.path.exists(os.path.join(out_dir, href)):
            links.append(f"<a href='{html.escape(href)}'>{html.escape(s)}</a>")
    if not rows:
        return ""
    links_html = (" · ".join(links)) if links else "<span class='sub'>(run view_agentic.py)</span>"
    table = ("<table class='grid'><thead><tr><th class='corner'>solver</th><th>accuracy</th>"
             "<th>n</th><th>avg zooms</th><th>used zoom<br><span class='sub'>frac &ge;1 crop</span></th>"
             "<th>zoom-count dist<br><span class='sub'>#crops:#examples</span></th></tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table>")
    return f"""
<section>
  <h2>{html.escape(label)} <span class='sub'>agentic vision — zoom tool</span></h2>
  <p class='note'><b>Active perception:</b> the solver may call a <code>zoom</code> tool to crop and
  magnify image regions (up to a budget), re-inspecting detail before answering. This is the third
  comparison leg alongside the verifier grid above and pass@N self-consistency.</p>
  <h3>Per-model summary</h3>
  {table}
  <h3>Per-example rollouts <span class='sub'>image with chosen zoom boxes + step-by-step tool calls</span></h3>
  <p class='links'>{links_html}</p>
</section>"""


CSS = """
:root { --bd:#dcdfe4; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; color:#1a1a1a; background:#f4f5f7;
       margin:0; line-height:1.5; }
header { background:#1f2937; color:#fff; padding:20px 28px; }
header h1 { margin:0; font-size:22px; }
header .sub { opacity:.8; font-size:14px; margin-top:6px; }
.wrap { max-width:1080px; margin:0 auto; padding:8px 24px 60px; }
section { background:#fff; border:1px solid var(--bd); border-radius:12px; margin:22px 0;
          padding:18px 22px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
h2 { margin:.2em 0 .6em; font-size:20px; } h2 .n { color:#6b7280; font-size:14px; font-weight:400; }
h3 { margin:1.3em 0 .5em; font-size:15px; }
.sub { color:#6b7280; font-size:12px; font-weight:400; }
table.grid { border-collapse:collapse; margin:6px 0 4px; font-size:14px; }
table.grid th, table.grid td { border:1px solid var(--bd); padding:8px 12px; text-align:center; }
table.grid thead th { background:#f3f4f6; font-weight:600; }
table.grid td { font-variant-numeric:tabular-nums; font-weight:600; min-width:74px; }
table.grid .rowh, table.grid .corner { background:#f3f4f6; text-align:left; font-weight:600; }
table.grid .sub { display:block; }
table.grid td.na { background:#f0f0f0; color:#999; font-weight:400; }
table.grid td.hist { font-weight:400; font-size:12px; color:#374151; }
.figs { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.figs figure { margin:0; border:1px solid var(--bd); border-radius:8px; padding:8px; background:#fff; }
.figs img { width:100%; height:auto; display:block; }
.note { background:#fff7e6; border:1px solid #ffe2a8; border-radius:8px; padding:8px 12px; font-size:13px; }
.warn { color:#c5221f; }
.missing { color:#999; font-style:italic; padding:20px; text-align:center; }
.links a { display:inline-block; margin-right:6px; color:#1a56db; text-decoration:none; font-weight:600; }
.links a:hover { text-decoration:underline; }
code { background:#f0f2f5; padding:1px 5px; border-radius:4px; font-size:.92em; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[],
                    help="LABEL:result_dir:fig_dir:views_rel  (repeatable)")
    ap.add_argument("--agentic", action="append", default=[],
                    help="LABEL:result_dir:views_rel  for agentic-vision runs (repeatable)")
    ap.add_argument("--out", default="vlm/viz/REPORT.html")
    args = ap.parse_args()
    if not args.dataset and not args.agentic:
        ap.error("need at least one --dataset or --agentic")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    sections = []
    for spec in args.dataset:
        label, result_dir, fig_dir, views_rel = spec.split(":", 3)
        sections.append(dataset_section(label, result_dir, fig_dir, views_rel, out_dir))
    for spec in args.agentic:
        label, result_dir, views_rel = spec.split(":", 2)
        sections.append(agentic_section(label, result_dir, views_rel, out_dir))

    from datetime import datetime
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VLM Solver–Verifier Report</title><style>{CSS}</style></head><body>
<header>
  <h1>VLM Solver–Verifier Report</h1>
  <div class="sub">Three VLMs (Qwen3-VL-2B, Gemma-4-E2B, LLaVA-1.5-7B) as both solver and
   verifier — every model verifies every model, on the full datasets.
   Generated {datetime.now():%Y-%m-%d %H:%M}.</div>
</header>
<div class="wrap">
  <section>
    <h2>How to read this</h2>
    <p>Each model <b>solves</b> the task, then each model acts as a <b>verifier</b> that judges
    whether a solver's answer is correct. A verifier is only useful if it beats the trivial
    "accept everything" baseline (the majority class). In the accuracy grid, <code>*</code> marks
    cells that clear that bar. <b>Recall</b> = fraction of truly-correct answers accepted;
    <b>specificity</b> = fraction of truly-wrong answers caught. A lenient "yes-man" has high
    recall but near-zero specificity.</p>
  </section>
  {''.join(sections)}
</div></body></html>"""
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
