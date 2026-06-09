"""Per-example viewer: one solver answer with every verifier's response stacked on top.

For each solver, this joins its three `verify_*.json` files (one per verifier) on the
example id and emits a self-contained HTML page. Each example is one card showing:

  * the image,
  * the exact prompt fed to the solver and the solver's full output,
  * whether the solver was *counted* right (extracted count vs. gold),
  * one row per verifier with that verifier's full response, its parsed verdict, and
    whether the verifier was *counted* right (verdict matches the solver's correctness).

Images are base64-embedded so each HTML file is fully portable (open it anywhere).

Usage:
    python vlm/view_examples.py                       # all solvers -> vlm/views/*.html
    python vlm/view_examples.py --solver Qwen         # only solvers whose name matches
    python vlm/view_examples.py --limit 20            # first 20 examples per page
"""

import argparse
import base64
import glob
import html
import json
import mimetypes
import os

VLM_DIR = os.path.dirname(os.path.abspath(__file__))

SHORT = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "google/gemma-4-E2B-it": "Gemma-4-E2B",
    "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
}
VERIFIER_ORDER = ["Qwen3-VL-2B", "Gemma-4-E2B", "LLaVA-1.5-7B"]


def short(name):
    return SHORT.get(name, name)


def slug(name):
    return short(name).replace("/", "_")


def embed_image(path):
    """Return a data: URI for the image, or None if it can't be read."""
    if not path or not os.path.exists(path):
        return None
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_solver_prompts(result_dir):
    """Map (solver_model, example_id) -> exact text prompt fed to the solver."""
    prompts = {}
    for f in glob.glob(os.path.join(result_dir, "countbench_*.json")):
        if f.endswith("_scores.json"):
            continue
        d = json.load(open(f))
        # solver run files store the model under metadata.model.name
        model = d.get("metadata", {}).get("model", {}).get("name")
        if not model:
            continue
        for r in d.get("records", []):
            prompts[(model, r["id"])] = r.get("text_prompt")
    return prompts


def collect(result_dir):
    """Group verify files by solver: solver_model -> {id -> example dict}."""
    solver_prompts = load_solver_prompts(result_dir)
    solvers = {}  # solver_model -> {id -> example}
    for f in sorted(glob.glob(os.path.join(result_dir, "verify_*.json"))):
        d = json.load(open(f))
        md = d["metadata"]
        s_model = md["solver_model"]
        v_model = md["verifier_model"]["name"]
        bucket = solvers.setdefault(s_model, {})
        for rec in d["records"]:
            ex = bucket.get(rec["id"])
            if ex is None:
                ex = {
                    "id": rec["id"],
                    "image": rec["image"],
                    "question": rec["question"],
                    "gold": rec["answer"],
                    "solver_prompt": solver_prompts.get((s_model, rec["id"])),
                    "solver_output": rec["solver_full_output"],
                    "solver_extracted": rec.get("solver_extracted_answer"),
                    "solver_correct": rec["solver_correct"],
                    "verifiers": {},
                }
                bucket[rec["id"]] = ex
            verdict = rec["verifier_verdict"]
            sc = bool(rec["solver_correct"])
            if verdict is None:
                v_correct = None  # no parseable verdict
            else:
                v_correct = (bool(verdict) == sc)
            ex["verifiers"][short(v_model)] = {
                "response": rec["verifier_response"],
                "verdict": verdict,
                "verifier_correct": v_correct,
            }
    return solvers


def badge(ok, true_txt="correct", false_txt="incorrect", none_txt="no verdict"):
    if ok is None:
        return f'<span class="badge bad">{none_txt}</span>'
    cls = "ok" if ok else "no"
    return f'<span class="badge {cls}">{true_txt if ok else false_txt}</span>'


def render_example(ex):
    img = embed_image(ex["image"])
    img_html = (f'<img src="{img}" alt="example {ex["id"]}">' if img
                else '<div class="noimg">[image not found]</div>')
    esc = html.escape

    rows = []
    for v in VERIFIER_ORDER:
        info = ex["verifiers"].get(v)
        if info is None:
            continue
        verdict = info["verdict"]
        verdict_txt = ("—" if verdict is None
                       else ("says CORRECT" if verdict else "says INCORRECT"))
        rows.append(f"""
        <tr>
          <td class="vname">{esc(v)}</td>
          <td class="vverdict">{verdict_txt}</td>
          <td class="vjudge">{badge(info["verifier_correct"], "verifier right", "verifier wrong")}</td>
          <td class="vresp"><pre>{esc(info["response"].strip())}</pre></td>
        </tr>""")

    solver_correct = badge(ex["solver_correct"])
    extracted = "—" if ex["solver_extracted"] is None else esc(str(ex["solver_extracted"]))
    return f"""
    <div class="card">
      <div class="head">
        <div class="imgwrap">{img_html}</div>
        <div class="meta">
          <div class="idline">#{ex["id"]} &nbsp; gold count = <b>{ex["gold"]}</b></div>
          <div class="prompt"><span class="lab">solver prompt</span>
            <pre>{esc((ex["solver_prompt"] or ex["question"]).strip())}</pre></div>
          <div class="solver"><span class="lab">solver output</span>
            <pre>{esc(ex["solver_output"].strip())}</pre></div>
          <div class="verdict-line">
            solver extracted = <b>{extracted}</b> vs gold <b>{ex["gold"]}</b> &rarr; {solver_correct}
          </div>
        </div>
      </div>
      <table class="verifiers">
        <thead><tr><th>verifier</th><th>verdict</th><th>scored</th><th>response</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f4f5f7;
       color: #1a1a1a; }
header { position: sticky; top: 0; background: #1f2937; color: #fff; padding: 14px 22px; z-index: 5; }
header h1 { margin: 0; font-size: 18px; }
header .sub { font-size: 13px; opacity: .8; margin-top: 4px; }
.wrap { max-width: 1100px; margin: 18px auto; padding: 0 16px; }
.card { background: #fff; border: 1px solid #dcdfe4; border-radius: 10px; margin: 16px 0;
        overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.head { display: flex; gap: 16px; padding: 14px; border-bottom: 1px solid #eceef1; }
.imgwrap { flex: 0 0 280px; }
.imgwrap img { width: 280px; border-radius: 6px; border: 1px solid #d0d3d8; }
.noimg { width: 280px; height: 180px; display: flex; align-items: center; justify-content: center;
         background: #f0f0f0; color: #999; border-radius: 6px; }
.meta { flex: 1; min-width: 0; }
.idline { font-size: 15px; margin-bottom: 8px; }
.lab { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
       color: #6b7280; font-weight: 600; margin-bottom: 2px; }
pre { margin: 2px 0 10px; white-space: pre-wrap; word-wrap: break-word; font-family:
      ui-monospace, Menlo, Consolas, monospace; font-size: 12.5px; background: #f8f9fb;
      border: 1px solid #eceef1; border-radius: 6px; padding: 8px; max-height: 240px; overflow: auto; }
.verdict-line { font-size: 14px; margin-top: 4px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 11px; font-size: 12px;
         font-weight: 700; }
.badge.ok { background: #d6f5dd; color: #137333; }
.badge.no { background: #fbe1e1; color: #c5221f; }
.badge.bad { background: #e6e6e6; color: #666; }
table.verifiers { width: 100%; border-collapse: collapse; font-size: 13px; }
table.verifiers th { text-align: left; background: #f3f4f6; padding: 7px 12px; color: #374151;
                     font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
table.verifiers td { padding: 9px 12px; border-top: 1px solid #eceef1; vertical-align: top; }
.vname { font-weight: 700; white-space: nowrap; }
.vverdict { white-space: nowrap; }
.vresp { width: 100%; }
.vresp pre { margin: 0; max-height: 180px; }
"""


def render_page(solver_model, examples, limit):
    examples = sorted(examples.values(), key=lambda e: e["id"])
    if limit:
        examples = examples[:limit]
    n = len(examples)
    n_solver_ok = sum(1 for e in examples if e["solver_correct"])
    cards = "\n".join(render_example(e) for e in examples)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{short(solver_model)} — solver vs. verifiers</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Solver: {html.escape(short(solver_model))} &nbsp;·&nbsp; every verifier stacked per example</h1>
  <div class="sub">{n} examples · solver counted right on {n_solver_ok}/{n} ·
    each row = one verifier's response and whether its verdict matched the solver's correctness</div>
</header>
<div class="wrap">{cards}</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default=os.path.join(VLM_DIR, "result"))
    ap.add_argument("--out_dir", default=os.path.join(VLM_DIR, "views"))
    ap.add_argument("--solver", default=None,
                    help="substring filter on solver name (e.g. 'Qwen', 'gemma', 'llava')")
    ap.add_argument("--limit", type=int, default=None, help="max examples per page")
    args = ap.parse_args()

    solvers = collect(args.result_dir)
    if not solvers:
        raise SystemExit(f"no verify_*.json found in {args.result_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    written = []
    for s_model, examples in sorted(solvers.items()):
        if args.solver and args.solver.lower() not in short(s_model).lower():
            continue
        out = os.path.join(args.out_dir, f"view_solver-{slug(s_model)}.html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_page(s_model, examples, args.limit))
        written.append(out)
        print(f"wrote {out}  ({min(len(examples), args.limit or len(examples))} examples)")

    if not written:
        print("nothing matched --solver filter")


if __name__ == "__main__":
    main()
