"""Per-example rollout viewer for the agentic-vision (zoom-tool) runs.

For each `agentic_vision/<dataset>/<model>/records.json` this emits one self-contained
HTML page where every example is a card showing the *whole rollout*:

  * the original image with every requested zoom region drawn on it (numbered, colored),
  * the question, gold answer, the model's final extracted answer, and correctness,
  * a step-by-step timeline of the conversation: each assistant turn (reasoning + the
    <tool_call> it emitted, with the zoom call highlighted) followed by the cropped,
    magnified image the tool returned for that call.

Crops are reconstructed from the original image + the stored pixel box (the runtime crops
aren't saved to disk), then base64-embedded, so each HTML file is fully portable.

Usage:
    # one dataset dir -> one page per model
    python vlm/viz/view_agentic.py --result_dir vlm/result/agentic_vision/countbench \
        --out_dir vlm/viz/views/agentic_countbench --limit 200
"""

import argparse
import base64
import glob
import html
import io
import json
import os
import re

from PIL import Image

VLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../vlm

SHORT = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
# Distinct colors for the numbered zoom boxes (cycled if there are more crops).
BOX_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
_TOOLCALL_RE = re.compile(r"(<tool_call>.*?</tool_call>)", re.S)


def short(name):
    return SHORT.get(name, name.split("/")[-1])


def slug(name):
    return short(name).replace("/", "_")


def embed_pil(im, fmt="JPEG", quality=85):
    """Return a data: URI for a PIL image."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def badge(ok, true_txt="correct", false_txt="incorrect"):
    cls = "ok" if ok else "no"
    return f'<span class="badge {cls}">{true_txt if ok else false_txt}</span>'


def overlay_boxes_html(img_uri, W, H, crops):
    """Original image with one numbered, colored rectangle per zoom (CSS % overlay)."""
    marks = []
    for k, c in enumerate(crops):
        x1, y1, x2, y2 = c["pixel_box"]
        col = BOX_COLORS[k % len(BOX_COLORS)]
        left, top = 100 * x1 / W, 100 * y1 / H
        w, h = 100 * (x2 - x1) / W, 100 * (y2 - y1) / H
        marks.append(
            f'<div class="bx" style="left:{left:.2f}%;top:{top:.2f}%;width:{w:.2f}%;'
            f'height:{h:.2f}%;border-color:{col}"><span style="background:{col}">{k + 1}</span></div>')
    return (f'<div class="imgbox"><img src="{img_uri}" alt="original">'
            f'{"".join(marks)}</div>')


def render_assistant(text):
    """Assistant turn: reasoning prose with any <tool_call> block highlighted."""
    parts = []
    for seg in _TOOLCALL_RE.split(text.strip()):
        if not seg.strip():
            continue
        if seg.lstrip().startswith("<tool_call>"):
            parts.append(f'<pre class="toolcall">{html.escape(seg.strip())}</pre>')
        else:
            parts.append(f'<pre class="say">{html.escape(seg.strip())}</pre>')
    return "".join(parts) or '<pre class="say">(empty)</pre>'


def render_rollout(ex, orig, W, H):
    """The step-by-step timeline of turns, with each zoom's crop image inline."""
    crops = ex.get("crops", [])
    steps, seen_first_user = [], False
    turns = ex.get("turns")
    if turns is None:  # fallback for records saved before `turns` existed
        turns = [{"role": "assistant", "text": ex["solver_full_output"]}]
    for t in turns:
        if t["role"] == "assistant":
            steps.append(f'<div class="step asst"><div class="who">model</div>'
                         f'<div class="body">{render_assistant(t["text"])}</div></div>')
        else:
            if not seen_first_user:
                seen_first_user = True
                continue  # the initial prompt turn (image shown in the header already)
            k = t.get("crop")
            if k is None:
                # text-only turn = the forced-answer instruction (budget exhausted)
                steps.append(f'<div class="step force"><div class="who">⏱ forced</div>'
                             f'<div class="body"><pre class="say">{html.escape(t["text"].strip())}'
                             f'</pre></div></div>')
                continue
            c = crops[k] if k < len(crops) else None
            col = BOX_COLORS[k % len(BOX_COLORS)]
            crop_html = ""
            if c is not None:
                x1, y1, x2, y2 = c["pixel_box"]
                crop_img = orig.crop((x1, y1, x2, y2))
                crop_html = (f'<img class="cropimg" style="border-color:{col}" '
                             f'src="{embed_pil(crop_img)}" alt="zoom {k + 1}">')
                req = c.get("requested")
                cap = (f'<div class="cap">zoom <b style="color:{col}">#{k + 1}</b> · '
                       f'requested {req} · pixel box {list(c["pixel_box"])}</div>')
            else:
                cap = '<div class="cap">zoom (box unavailable)</div>'
            steps.append(f'<div class="step tool"><div class="who">🔍 zoom</div>'
                         f'<div class="body">{cap}{crop_html}</div></div>')
    return "".join(steps)


def render_example(ex):
    esc = html.escape
    path = ex["image"]
    if path and os.path.exists(path):
        orig = Image.open(path).convert("RGB")
        W, H = orig.size
        header_img = overlay_boxes_html(embed_pil(orig), W, H, ex.get("crops", []))
    else:
        orig, W, H = None, 1, 1
        header_img = '<div class="noimg">[image not found]</div>'

    extracted = "—" if ex.get("solver_extracted_answer") is None else esc(str(ex["solver_extracted_answer"]))
    n_crops = ex.get("n_crops", len(ex.get("crops", [])))
    rollout = render_rollout(ex, orig, W, H) if orig is not None else ""
    return f"""
    <div class="card">
      <div class="head">
        <div class="imgwrap">{header_img}</div>
        <div class="meta">
          <div class="idline">#{esc(str(ex["id"]))} &nbsp; gold = <b>{esc(str(ex["answer"]))}</b>
            &nbsp;·&nbsp; <span class="zoomcount">{n_crops} zoom(s)</span></div>
          <div class="qline"><span class="lab">question</span>
            <div class="q">{esc(str(ex["question"]).strip())}</div></div>
          <div class="verdict-line">final answer = <b>{extracted}</b> vs gold
            <b>{esc(str(ex["answer"]))}</b> &rarr; {badge(ex["solver_correct"])}</div>
        </div>
      </div>
      <div class="rollout"><div class="rlabel">rollout</div>{rollout}</div>
    </div>"""


CSS = """
* { box-sizing: border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#f4f5f7; color:#1a1a1a; }
header { position:sticky; top:0; background:#1f2937; color:#fff; padding:14px 22px; z-index:5; }
header h1 { margin:0; font-size:18px; } header .sub { font-size:13px; opacity:.85; margin-top:4px; }
.wrap { max-width:1100px; margin:18px auto; padding:0 16px; }
.card { background:#fff; border:1px solid #dcdfe4; border-radius:10px; margin:16px 0; overflow:hidden;
        box-shadow:0 1px 3px rgba(0,0,0,.06); }
.head { display:flex; gap:16px; padding:14px; border-bottom:1px solid #eceef1; }
.imgwrap { flex:0 0 320px; }
.imgbox { position:relative; width:320px; line-height:0; }
.imgbox img { width:320px; border-radius:6px; border:1px solid #d0d3d8; }
.bx { position:absolute; border:2.5px solid; border-radius:3px; box-shadow:0 0 0 1px rgba(0,0,0,.25); }
.bx span { position:absolute; top:-11px; left:-2px; color:#fff; font-size:11px; font-weight:700;
           line-height:1; padding:2px 5px; border-radius:8px; }
.noimg { width:320px; height:200px; display:flex; align-items:center; justify-content:center;
         background:#f0f0f0; color:#999; border-radius:6px; }
.meta { flex:1; min-width:0; }
.idline { font-size:15px; margin-bottom:8px; }
.zoomcount { background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px; padding:2px 8px; border-radius:10px; }
.lab { display:inline-block; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
       color:#6b7280; font-weight:600; margin-bottom:2px; }
.q { font-size:14px; margin-bottom:10px; }
.verdict-line { font-size:14px; margin-top:6px; }
.badge { display:inline-block; padding:2px 9px; border-radius:11px; font-size:12px; font-weight:700; }
.badge.ok { background:#d6f5dd; color:#137333; } .badge.no { background:#fbe1e1; color:#c5221f; }
.rollout { padding:12px 14px 16px; }
.rlabel { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#6b7280;
          font-weight:600; margin-bottom:8px; }
.step { display:flex; gap:10px; margin:0 0 10px; }
.step .who { flex:0 0 70px; font-size:12px; font-weight:700; color:#374151; padding-top:3px; }
.step.tool .who { color:#b45309; }
.step.force .who { color:#6d28d9; }
.step .body { flex:1; min-width:0; border-left:3px solid #e5e7eb; padding-left:10px; }
.step.tool .body { border-left-color:#fbbf24; }
.step.force .body { border-left-color:#a78bfa; }
pre { margin:0 0 6px; white-space:pre-wrap; word-wrap:break-word; font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:12.5px; background:#f8f9fb; border:1px solid #eceef1; border-radius:6px; padding:8px;
      max-height:300px; overflow:auto; }
pre.toolcall { background:#fffbeb; border-color:#fde68a; color:#92400e; }
.cap { font-size:12px; color:#6b7280; margin-bottom:4px; }
.cropimg { max-width:360px; max-height:260px; border:3px solid; border-radius:6px; }
"""


def render_page(model, dataset, accuracy, examples, limit):
    examples = sorted(examples, key=lambda e: e["id"])
    if limit:
        examples = examples[:limit]
    n = len(examples)
    n_ok = sum(1 for e in examples if e["solver_correct"])
    avg_crops = sum(e.get("n_crops", 0) for e in examples) / n if n else 0
    cards = "\n".join(render_example(e) for e in examples)
    acc_txt = f" · full-run accuracy {accuracy:.2f}" if accuracy is not None else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(short(model))} — agentic-vision rollouts ({html.escape(dataset)})</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Agentic vision: {html.escape(short(model))} &nbsp;·&nbsp; {html.escape(dataset)}</h1>
  <div class="sub">{n} examples · correct on {n_ok}/{n}{acc_txt} · avg {avg_crops:.1f} zoom(s)/example ·
    boxes on the image are the regions the model chose to magnify (numbered in call order)</div>
</header>
<div class="wrap">{cards}</div>
</body></html>"""


def find_runs(result_dir):
    """Yield (model_name, dataset_name, accuracy, records, out_slug) for each records.json.

    Accepts either a single model dir (containing records.json) or a parent dir holding
    several model subdirs.
    """
    paths = []
    if os.path.exists(os.path.join(result_dir, "records.json")):
        paths.append(os.path.join(result_dir, "records.json"))
    else:
        paths += sorted(glob.glob(os.path.join(result_dir, "*", "records.json")))
    for p in paths:
        d = json.load(open(p))
        md = d.get("metadata", {})
        model = md.get("solver_model") or os.path.basename(os.path.dirname(p))
        dataset = md.get("dataset") or os.path.basename(os.path.dirname(os.path.dirname(p)))
        acc = None
        mpath = os.path.join(os.path.dirname(p), "metrics.json")
        if os.path.exists(mpath):
            acc = json.load(open(mpath)).get("accuracy")
        yield model, dataset, acc, d["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
                    help="agentic_vision/<dataset> dir (or a single model dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=200, help="max examples per page")
    args = ap.parse_args()

    runs = list(find_runs(args.result_dir))
    if not runs:
        raise SystemExit(f"no records.json found under {args.result_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    for model, dataset, acc, records in runs:
        out = os.path.join(args.out_dir, f"view_agentic-{slug(model)}.html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_page(model, dataset, acc, records, args.limit))
        print(f"wrote {out}  ({min(len(records), args.limit)} of {len(records)} examples)")


if __name__ == "__main__":
    main()
