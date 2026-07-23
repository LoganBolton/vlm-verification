#!/usr/bin/env python3
"""Verify the pushed `loganbolton/vlm-verification-logs` dataset against local disk,
in several independent formats, and emit an interactive HTML for manual spot-checking.

What it checks (all read the ACTUAL uploaded parquet, not the local JSON):
  1. Row-count parity: HF `logs` rows == rows the exporter would produce from disk.
  2. File parity: distinct `source_file` on HF == kept files on disk (no missing/stale).
  3. Per-experiment and per-record_type row breakdown (HF vs disk), side by side.
  4. Orphan-shard check: repo has no leftover parquet from an older shard layout.
  5. Losslessness: sampled rows' `raw_record` JSON parses and its core fields agree
     with the flattened columns.
  6. `images` config: every row decodes to a real image; join coverage with logs.
  7. load_dataset() streaming smoke test (what an external user would run).

Then writes `vlm/viz/hf_verify_sample.html`: a stratified subsample (a few rows per
experiment x record_type), each with its joined image (decoded from the HF images
config), all flattened fields, and the raw_record JSON — so you can eyeball that what
landed on HuggingFace matches reality.

Run:  .venv-vlm/bin/python vlm/verify_hf_upload.py
"""
import base64
import html
import io
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("HF_DATASET_REPO", "loganbolton/vlm-verification-logs")
ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = ROOT / "vlm" / "viz" / "hf_verify_sample.html"
PER_CELL = int(os.environ.get("PER_CELL", "2"))   # rows per (experiment,record_type)

api = HfApi()


def disk_truth():
    """Row/file counts the exporter would produce from the current disk state."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp", ROOT / "vlm" / "export_hf_dataset.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    files, rows = set(), 0
    by_exp, by_rt = Counter(), Counter()
    for fam, f in m.kept_files():
        rel = os.path.relpath(f, m.RESULT_DIR)
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        recs = doc["records"] if isinstance(doc, dict) and "records" in doc else doc
        if not isinstance(recs, list):
            continue
        rtype = m.record_type_of(fam, meta)
        n = sum(1 for r in recs if isinstance(r, dict))
        if n:
            files.add(rel.replace(os.sep, "/"))
            rows += n
            by_exp[rel.split(os.sep)[0]] += n
            by_rt[rtype] += n
    return {"files": files, "rows": rows, "by_exp": by_exp, "by_rt": by_rt}


def load_hf_logs():
    """Read all logs shards from HF into memory-light column stores + row sampler."""
    repo_files = api.list_repo_files(REPO, repo_type="dataset")
    log_shards = sorted(f for f in repo_files if f.startswith("logs/") and f.endswith(".parquet"))
    img_shards = sorted(f for f in repo_files if f.startswith("images/") and f.endswith(".parquet"))

    src_files, by_exp, by_rt = set(), Counter(), Counter()
    total = 0
    # stratified reservoir: (experiment, record_type) -> list of sampled full rows
    buckets = defaultdict(list)
    seen_per_cell = Counter()
    rng = random.Random(0)
    cols = None
    for sh in log_shards:
        p = hf_hub_download(REPO, sh, repo_type="dataset")
        t = pq.read_table(p)
        if cols is None:
            cols = t.column_names
        d = t.to_pydict()
        n = t.num_rows
        total += n
        for i in range(n):
            exp = d["experiment"][i]
            rt = d["record_type"][i]
            src_files.add(d["source_file"][i])
            by_exp[exp] += 1
            by_rt[rt] += 1
            key = (exp, rt)
            seen_per_cell[key] += 1
            # reservoir sampling per cell
            if len(buckets[key]) < PER_CELL:
                buckets[key].append({c: d[c][i] for c in cols})
            else:
                j = rng.randint(0, seen_per_cell[key] - 1)
                if j < PER_CELL:
                    buckets[key][j] = {c: d[c][i] for c in cols}
    return {
        "total": total, "src_files": src_files, "by_exp": by_exp, "by_rt": by_rt,
        "buckets": buckets, "log_shards": log_shards, "img_shards": img_shards,
    }


def load_hf_images():
    """(dataset,id) -> PNG bytes, decoded from the HF images config."""
    repo_files = api.list_repo_files(REPO, repo_type="dataset")
    img_shards = sorted(f for f in repo_files if f.startswith("images/") and f.endswith(".parquet"))
    idx = {}
    n_ok = 0
    from PIL import Image
    for sh in img_shards:
        p = hf_hub_download(REPO, sh, repo_type="dataset")
        t = pq.read_table(p).to_pydict()
        for ds, rid, img in zip(t["dataset"], t["id"], t["image"]):
            b = img["bytes"] if isinstance(img, dict) else img
            try:
                Image.open(io.BytesIO(b)).verify()
                n_ok += 1
            except Exception:
                pass
            idx[(ds, str(rid))] = b
    return idx, n_ok


def fmt_table(title, hf_counter, disk_counter):
    keys = sorted(set(hf_counter) | set(disk_counter))
    lines = [f"  {title:<22}{'HF':>10}{'disk':>10}   status"]
    ok = True
    for k in keys:
        h, dk = hf_counter.get(k, 0), disk_counter.get(k, 0)
        mark = "OK" if h == dk else f"MISMATCH (Δ{h-dk:+})"
        if h != dk:
            ok = False
        lines.append(f"    {str(k):<20}{h:>10}{dk:>10}   {mark}")
    return "\n".join(lines), ok


# ---------------------------------------------------------------- HTML rendering
def img_tag(img_bytes):
    if not img_bytes:
        return '<div class="noimg">no image</div>'
    b64 = base64.b64encode(img_bytes).decode()
    return f'<img src="data:image/png;base64,{b64}"/>'


def field(label, val, mono=True, pre=True):
    if val is None or val == "":
        return ""
    v = html.escape(str(val))
    cls = "mono" if mono else ""
    tag = "pre" if pre else "div"
    return f'<div class="fld"><span class="lab">{html.escape(label)}</span><{tag} class="val {cls}">{v}</{tag}></div>'


def render_html(hf, images_idx, checks_text, all_ok):
    cells = sorted(hf["buckets"].keys())
    cards = []
    for (exp, rt) in cells:
        for row in hf["buckets"][(exp, rt)]:
            ds = row.get("dataset")
            rid = row.get("id")
            img = images_idx.get((ds, str(rid)))
            correct = row.get("correct")
            badge = ""
            if correct is True:
                badge = '<span class="badge ok">correct</span>'
            elif correct is False:
                badge = '<span class="badge bad">incorrect</span>'
            vv = row.get("verifier_verdict")
            if vv not in (None, ""):
                badge += f'<span class="badge vv">verdict={html.escape(str(vv))}</span>'
            chips = "".join(
                f'<span class="chip">{html.escape(k)}={html.escape(str(row.get(k)))}</span>'
                for k in ("experiment", "record_type", "dataset", "solver_model", "verifier_model", "id")
                if row.get(k) not in (None, "")
            )
            raw = row.get("raw_record")
            try:
                raw_pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False) if raw else ""
            except Exception:
                raw_pretty = raw or ""
            body = "".join([
                field("question", row.get("question")),
                field("gold_answer", row.get("gold_answer")),
                field("task_prompt", row.get("task_prompt")),
                field("model_output", row.get("model_output")),
                field("extracted_answer", row.get("extracted_answer")),
                field("verifier_prompt", row.get("verifier_prompt")),
                field("verifier_output", row.get("verifier_output")),
                field("extra", row.get("extra")),
                field("image_path", row.get("image_path"), pre=False),
            ])
            card = f"""
            <div class="card" data-cell="{html.escape(exp+' / '+rt)}">
              <div class="hd">{chips}{badge}</div>
              <div class="cols">
                <div class="imgcol">{img_tag(img)}</div>
                <div class="txtcol">{body}
                  <details><summary>raw_record JSON</summary><pre class="val mono">{html.escape(raw_pretty)}</pre></details>
                </div>
              </div>
            </div>"""
            cards.append(card)

    cell_opts = "".join(f'<option value="{html.escape(e+" / "+r)}">{html.escape(e+" / "+r)} ({len(hf["buckets"][(e,r)])})</option>' for (e, r) in cells)
    status_color = "#137333" if all_ok else "#b3261e"
    status_word = "ALL CHECKS PASSED" if all_ok else "MISMATCHES FOUND — see report"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>HF upload verification sample</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1f2328}}
 header{{background:#0d1117;color:#eee;padding:16px 24px}}
 header h1{{margin:0 0 4px;font-size:18px}}
 .status{{font-weight:700;color:{status_color};background:#fff;display:inline-block;padding:4px 10px;border-radius:6px;margin-top:6px}}
 .report{{background:#0d1117;color:#c9d1d9;margin:0;padding:12px 24px;font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre;overflow-x:auto;border-top:1px solid #30363d}}
 .toolbar{{position:sticky;top:0;background:#fff;padding:10px 24px;border-bottom:1px solid #d0d7de;z-index:5}}
 .wrap{{padding:16px 24px;display:flex;flex-direction:column;gap:16px}}
 .card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;overflow:hidden}}
 .hd{{padding:8px 12px;background:#f0f3f6;border-bottom:1px solid #d0d7de;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
 .chip{{font:11px ui-monospace,monospace;background:#dde3ea;border-radius:20px;padding:2px 8px}}
 .badge{{font:11px sans-serif;border-radius:20px;padding:2px 8px;color:#fff}}
 .badge.ok{{background:#137333}} .badge.bad{{background:#b3261e}} .badge.vv{{background:#5a3e9e}}
 .cols{{display:flex;gap:16px;padding:12px}}
 .imgcol{{flex:0 0 360px}} .imgcol img{{max-width:360px;max-height:420px;border:1px solid #ccc;border-radius:6px}}
 .noimg{{width:360px;height:120px;display:flex;align-items:center;justify-content:center;background:#f0f0f0;color:#999;border-radius:6px}}
 .txtcol{{flex:1;min-width:0}}
 .fld{{margin-bottom:8px}} .lab{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#57606a;margin-bottom:2px}}
 .val{{margin:0;white-space:pre-wrap;word-break:break-word;background:#f6f8fa;border:1px solid #eaeef2;border-radius:6px;padding:6px 8px;max-height:320px;overflow:auto}}
 .mono{{font:12px/1.45 ui-monospace,Menlo,monospace}}
 details summary{{cursor:pointer;color:#0969da;font-size:12px;margin-top:4px}}
 select,input{{font-size:13px;padding:4px 6px}}
</style></head><body>
<header>
  <h1>{html.escape(REPO)} — upload verification</h1>
  <div>logs rows on HF: <b>{hf['total']:,}</b> &nbsp;·&nbsp; distinct source files: <b>{len(hf['src_files']):,}</b> &nbsp;·&nbsp; images decoded OK: <b>{len(images_idx):,}</b></div>
  <div class="status">{status_word}</div>
</header>
<pre class="report">{html.escape(checks_text)}</pre>
<div class="toolbar">
  Filter cell: <select id="cell" onchange="flt()"><option value="">(all {len(cards)} sampled rows)</option>{cell_opts}</select>
  &nbsp; Text search: <input id="q" oninput="flt()" placeholder="substring in card..."/>
</div>
<div class="wrap" id="wrap">
{''.join(cards)}
</div>
<script>
function flt(){{
  var c=document.getElementById('cell').value.toLowerCase();
  var q=document.getElementById('q').value.toLowerCase();
  document.querySelectorAll('.card').forEach(function(el){{
    var okc=!c||el.getAttribute('data-cell').toLowerCase()===c;
    var okq=!q||el.innerText.toLowerCase().indexOf(q)>=0;
    el.style.display=(okc&&okq)?'':'none';
  }});
}}
</script>
</body></html>"""


def main():
    print(f"Verifying {REPO} …\n")
    disk = disk_truth()
    hf = load_hf_logs()
    images_idx, img_ok = load_hf_images()

    lines = []
    lines.append("=" * 64)
    lines.append("FORMAT 1 — row-count parity (HF logs vs disk exporter)")
    lines.append(f"  HF rows:   {hf['total']:,}")
    lines.append(f"  disk rows: {disk['rows']:,}")
    c1 = hf['total'] == disk['rows']
    lines.append(f"  => {'OK — identical' if c1 else 'MISMATCH'}")

    lines.append("\nFORMAT 2 — file parity (source_file set)")
    missing = disk['files'] - hf['src_files']     # on disk, not uploaded
    stale = hf['src_files'] - disk['files']        # uploaded, gone from disk
    lines.append(f"  HF distinct files:   {len(hf['src_files'])}")
    lines.append(f"  disk distinct files: {len(disk['files'])}")
    lines.append(f"  missing (disk not on HF): {len(missing)}")
    for x in sorted(missing)[:10]:
        lines.append(f"      - {x}")
    lines.append(f"  stale  (on HF not on disk): {len(stale)}")
    for x in sorted(stale)[:10]:
        lines.append(f"      - {x}")
    c2 = not missing and not stale

    t3a, ok3a = fmt_table("rows by experiment", hf['by_exp'], disk['by_exp'])
    t3b, ok3b = fmt_table("rows by record_type", hf['by_rt'], disk['by_rt'])
    lines.append("\nFORMAT 3 — per-experiment breakdown (HF vs disk)")
    lines.append(t3a)
    lines.append("\n         — per-record_type breakdown (HF vs disk)")
    lines.append(t3b)
    c3 = ok3a and ok3b

    lines.append("\nFORMAT 4 — shard layout / orphan check")
    denoms = {f.split("-of-")[1].split(".")[0] for f in hf['log_shards']}
    lines.append(f"  logs shards: {len(hf['log_shards'])}  (of-NN groups: {sorted(denoms)})")
    lines.append(f"  images shards: {len(hf['img_shards'])}")
    c4 = len(denoms) == 1
    lines.append(f"  => {'OK — single consistent shard set' if c4 else 'WARN — mixed shard denominators (orphans?)'}")

    lines.append("\nFORMAT 5 — losslessness (raw_record vs flattened cols, sampled)")
    n_chk = n_bad = 0
    for rows in hf['buckets'].values():
        for r in rows:
            raw = r.get("raw_record")
            if not raw:
                continue
            n_chk += 1
            try:
                obj = json.loads(raw)
            except Exception:
                n_bad += 1
                continue
            # core cross-checks where applicable
            if r.get("question") is not None and obj.get("question") is not None:
                if str(obj.get("question")) != str(r.get("question")):
                    n_bad += 1
    lines.append(f"  sampled rows checked: {n_chk}   raw_record parse/mismatch failures: {n_bad}")
    c5 = n_bad == 0

    lines.append("\nFORMAT 6 — images config")
    lines.append(f"  image rows decoded OK: {img_ok} / {len(images_idx)}")
    # join coverage: do sampled log image_paths resolve to an image?
    join_hit = join_miss = 0
    for rows in hf['buckets'].values():
        for r in rows:
            if r.get("image_path"):
                if (r.get("dataset"), str(r.get("id"))) in images_idx:
                    join_hit += 1
                else:
                    join_miss += 1
    lines.append(f"  sampled (dataset,id) join hits/misses: {join_hit}/{join_miss}")
    c6 = img_ok == len(images_idx) and join_miss == 0

    lines.append("\nFORMAT 7 — external load_dataset() streaming smoke test")
    try:
        from datasets import load_dataset
        it = load_dataset(REPO, "logs", split="train", streaming=True)
        first = next(iter(it))
        lines.append(f"  OK — streamed 1 row, {len(first)} columns: {sorted(first)[:6]}…")
        c7 = True
    except Exception as e:
        lines.append(f"  FAILED: {e}")
        c7 = False

    all_ok = all([c1, c2, c3, c4, c5, c6, c7])
    lines.append("\n" + "=" * 64)
    lines.append(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    lines.append("=" * 64)
    checks_text = "\n".join(lines)
    print(checks_text)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(render_html(hf, images_idx, checks_text, all_ok))
    n_cards = sum(len(v) for v in hf['buckets'].values())
    print(f"\nWrote interactive sample ({n_cards} rows across {len(hf['buckets'])} cells) -> {OUT_HTML}")


if __name__ == "__main__":
    main()
