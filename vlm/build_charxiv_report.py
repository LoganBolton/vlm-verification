#!/usr/bin/env python3
"""Lean, self-contained CharXiv replication report — the 'good stuff' only.

Assembles ONE portable HTML page (figures base64-embedded) showcasing the completed
"When Does Verification Pay Off?" replication on CharXiv:
  1. Verifier gain by regime (self / intra / cross)          -> the headline finding
  2. §5.1 validation: predicted gain vs realized resampling  -> gain predicts payoff
  3. 13x13 gain / F1 / FNR matrices (rows=verifier, cols=solver), colour-coded
  4. Agentic-zoom accuracy vs budget curves (+ links to rollout viewers)

Inputs (already produced by verifier_gain.py / plot_gain_scatter.py and the runs):
  vlm/result/verifier_grid/charxiv_gain.csv
  vlm/result/plots/charxiv_gain_by_regime.png, charxiv_gain_vs_resampling.{png,csv}
  vlm/result/agentic_vision/charxiv_c{2,4,8}/<model>/metrics.json
Run:  .venv/bin/python vlm/build_charxiv_report.py
Out:  vlm/viz/REPORT.html  (+ regenerates plots/charxiv_zoom_budget.png)
"""
import base64, bisect, csv, glob, html, json, os, re, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "vlm/result"
GRID_CSV = f"{RES}/verifier_grid/charxiv_gain.csv"
PLOTS = f"{RES}/plots"
# Blog-authoring layout: figures written as separate files under report/figures/, an always-fresh
# canonical render at report/report_generated.html, and report/index.html seeded ONCE for you to
# hand-edit -- never overwritten on rebuild.
REPORT_DIR = "report"
REPORT_FIGS = f"{REPORT_DIR}/figures"
GEN_OUT = f"{REPORT_DIR}/report_generated.html"
EDIT_OUT = f"{REPORT_DIR}/index.html"
FAM_ORDER = {"qwen-vl": 0, "internvl": 1, "gemma": 2, "llava": 3, "other": 4}


def family(s):
    s = s.lower()
    for k, f in [("qwen3-vl", "qwen-vl"), ("internvl", "internvl"),
                 ("gemma", "gemma"), ("llava", "llava")]:
        if k in s:
            return f
    return "other"


def size(s):
    m = re.search(r"(\d+)\s*b", s.lower()) or re.search(r"-(\d+)b", s.lower())
    return int(m.group(1)) if m else 99


def short(m):
    return (m.replace("-Instruct", "").replace("InternVL3-5", "IVL")
            .replace("Qwen3-VL", "Q").replace("gemma-4-", "g").replace("llava-1.5-", "llava"))


def full_name(m):
    """CSV short-name -> clean full model name (just normalises the InternVL dotting)."""
    return m.replace("InternVL3-5", "InternVL3.5")


# --- tiny inline-org-logos (self-contained SVG, sized to 1em so they never break a line) ---
_GOOGLE_G = (
    "<svg viewBox='0 0 48 48' class='lg' aria-label='Google'>"
    "<path fill='#4285F4' d='M45.12 24.5c0-1.56-.14-3.06-.4-4.5H24v8.51h11.84c-.51 2.75-2.06 5.08"
    "-4.39 6.64v5.52h7.11c4.16-3.83 6.56-9.47 6.56-16.17z'/>"
    "<path fill='#34A853' d='M24 46c5.94 0 10.92-1.97 14.56-5.33l-7.11-5.52c-1.97 1.32-4.49 2.1"
    "-7.45 2.1-5.73 0-10.58-3.87-12.31-9.07H4.34v5.7C7.96 41.07 15.4 46 24 46z'/>"
    "<path fill='#FBBC05' d='M11.69 28.18C11.25 26.86 11 25.45 11 24s.25-2.86.69-4.18v-5.7H4.34"
    "C2.85 17.09 2 20.45 2 24s.85 6.91 2.34 9.88l7.35-5.7z'/>"
    "<path fill='#EA4335' d='M24 10.75c3.23 0 6.13 1.11 8.41 3.29l6.31-6.31C34.91 4.18 29.93 2 24 2"
    "C15.4 2 7.96 6.93 4.34 14.12l7.35 5.7c1.73-5.2 6.58-9.07 12.31-9.07z'/></svg>")


def _chip(bg, letter):
    return (f"<svg viewBox='0 0 16 16' class='lg'><rect width='16' height='16' rx='4' fill='{bg}'/>"
            f"<text x='8' y='11.6' font-size='10' font-weight='700' text-anchor='middle' "
            f"fill='#fff' font-family='Arial,Helvetica,sans-serif'>{letter}</text></svg>")


def _asset_b64(name):
    p = os.path.join(os.path.dirname(__file__), "viz", "assets", name)
    return base64.b64encode(open(p, "rb").read()).decode()


# Real org marks embedded ONCE via CSS background (referenced by class, so the logo isn't
# base64-duplicated per cell): Qwen pinwheel (QwenLM/Qwen-VL), the InternLM scholar mascot for
# InternVL, and the LLaVA volcano mascot. gemma keeps Google's official inline 4-colour G.
LOGO_CSS = (
    ".lgbg{display:inline-block;height:1em;width:1em;background-position:center;"
    "background-size:contain;background-repeat:no-repeat;vertical-align:-0.15em;flex:none}"
    f".lg-qwen{{background-image:url('data:image/png;base64,{_asset_b64('qwen.png')}')}}"
    f".lg-gv{{background-image:url('data:image/png;base64,{_asset_b64('internvl.png')}')}}"
    f".lg-llava{{background-image:url('data:image/png;base64,{_asset_b64('llava.png')}')}}")
LOGO = {"qwen-vl": "<span class='lgbg lg-qwen' title='Qwen'></span>",
        "internvl": "<span class='lgbg lg-gv' title='InternVL (InternLM)'></span>",
        "gemma": _GOOGLE_G, "llava": "<span class='lgbg lg-llava' title='LLaVA'></span>",
        "other": _chip("#888", "?")}


def label(m, vertical=False):
    """Logo + full model name. vertical=True => upright logo above rotated name (matrix columns)."""
    logo, name = LOGO[family(m)], html.escape(full_name(m))
    if vertical:
        return f"<span class=vlogo>{logo}</span><span class=vtext>{name}</span>"
    return f"<span class=mdl>{logo}<span>{name}</span></span>"


def extimg(path, style="max-width:100%"):
    """Copy a figure into report/figures/ and reference it as a separate file (not base64),
    so the HTML is hand-editable and the plots live as standalone images in report/."""
    if not os.path.exists(path):
        return f"<p><em>(missing: {html.escape(path)})</em></p>"
    os.makedirs(REPORT_FIGS, exist_ok=True)
    name = os.path.basename(path)
    shutil.copyfile(path, os.path.join(REPORT_FIGS, name))
    return f"<img src='figures/{name}' style='{style}'>"


def lerp(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def color_gain(v):  # diverging red(-)/white(0)/green(+), clip at +-0.15
    t = max(-1, min(1, v / 0.15))
    if t >= 0:
        r, g, b = lerp((255, 255, 255), (60, 160, 70), t)
    else:
        r, g, b = lerp((255, 255, 255), (200, 70, 70), -t)
    return f"rgb({r},{g},{b})"


def color_scale(v, lo, hi, good_high=True):  # white->green on [lo,hi]
    t = 0 if hi == lo else max(0, min(1, (v - lo) / (hi - lo)))
    if not good_high:
        t = 1 - t
    r, g, b = lerp((255, 255, 255), (60, 160, 70), t)
    return f"rgb({r},{g},{b})"


def delta_color(d):  # text colour for a Δ: green gain / red drop / grey ~0
    return "#1a7f37" if d > 5e-4 else ("#c0392b" if d < -5e-4 else "#888")


def acc_at_k(p, tpr, fpr, k):  # expected rejection-sampling accuracy with a budget of k tries
    a = p * tpr + (1 - p) * fpr
    if a <= 0:
        return p
    return (1 - (1 - a) ** k) * (p * tpr / a) + (1 - a) ** k * p


def rdylgn(t):  # soft red(worst)->cream->green(best) ramp; t in [0,1]
    t = max(0.0, min(1.0, t))
    t = 0.5 + (t - 0.5) * 0.78          # compress toward the middle -> gentler overall contrast
    if t < 0.5:
        r, g, b = lerp((222, 132, 122), (250, 248, 236), t / 0.5)   # muted red -> cream
    else:
        r, g, b = lerp((250, 248, 236), (120, 184, 130), (t - 0.5) / 0.5)  # cream -> muted green
    return f"rgb({r},{g},{b})"


def load_base_v3():
    """short-name -> v3-scored single-shot base accuracy, so the base column is apples-to-apples
    with the maj@5 (v3) and agentic-zoom (v3) numbers it is differenced against."""
    out = {}
    for sf in glob.glob(f"{RES}/charxiv*/charxiv_*_scores.json"):
        d = json.load(open(sf))
        if d.get("metadata", {}).get("extractor") != "charxiv_finalanswer_normalized_match_v3":
            continue
        m = re.search(r"charxiv_(.+?)_\d{8}-\d{6}_scores\.json", os.path.basename(sf))
        if m:
            out[m.group(1)] = d["metrics"]["solver"]["accuracy"]
    return out


def load_grid():
    rows = list(csv.DictReader(open(GRID_CSV)))
    for r in rows:
        for k in ("p", "f1", "fnr", "gain", "precision", "verifier_acc", "tpr", "fpr"):
            r[k] = float(r[k])
    return rows


def matrix_table(rows, field, good_high, title, note):
    models = sorted({r["solver"] for r in rows} | {r["verifier"] for r in rows},
                    key=lambda m: (FAM_ORDER[family(m)], size(m), m))
    cell = {(r["solver"], r["verifier"]): r for r in rows}
    N = len(models)
    fmt = (lambda x: f"{x:+.2f}") if field == "gain" else (lambda x: f"{x:.2f}")
    # high-contrast colouring: rank-based (empirical CDF) so the worst cells are reddest, the best
    # greenest, and the median yellow -- balanced spread even when the values are skewed/outliered.
    allv = sorted(r[field] for r in rows)
    n = len(allv)
    def colorfn(v):
        t = (bisect.bisect_left(allv, v) + bisect.bisect_right(allv, v)) / (2 * n)
        return rdylgn(t if good_high else 1 - t)
    h = [f"<h3>{title}</h3><p class=note>{note}</p>", "<table class=mx>",
         "<tr><th class=corner rowspan=2>JUDGE&nbsp;&darr;<br>\\ SOLVER&nbsp;&rarr;</th>"
         f"<th class=spantop colspan={N}>SOLVER model &nbsp;(generates the answer)</th>"
         "<th class=avgh rowspan=2>judge<br>avg</th></tr>", "<tr>"]
    for s in models:
        h.append(f"<th class=col>{label(s, vertical=True)}</th>")
    h.append("</tr>")
    for v in models:
        h.append(f"<tr><th class=rowh>{label(v)}</th>")
        vals = []
        for s in models:
            r = cell.get((s, v))
            if not r:
                h.append("<td class=na>–</td>"); continue
            val = r[field]; vals.append(val)
            diag = " diag" if s == v else ""
            h.append(f"<td class='c{diag}' style='background:{colorfn(val)}'>{fmt(val)}</td>")
        if vals:
            av = sum(vals) / len(vals)
            h.append(f"<td class='c avg' style='background:{colorfn(av)}'>{fmt(av)}</td>")
        else:
            h.append("<td class=na>–</td>")
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def summary_table(rows, base_v3):
    """One row per solver. base acc | maj@5 | avg intra-family judge (k=5) | best zoom.
    Comparison columns show ONLY the Δ vs base (number coloured green/red); bold = best in row.
    base acc is the v3-scored single-shot accuracy so every Δ is apples-to-apples."""
    models = sorted({r["solver"] for r in rows} | {r["verifier"] for r in rows},
                    key=lambda m: (FAM_ORDER[family(m)], size(m), m))
    base = {}
    for r in rows:
        base.setdefault(r["solver"], r["p"])
    base.update(base_v3)                      # prefer the v3-scored base where available
    # best zoom = highest accuracy across the c2/c4/c8 budgets (budget itself not shown)
    zoombest = {}
    for mp in glob.glob(f"{RES}/agentic_vision/charxiv_c*/*/metrics.json"):
        d = json.load(open(mp))
        acc = d.get("accuracy", d.get("metrics", {}).get("accuracy"))
        m = os.path.basename(os.path.dirname(mp))
        if acc is not None and acc > zoombest.get(m, -1):
            zoombest[m] = acc
    maj5 = {}  # solver -> maj@5 accuracy from the n>=5 independent self-consistency runs
    for mp in glob.glob(f"{RES}/self_consistency/charxiv/*/metrics.json"):
        d = json.load(open(mp))
        mk = d.get("maj_at_k") or []
        if len(mk) >= 5:
            maj5[os.path.basename(os.path.dirname(mp))] = mk[4]  # maj_at_k[4] = k=5

    intra = {}  # solver -> mean acc@5 over its INTRA-family judges (same family, diff size)
    icnt = {}
    for r in rows:
        if r["regime"] == "intra":
            s = r["solver"]
            intra[s] = intra.get(s, 0.0) + acc_at_k(r["p"], r["tpr"], r["fpr"], 5)
            icnt[s] = icnt.get(s, 0) + 1
    for s in list(intra):
        intra[s] /= icnt[s]

    def dcell(val, b, is_best):
        if val is None:
            return "<td class=na>NA</td>"
        d = val - b
        num = f"<b>{d:+.2f}</b>" if is_best else f"{d:+.2f}"
        return f"<td class=c><span style='color:{delta_color(d)}'>{num}</span></td>"

    h = ["<table class='mx sum'>",
         "<tr><th class=rowh>solver model</th><th>base<br>acc</th><th>maj@5</th>"
         "<th>avg intra-family<br>judge (k=5)</th><th>best<br>zoom</th></tr>"]
    for m in models:
        b = base[m]
        iv = intra.get(m)
        z = zoombest.get(m)
        mj = maj5.get(m)
        cands = [x for x in (b, mj, iv, z) if x is not None]
        best = max(cands) if cands else None
        bcell = f"<b>{b:.2f}</b>" if best is not None and b == best else f"{b:.2f}"
        h.append(f"<tr><th class=rowh>{label(m)}</th><td class=c>{bcell}</td>"
                 + dcell(mj, b, mj is not None and mj == best)
                 + dcell(iv, b, iv is not None and iv == best)
                 + dcell(z, b, z is not None and z == best) + "</tr>")
    h.append("</table>")
    return "".join(h)


def zoom_curves(base):
    """Build the acc-vs-budget figure + table from charxiv_c{2,4,8} metrics; return (png, html).
    `base` maps solver short-name -> single-shot base accuracy (for the Δ columns)."""
    data = {}
    for mp in glob.glob(f"{RES}/agentic_vision/charxiv_c*/*/metrics.json"):
        b = int(re.search(r"_c(\d+)/", mp).group(1))
        model = os.path.basename(os.path.dirname(mp))
        d = json.load(open(mp))
        acc = d.get("accuracy", d.get("metrics", {}).get("accuracy"))
        if acc is not None:
            data.setdefault(model, {})[b] = acc
    models = sorted(data, key=lambda m: (FAM_ORDER[family(m)], size(m), m))
    budgets = sorted({b for v in data.values() for b in v})

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        xs = [b for b in budgets if b in data[m]]
        ys = [data[m][b] for b in xs]
        ax.plot(xs, ys, "-o", color=cmap(i % 10), label=short(m), lw=1.8, ms=5)
    ax.set_xlabel("zoom budget (max crops)"); ax.set_ylabel("CharXiv accuracy")
    ax.set_xticks(budgets); ax.set_title("Agentic-zoom: accuracy vs budget")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = f"{PLOTS}/charxiv_zoom_budget.png"; fig.savefig(png, dpi=130)

    # base acc, then each budget as a Δ vs base (coloured number, same style as the summary table)
    t = ["<table class=mx><tr><th class=rowh>model</th><th>base<br>acc</th>" +
         "".join(f"<th>c{b}<br>(Δ)</th>" for b in budgets) + "</tr>"]
    for m in models:
        cells = data[m]
        b0 = base.get(m)
        row = f"<tr><th class=rowh>{label(m)}</th>"
        row += f"<td class=c>{b0:.2f}</td>" if b0 is not None else "<td class=na>NA</td>"
        for b in budgets:
            if b in cells and b0 is not None:
                d = cells[b] - b0
                row += f"<td class=c><span style='color:{delta_color(d)}'>{d:+.2f}</span></td>"
            else:
                row += "<td class=na>NA</td>"
        row += "</tr>"
        t.append(row)
    t.append("</table>")
    return png, "".join(t)


def regime_summary(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["regime"]].append(r["gain"])
    h = ["<table class=kv><tr><th>regime</th><th>n</th><th>mean gain</th><th>min</th><th>max</th></tr>"]
    for reg in ["self", "intra", "cross"]:
        g = by.get(reg, [])
        if g:
            h.append(f"<tr><td>{reg}</td><td>{len(g)}</td>"
                     f"<td><b>{sum(g)/len(g):+.2f}</b></td><td>{min(g):+.2f}</td><td>{max(g):+.2f}</td></tr>")
    h.append("</table>")
    return "".join(h)


def s51_summary():
    path = f"{PLOTS}/charxiv_gain_vs_resampling.csv"
    if not os.path.exists(path):
        return "<p><em>(no §5.1 csv)</em></p>", ""
    rows = list(csv.DictReader(open(path)))
    xs = [float(r["pred_gain_k"]) for r in rows]
    ys = [float(r["realized_gain"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    pear = cov/(vx*vy)**0.5 if vx and vy else float("nan")
    return (f"<p>Across <b>{n}</b> (solver, judge) cells with both a static-grid gain and a "
            f"realized k=5 rejection run: <b>Pearson r = {pear:+.2f}</b>. "
            f"Predicted judge gain tracks realized rejection-sampling improvement.</p>"), rows


def main():
    rows = load_grid()
    base_v3 = load_base_v3()                  # v3-scored single-shot base per model
    bacc = {}
    for r in rows:
        bacc.setdefault(r["solver"], r["p"])
    bacc.update(base_v3)                       # prefer v3 base where available
    zoom_png, zoom_tbl = zoom_curves(bacc)
    s51_txt, _ = s51_summary()

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;
         padding:0 1rem;color:#1a1a1a;line-height:1.5}
    h1{border-bottom:3px solid #2c3e50;padding-bottom:.3rem}
    h2{margin-top:2.5rem;color:#2c3e50;border-bottom:1px solid #ddd;padding-bottom:.2rem}
    .note{color:#666;font-size:.86rem;margin:.2rem 0 .6rem}
    table.mx{border-collapse:collapse;font-size:.8rem;margin:.5rem 0}
    table.mx td,table.mx th{border:1px solid #ccc;padding:3px 7px;text-align:center}
    table.mx th{background:#f4f6f8} .mx td.c{font-variant-numeric:tabular-nums}
    .mx td.diag{outline:2px solid #2c3e50;outline-offset:-2px;font-weight:600}
    .mx td.na{color:#c8ccd0;font-style:italic;font-size:.82em} .corner,.rowh{text-align:right!important;background:#f4f6f8;font-weight:600}
    /* org logos: sized to the text so they never alter line height/width */
    .lg{height:1em;width:auto;vertical-align:-0.15em;flex:none}
    """ + LOGO_CSS + """
    .mdl{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
    th.col{vertical-align:bottom;padding:5px 3px}
    th.col .vlogo{display:block;text-align:center;margin-bottom:4px}
    th.col .vtext{writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;
                  display:inline-block;font-weight:600}
    .rowh .mdl{justify-content:flex-end}
    .mx .spantop{background:#e8edf2;font-weight:700;letter-spacing:.04em;color:#2c3e50}
    .mx .avgh{background:#eef2f5;font-weight:700;color:#2c3e50}
    .mx td.avg{font-weight:700;border-left:2px solid #2c3e50}
    .mx .corner{font-size:.72rem;line-height:1.2}
    table.sum{font-size:.9rem} table.sum td,table.sum th{padding:3px 11px;vertical-align:middle;text-align:center}
    table.sum .rowh{min-width:150px}
    table.kv{border-collapse:collapse;margin:.5rem 0} table.kv td,table.kv th{border:1px solid #ccc;padding:4px 12px}
    .grid2{display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start}
    .card{background:#f8f9fa;border:1px solid #e3e6e8;border-radius:8px;padding:1rem;margin:.5rem 0}
    a.viewer{display:inline-block;margin:.2rem .4rem .2rem 0;padding:.3rem .7rem;background:#2c3e50;
             color:#fff;border-radius:5px;text-decoration:none;font-size:.85rem}
    """
    P = []
    P.append(f"<!doctype html><meta charset=utf-8><title>CharXiv — Verification Pay-Off Replication</title><style>{css}</style>")
    P.append("<h1>When Does Verification Pay Off? — CharXiv (VLM replication)</h1>")
    P.append("<p class=note>13 models · 4 families · solver×verifier grid (169 cells) · §5.1 rejection (k=5) · agentic-zoom. "
             "Self = model judging itself; intra = same family, different size; cross = different family.</p>")

    P.append("<h2>1 · Per-model summary <span class=note>(test-time compute vs single-shot base)</span></h2>")
    P.append("<p class=note>One row per solver. <b>base acc</b> = single-shot accuracy "
             "(v3 scorer, same as every other column); "
             "<b>maj@5</b> = majority vote of 5 independent samples; "
             "<b>avg intra-family judge (k=5)</b> = mean over same-family judges of rejection-sampling "
             "capped at 5 tries; <b>best zoom</b> = best accuracy across the 2/4/8-crop agentic-vision "
             "budgets. All comparison columns show only the Δ vs base — green number = gain, red = drop; "
             "<b>bold</b> = best accuracy in the row. "
             "Zoom n/a for llava (single-image only) and gemma-4-12B (vLLM bug).</p>")
    P.append(summary_table(rows, base_v3))

    P.append("<h2>2 · Judge gain by regime <span class=note>(the headline)</span></h2>")
    P.append("<div class=grid2><div>" + extimg(f"{PLOTS}/charxiv_gain_by_regime.png", "max-width:480px") + "</div>")
    P.append("<div class=card>" + regime_summary(rows) +
             "<p class=note>Gain = judge-accept precision − solver accuracy (asymptotic resampling lift). "
             "<b>Self-judging pays off least</b> — models rubber-stamp their own outputs; cross-family is most honest.</p></div></div>")

    P.append("<h2>3 · §5.1 — does gain predict realized resampling?</h2>")
    P.append("<div class=grid2><div>" + extimg(f"{PLOTS}/charxiv_gain_vs_resampling.png", "max-width:520px") + "</div>")
    P.append("<div class=card>" + s51_txt + "</div></div>")

    P.append("<h2>4 · Gain / F1 / FNR matrices <span class=note>(rows = JUDGE model, "
             "cols = SOLVER model; diagonal = self; last column = each judge's average across solvers)</span></h2>")
    P.append(matrix_table(rows, "gain", True,
                          "Judge gain (judge-accept precision − solver accuracy)",
                          "colour scaled worst→best across this matrix: green = the judge helps resampling most, red = least/hurts."))
    P.append(matrix_table(rows, "f1", True,
                          "Judge F1 (accept-decision)",
                          "colour worst→best: green = best accept/reject discrimination, red = worst."))
    P.append(matrix_table(rows, "fnr", False,
                          "Judge FNR (miss rate on correct answers)",
                          "lower is better, so colour is inverted: green = lenient (accepts correct), red = harsh (rejects correct, e.g. llava / over-strict judges)."))

    P.append("<h2>5 · Agentic-zoom — accuracy vs budget</h2>")
    P.append("<div class=card>" + zoom_tbl +
             "<p class=note>Δ vs the v3 base at each crop budget. Zoom helps gemma / small-Qwen but hurts the "
             "InternVL family (it won't emit the required &lt;tool_call&gt; markup, so crops never fire).</p></div>")

    doc = "\n".join(P)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(GEN_OUT, "w") as f:                     # always-fresh canonical render
        f.write(doc)
    if os.path.exists(EDIT_OUT):                       # never clobber your hand-edited copy
        note = f"left your {EDIT_OUT} untouched"
    else:
        with open(EDIT_OUT, "w") as f:
            f.write(doc)
        note = f"seeded editable {EDIT_OUT}"
    nfig = len(glob.glob(f"{REPORT_FIGS}/*"))
    print(f"wrote {GEN_OUT} ({os.path.getsize(GEN_OUT)//1024} KB) + {nfig} figure(s) in {REPORT_FIGS}/; {note}")


if __name__ == "__main__":
    main()
