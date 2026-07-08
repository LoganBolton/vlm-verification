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


def rel_pct(delta, base):  # relative change: a Δ expressed as a % of the base it's measured against
    if not base:
        return ""
    return f"{delta / base * 100:+.0f}%"


def rel_span(delta, base):  # muted "(±NN%)" parenthetical to sit beside a raw Δ
    r = rel_pct(delta, base)
    return f" <span class=rel>({r})</span>" if r else ""


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


GRID_FIELDS = ("p", "f1", "fnr", "gain", "precision", "verifier_acc", "tpr", "fpr")


def load_grid(ds):
    rows = list(csv.DictReader(open(f"{RES}/verifier_grid/{ds}_gain.csv")))
    for r in rows:
        for k in GRID_FIELDS:
            r[k] = float(r[k])
    return rows


def load_base(ds):
    """short-name -> single-shot base accuracy. CharXiv uses the v3 scorer (apples-to-apples with
    maj@5 / zoom); CountBench has one uniform scorer, so we just take each model's latest run."""
    out = {}
    if ds == "charxiv":
        for sf in glob.glob(f"{RES}/charxiv*/charxiv_*_scores.json"):
            d = json.load(open(sf))
            if d.get("metadata", {}).get("extractor") != "charxiv_finalanswer_normalized_match_v3":
                continue
            m = re.search(r"charxiv_(.+?)_\d{8}-\d{6}_scores\.json", os.path.basename(sf))
            if m:
                out[m.group(1)] = d["metrics"]["solver"]["accuracy"]
    else:
        best = {}  # name -> (timestamp, acc); keep the latest run per model
        for sf in glob.glob(f"{RES}/{ds}*/{ds}_*_scores.json"):
            m = re.search(rf"{ds}_(.+?)_(\d{{8}}-\d{{6}})_scores\.json", os.path.basename(sf))
            if not m:
                continue
            name, tstamp = m.group(1), m.group(2)
            if tstamp >= best.get(name, ("",))[0]:
                best[name] = (tstamp, json.load(open(sf))["metrics"]["solver"]["accuracy"])
        out = {n: v[1] for n, v in best.items()}
    return out


def load_maj5(ds):
    """short-name -> maj@5 accuracy from the n>=5 independent self-consistency runs."""
    out = {}
    for mp in glob.glob(f"{RES}/self_consistency/{ds}/*/metrics.json"):
        mk = json.load(open(mp)).get("maj_at_k") or []
        if len(mk) >= 5:
            out[os.path.basename(os.path.dirname(mp))] = mk[4]  # maj_at_k[4] = k=5
    return out


def load_zoom(ds):
    """short-name -> {budget: accuracy} across the agentic-zoom c2/c4/c8 runs."""
    data = {}
    for mp in glob.glob(f"{RES}/agentic_vision/{ds}_c*/*/metrics.json"):
        mm = re.search(r"_c(\d+)/", mp)
        if not mm:
            continue
        b = int(mm.group(1))
        d = json.load(open(mp))
        acc = d.get("accuracy", d.get("metrics", {}).get("accuracy"))
        if acc is not None:
            data.setdefault(os.path.basename(os.path.dirname(mp)), {})[b] = acc
    return data


def compute_intra(rows):
    """short-name -> mean acc@5 over the solver's INTRA-family judges (same family, diff size)."""
    tot, cnt = {}, {}
    for r in rows:
        if r["regime"] == "intra":
            s = r["solver"]
            tot[s] = tot.get(s, 0.0) + acc_at_k(r["p"], r["tpr"], r["fpr"], 5)
            cnt[s] = cnt.get(s, 0) + 1
    return {s: tot[s] / cnt[s] for s in tot}


def compute_realized(s51_rows, regime, base):
    """short-name -> resulting accuracy (base + mean REALIZED rejection gain) over the solver's
    <regime>-family judges. Uses the actual measured k=5 rejection gain (acc_final − base), NOT the
    predicted acc@k from the static grid. Returned as base+gain so the table's Δ-vs-base column shows
    exactly the realized gain, consistent with the base-acc anchor column."""
    tot, cnt = {}, {}
    for r in s51_rows:
        if r.get("regime") != regime:
            continue
        s = r["solver"]
        try:
            g = float(r["realized_gain"])
        except (KeyError, ValueError, TypeError):
            continue
        tot[s] = tot.get(s, 0.0) + g
        cnt[s] = cnt.get(s, 0) + 1
    out = {}
    for s in tot:
        b = base.get(s)
        if b is not None:
            out[s] = b + tot[s] / cnt[s]
    return out


def compute_realized_best(s51_rows, regime, base):
    """short-name -> accuracy of the solver's SINGLE BEST <regime>-family judge (base + the MAX
    realized rejection gain over that regime's judges). An oracle judge-selection ceiling per solver:
    'if you'd picked the best cross-family judge, how far does k=5 rejection get you?'"""
    best = {}
    for r in s51_rows:
        if r.get("regime") != regime:
            continue
        s = r["solver"]
        try:
            g = float(r["realized_gain"])
        except (KeyError, ValueError, TypeError):
            continue
        if s not in best or g > best[s]:
            best[s] = g
    out = {}
    for s in best:
        b = base.get(s)
        if b is not None:
            out[s] = b + best[s]
    return out


def base_with_fallback(ds, rows):
    """Base accuracy per solver: measured base (v3/latest) where available, else the grid's p."""
    b = {}
    for r in rows:
        b.setdefault(r["solver"], r["p"])
    b.update(load_base(ds))
    return b


# ---- "Average" builders: combine the two datasets cell-by-cell (intersection only) ----
def mean_dict(dicts):
    keys = set().union(*[d.keys() for d in dicts]) if dicts else set()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def avg_grid(rows_a, rows_b):
    da = {(r["solver"], r["verifier"]): r for r in rows_a}
    db = {(r["solver"], r["verifier"]): r for r in rows_b}
    out = []
    for k in da.keys() & db.keys():
        ra, rb = da[k], db[k]
        m = {"solver": k[0], "verifier": k[1], "regime": ra["regime"]}
        for f in GRID_FIELDS:
            m[f] = (ra[f] + rb[f]) / 2
        out.append(m)
    return out


def avg_zoom(zlist):
    out = {}
    for m in set().union(*[z.keys() for z in zlist]) if zlist else set():
        bud = {}
        for b in set().union(*[z.get(m, {}).keys() for z in zlist]):
            vals = [z[m][b] for z in zlist if b in z.get(m, {})]
            if vals:
                bud[b] = sum(vals) / len(vals)
        out[m] = bud
    return out


def avg_s51(rows_a, rows_b):
    da = {(r["solver"], r["verifier"]): r for r in rows_a}
    db = {(r["solver"], r["verifier"]): r for r in rows_b}
    out = []
    for k in da.keys() & db.keys():
        ra, rb = da[k], db[k]
        out.append({"solver": k[0], "verifier": k[1], "regime": ra["regime"],
                    "pred_gain_k": (float(ra["pred_gain_k"]) + float(rb["pred_gain_k"])) / 2,
                    "realized_gain": (float(ra["realized_gain"]) + float(rb["realized_gain"])) / 2,
                    "base": (float(ra["base"]) + float(rb["base"])) / 2})
    return out


def realized_rows(s51_rows):
    """Reshape §5.1 rejection rows into what matrix_table/regime_summary expect: the realized gain
    (final − base accuracy) as the `gain` field, and the base accuracy as `p` (for the rel-% column)."""
    out = []
    for r in s51_rows:
        try:
            g, b = float(r["realized_gain"]), float(r["base"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append({"solver": r["solver"], "verifier": r["verifier"],
                    "regime": r["regime"], "gain": g, "p": b})
    return out


def matrix_table(rows, field, good_high, title, note, show_rel=False):
    """show_rel: for the gain matrix, append each cell's gain as a % of the solver's base accuracy
    (relative lift), e.g. "+0.11 (+15%)" — makes cells comparable across easy/hard datasets."""
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
        vals, sg, sb = [], 0.0, 0.0
        for s in models:
            r = cell.get((s, v))
            if not r:
                h.append("<td class=na>–</td>"); continue
            val = r[field]; vals.append(val)
            diag = " diag" if s == v else ""
            rel = rel_span(val, r["p"]) if show_rel else ""
            if show_rel:
                sg += val; sb += r["p"]                        # for the base-weighted row avg
            h.append(f"<td class='c{diag}' style='background:{colorfn(val)}'>{fmt(val)}{rel}</td>")
        if vals:
            av = sum(vals) / len(vals)
            avrel = (f" <span class=rel>({sg/sb*100:+.0f}%)</span>"
                     if show_rel and sb else "")
            h.append(f"<td class='c avg' style='background:{colorfn(av)}'>{fmt(av)}{avrel}</td>")
        else:
            h.append("<td class=na>–</td>")
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def render_summary(models, base, maj5, cross, zoomdata, pct_only=False, crossbest=None):
    """One row per solver. base acc | maj@5 | avg cross-family judge (k=5) | best zoom.
    The cross-family judge column is the REALIZED rejection-sampling accuracy (base + actual measured
    gain), so its Δ-vs-base is the true payoff — not the predicted acc@k.
    crossbest: optional short-name->acc dict; when given, an extra "best cross-family judge (k=5)"
    column (the solver's single best cross-family judge) is inserted before best zoom.
    Comparison columns show the Δ vs base (number coloured green/red); bold = best in row.
    pct_only: show ONLY the relative % change (no raw Δ) — used for the summarised index.html.
    All inputs are precomputed short-name -> value dicts (zoomdata is {model:{budget:acc}})."""
    zoombest = {m: max(v.values()) for m, v in zoomdata.items() if v}
    show_best = crossbest is not None

    def dcell(val, b, is_best):
        if val is None:
            return "<td class=na>NA</td>"
        d = val - b
        if pct_only:
            r = rel_pct(d, b) or "–"
            inner = f"<b>{r}</b>" if is_best else r
            return f"<td class=c><span style='color:{delta_color(d)}'>{inner}</span></td>"
        num = f"<b>{d:+.2f}</b>" if is_best else f"{d:+.2f}"
        return (f"<td class=c><span style='color:{delta_color(d)}'>{num}</span>"
                f"{rel_span(d, b)}</td>")

    bestcol_h = "<th>best cross-family<br>judge (k=5)</th>" if show_best else ""
    h = ["<table class='mx sum'>",
         "<tr><th class=rowh>solver model</th><th>base<br>acc</th><th>maj@5</th>"
         "<th>avg cross-family<br>judge (k=5)</th>" + bestcol_h + "<th>best<br>zoom</th></tr>"]
    # accumulate per-column changes (Δ and % vs base) so we can average them in a footer row
    dlt = {"maj5": [], "cross": [], "crossbest": [], "zoom": []}
    pct = {"maj5": [], "cross": [], "crossbest": [], "zoom": []}
    bases = []
    for m in models:
        b = base.get(m)
        if b is None:
            continue
        bases.append(b)
        iv, z, mj = cross.get(m), zoombest.get(m), maj5.get(m)
        cbv = crossbest.get(m) if show_best else None
        for key, val in (("maj5", mj), ("cross", iv), ("crossbest", cbv), ("zoom", z)):
            if val is not None and b:
                dlt[key].append(val - b)
                pct[key].append((val - b) / b * 100)
        cands = [x for x in (b, mj, iv, cbv, z) if x is not None]
        best = max(cands) if cands else None
        bcell = f"<b>{b:.2f}</b>" if best is not None and b == best else f"{b:.2f}"
        bestcell = dcell(cbv, b, cbv is not None and cbv == best) if show_best else ""
        h.append(f"<tr><th class=rowh>{label(m)}</th><td class=c>{bcell}</td>"
                 + dcell(mj, b, mj is not None and mj == best)
                 + dcell(iv, b, iv is not None and iv == best)
                 + bestcell
                 + dcell(z, b, z is not None and z == best) + "</tr>")

    # footer: mean change per column (mean of each model's % change; raw mode also shows mean Δ)
    def fcell(key):
        if not pct[key]:
            return "<td class=na>–</td>"
        mp = sum(pct[key]) / len(pct[key])
        if pct_only:
            return f"<td class=c><span style='color:{delta_color(mp)}'><b>{mp:+.0f}%</b></span></td>"
        md = sum(dlt[key]) / len(dlt[key])
        return (f"<td class=c><span style='color:{delta_color(md)}'><b>{md:+.2f}</b></span>"
                f" <span class=rel>({mp:+.0f}%)</span></td>")
    mb = f"{sum(bases)/len(bases):.2f}" if bases else "–"
    bestfoot = fcell("crossbest") if show_best else ""
    h.append(f"<tr class=avgrow><th class=rowh>average</th><td class=c>{mb}</td>"
             + fcell("maj5") + fcell("cross") + bestfoot + fcell("zoom") + "</tr>")
    h.append("</table>")
    return "".join(h)


FAM_DISPLAY = {"qwen-vl": "Qwen3-VL", "internvl": "InternVL3.5", "gemma": "gemma-4", "llava": "llava-1.5"}


def render_best_cross_judges(s51_rows, topn=2):
    """One row per solver FAMILY. Rank the specific CROSS-family JUDGE MODELS by their AVERAGE realized
    k=5 gain over the whole solver family, and show the top-N: col 1 = best cross-family judge model,
    col 2 = 2nd best, ... Each judge model appears in at most one column. Each cell names the judge
    model, the solver family it judged, and the mean realized gain (Δ + %) across that family's solvers."""
    # {solver_family: {verifier_model: [(gain, base), ...]}}
    byfam = {}
    for r in s51_rows:
        if r.get("regime") != "cross":
            continue
        try:
            g, b = float(r["realized_gain"]), float(r["base"])
        except (KeyError, ValueError, TypeError):
            continue
        byfam.setdefault(family(r["solver"]), {}).setdefault(r["verifier"], []).append((g, b))
    if not byfam:
        return "<p class=note><em>(no cross-family judge data yet for this dataset)</em></p>"
    ranks = ["best", "2nd best", "3rd best", "4th best", "5th best"]
    h = ["<table class='mx sum'>",
         "<tr><th class=rowh>solver family</th>"
         + "".join(f"<th>{ranks[i]} cross-family<br>judge model (k=5)</th>" for i in range(topn))
         + "</tr>"]
    for sf in sorted(byfam, key=lambda f: FAM_ORDER.get(f, 9)):
        # average realized gain per specific judge model, then rank models
        agg = []
        for v, pairs in byfam[sf].items():
            gm = sum(g for g, _ in pairs) / len(pairs)
            bm = sum(b for _, b in pairs) / len(pairs)
            agg.append((gm, bm, v, len(pairs)))
        agg.sort(key=lambda x: x[0], reverse=True)
        cells = agg[:topn]
        famcell = f"<span class=mdl>{LOGO[sf]}<span>{FAM_DISPLAY.get(sf, sf)}</span></span>"
        h.append(f"<tr><th class=rowh>{famcell}</th>")
        for gm, bm, v, npair in cells:
            h.append(f"<td class=c><div>{label(v)}</div>"
                     f"<div class=rel style='margin:0'>avg over {npair} solver{'s' if npair != 1 else ''}</div>"
                     f"<span style='color:{delta_color(gm)}'><b>{gm:+.2f}</b></span>{rel_span(gm, bm)}</td>")
        h += ["<td class=na>–</td>"] * (topn - len(cells))
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def render_zoom(zoomdata, base, pngpath, acc_label):
    """acc-vs-budget figure + Δ table from a {model:{budget:acc}} dict; returns (png, html)."""
    data = {m: v for m, v in zoomdata.items() if v}
    models = sorted(data, key=lambda m: (FAM_ORDER[family(m)], size(m), m))
    budgets = sorted({b for v in data.values() for b in v})
    if not models:
        return None, "<p class=note><em>(no zoom runs for this dataset)</em></p>"

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        xs = [b for b in budgets if b in data[m]]
        ys = [data[m][b] for b in xs]
        ax.plot(xs, ys, "-o", color=cmap(i % 10), label=short(m), lw=1.8, ms=5)
    ax.set_xlabel("zoom budget (max crops)"); ax.set_ylabel(f"{acc_label} accuracy")
    ax.set_xticks(budgets); ax.set_title(f"Agentic-zoom: accuracy vs budget — {acc_label}")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(pngpath, dpi=130); plt.close(fig)

    t = ["<table class=mx><tr><th class=rowh>model</th><th>base<br>acc</th>" +
         "".join(f"<th>c{b}<br>(Δ)</th>" for b in budgets) + "</tr>"]
    for m in models:
        cells = data[m]; b0 = base.get(m)
        row = f"<tr><th class=rowh>{label(m)}</th>"
        row += f"<td class=c>{b0:.2f}</td>" if b0 is not None else "<td class=na>NA</td>"
        for b in budgets:
            if b in cells and b0 is not None:
                d = cells[b] - b0
                row += (f"<td class=c><span style='color:{delta_color(d)}'>{d:+.2f}</span>"
                        f"{rel_span(d, b0)}</td>")
            else:
                row += "<td class=na>NA</td>"
        t.append(row + "</tr>")
    t.append("</table>")
    return pngpath, "".join(t)


_REG_COLOR = {"self": "#c0392b", "intra": "#e2a53b", "cross": "#3ca846"}


def plot_regime_bar(rows, pngpath, title):
    """Mean judge gain by regime (self/intra/cross) as a small bar chart."""
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["regime"]].append(r["gain"])
    regs = [r for r in ("self", "intra", "cross") if by.get(r)]
    means = [sum(by[r]) / len(by[r]) for r in regs]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.bar(regs, means, color=[_REG_COLOR[r] for r in regs])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("mean judge gain"); ax.set_title(title)
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(pngpath, dpi=130); plt.close(fig)


def plot_actual_regime_points(rows, pngpath):
    """Relative actual rejection-sampling gain by regime, pooled across datasets."""
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        base = float(r["p"])
        if base:
            by[r["regime"]].append(float(r["gain"]) / base)
    regs = [r for r in ("self", "intra", "cross") if by.get(r)]
    means = [sum(by[r]) / len(by[r]) for r in regs]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    xs = list(range(len(regs)))
    ax.bar(xs, means, color=[_REG_COLOR[r] for r in regs], alpha=0.9, width=0.6)
    for i, reg in enumerate(regs):
        vals = by[reg]
        # Deterministic narrow spread so overlapping points remain visible without randomness.
        offsets = [((j % 9) - 4) * 0.012 for j in range(len(vals))]
        ax.scatter([i + off for off in offsets], vals, color="black", s=14, alpha=0.45, zorder=3)
        ax.text(i + 0.19, means[i], f"{means[i]*100:+.0f}%", ha="center",
                va="bottom" if means[i] >= 0 else "top", fontsize=11, color="#1a7f37")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r.capitalize()}\n(n={len(by[r])})" for r in regs])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.set_ylabel("Relative accuracy increase vs baseline")
    ax.set_title("Accuracy Gain by Judge Type")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pngpath, dpi=150)
    plt.close(fig)


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else float("nan")


def _linear_fit(xs, ys):
    """Least-squares y = slope*x + intercept."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    if vx == 0:
        return float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
    return slope, my - slope * mx


def _one_row_per_pair(rows):
    """Keep the last row encountered for each solver-verifier pair."""
    by_pair = {}
    for r in rows:
        by_pair[(r["solver"], r["verifier"])] = r
    return [by_pair[k] for k in sorted(by_pair)]


def plot_s51(rows, pngpath, title):
    """predicted gain@5 (static grid) vs realized k=5 rejection gain, coloured by regime."""
    rows = _one_row_per_pair(rows)
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for reg in ("self", "intra", "cross"):
        xs = [float(r["pred_gain_k"]) for r in rows if r["regime"] == reg]
        ys = [float(r["realized_gain"]) for r in rows if r["regime"] == reg]
        if xs:
            ax.scatter(xs, ys, s=24, color=_REG_COLOR[reg], label=reg, alpha=0.8, edgecolor="none")
    allx = [float(r["pred_gain_k"]) for r in rows]
    ally = [float(r["realized_gain"]) for r in rows]
    r = _pearson(allx, ally)
    slope, intercept = _linear_fit(allx, ally)
    xmin, xmax = min(allx + [0.0]), max(allx + [0.0])
    ymin, ymax = min(ally + [0.0]), max(ally + [0.0])
    xpad = 0.08 * (xmax - xmin) or 0.01
    ypad = 0.08 * (ymax - ymin) or 0.01
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    if slope == slope:
        ax.plot([xmin, xmax], [slope * xmin + intercept, slope * xmax + intercept],
                "-", color="#444", lw=1.1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(0.04, 0.96, f"r = {r:.2f}\nslope = {slope:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff7cc", edgecolor="#e7d98a", alpha=0.92))
    ax.set_xlabel("Predicted Gain")
    ax.set_ylabel("Actual Gain")
    ax.set_title("Predicted vs Actual Gain for VLM Judge\nRejection Sampling")
    ax.legend(fontsize=10, markerscale=1.2)
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(pngpath, dpi=300); plt.close(fig)
    return True


def regime_summary(rows):
    import collections
    by = collections.defaultdict(list)
    sum_g = collections.defaultdict(float)   # base-weighted relative = Σgain / Σbase (ratio of means,
    sum_b = collections.defaultdict(float)   # robust to low-base outliers like llava)
    for r in rows:
        by[r["regime"]].append(r["gain"])
        sum_g[r["regime"]] += r["gain"]
        sum_b[r["regime"]] += r["p"]
    h = ["<table class=kv><tr><th>regime</th><th>n</th><th>mean gain</th><th>min</th><th>max</th></tr>"]
    for reg in ["self", "intra", "cross"]:
        g = by.get(reg, [])
        if g:
            relstr = (f" <span class=rel>({sum_g[reg]/sum_b[reg]*100:+.0f}%)</span>"
                      if sum_b[reg] else "")
            h.append(f"<tr><td>{reg}</td><td>{len(g)}</td>"
                     f"<td><b>{sum(g)/len(g):+.2f}</b>{relstr}</td>"
                     f"<td>{min(g):+.2f}</td><td>{max(g):+.2f}</td></tr>")
    h.append("</table>")
    return "".join(h)


def load_s51(ds):
    path = f"{PLOTS}/{ds}_gain_vs_resampling.csv"
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def s51_text(rows):
    rows = _one_row_per_pair(rows)
    if not rows:
        return "<p class=note><em>(no §5.1 rejection data yet)</em></p>"
    xs = [float(r["pred_gain_k"]) for r in rows]
    ys = [float(r["realized_gain"]) for r in rows]
    pear = _pearson(xs, ys)
    slope, _ = _linear_fit(xs, ys)
    return (f"<p>Across <b>{len(xs)}</b> (solver, judge) cells with both a static-grid gain and an "
            f"actual k=5 rejection run: <b>Pearson r = {pear:+.2f}</b>, "
            f"<b>regression slope = {slope:+.2f}</b>. "
            f"Predicted judge gain tracks actual rejection-sampling improvement.</p>")


def models_of(rows):
    return sorted({r["solver"] for r in rows} | {r["verifier"] for r in rows},
                  key=lambda m: (FAM_ORDER[family(m)], size(m), m))


# variant render order + display labels; "avg" = the two datasets combined cell-by-cell.
VKEYS = ["countbench", "charxiv", "avg"]
VLABEL = {"countbench": "CountBench", "charxiv": "CharXiv", "avg": "Average (both datasets)"}


def load_all():
    """Load every per-dataset dict plus the combined 'avg' variant. Reusable by main() and by
    the index.html table-splicer (vlm/update_index_tables.py)."""
    DS = ["countbench", "charxiv"]
    grid = {ds: load_grid(ds) for ds in DS}
    base = {ds: base_with_fallback(ds, grid[ds]) for ds in DS}
    maj5 = {ds: load_maj5(ds) for ds in DS}
    zoom = {ds: load_zoom(ds) for ds in DS}
    s51 = {ds: load_s51(ds) for ds in DS}
    # summary-table judge column = REALIZED cross-family rejection gain (actual, not predicted acc@k)
    cross = {ds: compute_realized(s51[ds], "cross", base[ds]) for ds in DS}
    crossbest = {ds: compute_realized_best(s51[ds], "cross", base[ds]) for ds in DS}
    # the "Average" variant: combine the two datasets cell-by-cell (intersection of shared cells)
    grid["avg"] = avg_grid(grid["countbench"], grid["charxiv"])
    base["avg"] = mean_dict([base["countbench"], base["charxiv"]])
    maj5["avg"] = mean_dict([maj5["countbench"], maj5["charxiv"]])
    zoom["avg"] = avg_zoom([zoom["countbench"], zoom["charxiv"]])
    s51["avg"] = avg_s51(s51["countbench"], s51["charxiv"])
    cross["avg"] = compute_realized(s51["avg"], "cross", base["avg"])
    crossbest["avg"] = compute_realized_best(s51["avg"], "cross", base["avg"])
    return dict(grid=grid, base=base, maj5=maj5, zoom=zoom, cross=cross, crossbest=crossbest, s51=s51)


def main():
    d = load_all()
    grid, base, maj5, zoom, cross, crossbest, s51 = (d["grid"], d["base"], d["maj5"],
                                          d["zoom"], d["cross"], d["crossbest"], d["s51"])

    # generate a consistent trio of plots (report-scoped names, so standalone plots aren't clobbered)
    zoom_tbl = {}
    for vk in VKEYS:
        plot_regime_bar(grid[vk], f"{PLOTS}/{vk}_regime_report.png", VLABEL[vk])
        if s51[vk]:
            plot_s51(s51[vk], f"{PLOTS}/{vk}_s51_report.png", VLABEL[vk])
        _, zoom_tbl[vk] = render_zoom(zoom[vk], base[vk], f"{PLOTS}/{vk}_zoom_report.png", VLABEL[vk])
    actual_regime_png = f"{PLOTS}/both_datasets_actual_gain_by_regime.png"
    plot_actual_regime_points(realized_rows(s51["countbench"]) + realized_rows(s51["charxiv"]),
                              actual_regime_png)
    os.makedirs(REPORT_FIGS, exist_ok=True)
    shutil.copyfile(actual_regime_png, f"{REPORT_FIGS}/both_datasets_actual_gain_by_regime.png")

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
    h3.ds{margin:1.5rem 0 .3rem;color:#2c3e50;font-size:1.02rem;font-weight:700;
          border-left:4px solid #2c3e50;padding-left:.55rem;letter-spacing:.01em}
    h3.ds.avg{border-left-color:#8e44ad;color:#6c3483}
    .rel{color:#8a8f96;font-size:.82em;font-weight:400;white-space:nowrap}
    .mx tr.avgrow td,.mx tr.avgrow th{border-top:2px solid #2c3e50;background:#eef2f5;font-weight:700}
    """
    P = []
    P.append(f"<!doctype html><meta charset=utf-8><title>Verification Pay-Off — CountBench &amp; CharXiv</title><style>{css}</style>")
    P.append("<h1>When Does Verification Pay Off? — VLM replication</h1>")
    P.append("<p class=note>13 models · 4 families · solver×verifier grid (169 cells / dataset) · §5.1 rejection (k=5) · agentic-zoom. "
             "Self = model judging itself; intra = same family, different size; cross = different family. "
             "<b>Every plot and table below is shown three ways: CountBench, CharXiv, then the two averaged together</b> "
             "(the Average combines shared cells — mean of the CountBench and CharXiv value).</p>")

    def ds_head(vk):
        cls = "ds avg" if vk == "avg" else "ds"
        return f"<h3 class='{cls}'>{VLABEL[vk]}</h3>"

    # 1 · Per-model summary
    P.append("<h2>1 · Per-model summary <span class=note>(test-time compute vs single-shot base)</span></h2>")
    P.append("<p class=note>One row per solver. <b>base acc</b> = single-shot accuracy; "
             "<b>maj@5</b> = majority vote of 5 independent samples; "
             "<b>avg cross-family judge (k=5)</b> = mean over different-family judges of the "
             "<i>realized</i> rejection-sampling accuracy (actual measured gain over base, capped at 5 tries) "
             "— not the predicted acc@k; <b>best cross-family judge (k=5)</b> = same but taking only each "
             "solver's single best cross-family judge (oracle judge selection); "
             "<b>best zoom</b> = best accuracy across the 2/4/8-crop agentic-vision "
             "budgets. Comparison columns show only the Δ vs base — green = gain, red = drop; "
             "<b>bold</b> = best accuracy in the row. Zoom n/a for llava (single-image only) and gemma-4-12B (vLLM bug).</p>")
    for vk in VKEYS:
        P.append(ds_head(vk))
        P.append(render_summary(models_of(grid[vk]), base[vk], maj5[vk], cross[vk], zoom[vk],
                                crossbest=crossbest[vk]))

    # 1b · Best cross-family judges per family
    P.append("<h2>1b · Best cross-family judge model per solver family <span class=note>(judged by a different family)</span></h2>")
    P.append("<p class=note>One row per solver family. For each specific cross-family <i>judge model</i> (from a "
             "different family) we average its <i>realized</i> k=5 rejection gain over all solvers in that family, "
             "then rank the judge models and show the top two: <b>best</b> cross-family judge model, then "
             "<b>2nd best</b>. Each judge model appears in at most one column. Each cell = the judge model, the "
             "number of solvers averaged, and the mean realized gain (Δ vs base, and % lift).</p>")
    for vk in VKEYS:
        P.append(ds_head(vk))
        P.append(render_best_cross_judges(s51[vk], topn=2))

    # 2 · Judge gain by regime
    P.append("<h2>2 · Judge gain by regime <span class=note>(the headline)</span></h2>")
    P.append("<p class=note>Gain = judge-accept precision − solver accuracy (asymptotic resampling lift). "
             "<b>Self-judging pays off least</b> — models rubber-stamp their own outputs; cross-family is most honest.</p>")
    for vk in VKEYS:
        P.append(ds_head(vk))
        P.append("<div class=grid2><div>" + extimg(f"{PLOTS}/{vk}_regime_report.png", "max-width:440px") + "</div>")
        P.append("<div class=card>" + regime_summary(grid[vk]) + "</div></div>")

    # 3 · §5.1 validation
    P.append("<h2>3 · §5.1 — does gain predict realized resampling?</h2>")
    P.append("<p class=note>Each point is a (solver, judge) cell: x = gain the static grid predicts at k=5, "
             "y = the accuracy lift actually realized by running judge-gated rejection sampling (k=5). "
             "A tight y≈x band means the cheap static gain forecasts the expensive realized payoff.</p>")
    for vk in VKEYS:
        P.append(ds_head(vk))
        if s51[vk]:
            P.append("<div class=grid2><div>" + extimg(f"{PLOTS}/{vk}_s51_report.png", "max-width:520px") + "</div>")
            P.append("<div class=card>" + s51_text(s51[vk]) + "</div></div>")
        else:
            P.append("<p class=note><em>(no §5.1 rejection data yet for this dataset)</em></p>")

    # 4 · Realized (actual) judge gain from the k=5 rejection loop
    P.append("<h2>4 · Realized judge gain <span class=note>(the ACTUAL measured payoff)</span></h2>")
    P.append("<p class=note>§2 above is the gain <i>predicted</i> from a one-shot judge pass. "
             "<b>This is the gain actually realized</b> by running the full judge-gated rejection loop "
             "(solve → judge → re-solve only the rejected, up to 5 attempts): "
             "<b>final accuracy − base accuracy</b>. Parentheses show it as a % of base. "
             "Rows = JUDGE, cols = SOLVER; diagonal = self; last column = judge's base-weighted average. "
             "Cells fill in as rejection runs land — sparse cells (–) just aren't computed yet.</p>")
    for vk in VKEYS:
        rr = realized_rows(s51[vk])
        P.append(ds_head(vk))
        if not rr:
            P.append("<p class=note><em>(no rejection data yet for this dataset)</em></p>")
            continue
        P.append("<div class=grid2><div class=card>" + regime_summary(rr) +
                 "<p class=note>mean realized gain by regime (base-weighted % in parens).</p></div></div>")
        P.append(matrix_table(rr, "gain", True,
                              "Realized gain (final − base accuracy, k=5 rejection loop)",
                              "green = the judge loop lifts accuracy most, red = least/hurts; % of base in parens.",
                              True))

    # 5 · Matrices
    P.append("<h2>5 · Gain / F1 / FNR matrices <span class=note>(rows = JUDGE model, "
             "cols = SOLVER model; diagonal = self; last column = each judge's average across solvers)</span></h2>")
    mspecs = [("gain", True, "Judge gain (judge-accept precision − solver accuracy)",
               "colour scaled worst→best across this matrix: green = the judge helps resampling most, red = least/hurts. "
               "Each cell also shows the gain as a % of the solver's base accuracy (relative lift), in parentheses.", True),
              ("f1", True, "Judge F1 (accept-decision)",
               "colour worst→best: green = best accept/reject discrimination, red = worst.", False),
              ("fnr", False, "Judge FNR (miss rate on correct answers)",
               "lower is better, so colour is inverted: green = lenient (accepts correct), red = harsh (rejects correct, e.g. llava / over-strict judges).", False)]
    for vk in VKEYS:
        P.append(ds_head(vk))
        for field, good_high, title, note, show_rel in mspecs:
            P.append(matrix_table(grid[vk], field, good_high, title, note, show_rel))

    # 6 · Agentic-zoom
    P.append("<h2>6 · Agentic-zoom — accuracy vs budget</h2>")
    P.append("<p class=note>Δ vs base at each crop budget. Zoom helps gemma / small-Qwen but hurts the "
             "InternVL family (it won't emit the required &lt;tool_call&gt; markup, so crops never fire).</p>")
    for vk in VKEYS:
        P.append(ds_head(vk))
        P.append("<div class=grid2><div>" + extimg(f"{PLOTS}/{vk}_zoom_report.png", "max-width:520px") + "</div>")
        P.append("<div class=card>" + zoom_tbl[vk] + "</div></div>")

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
