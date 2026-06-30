#!/usr/bin/env python3
"""§5.1 validation: does the static-grid VERIFIER GAIN predict realized rejection-sampling
improvement?  (Replicates the paper's gain-vs-resampling check for VLMs.)

For every (solver, verifier) cell that has BOTH
  - a static-grid confusion matrix  (vlm/result/verifier_grid/<ds>/verify_*.json), and
  - a realized rejection-sampling run (vlm/result/rejection/<ds>/<S>__<V>/metrics.json),
we plot:
  x = PREDICTED gain  = acc@k(p,tpr,fpr) - p   from the static grid (k = --k, default 5,
      matching the rejection runs' --max_attempts 5); also reports asymptotic prec - p.
  y = REALIZED gain   = acc(final attempt) - acc(attempt 0)   from the rejection metrics.

Each point is coloured by regime (self / intra / cross). Prints Pearson + Spearman over
the matched cells and writes a scatter PNG + a tidy CSV.

Run:  .venv/bin/python vlm/plot_gain_scatter.py [--dataset charxiv] [--k 5]
Outputs: vlm/result/plots/<ds>_gain_vs_resampling.{png,csv}
"""
import argparse, csv, glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
from verifier_gain import family, regime, acc_at_k  # reuse the exact gain math

REGIME_COLOR = {"self": "#d62728", "intra": "#1f77b4", "cross": "#2ca02c"}


def short(hf_id):
    s = hf_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9.-]+", "-", s)


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):  # average ties
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def predicted_gain(grid_dir, ds, k):
    """short(solver),short(verifier) -> dict(p, gain_k, gain_inf) from latest grid file."""
    pat = re.compile(rf"verify_{ds}_solver-(.+?)_verifier-(.+?)_(\d{{8}}-\d{{6}})\.json")
    latest = {}
    for f in glob.glob(f"{grid_dir}/verify_{ds}_*.json"):
        m = pat.search(os.path.basename(f))
        if not m: continue
        key = (m.group(1), m.group(2)); tss = m.group(3)
        if key not in latest or tss > latest[key][0]: latest[key] = (tss, f)
    out = {}
    for (s, v), (_, f) in latest.items():
        d = json.load(open(f)); vm = d["metrics"]["verifier"]; p = d["metrics"]["solver_accuracy"]
        tp, tn, fp, fn = vm["tp"], vm["tn"], vm["fp"], vm["fn"]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        prec = (p * tpr) / (p * tpr + (1 - p) * fpr) if (p * tpr + (1 - p) * fpr) > 0 else p
        out[(s, v)] = dict(p=p, gain_k=acc_at_k(p, tpr, fpr, k) - p, gain_inf=prec - p)
    return out


def realized_gain(metrics_path):
    """acc(final attempt) - acc(attempt 0) from a rejection metrics.json. None if unusable."""
    d = json.load(open(metrics_path))
    its = {int(k): v for k, v in d["iterations"].items()
           if isinstance(v, dict) and "solver" in v}
    if 0 not in its or not its: return None
    base = its[0]["solver"]["accuracy"]
    final = its[max(its)]["solver"]["accuracy"]
    return dict(base=base, final=final, realized=final - base,
                attempts=max(its) + 1, total=its[max(its)]["solver"]["total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="charxiv")
    ap.add_argument("--k", type=int, default=5, help="resampling budget to predict for")
    args = ap.parse_args()
    ds = args.dataset
    grid_dir = f"vlm/result/verifier_grid/{ds}"
    rej_dir = f"vlm/result/rejection/{ds}"
    out_dir = "vlm/result/plots"; os.makedirs(out_dir, exist_ok=True)

    pred = predicted_gain(grid_dir, ds, args.k)

    rows = []
    for mp in glob.glob(f"{rej_dir}/*/metrics.json"):
        d = json.load(open(mp)); md = d["metadata"]
        if md.get("verifier_model") in (None, "oracle"): continue
        s, v = short(md["solver_model"]), short(md["verifier_model"])
        if (s, v) not in pred: continue            # not a static-grid cell
        rg = realized_gain(mp)
        if rg is None: continue
        pg = pred[(s, v)]
        rows.append(dict(solver=s, verifier=v, regime=regime(s, v),
                         p=pg["p"], pred_gain_k=pg["gain_k"], pred_gain_inf=pg["gain_inf"],
                         base=rg["base"], final=rg["final"], realized_gain=rg["realized"],
                         attempts=rg["attempts"], total=rg["total"]))

    rows.sort(key=lambda r: (r["solver"], r["verifier"]))
    csv_path = f"{out_dir}/{ds}_gain_vs_resampling.csv"
    cols = ["solver", "verifier", "regime", "p", "pred_gain_k", "pred_gain_inf",
            "base", "final", "realized_gain", "attempts", "total"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in cols})

    print(f"matched {len(rows)} (solver,verifier) cells with both grid + rejection data")
    if not rows:
        print("no matched cells yet (rejection runs still in flight?) -- wrote empty CSV")
        return

    xs = [r["pred_gain_k"] for r in rows]; ys = [r["realized_gain"] for r in rows]
    pr, sr = pearson(xs, ys), spearman(xs, ys)
    print(f"predicted gain@{args.k} vs realized:  Pearson r={pr:+.3f}  Spearman rho={sr:+.3f}")
    for reg in ["self", "intra", "cross"]:
        g = [r for r in rows if r["regime"] == reg]
        if g:
            print(f"  {reg:6} n={len(g):2}  mean pred@{args.k}={sum(r['pred_gain_k'] for r in g)/len(g):+.3f}"
                  f"  mean realized={sum(r['realized_gain'] for r in g)/len(g):+.3f}")

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(7, 6))
    lo = min(xs + ys + [0]); hi = max(xs + ys + [0])
    pad = 0.02 + 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1, label="y = x")
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    for reg in ["self", "intra", "cross"]:
        g = [r for r in rows if r["regime"] == reg]
        if not g: continue
        ax.scatter([r["pred_gain_k"] for r in g], [r["realized_gain"] for r in g],
                   s=55, alpha=0.8, color=REGIME_COLOR[reg], edgecolor="white", lw=0.7,
                   label=f"{reg} (n={len(g)})")
    ax.set_xlabel(f"predicted verifier gain@{args.k}  (static grid: acc@{args.k} - p)")
    ax.set_ylabel("realized rejection-sampling gain  (acc_final - acc_0)")
    ax.set_title(f"§5.1  {ds}: predicted gain vs realized resampling\n"
                 f"Pearson r={pr:+.3f}  Spearman ρ={sr:+.3f}  (n={len(rows)})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    png = f"{out_dir}/{ds}_gain_vs_resampling.png"
    fig.savefig(png, dpi=130); print(f"wrote {png}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
