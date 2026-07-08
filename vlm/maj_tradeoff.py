#!/usr/bin/env python3
"""Compute-vs-accuracy tradeoff for MAJORITY VOTING (maj@n self-consistency).

Each self_consistency/<ds>/<model>/metrics.json logs maj_at_k (accuracy of a maj vote over the
first k samples, k=1..N) and metadata.solver_gflops (total FLOPs for the N-sample run). maj@k
compute = k * (solver_gflops / n_samples). We draw one curve per model: accuracy vs compute as N
grows 1->N, so you can see where extra samples stop paying off. Same axes/style as compute_tradeoff.py.

Writes report/figures/{ds}_maj_tradeoff.png (+ an 'avg' over both datasets). Run:
  .venv/bin/python vlm/maj_tradeoff.py
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import build_charxiv_report as R
import compute_tradeoff as T   # for the shared fam_color palette

DATASETS = ["charxiv", "countbench"]
FIGDIR = "report/figures/tradeoff"
NMAX = 5   # cap the curve at 5 samples so all models are comparable


def name(m):
    """full model name exactly as the report displays it (the HF id)."""
    return m


def collect(ds):
    """model -> list of (compute_gflops, acc) for k=1..min(N,NMAX)."""
    out = {}
    for f in glob.glob(f"vlm/result/self_consistency/{ds}/*/metrics.json"):
        d = json.load(open(f))
        mk = d.get("maj_at_k") or []
        md = d.get("metadata", {})
        n = md.get("n_samples") or len(mk)
        g = md.get("solver_gflops")
        if not mk or not g or not n:
            continue
        per = g / n                       # FLOPs for one sample over the whole dataset
        kmax = min(len(mk), NMAX)
        m = md.get("solver_model", os.path.basename(os.path.dirname(f)))
        out[m] = [((k + 1) * per, mk[k]) for k in range(kmax)]
    return out


def render(ds, curves, title=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = sorted(curves, key=lambda m: (R.FAM_ORDER.get(R.family(m), 4), R.size(m)))
    for m in order:
        pts = curves[m]
        c = T.fam_color(m)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=c, lw=1.6, ms=4, alpha=0.9)
        # start (maj@1) marker + label; end (maj@N) hollow marker
        ax.scatter(xs[0], ys[0], s=70, marker="o", color=c, edgecolor="k", zorder=4)
        ax.annotate(name(m), (xs[0], ys[0]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total inference compute  (GFLOPs, whole dataset, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title or f"Compute vs accuracy — maj@n voting — {ds}\n"
                          f"(each curve = one model, N=1→{NMAX} samples)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = f"{FIGDIR}/{ds}_maj_tradeoff.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def dump_table(ds, curves):
    print(f"\n===== {ds} =====")
    print(f"{'model':16s} {'maj@1':>6s} {'maj@'+str(NMAX):>6s} {'Δacc':>6s} "
          f"{'@1 TFLOP':>9s} {'@'+str(NMAX)+' TFLOP':>9s}")
    for m in sorted(curves, key=lambda m: curves[m][-1][1], reverse=True):
        pts = curves[m]
        g1, a1 = pts[0]; gN, aN = pts[-1]
        print(f"  {name(m):16s} {a1:6.3f} {aN:6.3f} {aN-a1:+6.3f} "
              f"{g1/1e6:9.2f} {gN/1e6:9.2f}")


def average(per_ds):
    dss = list(per_ds)
    common = set.intersection(*[set(per_ds[d]) for d in dss])
    avg = {}
    for m in common:
        k = min(len(per_ds[d][m]) for d in dss)   # align curve lengths
        avg[m] = [(sum(per_ds[d][m][i][0] for d in dss) / len(dss),
                   sum(per_ds[d][m][i][1] for d in dss) / len(dss)) for i in range(k)]
    return avg


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    per_ds = {}
    for ds in DATASETS:
        curves = collect(ds)
        if not curves:
            print(f"[{ds}] no data"); continue
        per_ds[ds] = curves
        out = render(ds, curves)
        dump_table(ds, curves)
        print(f"  -> wrote {out}")
    if len(per_ds) == len(DATASETS):
        avg = average(per_ds)
        out = render("avg", avg,
                     title=f"Compute vs accuracy — maj@n voting — average of both datasets\n"
                           f"(each curve = one model, N=1→{NMAX} samples)")
        dump_table("avg", avg)
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
