"""Verifier Gain: turn the solver x verifier confusion matrices into the paper's metric.

Replicates the core analysis of "When Does Verification Pay Off?" (Lu et al.) for VLMs.
Reads the per-pair files produced by vlm/vlm_verify.py (vlm/result/verifier_grid/<ds>/), and
for each (solver, verifier) computes, from the verifier's confusion matrix on the solver's
solutions + the solver pass rate p:

    TPR  = tp/(tp+fn)            P(verifier accepts | solution correct)   "sensitivity"
    FPR  = fp/(fp+tn)            P(verifier accepts | solution incorrect)
    prec = p*TPR / (p*TPR + (1-p)*FPR)   precision of an ACCEPT decision

Rejection sampling (draw k iid solver samples, return the first the verifier accepts; if
none accepted, fall back to a random sample) has expected accuracy:

    a       = p*TPR + (1-p)*FPR                 P(a sample is accepted)
    acc(k)  = (1-(1-a)^k)*prec + (1-a)^k * p
    acc(inf)= prec                              (unlimited resampling)

We report VERIFIER GAIN = acc(inf) - p = prec - p  (headline), plus acc(k) for k=1..16.
A negative gain means the verifier HURTS rejection sampling (accepts wrong answers as
readily as right ones). Each pair is tagged self / intra-family / cross-family.

Run:  .venv/bin/python vlm/verifier_gain.py [--dataset charxiv]
Outputs: vlm/result/verifier_grid/<ds>_gain.csv  + printed matrices.
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict


def family(short: str) -> str:
    s = short.lower()
    if "qwen3-vl" in s: return "qwen-vl"
    if "internvl" in s: return "internvl"
    if "gemma" in s:    return "gemma"
    if "llava" in s:    return "llava"
    return "other"


def regime(solver: str, verifier: str) -> str:
    if solver == verifier: return "self"
    return "intra" if family(solver) == family(verifier) else "cross"


def acc_at_k(p, tpr, fpr, k):
    a = p * tpr + (1 - p) * fpr
    if a <= 0:                      # verifier accepts nothing -> always fall back to random
        return p
    prec = p * tpr / a
    return (1 - (1 - a) ** k) * prec + (1 - a) ** k * p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="charxiv")
    ap.add_argument("--ks", default="1,2,4,8,16")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    griddir = f"vlm/result/verifier_grid/{args.dataset}"

    # latest file per (solver, verifier)
    pat = re.compile(rf"verify_{args.dataset}_solver-(.+?)_verifier-(.+?)_(\d{{8}}-\d{{6}})\.json")
    latest = {}
    for f in glob.glob(f"{griddir}/verify_{args.dataset}_*.json"):
        m = pat.search(os.path.basename(f))
        if not m: continue
        solver, verifier, tss = m.group(1), m.group(2), m.group(3)
        key = (solver, verifier)
        if key not in latest or tss > latest[key][0]:
            latest[key] = (tss, f)

    rows = []
    for (solver, verifier), (_, f) in sorted(latest.items()):
        d = json.load(open(f))
        v = d["metrics"]["verifier"]
        p = d["metrics"]["solver_accuracy"]
        tp, tn, fp, fn = v["tp"], v["tn"], v["fp"], v["fn"]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (tp + fn) if (tp + fn) else 0.0   # 1 - tpr = miss rate on correct answers
        # F1 of the ACCEPT decision (treat "verifier accepts a correct answer" as the positive
        # class): precision_clf = tp/(tp+fp), recall_clf = tpr; f1 = harmonic mean.
        prec_clf = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec_clf * tpr / (prec_clf + tpr)) if (prec_clf + tpr) > 0 else 0.0
        prec = (p * tpr) / (p * tpr + (1 - p) * fpr) if (p * tpr + (1 - p) * fpr) > 0 else p
        row = dict(solver=solver, verifier=verifier, regime=regime(solver, verifier),
                   solver_fam=family(solver), verifier_fam=family(verifier),
                   p=p, verifier_acc=v["accuracy"], tpr=tpr, fpr=fpr, fnr=fnr,
                   f1=f1, precision=prec,
                   gain=prec - p, bad=v.get("bad_count", 0), total=v["total"])
        for k in ks:
            row[f"gain@{k}"] = acc_at_k(p, tpr, fpr, k) - p
        rows.append(row)

    if not rows:
        print(f"no verifier-grid files in {griddir}"); return

    cols = (["solver", "verifier", "regime", "solver_fam", "verifier_fam", "p",
             "verifier_acc", "tpr", "fpr", "fnr", "f1", "precision", "gain"]
            + [f"gain@{k}" for k in ks] + ["bad", "total"])
    out = f"{griddir}_gain.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} pairs -> {out}\n")

    models = sorted({r["solver"] for r in rows} | {r["verifier"] for r in rows},
                    key=lambda m: (family(m), m))
    cell = {(r["solver"], r["verifier"]): r for r in rows}

    def matrix(title, fn):
        print(f"===== {title} (rows=verifier, cols=solver) =====")
        sh = lambda m: m.replace("-Instruct", "").replace("InternVL3-5", "IVL").replace("Qwen3-VL", "Q")[:9]
        corner = "verif\\solv"
        print(f"{corner:11}" + "".join(f"{sh(s):>9}" for s in models))
        for vmod in models:
            line = [f"{sh(vmod):11}"]
            for smod in models:
                r = cell.get((smod, vmod))
                line.append(f"{fn(r):>9}" if r else f"{'-':>9}")
            print("".join(line))
        print()

    matrix("VERIFIER ACCURACY", lambda r: f"{r['verifier_acc']:.3f}")
    matrix("VERIFIER F1 (accept-decision)", lambda r: f"{r['f1']:.3f}")
    matrix("VERIFIER FNR (miss rate on correct)", lambda r: f"{r['fnr']:.3f}")
    matrix("VERIFIER GAIN (prec - p)", lambda r: f"{r['gain']:+.3f}")

    print("===== mean verifier gain by regime =====")
    byreg = defaultdict(list)
    for r in rows: byreg[r["regime"]].append(r["gain"])
    means = {}
    for reg in ["self", "intra", "cross"]:
        g = byreg.get(reg, [])
        if g:
            means[reg] = sum(g) / len(g)
            print(f"  {reg:6} n={len(g):2}  mean gain={means[reg]:+.3f}  "
                  f"(min {min(g):+.3f}, max {max(g):+.3f})")

    # ---- by-regime bar figure (the paper's self/intra/cross headline) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = "vlm/result/plots"; os.makedirs(out_dir, exist_ok=True)
        regs = [r for r in ["self", "intra", "cross"] if r in means]
        colors = {"self": "#d62728", "intra": "#1f77b4", "cross": "#2ca02c"}
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        xs = range(len(regs))
        ax.bar(xs, [means[r] for r in regs], color=[colors[r] for r in regs], alpha=0.85,
               width=0.6)
        for i, r in enumerate(regs):
            ax.scatter([i] * len(byreg[r]), byreg[r], color="black", s=12, alpha=0.5, zorder=3)
            ax.text(i, means[r], f"{means[r]:+.3f}", ha="center",
                    va="bottom" if means[r] >= 0 else "top", fontsize=10)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(list(xs)); ax.set_xticklabels([f"{r}\n(n={len(byreg[r])})" for r in regs])
        ax.set_ylabel("verifier gain (precision - solver acc)")
        ax.set_title(f"{args.dataset}: verifier gain by regime")
        fig.tight_layout()
        png = f"{out_dir}/{args.dataset}_gain_by_regime.png"
        fig.savefig(png, dpi=130); print(f"\nwrote {png}")
    except Exception as e:
        print(f"\n[skip by-regime plot: {e}]")


if __name__ == "__main__":
    main()
