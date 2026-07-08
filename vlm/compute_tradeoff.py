#!/usr/bin/env python3
"""Compute-vs-accuracy tradeoff from rejection metrics.json (real measured GFLOPs).

Each rejection cell logs per-attempt gflops for the solver and verifier. We turn that into:
  * SOLO points   — run model M by itself (attempt-0 solver gflops, base accuracy).
  * PIPELINE pts  — small solver + judge rejection (sum of all solver+verifier gflops, final acc).
Then we plot accuracy vs total GFLOPs (log-x) and mark the Pareto frontier (best acc per FLOP).

Writes report/figures/{ds}_compute_tradeoff.png and prints a per-model table.
Run:  .venv/bin/python vlm/compute_tradeoff.py
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import build_charxiv_report as R

DATASETS = ["charxiv", "countbench"]
FIGDIR = "report/figures/tradeoff"


def name(m):
    """full model name with the lab/org prefix stripped (e.g. 'gemma-4-12B-it',
    'Qwen3-VL-8B-Instruct', 'InternVL3_5-1B', 'llava-1.5-7b-hf')."""
    return m.split("/", 1)[-1]


FAM_COLOR = {"qwen-vl": "#9467bd", "gemma": "#1f77b4", "internvl": "#ff7f0e",
             "llava": "#d62728", "other": "#7f7f7f"}


def fam_color(m):
    """fixed per-family color: Qwen purple, gemma blue, InternVL orange, llava red."""
    return FAM_COLOR.get(R.family(m), FAM_COLOR["other"])


def canonical_cost(ds):
    """One canonical single-pass FLOP cost per model = self-consistency solver_gflops / n_samples.
    Same anchor used by the maj@n plot, so 'solo' lines up across both figures. Returns {model: C}."""
    C = {}
    for f in glob.glob(f"vlm/result/self_consistency/{ds}/*/metrics.json"):
        d = json.load(open(f)); md = d.get("metadata", {})
        g, n = md.get("solver_gflops"), md.get("n_samples")
        if g and n:
            C[md["solver_model"]] = g / n
    return C


def cell(f):
    d = json.load(open(f))
    md = d["metadata"]
    M = d["iterations"].get("-1", {}).get("total_in_original_data")
    sp = vp = 0.0                       # solver / verifier passes, in whole-dataset units
    base = final = None
    for k, v in d["iterations"].items():
        if k == "-1":
            continue
        s = v.get("solver", {})
        items = s.get("total_this_iteration", 0) or 0
        sp += items
        if "verifier" in v:             # verifier judges that same fresh set of items
            vp += items
        if v.get("attempt") == 0:
            base = s.get("accuracy")
        if s.get("accuracy") is not None:
            final = s["accuracy"]
    if not M:
        return md, base, final, None, None
    return md, base, final, sp / M, vp / M


def collect(ds):
    """Compute priced at a CANONICAL single-pass cost per model (self-consistency), so 'solo'
    matches the maj@n plot. solo = C[solver]*1 pass; pipeline = C[solver]*sp + C[verifier]*vp."""
    C = canonical_cost(ds)
    solo = {}       # model -> (gflops, acc)
    pipes = []      # (solver, verifier, gflops_total, final_acc, base_acc)
    for f in glob.glob(f"vlm/result/rejection/{ds}/*/metrics.json"):
        md, base, final, sp, vp = cell(f)
        if base is None or final is None or sp is None:
            continue
        s, v = md["solver_model"], md["verifier_model"]
        cs = C.get(s)
        if cs is None:                  # no canonical anchor for this solver -> skip
            continue
        solo[s] = (cs, base)            # one canonical solver pass
        cv = C.get(v, 0.0) if v != "oracle" else 0.0   # oracle judge is free/aspirational
        pipes.append((s, v, cs * sp + cv * vp, final, base))
    return solo, pipes


def pareto(points):
    """points: list of (gflops, acc, label). Return subset on the upper-left frontier."""
    pts = sorted(points, key=lambda p: p[0])
    front, best = [], -1
    for g, a, lab in pts:
        if a > best:
            front.append((g, a, lab))
            best = a
    return front


def best_judge(pipes):
    """best REAL judge per solver model (max final accuracy, oracle excluded)."""
    best = {}   # solver -> (gflops_total, final_acc, verifier)
    for s, v, g, a, b in pipes:
        if v == "oracle":
            continue
        if s not in best or a > best[s][1]:
            best[s] = (g, a, v)
    return best


def render(ds, solo, best, title=None):
    """solo: {model:(gflops,acc)}   best: {model:(gflops,acc,verifier)}."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    allpts = []
    for m, (g, a) in solo.items():
        c = fam_color(m)
        ax.scatter(g, a, s=95, marker="o", color=c, edgecolor="k", zorder=4)
        ax.annotate(name(m), (g, a), fontsize=7, xytext=(4, 3), textcoords="offset points")
        allpts.append((g, a, "solo:" + name(m)))
        if m in best:                          # model + its best judge (judge name not shown)
            bg, ba = best[m][0], best[m][1]
            ax.plot([g, bg], [a, ba], "-", color=c, lw=1, alpha=0.5, zorder=2)
            ax.scatter(bg, ba, s=55, marker="D", color=c, edgecolor="k", alpha=0.9, zorder=3)
            allpts.append((bg, ba, f"{name(m)}+judge"))

    fr = pareto(allpts)
    ax.plot([p[0] for p in fr], [p[1] for p in fr], "-", color="crimson",
            lw=2, zorder=1, label="Pareto frontier (real)")
    ax.set_xscale("log")
    ax.set_xlabel("total inference compute  (GFLOPs, whole dataset, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title or f"Compute vs accuracy — {ds}\n"
                          "(o = model solo,  ◇ = same model + its best judge)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = f"{FIGDIR}/{ds}_compute_tradeoff.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out, fr


def dump_table(ds, solo, best, fr):
    print(f"\n===== {ds} =====")
    print(f"{'model':16s} {'solo acc':>8s} {'solo TFLOP':>10s} | "
          f"{'+best judge acc':>15s} {'TFLOP':>7s} {'Δacc':>6s}")
    for m, (g, a) in sorted(solo.items(), key=lambda kv: kv[1][1], reverse=True):
        if m in best:
            bg, ba = best[m][0], best[m][1]
            print(f"  {name(m):16s} {a:8.3f} {g/1e6:10.2f} | {ba:15.3f} {bg/1e6:7.2f} {ba-a:+6.3f}")
        else:
            print(f"  {name(m):16s} {a:8.3f} {g/1e6:10.2f} | {'(no judge yet)':>15s}")
    print("Pareto frontier — REAL judges (cheapest deployable way to reach each accuracy):")
    for g, a, lab in fr:
        print(f"  {g/1e6:8.2f} TFLOP   acc {a:.3f}   {lab}")


def average(per_ds):
    """per_ds: {ds: (solo, best)}. Return (solo_avg, best_avg) over models present in ALL datasets."""
    dss = list(per_ds)
    solo_avg, best_avg = {}, {}
    common_solo = set.intersection(*[set(per_ds[d][0]) for d in dss])
    for m in common_solo:
        gs = [per_ds[d][0][m][0] for d in dss]
        as_ = [per_ds[d][0][m][1] for d in dss]
        solo_avg[m] = (sum(gs) / len(gs), sum(as_) / len(as_))
    common_best = set.intersection(*[set(per_ds[d][1]) for d in dss]) & common_solo
    for m in common_best:
        gs = [per_ds[d][1][m][0] for d in dss]
        as_ = [per_ds[d][1][m][1] for d in dss]
        best_avg[m] = (sum(gs) / len(gs), sum(as_) / len(as_), "avg")
    return solo_avg, best_avg


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    per_ds = {}
    for ds in DATASETS:
        solo, pipes = collect(ds)
        if not solo:
            print(f"[{ds}] no data"); continue
        best = best_judge(pipes)
        per_ds[ds] = (solo, best)
        out, fr = render(ds, solo, best)
        dump_table(ds, solo, best, fr)
        print(f"  -> wrote {out}")

    if len(per_ds) == len(DATASETS):
        solo_avg, best_avg = average(per_ds)
        out, fr = render("avg", solo_avg, best_avg,
                         title="Compute vs accuracy — average of both datasets\n"
                               "(o = model solo,  ◇ = same model + its best judge)")
        dump_table("avg", solo_avg, best_avg, fr)
        print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
