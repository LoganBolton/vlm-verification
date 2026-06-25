"""Collate every VLM result into one long-format table.

The experiments wrote results into several directory conventions over time:
  - base single-shot:  vlm/result/<ds>[ _tier2 | _tier3 | _intern ]/<ds>_<model>_<ts>_scores.json
  - majority vote:     vlm/result/self_consistency/<ds>/<model>/metrics.json   (avg_at_1, maj_at_k[])
  - VLM judge:         vlm/result/rejection/<ds>/<solver>__<verifier>/metrics.json (iterations.*)
  - agentic zoom:      vlm/result/agentic_vision/<ds>_c<budget>/<model>/metrics.json

This script scans all of them and emits ONE csv (vlm/result/ALL_RESULTS.csv) with a row per
(dataset, model, method, variant) plus the source path, then prints a coverage matrix so the
gaps are obvious. It is the single source of truth -- read this, not the scattered dirs.

Run:  .venv/bin/python vlm/collate_results.py
"""
import csv, glob, json, os, re

RESULT = "vlm/result"
DATASETS = ["countbench", "charxiv"]
OUT = os.path.join(RESULT, "ALL_RESULTS.csv")


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def norm_model(name: str) -> str:
    """Strip HF org prefix and unify hyphen/underscore so every source agrees."""
    n = name.split("/")[-1]
    return n.replace("_", "-")


# ---- source scanners: each yields dict rows ----

def scan_base():
    # tier1 = bare <ds>; plus _tier2/_tier3/_intern. Skip verify_* files (those are old judge runs).
    for ds in DATASETS:
        for sub in [ds, f"{ds}_tier2", f"{ds}_tier3", f"{ds}_intern"]:
            for f in glob.glob(f"{RESULT}/{sub}/{ds}_*_scores.json"):
                if os.path.basename(f).startswith("verify_"):
                    continue
                d = jload(f)
                if not d:
                    continue
                acc = d.get("metrics", {}).get("solver", {}).get("accuracy")
                n = d.get("metrics", {}).get("solver", {}).get("total")
                m = re.match(rf"{ds}_(.+?)_\d{{8}}-\d{{6}}_scores\.json", os.path.basename(f))
                if acc is None or not m:
                    continue
                yield dict(dataset=ds, model=norm_model(m.group(1)), method="base",
                           variant="single-shot", accuracy=acc, n=n, source=f)


def scan_majority():
    for ds in DATASETS:
        for mdir in sorted(glob.glob(f"{RESULT}/self_consistency/{ds}/*")):
            d = jload(os.path.join(mdir, "metrics.json"))
            if not d:
                continue
            mk = d.get("maj_at_k") or []
            model = norm_model(os.path.basename(mdir))
            ns = len(mk)
            if d.get("avg_at_1") is not None:
                yield dict(dataset=ds, model=model, method="maj", variant="avg@1",
                           accuracy=d["avg_at_1"], n=1, source=mdir)
            if mk:
                yield dict(dataset=ds, model=model, method="maj", variant=f"maj@{ns}",
                           accuracy=mk[-1], n=ns, source=mdir)
                peak = max(mk); pk = mk.index(peak) + 1
                yield dict(dataset=ds, model=model, method="maj", variant=f"maj@peak(k={pk})",
                           accuracy=peak, n=pk, source=mdir)


def scan_rejection():
    for ds in DATASETS:
        for mdir in sorted(glob.glob(f"{RESULT}/rejection/{ds}/*")):
            d = jload(os.path.join(mdir, "metrics.json"))
            if not d:
                continue
            its = d.get("iterations", {})
            keys = sorted((int(k) for k in its if int(k) >= 0))
            if not keys:
                continue
            last = its[str(keys[-1])]["solver"]
            solver = norm_model(d.get("metadata", {}).get("solver_model", os.path.basename(mdir)))
            verifier = os.path.basename(mdir).split("__")[-1]
            yield dict(dataset=ds, model=solver, method="judge",
                       variant=f"reject/{verifier}", accuracy=last.get("accuracy"),
                       n=keys[-1] + 1, source=mdir)


def scan_agentic():
    for ds in DATASETS:
        for cdir in sorted(glob.glob(f"{RESULT}/agentic_vision/{ds}_c*")):
            b = cdir.rsplit("_c", 1)[-1]
            if "demo" in cdir:
                continue
            for mdir in sorted(glob.glob(f"{cdir}/*")):
                d = jload(os.path.join(mdir, "metrics.json"))
                if not d:
                    continue
                acc = d.get("accuracy")
                if acc is None:
                    continue
                recs = jload(os.path.join(mdir, "records.json")) or []
                if isinstance(recs, dict):
                    recs = recs.get("records", [])
                avgz = sum(r.get("n_crops", 0) for r in recs) / len(recs) if recs else None
                yield dict(dataset=ds, model=norm_model(os.path.basename(mdir)),
                           method="zoom", variant=f"c{b}", accuracy=acc, n=len(recs),
                           source=mdir, avg_zoom=avgz)


def main():
    rows = []
    for scan in (scan_base, scan_majority, scan_rejection, scan_agentic):
        rows.extend(scan())

    cols = ["dataset", "model", "method", "variant", "accuracy", "n", "avg_zoom", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} rows -> {OUT}\n")

    # ---- coverage matrix: best accuracy per (model, method) per dataset ----
    MODEL_ORDER = ["Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct",
                   "InternVL3-5-1B", "InternVL3-5-2B", "InternVL3-5-4B", "InternVL3-5-8B",
                   "InternVL3-5-14B", "gemma-4-E2B-it", "gemma-4-E4B-it", "gemma-4-12B-it",
                   "llava-1.5-7b-hf", "llava-1.5-13b-hf"]
    METHODS = ["base", "maj", "judge", "zoom"]
    for ds in DATASETS:
        print(f"===== {ds}: best accuracy per method (- = no run) =====")
        print(f"{'model':22} {'base':>7} {'maj':>14} {'judge':>14} {'zoom':>12}")
        models = [m for m in MODEL_ORDER if any(r["dataset"] == ds and r["model"] == m for r in rows)]
        for m in models:
            cells = []
            for meth in METHODS:
                cand = [r for r in rows if r["dataset"] == ds and r["model"] == m
                        and r["method"] == meth and isinstance(r["accuracy"], (int, float))]
                if not best_lbl(cand, meth):
                    cells.append("    -    " if meth in ("maj", "judge") else "   -  ")
                else:
                    cells.append(best_lbl(cand, meth))
            print(f"{m:22} {cells[0]:>7} {cells[1]:>14} {cells[2]:>14} {cells[3]:>12}")
        print()


def best_lbl(cand, meth):
    """Pick the headline cell for a method and format it with its qualifier."""
    if not cand:
        return ""
    if meth == "base":
        r = cand[0]
        return f"{r['accuracy']:.3f}"
    if meth == "maj":
        # prefer the converged maj@N variant
        conv = [r for r in cand if r["variant"].startswith("maj@") and "peak" not in r["variant"]]
        r = max(conv or cand, key=lambda x: x["accuracy"])
        return f"{r['accuracy']:.3f}/{r['variant'].replace('maj@','k=')}"
    if meth == "judge":
        # exclude oracle (upper bound) from the headline; show real verifier
        real = [r for r in cand if "oracle" not in r["variant"]]
        r = max(real or cand, key=lambda x: x["accuracy"])
        return f"{r['accuracy']:.3f}"
    if meth == "zoom":
        r = max(cand, key=lambda x: x["accuracy"])
        return f"{r['accuracy']:.3f}/{r['variant']}"
    return ""


if __name__ == "__main__":
    main()
