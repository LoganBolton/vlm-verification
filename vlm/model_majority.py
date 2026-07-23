#!/usr/bin/env python3
"""Cross-MODEL majority vote -- the ensemble counterpart to vlm/self_consistency.py.

Self-consistency takes N samples from ONE model and votes. Here we take ONE answer from
each of MANY DIFFERENT models and let them vote: every model submits its single base-run
answer, and the most common answer across models wins. This reuses the existing base
single-shot runs (vlm/result/<ds>[_tier2|_tier3|_intern]/<ds>_<model>_<ts>_scores.json),
so nothing new is generated -- it's a pure aggregation over answers we already have.

For a fixed set of M models we compute, as a function of ensemble size k = 1..M:
  - maj@k (best)   : majority vote over the k STRONGEST models (added by descending single-
                     model accuracy). This is the realistic "use your k best models" curve.
  - maj@k (random) : expected majority-vote accuracy over a random k-subset of models
                     (mean over many random model orderings) -- what you'd get picking k
                     models blind. Ties -> the earliest model (in the sampled order) that
                     reached the top count, matching self_consistency's tie rule.
  - pass@k         : oracle coverage -- fraction where ANY of k random models is correct.
  - avg@1          : mean single-model accuracy (the "pick one model at random" baseline).
Samples with no extractable answer abstain (don't cast a vote), exactly as in maj@n.

Compute per model is the canonical single-pass GFLOPs shared with the other tradeoff plots
(compute_tradeoff.canonical_cost); the k-model ensemble costs the SUM over its members.

Writes metrics.json + records.json under --output_dir and one figure per dataset. Run:
  .venv/bin/python vlm/model_majority.py --dataset charxiv
  .venv/bin/python vlm/model_majority.py --dataset countbench
  .venv/bin/python vlm/model_majority.py            # both, + report figures
"""
from collections import Counter
from pprint import pprint
import argparse, glob, json, os, random, re, sys

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
import score_charxiv                                    # noqa: E402
import compute_tradeoff as T                            # canonical_cost, fam_color  # noqa: E402

DATASETS = ["charxiv", "countbench"]
RESULT = "vlm/result"
FIGDIR = "report/figures/tradeoff"
N_RANDOM_ORDERS = 400                                   # bootstrap orderings for the random curves


def pretty_model(hf_id):
    """Short display name: drop org prefix + instruct/format suffixes, capitalize family.
    e.g. 'google/gemma-4-12B-it' -> 'Gemma-4-12B'."""
    m = hf_id.split("/")[-1]
    for suf in ("-Instruct", "-it", "-hf"):
        if m.endswith(suf):
            m = m[: -len(suf)]
    for fam in ("gemma", "qwen", "llava"):
        if m.lower().startswith(fam):
            m = m[: len(fam)].capitalize() + m[len(fam):]
    return m


def answer_key(dataset_name, extracted):
    """Canonical vote key: same answer -> same key. None -> abstain. Mirrors self_consistency."""
    if extracted is None or extracted == "":
        return None
    if dataset_name == "countbench":
        try:
            return str(float(extracted))
        except (TypeError, ValueError):
            return None
    return score_charxiv.normalize(str(extracted))


def find_base_runs(ds):
    """model_hf_id -> per-id {key, correct} from the LATEST base scores file for each model.
    Scans tier1/tier2/tier3/intern, skipping verify_* judge files."""
    latest = {}                                          # model -> (timestamp, path)
    for sub in [ds, f"{ds}_tier2", f"{ds}_tier3", f"{ds}_intern"]:
        for f in glob.glob(f"{RESULT}/{sub}/{ds}_*_scores.json"):
            base = os.path.basename(f)
            if base.startswith("verify_"):
                continue
            m = re.match(rf"{ds}_(.+?)_(\d{{8}}-\d{{6}})_scores\.json", base)
            if not m:
                continue
            ts = m.group(2)
            d = json.load(open(f))
            model = d.get("metadata", {}).get("args", {}).get("solver_model_name") \
                or d.get("metadata", {}).get("model", {}).get("name") or m.group(1)
            if model not in latest or ts > latest[model][0]:
                latest[model] = (ts, f)

    runs = {}
    for model, (ts, f) in latest.items():
        d = json.load(open(f))
        per_id = {}
        for r in d["records"]:
            key = answer_key(ds, r.get("solver_extracted_answer"))
            correct = str(r.get("solver_correct")).lower() == "true"
            per_id[str(r["id"])] = {"key": key, "correct": correct if key is not None else False}
        runs[model] = per_id
    return runs


def maj_vote(votes):
    """votes: list of (key, correct) with key != None. Returns (winner_correct, winner_key)
    or (False, None) if no votes. Tie -> earliest key to reach the top count (self_consistency rule)."""
    ks = [k for k, _ in votes]
    if not ks:
        return False, None
    cnt = Counter(ks)
    top = max(cnt.values())
    winner = next(k for k in ks if cnt[k] == top)       # first key achieving the top count
    correct = next(c for k, c in votes if k == winner)  # same key -> same correctness
    return bool(correct), winner


def compute_metrics(ds, runs):
    """All cross-model majority metrics over the common problem set of the given runs."""
    models = list(runs)
    ids = sorted(set.intersection(*[set(r) for r in runs.values()]), key=lambda x: (len(x), x))
    M, P = len(models), len(ids)

    # per-model single-shot accuracy over the common ids
    indiv = {m: sum(runs[m][i]["correct"] for i in ids) / P for m in models}
    avg_at_1 = sum(indiv.values()) / M

    def votes_for(id_, subset):
        return [(runs[m][id_]["key"], runs[m][id_]["correct"])
                for m in subset if runs[m][id_]["key"] is not None]

    # ---- best-k / worst-k curves: add models in descending / ascending individual accuracy ----
    best_order = sorted(models, key=lambda m: indiv[m], reverse=True)
    worst_order = best_order[::-1]
    maj_best, maj_worst = [], []
    for k in range(1, M + 1):
        maj_best.append(sum(maj_vote(votes_for(i, best_order[:k]))[0] for i in ids) / P)
        maj_worst.append(sum(maj_vote(votes_for(i, worst_order[:k]))[0] for i in ids) / P)

    # ---- random-k curves (bootstrap over model orderings): expected maj@k and pass@k ----
    rng = random.Random(42)
    maj_sum = [0.0] * (M + 1)
    pass_sum = [0.0] * (M + 1)
    for _ in range(N_RANDOM_ORDERS):
        order = models[:]
        rng.shuffle(order)
        for k in range(1, M + 1):
            sub = order[:k]
            mc = pc = 0
            for i in ids:
                v = votes_for(i, sub)
                if maj_vote(v)[0]:
                    mc += 1
                if any(c for _, c in v):
                    pc += 1
            maj_sum[k] += mc / P
            pass_sum[k] += pc / P
    maj_random = [maj_sum[k] / N_RANDOM_ORDERS for k in range(1, M + 1)]
    pass_random = [pass_sum[k] / N_RANDOM_ORDERS for k in range(1, M + 1)]

    # ---- full-ensemble vote (all M models) + the winning answer per problem, for records ----
    # iterate in best_order so ties break toward the STRONGER model (and this matches
    # maj_at_k_best[-1], the k=M point of the strongest-first curve).
    records = []
    ens_correct = 0
    for i in ids:
        v = votes_for(i, best_order)
        c, winner = maj_vote(v)
        ens_correct += c
        records.append({"id": i, "winner_key": winner, "correct": c,
                        "n_votes": len(v), "n_models": M})
    ensemble_acc = ens_correct / P

    # ---- compute: canonical single-pass GFLOPs summed over the ensemble members ----
    C = T.canonical_cost(ds)                             # whole-dataset single-pass GFLOPs per model
    cost_best = []
    cum = 0.0
    have_all_cost = all(m in C for m in models)
    for k in range(1, M + 1):
        cum += C.get(best_order[k - 1], 0.0)
        cost_best.append(cum if have_all_cost else None)
    total_cost = sum(C.get(m, 0.0) for m in models) if have_all_cost else None

    return {
        "metadata": {
            "dataset": ds, "n_models": M, "n_problems": P,
            "models": models, "best_order": best_order,
            "indiv_acc": {m: round(indiv[m], 4) for m in models},
            "n_random_orders": N_RANDOM_ORDERS,
            "total_ensemble_gflops": total_cost,
            "cost_best_gflops": cost_best,
            "scorer": score_charxiv.EXTRACTOR_NAME if ds == "charxiv" else "count_exact",
        },
        "avg_at_1": avg_at_1,
        "best_single_acc": max(indiv.values()),
        "ensemble_acc": ensemble_acc,          # == maj_best[-1] == vote over all models
        "maj_at_k_best": maj_best,
        "maj_at_k_worst": maj_worst,
        "maj_at_k_random": maj_random,
        "pass_at_k_random": pass_random,
    }, records


def render(ds, metrics, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = metrics["metadata"]["n_models"]
    ks = list(range(1, M + 1))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ks, metrics["maj_at_k_best"], "-o", color="#2ca02c", lw=1.8, ms=5,
            label="maj@k — k strongest models")
    ax.plot(ks, metrics["maj_at_k_random"], "--s", color="#1f77b4", lw=1.6, ms=4,
            label="maj@k — k random models")
    ax.plot(ks, metrics["maj_at_k_worst"], "-D", color="#ff7f0e", lw=1.8, ms=4,
            label="maj@k — k weakest models")
    best_model = metrics["metadata"].get("best_model") \
        or pretty_model(metrics["metadata"].get("best_order", ["?"])[0])
    ax.axhline(metrics["best_single_acc"], color="#d62728", lw=1.2, ls="-.",
               label=f"best single model ({best_model})")
    ax.axhline(metrics["avg_at_1"], color="0.5", lw=1.0, ls="-.",
               label="avg single model")
    ax.set_xticks(ks)
    ax.set_xlabel("number of models in the ensemble")
    ax.set_ylabel("accuracy")
    ax.set_title(title or f"Cross-model majority vote — {ds}  "
                 f"(ensemble of all {M} = {metrics['ensemble_acc']:.3f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = f"{FIGDIR}/{ds}_model_majority.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def run(ds, output_dir):
    runs = find_base_runs(ds)
    if len(runs) < 2:
        print(f"[{ds}] need >=2 models, found {len(runs)}"); return None
    metrics, records = compute_metrics(ds, runs)
    os.makedirs(output_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"), indent=2)
    json.dump({"metadata": metrics["metadata"], "records": records},
              open(os.path.join(output_dir, "records.json"), "w"), indent=2)
    md = metrics["metadata"]
    print(f"\n===== {ds}  ({md['n_models']} models, {md['n_problems']} problems) =====")
    pprint({"avg@1 (mean single model)": round(metrics["avg_at_1"], 4),
            "best single model": round(metrics["best_single_acc"], 4),
            "maj@2 (best)": round(metrics["maj_at_k_best"][1], 4),
            f"maj@{md['n_models']} (all models ensemble)": round(metrics["ensemble_acc"], 4),
            f"pass@{md['n_models']} (oracle)": round(metrics["pass_at_k_random"][-1], 4)})
    fig = render(ds, metrics)
    print(f"  -> wrote metrics/records to {output_dir}")
    print(f"  -> wrote {fig}")
    return metrics


def average(per_ds):
    """Average the metric curves across datasets, aligned by ensemble size k (1..M).
    maj_at_k_best[k] is 'accuracy of the k strongest models' on each dataset, so averaging by
    index is meaningful even though the per-dataset model ordering differs."""
    metas = list(per_ds.values())
    M = min(m["metadata"]["n_models"] for m in metas)
    n = len(metas)
    avg_curve = lambda key: [sum(m[key][i] for m in metas) / n for i in range(M)]
    avg_scalar = lambda key: sum(m[key] for m in metas) / n
    # best single model across datasets: argmax of per-model accuracy averaged over datasets
    models = metas[0]["metadata"]["indiv_acc"].keys()
    mean_indiv = {mo: sum(m["metadata"]["indiv_acc"][mo] for m in metas) / n for mo in models}
    best_model = pretty_model(max(mean_indiv, key=mean_indiv.get))
    return {
        "metadata": {"n_models": M, "datasets": list(per_ds), "averaged": True,
                     "best_model": best_model},
        "avg_at_1": avg_scalar("avg_at_1"),
        "best_single_acc": avg_scalar("best_single_acc"),
        "ensemble_acc": avg_scalar("ensemble_acc"),
        "maj_at_k_best": avg_curve("maj_at_k_best"),
        "maj_at_k_worst": avg_curve("maj_at_k_worst"),
        "maj_at_k_random": avg_curve("maj_at_k_random"),
        "pass_at_k_random": avg_curve("pass_at_k_random"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="run one dataset; omit to run all")
    p.add_argument("--output_dir", type=str, default=None,
                   help="default vlm/result/model_majority/<ds>")
    return p.parse_args()


def main():
    args = parse_args()
    targets = [args.dataset] if args.dataset else DATASETS
    per_ds = {}
    for ds in targets:
        out = args.output_dir or f"{RESULT}/model_majority/{ds}"
        m = run(ds, out)
        if m is not None:
            per_ds[ds] = m
    if len(per_ds) == len(DATASETS):     # combined figure averaging both datasets
        avg = average(per_ds)
        fig = render("avg", avg,
                     title="Cross Model Majority Vote - Average Across Datasets")
        print(f"\n===== avg (both datasets) =====\n  -> wrote {fig}")


if __name__ == "__main__":
    main()
