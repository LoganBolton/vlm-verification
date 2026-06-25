"""Self-consistency / majority-vote@N baseline -- the verification-free counterpart
to vlm/rejection_sampling.py.

Each problem is solved N times (one vLLM call with n=N, temperature sampling). From the
N samples we compute, as a function of budget k = 1..N:
  - maj@k   : accuracy of the majority-vote answer over the first k samples
              (ties broken by first occurrence; samples with no extractable answer abstain)
  - pass@k  : coverage -- fraction where ANY of the first k samples is correct
              (this is the self-consistency analogue of the oracle ceiling)
  - avg@1   : mean single-sample accuracy (expected pass@1)

Compare maj@k against the rubber-stamp verifier's final accuracy (from the matching
rejection run) and pass@k against the oracle ceiling: does spending the same extra
solver compute on majority vote capture more of the headroom than the verifier does?

Usage:
    python vlm/self_consistency.py --solver_model_name OpenGVLab/InternVL3_5-8B \
        --data_dir data/countbench --n_samples 16 \
        --output_dir vlm/result/self_consistency/countbench/InternVL3-5-8B
"""
from collections import Counter
from pprint import pprint
from types import SimpleNamespace
import argparse, json, math, os, sys

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
from vlm_inference import (load_image_qa, load_chat_renderer, build_messages,  # noqa: E402
                           run_vllm_backend)
from rejection_sampling import MODEL_SIZES, DATASET_FNS, free_gpu_wait  # noqa: E402
import score_charxiv  # noqa: E402


def answer_key(dataset_name, extracted):
    """Canonical key for grouping votes; same answer -> same key -> same correctness."""
    if extracted is None:
        return None
    if dataset_name == "countbench":
        try:
            return float(extracted)
        except (TypeError, ValueError):
            return None
    return score_charxiv.normalize(str(extracted))  # charxiv: normalized text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver_model_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--prompt_dir", type=str, default="prompts")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--solver_temperature", type=float, default=0.7)
    p.add_argument("--solver_top_p", type=float, default=0.9)
    p.add_argument("--solver_top_k", type=int, default=-1)
    p.add_argument("--solver_max_new_tokens", type=int, default=8192)
    p.add_argument("--solver_repetition_penalty", type=float, default=1.0)
    p.add_argument("--solver_max_model_len", type=int, default=None)
    p.add_argument("--solver_disable_chunked_mm", action="store_true")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert args.solver_model_name in MODEL_SIZES
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80); pprint(vars(args)); print("=" * 80)

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    fns = DATASET_FNS[dataset_name]
    N = args.n_samples

    # Identical loading/shuffle to rejection_sampling so the problem set matches exactly.
    dataset = load_image_qa(args.data_dir).shuffle(seed=args.seed)
    print(f"Dataset size: {len(dataset)}  | n_samples={N}")

    with open(f"{args.prompt_dir}/inference_prompt.md") as f:
        inference_prompt = f.read()
    solver_render = load_chat_renderer(args.solver_model_name)

    rendered, images = [], []
    for ex in dataset:
        rendered.append(solver_render(build_messages(
            inference_prompt.format(question=ex["question"]))))
        images.append(ex["image_path"])

    sargs = SimpleNamespace(
        solver_model_name=args.solver_model_name,
        solver_temperature=args.solver_temperature,
        solver_max_new_tokens=args.solver_max_new_tokens,
        solver_top_k=args.solver_top_k, solver_top_p=args.solver_top_p,
        solver_repetition_penalty=args.solver_repetition_penalty,
        solver_n_samples=N,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.solver_max_model_len,
        disable_chunked_mm=args.solver_disable_chunked_mm,
        seed=args.seed,
    )
    free_gpu_wait()
    outputs, solver_tokens = run_vllm_backend(sargs, rendered, images)  # flat, N per prompt
    free_gpu_wait()
    assert len(outputs) == len(dataset) * N

    # Per-problem: list of (key, correct_bool) for the N samples.
    records = []
    for i, ex in enumerate(dataset):
        samples = outputs[i * N:(i + 1) * N]
        keys, corr = [], {}
        for o in samples:
            k = answer_key(dataset_name, fns["extract"](o))
            keys.append(k)
            if k is not None and k not in corr:
                corr[k] = bool(fns["correct"](ex["answer"], o))
        records.append({"id": ex["id"], "answer": ex["answer"],
                        "keys": [None if k is None else str(k) for k in keys],
                        "key_correct": {str(k): v for k, v in corr.items()},
                        "_keys": keys, "_corr": corr})

    # Metric curves over budget k = 1..N.
    maj_at_k, pass_at_k = [], []
    for k in range(1, N + 1):
        maj_correct = cover_correct = 0
        for r in records:
            ks = [x for x in r["_keys"][:k] if x is not None]
            # coverage / pass@k: any correct among first k
            if any(r["_corr"].get(x, False) for x in ks):
                cover_correct += 1
            # majority vote: modal key, ties -> earliest to reach the count
            if ks:
                cnt = Counter(ks)
                top = max(cnt.values())
                for x in ks:                      # first key achieving the top count
                    if cnt[x] == top:
                        winner = x; break
                if r["_corr"].get(winner, False):
                    maj_correct += 1
        maj_at_k.append(maj_correct / len(records))
        pass_at_k.append(cover_correct / len(records))
    avg1 = sum(r["_corr"].get(r["_keys"][0], False) if r["_keys"][0] is not None else False
               for r in records) / len(records)

    metrics = {
        "metadata": {"solver_model": args.solver_model_name, "dataset": dataset_name,
                     "n_samples": N, "n_problems": len(records),
                     "temperature": args.solver_temperature, "seed": args.seed,
                     "solver_gflops": solver_tokens * 2 * MODEL_SIZES[args.solver_model_name],
                     "scorer": score_charxiv.EXTRACTOR_NAME if dataset_name == "charxiv" else "count_exact"},
        "avg_at_1": avg1, "maj_at_k": maj_at_k, "pass_at_k": pass_at_k,
    }
    pprint({"avg@1": round(avg1, 4),
            "maj@1": round(maj_at_k[0], 4), f"maj@{N}": round(maj_at_k[-1], 4),
            f"pass@{N}": round(pass_at_k[-1], 4)})

    json.dump(metrics, open(os.path.join(args.output_dir, "metrics.json"), "w"), indent=2)
    for r in records:           # drop transient fields before saving
        r.pop("_keys", None); r.pop("_corr", None)
    json.dump({"metadata": metrics["metadata"], "records": records},
              open(os.path.join(args.output_dir, "records.json"), "w"), indent=2)
    print(f"\nwrote metrics + records to {args.output_dir}")


if __name__ == "__main__":
    main()
