"""VLM port of src/rejection_sampling.py: iterative rejection sampling with a VLM
solver and a VLM (or oracle) verifier.

Loop semantics mirror the original exactly:
  - each attempt re-solves the still-active problems with seed = base_seed + attempt;
  - attempt 0 permanently drops problems whose output has no extractable answer;
  - answers the verifier accepts leave the pool; the rest are re-solved next round;
  - the last attempt is solve-only (no verification);
  - per-round GFLOPs = total tokens x 2 x params(B), solver and verifier separately.

VLM specifics: prompts carry the image (solver and verifier both see it), prompt
rendering goes through load_chat_renderer (AutoProcessor or ChatML fallback), and
correctness comes from the dataset-specific scorer (countbench integer match /
charxiv strict final-answer match).

Usage:
    python vlm/rejection_sampling.py \
        --solver_model_name Qwen/Qwen3-VL-8B-Instruct \
        --verifier_model_name google/gemma-4-12B-it \
        --data_dir data/countbench --max_attempts 5 \
        --output_dir vlm/result/rejection/countbench/qwen8b__gemma12b
"""

from pprint import pprint
from types import SimpleNamespace
import argparse
import gc
import json
import math
import os
import sys
import time

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
from vlm_inference import (load_image_qa, load_chat_renderer, build_messages,  # noqa: E402
                           run_vllm_backend)
from vlm_verify import build_verifier_messages, truncate_response, extract_verdict  # noqa: E402
import score_results  # noqa: E402
import score_charxiv  # noqa: E402

# Approximate (active) parameter counts in billions, for the paper's GFLOPs convention
# (gflops = total_tokens * 2 * params_B). Gemma E-sizes use effective params.
MODEL_SIZES = {
    "Qwen/Qwen3-VL-2B-Instruct": 2, "Qwen/Qwen3-VL-4B-Instruct": 4,
    "Qwen/Qwen3-VL-8B-Instruct": 8,
    "google/gemma-4-E2B-it": 2, "google/gemma-4-E4B-it": 4, "google/gemma-4-12B-it": 12,
    "llava-hf/llava-1.5-7b-hf": 7, "llava-hf/llava-1.5-13b-hf": 13,
    "OpenGVLab/InternVL3_5-1B": 1, "OpenGVLab/InternVL3_5-2B": 2,
    "OpenGVLab/InternVL3_5-4B": 4, "OpenGVLab/InternVL3_5-8B": 8,
    "OpenGVLab/InternVL3_5-14B": 14,
}


def countbench_extract(text):
    return score_results.extract_count(text)


def countbench_correct(gold, text):
    ex = score_results.extract_count(text)
    try:
        return ex is not None and float(ex) == float(gold)
    except (TypeError, ValueError):
        return False


def charxiv_extract(text):
    ex = score_charxiv.extract_answer(text)
    return ex if ex else None


def charxiv_correct(gold, text):
    return score_charxiv.is_correct(gold, text)


DATASET_FNS = {
    "countbench": {"extract": countbench_extract, "correct": countbench_correct},
    "charxiv": {"extract": charxiv_extract, "correct": charxiv_correct},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver_model_name", type=str, required=True)
    p.add_argument("--verifier_model_name", type=str, default=None,
                   help="Required unless --oracle_verifier")
    p.add_argument("--prompt_dir", type=str, default="prompts")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset dir (data/countbench or data/charxiv)")
    p.add_argument("--dataset_subset_ratio", type=float, default=1.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    p.add_argument("--solver_max_new_tokens", type=int, default=8192)
    p.add_argument("--solver_temperature", type=float, default=0.7)
    p.add_argument("--solver_top_k", type=int, default=-1)
    p.add_argument("--solver_top_p", type=float, default=0.9)
    p.add_argument("--solver_repetition_penalty", type=float, default=1.0)
    p.add_argument("--solver_max_model_len", type=int, default=None)
    p.add_argument("--solver_disable_chunked_mm", action="store_true")

    p.add_argument("--verifier_max_new_tokens", type=int, default=512)
    p.add_argument("--verifier_temperature", type=float, default=0.7)
    p.add_argument("--verifier_top_k", type=int, default=-1)
    p.add_argument("--verifier_top_p", type=float, default=0.9)
    p.add_argument("--verifier_repetition_penalty", type=float, default=1.0)
    p.add_argument("--verifier_max_model_len", type=int, default=None)
    p.add_argument("--verifier_disable_chunked_mm", action="store_true")
    p.add_argument("--max_response_chars", type=int, default=6000)

    p.add_argument("--max_attempts", type=int, default=5)
    p.add_argument("--oracle_verifier", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert args.oracle_verifier or args.verifier_model_name, "need a verifier or --oracle_verifier"
    assert args.solver_model_name in MODEL_SIZES
    assert args.oracle_verifier or args.verifier_model_name in MODEL_SIZES
    return args


def free_gpu_wait(timeout_s=90):
    """Block until the previous vLLM engine's workers have released the GPUs."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free, total = torch.cuda.mem_get_info(0)
        if free > 0.8 * total:
            return
        time.sleep(2)
    print(f"[warn] GPUs still busy after {timeout_s}s; proceeding anyway")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    pprint(vars(args))
    print("=" * 80)

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    fns = DATASET_FNS[dataset_name]

    dataset = load_image_qa(args.data_dir)
    dataset = dataset.shuffle(seed=args.seed)
    if args.dataset_subset_ratio < 1.0:
        dataset = dataset.select(range(math.ceil(len(dataset) * args.dataset_subset_ratio)))
    assert len(dataset) > 0
    print(f"Dataset size: {len(dataset)}")

    with open(f"{args.prompt_dir}/inference_prompt.md") as f:
        inference_prompt = f.read()
    with open(f"{args.prompt_dir}/verification_prompt.md") as f:
        verification_prompt = f.read()

    solver_render = load_chat_renderer(args.solver_model_name)
    verifier_render = None if args.oracle_verifier else load_chat_renderer(args.verifier_model_name)

    active = list(range(len(dataset)))
    records = {}            # idx -> latest-attempt record (accepted ones keep theirs)
    accepted_attempt = {}   # idx -> attempt at which the verifier accepted it
    all_metrics = {-1: {"total_in_original_data": len(dataset)}}

    for attempt in range(args.max_attempts):
        print(f"\n========= ATTEMPT {attempt + 1}/{args.max_attempts}: {len(active)} active =========")
        it_seed = args.seed + attempt

        # ------------------------------- SOLVE -------------------------------
        rendered, images = [], []
        for idx in active:
            ex = dataset[idx]
            rendered.append(solver_render(build_messages(
                inference_prompt.format(question=ex["question"]))))
            images.append(ex["image_path"])
        sargs = SimpleNamespace(
            solver_model_name=args.solver_model_name,
            solver_temperature=args.solver_temperature,
            solver_max_new_tokens=args.solver_max_new_tokens,
            solver_top_k=args.solver_top_k, solver_top_p=args.solver_top_p,
            solver_repetition_penalty=args.solver_repetition_penalty,
            solver_n_samples=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.solver_max_model_len,
            disable_chunked_mm=args.solver_disable_chunked_mm,
            seed=it_seed,
        )
        free_gpu_wait()
        outputs, solver_tokens = run_vllm_backend(sargs, rendered, images)
        free_gpu_wait()

        # --------------------------- EVALUATE SOLVER --------------------------
        bad = 0
        verify_jobs = []  # (idx, record)
        for out, idx in zip(outputs, active):
            extracted = fns["extract"](out)
            if extracted is None:
                bad += 1
                continue
            correct = bool(fns["correct"](dataset[idx]["answer"], out))
            records[idx] = {
                "id": dataset[idx]["id"], "image": dataset[idx]["image_path"],
                "question": dataset[idx]["question"], "answer": dataset[idx]["answer"],
                "attempt": attempt,
                "solver_full_output": out,
                "solver_extracted_answer": extracted,
                "solver_correct": correct,
            }
            verify_jobs.append(idx)
        if attempt == 0:
            active = sorted(records.keys())
            print(f"attempt 0 drops malformed outputs; {len(active)} problems remain")

        metrics = {
            "attempt": attempt,
            "solver": {
                "total": len(records),
                "accuracy": sum(r["solver_correct"] for r in records.values()) / len(records),
                "total_this_iteration": len(outputs),
                "bad_count_this_iteration": bad,
                "gflops": solver_tokens * 2 * MODEL_SIZES[args.solver_model_name],
            },
        }

        if attempt == args.max_attempts - 1 or not verify_jobs:
            pprint(metrics)
            all_metrics[attempt] = metrics
            break

        # ------------------------------- VERIFY -------------------------------
        if args.oracle_verifier:
            accepted = [i for i in verify_jobs if records[i]["solver_correct"]]
            verifier_tokens = 0
        else:
            v_rendered, v_images = [], []
            for idx in verify_jobs:
                r = records[idx]
                vtext = verification_prompt.format(
                    question=r["question"],
                    response=truncate_response(r["solver_full_output"], args.max_response_chars))
                v_rendered.append(verifier_render(build_verifier_messages(vtext)))
                v_images.append(r["image"])
            vargs = SimpleNamespace(
                solver_model_name=args.verifier_model_name,
                solver_temperature=args.verifier_temperature,
                solver_max_new_tokens=args.verifier_max_new_tokens,
                solver_top_k=args.verifier_top_k, solver_top_p=args.verifier_top_p,
                solver_repetition_penalty=args.verifier_repetition_penalty,
                solver_n_samples=1,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.verifier_max_model_len,
                disable_chunked_mm=args.verifier_disable_chunked_mm,
                seed=it_seed,
            )
            v_out, verifier_tokens = run_vllm_backend(vargs, v_rendered, v_images)
            free_gpu_wait()
            accepted = [i for i, o in zip(verify_jobs, v_out) if extract_verdict(o) is True]

        for idx in accepted:
            active.remove(idx)
            accepted_attempt[idx] = attempt

        metrics["verifier"] = {
            "gflops": verifier_tokens * 2 * (0 if args.oracle_verifier
                                             else MODEL_SIZES[args.verifier_model_name]),
            "problems_newly_accepted": len(accepted),
        }
        pprint(metrics)
        all_metrics[attempt] = metrics

        if not active:
            print("No more problems to solve!")
            break

    # ------------------------------- SAVE -------------------------------
    for idx, r in records.items():
        r["accepted_attempt"] = accepted_attempt.get(idx)

    final = {
        "metadata": {
            "solver_model": args.solver_model_name,
            "verifier_model": "oracle" if args.oracle_verifier else args.verifier_model_name,
            "dataset": dataset_name, "max_attempts": args.max_attempts,
            "seed": args.seed, "max_response_chars": args.max_response_chars,
            "solver_repetition_penalty": args.solver_repetition_penalty,
            "verifier_repetition_penalty": args.verifier_repetition_penalty,
        },
        "iterations": all_metrics,
    }
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(final, f, indent=4)
    with open(f"{args.output_dir}/records.json", "w") as f:
        json.dump({"metadata": final["metadata"],
                   "records": [records[k] for k in sorted(records)]}, f, indent=4)
    print(f"\nsaved metrics + records to {args.output_dir}")


if __name__ == "__main__":
    main()
