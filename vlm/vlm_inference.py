"""Vision-Language-Model solver for the CountBenchQA task.

This mirrors the *solver* half of `src/inference.py`, but for a VLM: each example
carries an image plus a counting question, and the model must answer with the count.

The procedure follows the normal LLM solver exactly:

    load data -> shuffle(seed) -> (optional) subset
              -> format `prompts/inference_prompt.md` with the question
              -> wrap in a single user message (now image + text)
              -> apply the model's chat template (add_generation_prompt=True)
              -> generate (transformers or vLLM) -> save raw outputs

This script does NOT judge correctness. It only records what the model produced.
Scoring (answer extraction + accuracy) is a separate step: `vlm/score_results.py`.

Each run writes ONE self-describing JSON file named `<dataset>_<model>_<time>.json`
(e.g. `countbench_Qwen3-VL-2B-Instruct_20260607-223800.json`) so results are easy to
track down later. The file contains two top-level keys:

    {
      "metadata": {                # what data, what prompt, what model, what params
        "timestamp": "...",
        "dataset": {"name", "data_dir", "num_examples", "dataset_subset_ratio", ...},
        "model":   {"name", "backend"},
        "generation_params": {"temperature", "top_p", "top_k", "max_new_tokens", ...},
        "prompt":  {"template", "sample_text_prompt", "sample_rendered_prompt", ...},
        "args":    { ...full argparse namespace... }
      },
      "records": [ {                # per example: the exact prompt fed in + raw output
        "id", "image", "question", "answer",   # answer = ground truth, for the scorer
        "text_prompt", "rendered_prompt",
        "solver_full_output"
      }, ... ]
    }

Use `--dump_prompts_only` to stop after building prompts: it writes
`<dataset>_<model>_<time>_prompts.json` (metadata + every prompt, no model loaded).

Verifier is intentionally out of scope for now.
"""

from pprint import pprint
from typing import List, Dict, Any
from datetime import datetime
import argparse
import json
import math
import os
import re

from datasets import Dataset
from transformers import AutoProcessor, set_seed


def load_image_qa(data_dir: str) -> Dataset:
    """Load a locally-saved image-QA subset into a Dataset.

    Works for any dataset laid out like the download scripts produce: a
    `<data_dir>/metadata.jsonl` (one record per line) with at least 'id', 'image'
    (path relative to `data_dir`), 'question', and 'answer' fields. Used for both
    CountBenchQA (`scripts/download_countbench.py`) and CharXiv
    (`scripts/download_charxiv.py`).

    Returns:
        Dataset with the metadata columns plus 'image_path' (absolute path).
    """
    meta_path = os.path.join(data_dir, "metadata.jsonl")
    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["image_path"] = os.path.abspath(os.path.join(data_dir, r["image"]))
            records.append(r)
    return Dataset.from_list(records)


def build_messages(question_text: str) -> List[Dict[str, Any]]:
    """Build a single-turn multimodal user message: one image followed by the text.

    Mirrors the normal LLM's single user message, but the content is now a list
    holding an image placeholder and the formatted text prompt.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question_text},
        ],
    }]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model and prompts
    parser.add_argument("--solver_model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HuggingFace VLM ID for solving")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Prompt directory, should contain inference_prompt.md")

    # dataset
    parser.add_argument("--data_dir", type=str, default="data/countbench",
                        help="Directory with metadata.jsonl and images/ (download_countbench.py output)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Human-readable dataset name used in output filenames/metadata. "
                             "Defaults to the basename of --data_dir (e.g. 'countbench').")
    parser.add_argument("--dataset_subset_ratio", type=float, default=1.0)

    # vllm server initialization
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # solver sampling params (match src/inference.py defaults)
    parser.add_argument("--solver_max_new_tokens", type=int, default=8192)
    parser.add_argument("--solver_temperature", type=float, default=0.7)
    parser.add_argument("--solver_top_k", type=int, default=-1)
    parser.add_argument("--solver_top_p", type=float, default=0.9)
    parser.add_argument("--solver_n_samples", type=int, default=1)

    # backend
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"],
                        help="Generation backend. 'vllm' requires a vLLM build matching the GPU driver; "
                             "'transformers' is torch-version-agnostic and used for newer archs (e.g. Qwen3-VL).")

    # miscellaneous
    parser.add_argument("--output_dir", type=str, default="vlm/result", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump_prompts_only", action="store_true",
                        help="Write prompts.jsonl and exit without loading/running the model")

    args = parser.parse_args()

    if args.solver_n_samples > 1:
        assert args.solver_temperature > 0.0
    return args


def run_vllm_backend(args, rendered_prompts: List[str], image_paths: List[str]) -> List[str]:
    """Generate with vLLM (requires a vLLM build matching the GPU driver's CUDA version)."""
    import torch
    from vllm import LLM, SamplingParams
    from PIL import Image

    model = LLM(
        model=args.solver_model_name,
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    vllm_inputs = [
        {"prompt": rendered, "multi_modal_data": {"image": Image.open(p).convert("RGB")}}
        for rendered, p in zip(rendered_prompts, image_paths)
    ]
    sampling_params = SamplingParams(
        temperature=args.solver_temperature,
        max_tokens=args.solver_max_new_tokens,
        top_k=args.solver_top_k,
        top_p=args.solver_top_p,
        n=args.solver_n_samples,
        seed=args.seed,
    )
    generations = model.generate(vllm_inputs, sampling_params)
    outputs = []
    for g in generations:
        assert len(g.outputs) == args.solver_n_samples
        for o in g.outputs:
            outputs.append(o.text)
    return outputs


def run_transformers_backend(args, prompt_records: List[Dict[str, Any]]) -> List[str]:
    """Generate with HuggingFace transformers.

    Torch-version-agnostic, so it works for brand-new architectures (e.g. Qwen3-VL)
    even when no vLLM build matches the installed CUDA driver. Each example is run
    individually: the image is attached to the user message and the model's processor
    handles tokenization + image preprocessing.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.solver_model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.solver_model_name,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    do_sample = args.solver_temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.solver_max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": args.solver_n_samples,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.solver_temperature
        gen_kwargs["top_p"] = args.solver_top_p
        if args.solver_top_k and args.solver_top_k > 0:
            gen_kwargs["top_k"] = args.solver_top_k

    outputs: List[str] = []
    n = len(prompt_records)
    for i, rec in enumerate(prompt_records):
        # Same single user message as the prompt dump, but with the real image attached.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rec["image"]},
                {"type": "text", "text": rec["text_prompt"]},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens; decode only the newly generated continuation.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        outputs.extend(decoded)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  generated {i + 1}/{n}")
    return outputs


def _slug(text: str) -> str:
    """Filesystem-safe slug (keep alnum, dot, dash; collapse everything else to '-')."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", text).strip("-")


def build_metadata(args, dataset_name: str, num_examples: int, inference_prompt: str,
                   sample_record: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
    """Assemble a self-describing metadata block so any run can be understood in isolation.

    Captures: the data used, the exact prompt fed to the model (template + a rendered
    sample), which model ran, and the generation parameters.
    """
    do_sample = args.solver_temperature > 0.0
    return {
        "timestamp": timestamp,
        "dataset": {
            "name": dataset_name,
            "data_dir": os.path.abspath(args.data_dir),
            "num_examples": num_examples,
            "dataset_subset_ratio": args.dataset_subset_ratio,
            "shuffle_seed": args.seed,
        },
        "model": {
            "name": args.solver_model_name,
            "backend": args.backend,
        },
        "generation_params": {
            "do_sample": do_sample,
            "temperature": args.solver_temperature,
            "top_p": args.solver_top_p,
            "top_k": args.solver_top_k,
            "max_new_tokens": args.solver_max_new_tokens,
            "n_samples": args.solver_n_samples,
            "seed": args.seed,
        },
        "prompt": {
            "template_file": f"{args.prompt_dir}/inference_prompt.md",
            "template": inference_prompt,
            "sample_text_prompt": sample_record["text_prompt"],
            "sample_rendered_prompt": sample_record["rendered_prompt"],
            "sample_image": sample_record["image"],
        },
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("==============================================================================")
    pprint(vars(args))
    print("==============================================================================")

    # ----------------------------- DATA (same procedure) -----------------------------
    dataset = load_image_qa(args.data_dir)
    dataset = dataset.shuffle(seed=args.seed)
    assert isinstance(dataset, Dataset) and {"question", "answer"}.issubset(set(dataset.column_names))
    if args.dataset_subset_ratio < 1.0:
        subset_size = math.ceil(len(dataset) * args.dataset_subset_ratio)
        dataset = dataset.select(range(subset_size))
    assert len(dataset) > 0
    print(f"Dataset size: {len(dataset)}")

    with open(f"{args.prompt_dir}/inference_prompt.md", "r") as f:
        inference_prompt = f.read()

    # The processor owns the model-specific chat template (just like the tokenizer does
    # for the normal LLM). Loading it only downloads config/tokenizer, not the weights.
    processor = AutoProcessor.from_pretrained(args.solver_model_name, trust_remote_code=True)

    # --------------------------- BUILD + DUMP PROMPTS --------------------------------
    prompt_records = []
    rendered_prompts = []
    image_paths = []
    for ex in dataset:
        question_text = inference_prompt.format(question=ex["question"])
        messages = build_messages(question_text)
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered_prompts.append(rendered)
        image_paths.append(ex["image_path"])
        prompt_records.append({
            "id": ex["id"],
            "image": ex["image_path"],
            "question": ex["question"],
            "answer": ex["answer"],
            "text_prompt": question_text,
            "rendered_prompt": rendered,
        })

    # Identity for this run: dataset_model_time -> easy to find/sort later.
    dataset_name = args.dataset_name or os.path.basename(os.path.normpath(args.data_dir))
    model_short = args.solver_model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{_slug(dataset_name)}_{_slug(model_short)}_{timestamp}"

    metadata = build_metadata(args, dataset_name, len(dataset), inference_prompt,
                              prompt_records[0], timestamp)

    print("============================ Sample prompt begin ============================")
    print(f"image: {prompt_records[0]['image']}")
    print(prompt_records[0]["rendered_prompt"])
    print("============================ Sample prompt end ============================")

    if args.dump_prompts_only:
        # Prompts-only artifact: metadata + every constructed prompt, no generation.
        out_path = os.path.join(args.output_dir, f"{run_id}_prompts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "prompts": prompt_records}, f, indent=4)
        print(f"--dump_prompts_only set; wrote {len(prompt_records)} prompts to {out_path}")
        return

    # ------------------------------ SOLVER ------------------------------------------
    if args.backend == "vllm":
        outputs = run_vllm_backend(args, rendered_prompts, image_paths)
    else:
        outputs = run_transformers_backend(args, prompt_records)

    print("============================ Sample solver output begin ============================")
    print(outputs[0])
    print("============================ Sample solver output end ============================")

    # ------------------------------ COLLECT RAW OUTPUTS ------------------------------
    # This file holds raw model generations only. Correctness/accuracy is intentionally
    # NOT computed here -- scoring is a separate step (see vlm/score_results.py) so the
    # generation artifact and the evaluation can be regenerated/changed independently.
    solver_total = len(dataset) * args.solver_n_samples
    assert len(outputs) == solver_total, (len(outputs), solver_total)

    records = []
    for output_i, output in enumerate(outputs):
        data_i = output_i // args.solver_n_samples
        pr = prompt_records[data_i]
        records.append({
            "id": pr["id"],
            "image": pr["image"],
            "question": pr["question"],
            "answer": pr["answer"],          # ground truth, kept for the scorer
            # exact prompt fed to the model for this example
            "text_prompt": pr["text_prompt"],
            "rendered_prompt": pr["rendered_prompt"],
            "solver_full_output": output,
        })

    # Single self-describing artifact: metadata + per-example raw generations.
    out_path = os.path.join(args.output_dir, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "records": records}, f, indent=4)

    print(f"\nSaved {len(records)} raw generations to {out_path}")
    print("Run vlm/score_results.py on this file to compute correctness.")


if __name__ == "__main__":
    main()
