"""Vision-Language-Model solver for the CountBenchQA task.

This mirrors the *solver* half of `src/inference.py`, but for a VLM: each example
carries an image plus a counting question, and the model must answer with the count.

The procedure follows the normal LLM solver exactly:

    load data -> shuffle(seed) -> (optional) subset
              -> format `prompts/inference_prompt.md` with the question
              -> wrap in a single user message (now image + text)
              -> apply the model's chat template (add_generation_prompt=True)
              -> vLLM generate
              -> extract_float_answer -> exact_match -> accuracy

Before anything is sent to the model, every constructed prompt is written to
`<output_dir>/prompts.jsonl`, one record per line, so the exact image and exact
prompt going to the model can be inspected:

    {
        "id": 0,
        "image": "/abs/path/to/data/countbench/images/042.jpg",
        "question": "How many headsets are there in the image?",
        "answer": 10,
        "text_prompt": "Please reason step by step, ...\n\nHow many headsets ...",
        "rendered_prompt": "<|im_start|>user\n<|vision_start|><|image_pad|>..."
    }

Use `--dump_prompts_only` to stop after writing prompts.jsonl (no model is loaded).

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
import sys

# Reuse the exact answer-extraction and matching logic from the normal LLM pipeline.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
from answer_extractors import extract_float_answer  # noqa: E402
from oracle_verifiers import exact_match  # noqa: E402

from datasets import Dataset  # noqa: E402
from transformers import AutoProcessor, set_seed  # noqa: E402


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

    prompts_path = os.path.join(args.output_dir, "prompts.jsonl")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for r in prompt_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(prompt_records)} prompts to {prompts_path}")
    print("============================ Sample prompt begin ============================")
    print(f"image: {prompt_records[0]['image']}")
    print(prompt_records[0]["rendered_prompt"])
    print("============================ Sample prompt end ============================")

    if args.dump_prompts_only:
        print("--dump_prompts_only set; skipping model load and generation.")
        return

    # ------------------------------ SOLVER ------------------------------------------
    if args.backend == "vllm":
        outputs = run_vllm_backend(args, rendered_prompts, image_paths)
    else:
        outputs = run_transformers_backend(args, prompt_records)

    print("============================ Sample solver output begin ============================")
    print(outputs[0])
    print("============================ Sample solver output end ============================")

    # ------------------------------ EVALUATE SOLVER ----------------------------------
    solver_total = len(dataset) * args.solver_n_samples
    assert len(outputs) == solver_total, (len(outputs), solver_total)

    records = []
    solver_correct_count, bad_solve_count = 0, 0
    for output_i, output in enumerate(outputs):
        data_i = output_i // args.solver_n_samples
        extracted_answer = extract_float_answer(output)
        if extracted_answer is None:
            bad_solve_count += 1
        is_correct = exact_match(data_row=dataset[data_i], solver_extracted_answer=extracted_answer)
        solver_correct_count += is_correct
        records.append({
            "data_row": {k: v for k, v in dataset[data_i].items()},
            "solver_correct": is_correct,
            "solver_full_output": output,
            "solver_extracted_answer": extracted_answer,
        })

    metrics = {
        "solver": {
            "total": solver_total,
            "bad_count": bad_solve_count,
            "correct_count": solver_correct_count,
            "incorrect_count": solver_total - solver_correct_count,
            "accuracy": solver_correct_count / solver_total,
        },
    }

    with open(f"{args.output_dir}/record.json", "w") as f:
        json.dump(records, f, indent=4)
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n============================ Final Metrics ============================")
    pprint(metrics)


if __name__ == "__main__":
    main()
