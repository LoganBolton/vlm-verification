"""VLM verifier for the CountBenchQA task -- the multimodal analogue of the verifier
half of `src/inference.py`.

A *verifier* model is shown the SAME image, the question, and a *solver* model's full
response, and must judge whether the solver's answer is correct (`\\boxed{correct}`) or
incorrect (`\\boxed{incorrect}`). We then compare that verdict against the ground-truth
label (`solver_correct`, from `vlm/score_results.py`) to measure verifier quality.

The procedure mirrors the LLM verifier exactly, but the verifier message now carries the
image alongside the text:

    for each solved example:
        verification_prompt.format(question=..., response=<solver output>)
        -> user message = [image, that text]
        -> apply verifier's chat template
        -> generate -> extract_verifier_answer -> compare to solver_correct
        -> accuracy / TP / TN / FP / FN / precision / recall / F1

Input is a *scored* solver run (`*_scores.json`, which contains `solver_correct`). One
verifier can be run against several solver runs in a single model load. Each
(solver, verifier) pair is written to its own self-describing file:

    verify_<dataset>_solver-<solver>_verifier-<verifier>_<time>.json

Long solver responses are truncated (head+tail, keeping the final answer) to a uniform
character budget so every verifier -- including short-context ones like LLaVA-1.5 -- can
read every solver's output on equal terms.
"""

from pprint import pprint
from typing import List, Dict, Any
from datetime import datetime
from types import SimpleNamespace
import argparse
import json
import os
import sys

# Reuse the verdict extractor from the normal LLM pipeline, and the backend runners +
# slug helper from the VLM solver script (same dir).
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, VLM_DIR)
from answer_extractors import get_final_box_match  # noqa: E402
from vlm_inference import run_vllm_backend, run_transformers_backend, _slug, load_chat_renderer  # noqa: E402

from transformers import AutoProcessor, set_seed  # noqa: E402

VERDICT_EXTRACTOR = "verdict_boxed_or_prose"


def extract_verdict(text: str):
    """Extract a correct/incorrect verdict from a verifier response.

    Prefers a clean `\\boxed{correct|incorrect}`; otherwise falls back to scanning the
    prose. The strict boxed-only extractor returns None whenever there is no box, which
    wrongly discards models (e.g. LLaVA-1.5) that state the verdict in plain text
    ("The student's answer is correct."). 'incorrect'/'wrong' take precedence over
    'correct' (since "incorrect" contains "correct").

    Returns True (correct), False (incorrect), or None (no verdict found).
    """
    boxed = get_final_box_match(text)
    if boxed is not None and boxed.strip().lower() in ("correct", "incorrect"):
        return boxed.strip().lower() == "correct"
    low = text.lower()
    if "incorrect" in low or "wrong" in low:
        return False
    if "correct" in low:
        return True
    return None


def truncate_response(text: str, max_chars: int) -> str:
    """Trim an over-long solver response to `max_chars`, keeping head and tail.

    The final answer usually sits at the end, so we keep mostly the tail plus a bit of
    the opening, joined by an explicit marker.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def build_verifier_messages(verification_text: str) -> List[Dict[str, Any]]:
    """Single multimodal user message: the image followed by the verification text."""
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": verification_text},
        ],
    }]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_model_name", type=str, required=True,
                        help="HuggingFace VLM ID used as the verifier")
    parser.add_argument("--solver_run_files", type=str, nargs="+", required=True,
                        help="One or more scored solver runs (*_scores.json with solver_correct)")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Directory containing verification_prompt.md")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])

    # verifier sampling params (match src/inference.py solver defaults)
    parser.add_argument("--verifier_max_new_tokens", type=int, default=512)
    parser.add_argument("--verifier_temperature", type=float, default=0.7)
    parser.add_argument("--verifier_top_k", type=int, default=-1)
    parser.add_argument("--verifier_top_p", type=float, default=0.9)
    parser.add_argument("--verifier_repetition_penalty", type=float, default=1.0,
                        help="vLLM repetition penalty; 1.1 tames the repetition loops of "
                             "small InternVL3.5 models")
    parser.add_argument("--max_response_chars", type=int, default=6000,
                        help="Uniform truncation budget for the embedded solver response")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Cap vLLM context length (needed for huge-context models like "
                             "Qwen3-VL whose full KV cache won't fit on one GPU)")
    parser.add_argument("--disable_chunked_mm", action="store_true",
                        help="See vlm_inference.py: works around vLLM mm-chunking crashes")
    parser.add_argument("--output_dir", type=str, default="vlm/result")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("==============================================================================")
    pprint(vars(args))
    print("==============================================================================")

    with open(f"{args.prompt_dir}/verification_prompt.md", "r") as f:
        verification_prompt = f.read()

    render_chat = load_chat_renderer(args.verifier_model_name)
    verifier_short = args.verifier_model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --------- Build verifier prompts for every record of every solver run ----------
    # Accumulate across all solver files so the verifier model is loaded only once.
    all_rendered: List[str] = []      # chat-templated text, for the vLLM backend
    all_vtexts: List[str] = []        # raw verification text, for the transformers backend
    all_image_paths: List[str] = []
    per_file = []  # list of dicts: {solver_file, run, records, start, end, sample}

    for solver_file in args.solver_run_files:
        with open(solver_file, "r", encoding="utf-8") as f:
            run = json.load(f)
        records = run["records"]
        assert all("solver_correct" in r for r in records), \
            f"{solver_file} has no solver_correct -- pass a *_scores.json file"

        start = len(all_rendered)
        first_text = None
        for rec in records:
            response = truncate_response(rec["solver_full_output"], args.max_response_chars)
            vtext = verification_prompt.format(question=rec["question"], response=response)
            rendered = render_chat(build_verifier_messages(vtext))
            all_rendered.append(rendered)
            all_vtexts.append(vtext)
            all_image_paths.append(rec["image"])
            if first_text is None:
                first_text = (vtext, rendered, rec["image"])
        per_file.append({
            "solver_file": solver_file,
            "run": run,
            "records": records,
            "start": start,
            "end": len(all_rendered),
            "sample": first_text,
        })

    print(f"Built {len(all_rendered)} verifier prompts across {len(per_file)} solver run(s).")

    # ------------------------------- GENERATE ---------------------------------------
    # Reuse the solver-script backends by mapping verifier params onto their `solver_*` API.
    backend_args = SimpleNamespace(
        solver_model_name=args.verifier_model_name,
        solver_temperature=args.verifier_temperature,
        solver_max_new_tokens=args.verifier_max_new_tokens,
        solver_top_k=args.verifier_top_k,
        solver_top_p=args.verifier_top_p,
        solver_repetition_penalty=args.verifier_repetition_penalty,
        solver_n_samples=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_chunked_mm=args.disable_chunked_mm,
        seed=args.seed,
    )
    if args.backend == "vllm":
        outputs = run_vllm_backend(backend_args, all_rendered, all_image_paths)
    else:
        # transformers backend applies the chat template itself, so it needs the RAW
        # verification text (not the already-rendered prompt) plus the image path.
        prompt_records = [{"image": p, "text_prompt": t}
                          for p, t in zip(all_image_paths, all_vtexts)]
        outputs = run_transformers_backend(backend_args, prompt_records)

    assert len(outputs) == len(all_rendered)

    # --------------------------- EVALUATE + WRITE PER PAIR ---------------------------
    for pf in per_file:
        run, records = pf["run"], pf["records"]
        file_outputs = outputs[pf["start"]:pf["end"]]
        file_vtexts = all_vtexts[pf["start"]:pf["end"]]
        file_rendered = all_rendered[pf["start"]:pf["end"]]
        run_md = run.get("metadata", {})
        solver_model = run_md.get("model", {}).get("name", "unknown")
        solver_short = solver_model.split("/")[-1]
        dataset_name = (run_md.get("dataset") or {}).get("name", "dataset")

        tp = tn = fp = fn = bad = 0
        scored_records = []
        for rec, output, vtext, rendered in zip(records, file_outputs, file_vtexts, file_rendered):
            verdict = extract_verdict(output)
            solver_correct = bool(rec["solver_correct"])
            if verdict is None:
                bad += 1
            elif solver_correct and verdict:
                tp += 1
            elif (not solver_correct) and (not verdict):
                tn += 1
            elif (not solver_correct) and verdict:
                fp += 1
            else:
                fn += 1
            # Keep the solver context, then the EXACT prompt fed to the verifier, then its
            # response/verdict. (The solver's own prompt fields are dropped here -- they
            # describe the solver run, not the verification.)
            scored_records.append({
                "id": rec["id"],
                "image": rec["image"],
                "question": rec["question"],
                "answer": rec["answer"],
                "solver_full_output": rec["solver_full_output"],
                "solver_extracted_answer": rec.get("solver_extracted_answer"),
                "solver_correct": rec["solver_correct"],
                "verifier_prompt": vtext,                 # verification text fed to the verifier
                "verifier_rendered_prompt": rendered,     # exact chat-templated string sent
                "verifier_response": output,
                "verifier_verdict": verdict,
            })

        total = len(records)
        correct = tp + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        solver_correct_count = sum(bool(r["solver_correct"]) for r in records)

        sample_vtext, sample_rendered, sample_image = pf["sample"]
        metadata = {
            "timestamp": timestamp,
            "dataset": run.get("metadata", {}).get("dataset"),
            "solver_model": solver_model,
            "verifier_model": {"name": args.verifier_model_name, "backend": args.backend},
            "verification_params": {
                "temperature": args.verifier_temperature,
                "top_p": args.verifier_top_p,
                "top_k": args.verifier_top_k,
                "max_new_tokens": args.verifier_max_new_tokens,
                "repetition_penalty": args.verifier_repetition_penalty,
                "max_response_chars": args.max_response_chars,
                "seed": args.seed,
            },
            "verdict_extractor": VERDICT_EXTRACTOR,
            "prompt": {
                "template_file": f"{args.prompt_dir}/verification_prompt.md",
                "template": verification_prompt,
                "sample_verification_text": sample_vtext,
                "sample_rendered_prompt": sample_rendered,
                "sample_image": sample_image,
            },
            "scored_from": os.path.abspath(pf["solver_file"]),
        }
        metrics = {
            "solver_correct_count": solver_correct_count,
            "solver_accuracy": solver_correct_count / total if total else 0.0,
            "verifier": {
                "total": total,
                "bad_count": bad,
                "correct_count": correct,
                "accuracy": correct / total if total else 0.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        }

        run_id = f"verify_{_slug(dataset_name)}" \
                 f"_solver-{_slug(solver_short)}_verifier-{_slug(verifier_short)}_{timestamp}"
        out_path = os.path.join(args.output_dir, f"{run_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "metrics": metrics, "records": scored_records}, f, indent=4)

        print(f"\n[solver={solver_short}  verifier={verifier_short}] "
              f"verifier_acc={metrics['verifier']['accuracy']:.2f} "
              f"f1={f1:.2f} (tp={tp} tn={tn} fp={fp} fn={fn} bad={bad}) "
              f"| solver_acc={metrics['solver_accuracy']:.2f}")
        print(f"  saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
