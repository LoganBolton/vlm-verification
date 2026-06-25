"""Agentic-vision solver -- the third leg of the comparison.

A Qwen-VL model answers each image-QA problem while it can call a `zoom` tool that
crops and magnifies any region of the image, as many times as a budget allows. This is
*active perception*: instead of one fixed look at the (downscaled) image, the model can
re-inspect fine detail (small objects to count, tiny axis labels to read) on demand.

The three things we compare, all on the same problem set / scorers:
  - pass@N / majority vote (vlm/self_consistency.py)  -- spend extra compute on more samples
  - VLM judge / rejection sampling (vlm/rejection_sampling.py) -- spend it on a verifier
  - agentic vision (this file)                         -- spend it on re-looking at the image

The question: does letting the model zoom capture more of the headroom than majority
vote or a rubber-stamp verifier does?

How the loop works (one persistent vLLM engine, all problems batched per round):
  round 0: render [system+zoom-instructions, user(image, question)] for every problem and
           generate. Generation stops at `</tool_call>` (so a zoom request ends the turn)
           or at EOS (a final boxed answer).
  each round: for problems that emitted a <tool_call>{...zoom...}, crop+magnify the
           requested region of the ORIGINAL image, append it as a new image turn, and
           re-generate only those problems next round. Problems that produced a final
           answer (or no parseable tool call, or hit the crop budget) are done.

Coordinates are fractions of the original image in [0,1] (we also accept the 0-1000
grounding convention and rescale). Crops are upscaled so the model actually sees detail.

Usage:
    python vlm/agentic_vision.py --solver_model_name Qwen/Qwen3-VL-8B-Instruct \
        --data_dir data/countbench --max_crops 4 --solver_max_model_len 32768 \
        --output_dir vlm/result/agentic_vision/countbench/Qwen3-VL-8B-Instruct
"""
from pprint import pprint
import argparse, json, os, re, sys

from PIL import Image

VLM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VLM_DIR)
from vlm_inference import load_image_qa, load_chat_renderer  # noqa: E402
from rejection_sampling import MODEL_SIZES, DATASET_FNS, free_gpu_wait  # noqa: E402
import score_charxiv  # noqa: E402


# --------------------------------------------------------------------------------------
# Tool-call parsing + cropping
# --------------------------------------------------------------------------------------
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*)", re.S)
# Strip whole tool-call blocks AND any dangling/unclosed one (a turn that stopped at
# </tool_call> mid-call) so the answer scorer never reads coordinate numbers as the answer.
_STRIP_TOOL_RE = re.compile(r"<tool_call>.*?</tool_call>|<tool_call>.*\Z", re.S)


def answer_text(turn_text):
    """One assistant turn with its tool-call markup removed."""
    return _STRIP_TOOL_RE.sub("", turn_text).strip()


def conv_answer(conv):
    """Answer-bearing text across a conversation's assistant turns (tool calls removed).

    Stripping per-turn (not on the joined transcript) is essential: a dangling, unclosed
    tool call in one turn must not swallow a real answer that arrives in a later turn.
    """
    parts = [answer_text(m["content"]) for m in conv if m["role"] == "assistant"]
    return "\n".join(p for p in parts if p).strip()


def parse_zoom_box(text):
    """Extract the [x1,y1,x2,y2] box from the last <tool_call> in `text`.

    Generation is stopped at `</tool_call>`, so the JSON object is complete but the
    closing tag is absent. We grab from the last `<tool_call>` to the end and json-load
    the first balanced {...}. Returns a list of 4 floats, or None if nothing parseable.
    """
    m = None
    for m in _TOOL_RE.finditer(text):
        pass
    if m is None:
        return None
    blob = m.group(1)
    # Take the first balanced top-level {...} object.
    depth = 0
    end = None
    for i, ch in enumerate(blob):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        call = json.loads(blob[:end])
    except json.JSONDecodeError:
        return None
    args = call.get("arguments", call) if isinstance(call, dict) else {}
    box = args.get("box") or args.get("bbox") or args.get("region")
    if box is None and all(k in args for k in ("x1", "y1", "x2", "y2")):
        box = [args["x1"], args["y1"], args["x2"], args["y2"]]
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def crop_region(image, box, min_out=512, max_out=1024):
    """Crop `image` to the fractional `box` and upscale so detail is visible.

    `box` is [x1,y1,x2,y2]. Values in [0,1] are treated as fractions; if any value is
    >1 we assume the 0-1000 grounding convention and divide by 1000. The box is clamped
    to the image, given a minimum size, then the crop's long side is scaled into
    [min_out, max_out] px (never downscaling below the native crop). Returns (PIL, px_box).
    """
    W, H = image.size
    vals = list(box)
    if max(abs(v) for v in vals) > 1.0:
        vals = [v / 1000.0 for v in vals]
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # to pixels, clamp
    px1 = max(0, min(W - 1, int(round(x1 * W))))
    py1 = max(0, min(H - 1, int(round(y1 * H))))
    px2 = max(0, min(W, int(round(x2 * W))))
    py2 = max(0, min(H, int(round(y2 * H))))
    # enforce a minimum crop so a degenerate/tiny box still yields something usable
    min_px = max(16, int(0.05 * min(W, H)))
    if px2 - px1 < min_px:
        cx = (px1 + px2) // 2
        px1 = max(0, cx - min_px // 2); px2 = min(W, px1 + min_px)
    if py2 - py1 < min_px:
        cy = (py1 + py2) // 2
        py1 = max(0, cy - min_px // 2); py2 = min(H, py1 + min_px)
    crop = image.crop((px1, py1, px2, py2))
    cw, ch = crop.size
    long_side = max(cw, ch)
    if long_side < min_out:
        scale = min_out / long_side
    elif long_side > max_out:
        scale = max_out / long_side
    else:
        scale = 1.0
    if scale != 1.0:
        crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))),
                           Image.LANCZOS)
    return crop, (px1, py1, px2, py2)


# --------------------------------------------------------------------------------------
# Persistent vLLM engine (kept loaded across all rounds; allows multiple images/prompt)
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self, args, max_images):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import torch
        from vllm import LLM, SamplingParams
        mm_extra = {}
        if args.solver_disable_chunked_mm:
            mm_extra["disable_chunked_mm_input"] = True
        self.LLM = LLM(
            model=args.solver_model_name,
            dtype=torch.bfloat16,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.solver_max_model_len,
            seed=args.seed,
            limit_mm_per_prompt={"image": max_images, "video": 0},
            **mm_extra,
        )
        common = dict(temperature=args.solver_temperature, top_p=args.solver_top_p,
                      top_k=args.solver_top_k, repetition_penalty=args.solver_repetition_penalty,
                      max_tokens=args.solver_max_new_tokens, seed=args.seed)
        # Zoom turns stop the moment a tool call closes, so we can act on it.
        self.sampling = SamplingParams(stop=["</tool_call>"], **common)
        # Forced-answer turns must NOT stop at a tool call -- we want a real \boxed{} answer.
        self.sampling_answer = SamplingParams(**common)

    def generate(self, prompts, images, sampling=None):
        """prompts: list[str]; images: list[list[PIL]] (one list per prompt)."""
        inputs = [{"prompt": p, "multi_modal_data": {"image": imgs}}
                  for p, imgs in zip(prompts, images)]
        gens = self.LLM.generate(inputs, sampling or self.sampling)
        out_text, tokens = [], 0
        for g in gens:
            o = g.outputs[0]
            out_text.append(o.text)
            tokens += len(g.prompt_token_ids or []) + len(o.token_ids)
        return out_text, tokens


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver_model_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--prompt_dir", type=str, default="prompts")
    p.add_argument("--max_crops", type=int, default=4,
                   help="Max zoom tool calls the agent may make per problem")
    p.add_argument("--dataset_subset_ratio", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="Cap #problems (debug)")
    p.add_argument("--solver_temperature", type=float, default=0.7)
    p.add_argument("--solver_top_p", type=float, default=0.9)
    p.add_argument("--solver_top_k", type=int, default=-1)
    p.add_argument("--solver_max_new_tokens", type=int, default=2048)
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

    # Same loading/shuffle as self_consistency & rejection_sampling -> identical problem set.
    dataset = load_image_qa(args.data_dir).shuffle(seed=args.seed)
    import math
    if args.dataset_subset_ratio < 1.0:
        dataset = dataset.select(range(math.ceil(len(dataset) * args.dataset_subset_ratio)))
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    N = len(dataset)
    print(f"Dataset size: {N}  | max_crops={args.max_crops}")

    with open(f"{args.prompt_dir}/agentic_vision_prompt.md") as f:
        prompt_tmpl = f.read()
    render = load_chat_renderer(args.solver_model_name)

    # Per-problem state.
    convs = []      # message lists
    imgs = []       # list[PIL]; index aligns with <image> placeholders in convs[i]
    boxes = []      # list of pixel boxes cropped, per problem
    done = [False] * N
    finished_round = [None] * N
    for ex in dataset:
        img = Image.open(ex["image_path"]).convert("RGB")
        text = prompt_tmpl.format(question=ex["question"], max_crops=args.max_crops)
        convs.append([{"role": "user",
                       "content": [{"type": "image"}, {"type": "text", "text": text}]}])
        imgs.append([img])
        boxes.append([])

    engine = Engine(args, max_images=args.max_crops + 1)

    total_tokens = 0
    # max_crops zoom rounds + 1 final answer round.
    for rnd in range(args.max_crops + 1):
        active = [i for i in range(N) if not done[i]]
        if not active:
            break
        last_round = (rnd == args.max_crops)
        print(f"\n===== round {rnd}: {len(active)} active "
              f"({'final, no more zoom' if last_round else 'may zoom'}) =====")
        prompts = [render(convs[i]) for i in active]
        batch_imgs = [imgs[i] for i in active]
        # On the final round there is no more zooming, so generate a real answer (no tool
        # stop) instead of letting the turn end at another <tool_call>.
        outs, toks = engine.generate(prompts, batch_imgs,
                                     sampling=engine.sampling_answer if last_round else None)
        total_tokens += toks

        for i, text in zip(active, outs):
            box = None if last_round else parse_zoom_box(text)
            if box is not None and len(boxes[i]) < args.max_crops:
                # Record assistant zoom turn (re-add the stripped stop tag), then crop.
                convs[i].append({"role": "assistant", "content": text + "</tool_call>"})
                ex = dataset[i]
                crop, px = crop_region(imgs[i][0], box)
                imgs[i].append(crop)
                boxes[i].append({"requested": box, "pixel_box": px})
                left = args.max_crops - len(boxes[i])
                tail = (f"You have {left} zoom(s) left. Zoom again into a different region, "
                        f"or give your final answer in \\boxed{{}}." if left > 0 else
                        "You have NO zooms left. Do NOT call the tool again; give your final "
                        "answer now in \\boxed{}.")
                obs = (f"Tool result: here is the zoomed crop of region {box} "
                       f"(pixel box {px} of the original {imgs[i][0].size} image). " + tail)
                convs[i].append({"role": "user",
                                 "content": [{"type": "image"}, {"type": "text", "text": obs}]})
            else:
                # Final answer (or unparseable tool call / budget hit): close out.
                convs[i].append({"role": "assistant", "content": text})
                done[i] = True
                finished_round[i] = rnd

    # --------------------------- FORCED-ANSWER PASS ---------------------------
    # A budget-exhausted model can finish having only emitted tool calls (no \boxed{}).
    # Give each such problem ONE tool-free turn to commit to an answer using what it has
    # already seen; if it still doesn't answer, it is honestly scored wrong below.
    need = [i for i in range(N) if fns["extract"](conv_answer(convs[i])) is None]
    if need:
        print(f"\n===== forced-answer pass: {len(need)} problems never gave an answer =====")
        force_msg = ("You must answer the question NOW using only what you have already seen. "
                     "Do NOT call the zoom tool again. Give a brief justification, then your "
                     "final answer as \\boxed{...}.")
        for i in need:
            convs[i].append({"role": "user", "content": [{"type": "text", "text": force_msg}]})
        prompts = [render(convs[i]) for i in need]
        outs, toks = engine.generate(prompts, [imgs[i] for i in need],
                                     sampling=engine.sampling_answer)
        total_tokens += toks
        for i, text in zip(need, outs):
            convs[i].append({"role": "assistant", "content": text})

    # ----------------------------- SCORE + SAVE -----------------------------
    records = []
    n_correct = 0
    for i, ex in enumerate(dataset):
        transcript = "\n".join(m["content"] for m in convs[i] if m["role"] == "assistant")
        ans = conv_answer(convs[i])   # tool calls stripped per-turn -> no coordinate false matches
        extracted = fns["extract"](ans)
        correct = bool(fns["correct"](ex["answer"], ans)) if extracted is not None else False
        n_correct += correct
        # Structured rollout for the viewer: each turn is one message. An image-bearing user
        # turn carries either the original image (crop=None) or the k-th zoom crop (crop=k);
        # a text-only user turn is the forced-answer prompt. Assistant turns carry the model
        # text (reasoning + any tool call).
        turns, crop_i, seen_img = [], 0, False
        for m in convs[i]:
            if m["role"] == "user":
                has_img = any(c.get("type") == "image" for c in m["content"])
                txt = next((c["text"] for c in m["content"] if c.get("type") == "text"), "")
                if has_img and not seen_img:
                    seen_img = True
                    turns.append({"role": "user", "text": txt, "crop": None})
                elif has_img:
                    turns.append({"role": "user", "text": txt, "crop": crop_i}); crop_i += 1
                else:
                    turns.append({"role": "user", "text": txt, "crop": None})
            else:
                turns.append({"role": "assistant", "text": m["content"]})
        records.append({
            "id": ex["id"], "image": ex["image_path"],
            "question": ex["question"], "answer": ex["answer"],
            "solver_full_output": transcript,
            "solver_answer_text": ans,
            "solver_extracted_answer": extracted,
            "solver_correct": correct,
            "n_crops": len(boxes[i]),
            "crops": boxes[i],
            "finished_round": finished_round[i],
            "turns": turns,
        })

    n_crops_list = [len(b) for b in boxes]
    metrics = {
        "metadata": {
            "solver_model": args.solver_model_name, "dataset": dataset_name,
            "n_problems": N, "max_crops": args.max_crops,
            "temperature": args.solver_temperature, "seed": args.seed,
            "solver_gflops": total_tokens * 2 * MODEL_SIZES[args.solver_model_name],
            "scorer": score_charxiv.EXTRACTOR_NAME if dataset_name == "charxiv" else "count_exact",
        },
        "accuracy": n_correct / N if N else 0.0,
        "avg_crops": sum(n_crops_list) / N if N else 0.0,
        "frac_used_zoom": sum(c > 0 for c in n_crops_list) / N if N else 0.0,
        "crop_count_hist": {str(k): n_crops_list.count(k) for k in range(args.max_crops + 1)},
    }
    pprint({"accuracy": round(metrics["accuracy"], 4),
            "avg_crops": round(metrics["avg_crops"], 3),
            "frac_used_zoom": round(metrics["frac_used_zoom"], 3),
            "crop_hist": metrics["crop_count_hist"]})

    json.dump(metrics, open(os.path.join(args.output_dir, "metrics.json"), "w"), indent=2)
    json.dump({"metadata": metrics["metadata"], "records": records},
              open(os.path.join(args.output_dir, "records.json"), "w"), indent=2)
    print(f"\nwrote metrics + records to {args.output_dir}")


if __name__ == "__main__":
    main()
