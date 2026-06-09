# VLM Solver–Verifier Experiments on CountBenchQA

**Date:** 2026-06-08
**Task:** [CountBenchQA](https://huggingface.co/datasets/vikhyatk/CountBenchQA) — given an image and a question
("How many X are in the image?"), answer with the count. 100 examples (first 100 of the `test` split).
**Models (used as both solver and verifier):**

| Short name | HF id |
|---|---|
| Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` |
| Gemma-4-E2B | `google/gemma-4-E2B-it` |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |

This mirrors the LLM solver–verifier framework in `src/inference.py`, but multimodal: **both the solver and
the verifier see the image.** Every model verified every model (3×3 = 9 combinations), including itself.

**Charts:** see [`RESULTS_VISUAL.md`](RESULTS_VISUAL.md) for all figures. **Attention items:** [§6](#6-things-that-need-attention-caveats--follow-ups).

---

## 1. Method

**Solver** (`vlm/vlm_inference.py`): image + `prompts/inference_prompt.md` ("reason step by step, put the final
answer in `\boxed{}`") → model generates a response. Raw generations are saved with no correctness judgment.

**Scoring** (`vlm/score_results.py`): a count-aware extractor reads the answer from `\boxed{}` *or* prose
(digits **and** number-words, since LLaVA answers "There are six chairs" and never boxes), compares to ground
truth, and writes `solver_correct` per example.

**Verifier** (`vlm/vlm_verify.py`): the verifier is shown **the same image**, the question, and the solver's full
response via `prompts/verification_prompt.md`, and must output `\boxed{correct}` or `\boxed{incorrect}`. Its verdict
is parsed with a prose-aware extractor (`\boxed{}` **or** plain-text "correct"/"incorrect") and compared against
`solver_correct`. Long solver responses are truncated head+tail to a uniform 6000-char budget so short-context
verifiers (LLaVA, 4096 tokens) can read every solver's output on equal terms. Each output record stores the **exact
prompt fed to the verifier** (`verifier_prompt` + the chat-templated `verifier_rendered_prompt`) alongside its
`verifier_response` and `verifier_verdict`.

**Convention** (same as the repo's LLM verifier): the *positive* class is "solver was correct."
- **TP** = verifier said correct, solver was correct
- **FP** = verifier said correct, but solver was **wrong** ← *failed to catch an error*
- **TN** = verifier said incorrect, solver was wrong ← *caught the error*
- **FN** = verifier said incorrect, but solver was correct ← *too harsh*
- **bad** = no parseable verdict (counts against accuracy)

All verifier runs used vLLM (nightly cu129 build) with `temperature=0.7`, `max_new_tokens=512`.

---

## 2. Solver results (baseline)

| Solver | Accuracy | Trivial baseline* |
|---|---|---|
| Qwen3-VL-2B | **0.81** | 0.81 |
| Gemma-4-E2B | **0.61** | 0.61 |
| LLaVA-1.5-7B | **0.48** | 0.52 |

\*Trivial baseline = `max(acc, 1-acc)` — the accuracy of a verifier that blindly says "correct" (or "incorrect")
for every example. A verifier is only useful if it **beats** this.

LLaVA is a weak counter and ignores the format instruction (answers in prose); Qwen3-VL is the strongest solver.

---

## 3. Verifier accuracy grid

Rows = solver being judged; columns = verifier. **Bold** beats that solver's trivial baseline.

| solver ↓ \ verifier → | Qwen3-VL-2B | Gemma-4-E2B | LLaVA-1.5-7B | trivial |
|---|---|---|---|---|
| Qwen3-VL-2B  | **0.85** | **0.83** | 0.72 | 0.81 |
| Gemma-4-E2B  | **0.66** | **0.62** | 0.59 | 0.61 |
| LLaVA-1.5-7B | **0.89** | **0.70** | 0.42 | 0.52 |

> Verdicts are parsed with a prose-aware extractor (`\boxed{}` **or** plain text). The original strict boxed-only
> extractor discarded many parseable LLaVA verdicts as "bad" — see [§6](#6-things-that-need-attention-caveats--follow-ups);
> the numbers above are post-fix.

### Verifier averages (across the 3 solvers)

| Verifier | mean acc | mean F1 | specificity TN/(TN+FP)** | unparseable verdicts |
|---|---|---|---|---|
| **Qwen3-VL-2B** | **0.80** | 0.86 | **0.47** | 0 / 300 |
| Gemma-4-E2B | 0.72 | 0.81 | 0.29 | 5 / 300 |
| LLaVA-1.5-7B | 0.58 | 0.70 | 0.22 | 10 / 300 |

\*\*Specificity = of the solver answers that were truly **wrong**, the fraction the verifier actually caught. This is
the key "does the verifier add value" metric, and it is low for everyone except Qwen3-VL.

---

## 4. Per-cell detail

| Verifier | Judging | acc | F1 | precision | recall | TP | TN | FP | FN | bad |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | Qwen3-VL-2B | 0.85 | 0.92 | 0.84 | 1.00 | 81 | 4 | 15 | 0 | 0 |
| Qwen3-VL-2B | Gemma-4-E2B | 0.66 | 0.78 | 0.64 | 1.00 | 61 | 5 | 34 | 0 | 0 |
| Qwen3-VL-2B | LLaVA-1.5-7B | 0.89 | 0.89 | 0.84 | 0.96 | 46 | 43 | 9 | 2 | 0 |
| Gemma-4-E2B | Qwen3-VL-2B | 0.83 | 0.91 | 0.85 | 0.99 | 80 | 3 | 14 | 1 | 2 |
| Gemma-4-E2B | Gemma-4-E2B | 0.62 | 0.76 | 0.62 | 1.00 | 61 | 1 | 38 | 0 | 0 |
| Gemma-4-E2B | LLaVA-1.5-7B | 0.70 | 0.77 | 0.66 | 0.92 | 44 | 26 | 23 | 4 | 3 |
| LLaVA-1.5-7B | Qwen3-VL-2B | 0.72 | 0.82 | 0.87 | 0.77 | 62 | 10 | 9 | 19 | 0 |
| LLaVA-1.5-7B | Gemma-4-E2B | 0.59 | 0.74 | 0.61 | 0.93 | 57 | 2 | 37 | 4 | 0 |
| LLaVA-1.5-7B | LLaVA-1.5-7B | 0.42 | 0.56 | 0.48 | 0.69 | 31 | 11 | 34 | 14 | 10 |

---

## 5. Key findings

1. **Qwen3-VL-2B is the strongest verifier and the only one that adds real discriminative value.** It beats the
   trivial baseline on all three solvers (0.85 vs 0.81, 0.66 vs 0.61, 0.89 vs 0.52) and is the only model with
   non-trivial specificity (**0.47**) — it actually re-examines the image and catches wrong counts rather than
   rubber-stamping. It also never produced an unparseable verdict (0/300).

2. **The standout cell is Qwen3-VL verifying LLaVA (0.89, TN=43).** Because it independently re-counts from the
   image, it rejects the bulk of LLaVA's wrong answers — verification adds the most value precisely when the solver
   is weak. This is the solver–verifier "gain" effect from the paper, reproduced in the multimodal setting.

3. **Gemma-4-E2B beats baseline everywhere too (mean 0.72), but largely by being lenient.** Its recall is ≈1.0 and
   its specificity is only 0.29 (FP of 14/38/23): it says "correct" by default and rarely flags an error. On the
   strong solvers its accuracy therefore mostly *tracks* the solver's own accuracy rather than reflecting genuine
   error-catching — useful, but for a different reason than Qwen3-VL.

4. **LLaVA-1.5 is the weakest verifier (mean 0.58) and the only one that beats baseline on *no* solver, but it is
   not broken** — that earlier impression was a parsing artifact (see finding 6). It is *erratic* rather than
   simply lenient: harsh on the strong Qwen3-VL solver (FN=19 — it rejects correct answers) yet over-accepting on
   weaker solvers (FP=37 on Gemma). Its 4096-token context is also tight when reading long reasoning.

5. **A model verifying itself is not special.** Self-verification accuracies (diagonal) are 0.85 / 0.62 / 0.42 —
   they simply track model strength. A weak model is a weak verifier of anyone, including itself.

6. **Parsing matters as much as the model.** Scoring verdicts with a strict `\boxed{}`-only extractor made LLaVA
   look catastrophic (mean 0.38, 90/300 "bad") because it states verdicts in prose. A prose-aware extractor recovers
   almost all of those (10/300 bad) and lifts LLaVA-as-verifier to 0.58. The headline ranking
   (Qwen3-VL > Gemma-4-E2B > LLaVA-1.5) is unchanged, but the gap is far smaller than it first appeared — a reminder
   to separate *judgment quality* from *format compliance*.

---

## 6. Things that need attention (caveats & follow-ups)

Ordered roughly by how much each could change the conclusions.

### Statistical validity
- **Sampling, not greedy.** Both solver and verifier ran at `temperature=0.7`, so runs are non-deterministic.
  Re-run with `temperature=0` for stable, comparable numbers before drawing firm conclusions. *Action:* add
  `--solver_temperature 0` / `--verifier_temperature 0`.
- **n=100, single seed.** Treat gaps < ~5 points as noise. *Action:* scale to the full 491-example split and ≥3
  seeds; report confidence intervals.
- **Accuracy rewards "yes-men."** On a solver with 0.81 accuracy, a verifier that says "correct" every time scores
  0.81. Raw accuracy therefore flatters lenient verifiers. *Action:* lead with **balanced accuracy / MCC /
  specificity**, not raw accuracy. (The verdict-breakdown and leniency charts already expose this.)

### Mixed backends — the one real comparability gap
- **The three solver runs did not all use the same backend.** Qwen3-VL was solved via **transformers (cu126)**
  while Gemma-4 and LLaVA were solved via **vLLM (cu129)** (Qwen3-VL was generated before the vLLM env existed).
  The *verifier* runs were all vLLM, so the 3×3 grid is internally consistent, but the solver baselines aren't
  strictly apples-to-apples. *Action:* re-run the Qwen3-VL **solver** under vLLM (`--max_model_len 16384`) so all
  solver outputs share one backend, then re-score + re-verify.

### Scoring / extraction
- **Verdict extraction was a bug, now fixed.** The original verifier scoring used the repo's boxed-only
  `extract_verifier_answer`, which returns `None` whenever there is no `\boxed{}` — so LLaVA's prose verdicts
  ("The student's answer is correct.") were all dropped as `bad` (90/300). Now parsed with `extract_verdict`
  (boxed → else prose; "incorrect"/"wrong" beat "correct"), recovering all but 10/300. Existing files were
  re-parsed in place by `vlm/rescore_verify.py` (deterministic — generations untouched).
- **Remaining 10 `bad` are genuine** — LLaVA self-verification responses that ramble without a clear verdict. `bad`
  still counts as wrong (an unusable verdict *is* a failure); a "graded-only" accuracy could separate judgment
  quality from format compliance.
- **Solver-side last-number heuristic.** `extract_count` prefers `\boxed{}` then falls back to the *last* number in
  the text. A response that restates the question or shows intermediate counts can have the wrong "last number."
  Low risk on these terse outputs, but real. *Action:* prefer an explicit "Final Answer:" pattern before the fallback.
- **Number-words only cover 0–20**, and compound words ("twenty-one") parse to `1`. CountBench answers are small so
  this didn't bite, but it will on datasets with larger counts.

### Solver/verifier fairness
- **LLaVA-1.5 ignores the "reason then box" instruction** in both roles; its scores partly reflect prompt-format
  sensitivity, not just capability. A model-specific or simpler prompt would be fairer.
- **Response truncation (6000 chars, head+tail)** for the verifier could, in rare cases, drop a final answer buried
  mid-text. Only a handful of very long Qwen3-VL outputs were affected; head+tail keeps the ending.

### Infrastructure / reproducibility
- **vLLM is a pinned nightly `rc/dev` wheel** (`0.22.1rc1.dev258+...cu129`) installed by URL — not a stable release.
  It can vanish from the index or behave differently than a release. *Action:* record the exact wheel hash (done in
  the env) and ideally vendor it or move to a stable cu129 release when one ships.
- **flashinfer sampler is force-disabled** (`VLLM_USE_FLASHINFER_SAMPLER=0`) because it JIT-compiles with `nvcc`,
  which isn't installed. Fine, but anyone re-enabling flashinfer (e.g., for speed) needs a CUDA toolkit.
- **Qwen3-VL needs `--max_model_len`** on a 24 GB GPU (its 262k context KV cache won't fit). Documented in the
  verifier; the solver path now also accepts it.
- **Driver caps at CUDA 12.6.** That's why we run cu129 (not the cu130 stable wheel). A driver upgrade (root +
  reboot) would unlock stable vLLM releases.
- **Two environments** (`.venv-vlm` = transformers/cu126, `.venv-vllm` = vLLM/cu129) — easy to run something in the
  wrong one. Consolidating onto the vLLM env for everything would simplify.

---

## 7. Artifacts (all in `vlm/result/`)

**Solver runs** (`<dataset>_<model>_<time>.json` = metadata + raw generations; `_scores.json` adds the extracted
answer + `solver_correct`):
- `countbench_Qwen3-VL-2B-Instruct_20260607-235008.json` (+ `_scores.json`)
- `countbench_gemma-4-E2B-it_20260608-001225.json` (+ `_scores.json`)
- `countbench_llava-1.5-7b-hf_20260608-001222.json` (+ `_scores.json`)

**Verifier runs** (`verify_countbench_solver-<S>_verifier-<V>_<time>.json` = metadata + metrics + per-example
records with the **exact verifier prompt** (`verifier_prompt`, `verifier_rendered_prompt`), `verifier_response`,
and `verifier_verdict`): 9 files, one per (solver, verifier) pair.

**Figures + visual writeup:** `vlm/figures/*.png`, embedded in [`vlm/RESULTS_VISUAL.md`](RESULTS_VISUAL.md).

## 8. Reproduce

```bash
source .venv-vllm/bin/activate
SCORED="vlm/result/countbench_Qwen3-VL-2B-Instruct_*_scores.json \
        vlm/result/countbench_gemma-4-E2B-it_*_scores.json \
        vlm/result/countbench_llava-1.5-7b-hf_*_scores.json"

# one verifier vs all solvers (loads the verifier once); add --max_model_len for Qwen3-VL
python vlm/vlm_verify.py --verifier_model_name Qwen/Qwen3-VL-2B-Instruct --solver_run_files $SCORED --max_model_len 16384
python vlm/vlm_verify.py --verifier_model_name google/gemma-4-E2B-it     --solver_run_files $SCORED
python vlm/vlm_verify.py --verifier_model_name llava-hf/llava-1.5-7b-hf  --solver_run_files $SCORED

# re-parse verdicts / prompts on existing files without re-running the GPUs:
python vlm/rescore_verify.py "vlm/result/verify_*.json"
# regenerate charts (CPU):
.venv/bin/python vlm/plot_results.py
```
