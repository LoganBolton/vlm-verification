#!/usr/bin/env bash
# Scale-up sibling of run_full_pipeline.sh: same solve -> score -> verify grid, but for a
# named TIER of bigger models from the same families. Results land in tier-suffixed dirs
# (vlm/result/<dataset>_<tier>) so each tier stays a self-contained NxN grid and the
# original full-run dirs are untouched.
#
#   tier2:  Qwen3-VL-4B-Instruct, gemma-4-E4B-it, llava-1.5-13b-hf
#   tier3:  Qwen3-VL-8B-Instruct, gemma-4-12B-it   (LLaVA-1.5 has no size above 13b;
#           the 30B+ models in each family don't fit unquantized in 2x24GB)
#   intern: the whole InternVL3.5 dense ladder, 1B-14B, as a self-contained 5x5 family
#           study. Uses the ORIGINAL (non-HF) checkpoints: vLLM's interns1.py path for
#           the -HF conversions emits corrupted logits (mid-word gibberish on ~60% of
#           CountBench), while the internvl.py path for these is correct. Prompt
#           rendering falls back to ChatML + <image> (no AutoProcessor on these repos).
#           30B-A3B/38B don't fit unquantized.
#
# RESUMABLE: solve steps are skipped when a scored run already exists, verify steps when
# every (solver, verifier) output file already exists. A crashed vLLM engine can leave
# zombie workers holding all GPU memory, which made every later step fail instantly --
# so each GPU step is preceded by a sweep that kills leftovers and waits for free VRAM.
#
# Usage:  bash vlm/run_scale_pipeline.sh tier2
set -u

TIER="${1:?usage: run_scale_pipeline.sh tier2|tier3|intern}"
case "$TIER" in
  tier2) MODELS=("Qwen/Qwen3-VL-4B-Instruct" "google/gemma-4-E4B-it" "llava-hf/llava-1.5-13b-hf") ;;
  tier3) MODELS=("Qwen/Qwen3-VL-8B-Instruct" "google/gemma-4-12B-it") ;;
  intern) MODELS=("OpenGVLab/InternVL3_5-1B" "OpenGVLab/InternVL3_5-2B"
                  "OpenGVLab/InternVL3_5-4B" "OpenGVLab/InternVL3_5-8B"
                  "OpenGVLab/InternVL3_5-14B") ;;
  *) echo "unknown tier: $TIER"; exit 1 ;;
esac

PY=.venv-vllm/bin/python
BACKEND=vllm
# FlashInfer's sampling kernel wants to JIT with nvcc, which this box doesn't have;
# fall back to the torch sampler (same sampling semantics).
export VLLM_USE_FLASHINFER_SAMPLER=0
SUBSET="${SUBSET:-}"
SUBSET_ARG=""
[[ -n "$SUBSET" ]] && SUBSET_ARG="--dataset_subset_ratio $SUBSET"
LOGDIR=vlm/result/_run_logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_${TIER}.txt"

# Big declared contexts must be capped to fit KV cache in 2x24GB (short prompts anyway).
mml_for() {
  case "$1" in
    Qwen/Qwen3-VL-*) echo "--max_model_len 32768" ;;
    google/gemma-4-12B-it|google/gemma-4-E4B-it) echo "--max_model_len 32768" ;;
    OpenGVLab/InternVL3_5-*) echo "--max_model_len 32768" ;;
    *) echo "" ;;
  esac
}

# CharXiv solver responses are long; LLaVA's hard 4096-token context forces a tighter
# response cap there. Kept uniform across verifiers within a dataset (matches tier-1 runs).
mrc_for() {
  case "$1" in
    charxiv) echo "--max_response_chars 2500" ;;
    *) echo "" ;;
  esac
}

# InternVL3.5 quirks, all sizes: repetition_penalty 1.1 (OpenGVLab-recommended; the small
# sizes loop without it) and --disable_chunked_mm (vLLM 0.22 'Encoder cache miss'
# workaround; also declares image-only inputs so the per-step budget stays at 8192 --
# the model's max VIDEO item demanded a 29k budget, which OOM'd the 8B at solve time).
INTERN_COMMON="--disable_chunked_mm"
rp_solve_for()  { case "$1" in OpenGVLab/InternVL3_5-*) echo "--solver_repetition_penalty 1.1 $INTERN_COMMON" ;; *) echo "" ;; esac; }
rp_verify_for() { case "$1" in OpenGVLab/InternVL3_5-*) echo "--verifier_repetition_penalty 1.1 $INTERN_COMMON" ;; *) echo "" ;; esac; }

short_for() { echo "${1##*/}"; }
# Mirror vlm_inference._slug: result filenames collapse any non [A-Za-z0-9.-] run to '-'
# (e.g. InternVL3_5-1B-HF -> InternVL3-5-1B-HF), so file globs must use the slugged name.
slug_for() { echo "$1" | sed -E 's/[^A-Za-z0-9.-]+/-/g; s/^-+//; s/-+$//'; }
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_status() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

# Kill any leftover vLLM processes and wait until both GPUs are actually free.
gpu_cleanup() {
  pkill -f "vlm_inference.py|vlm_verify.py" 2>/dev/null
  pkill -f "VLLM::EngineCore|VLLM::Worker" 2>/dev/null
  local used
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-0}" -lt 1500 ]] && return 0
    sleep 2
  done
  log_status "WARN   GPU memory still in use after cleanup sweep: ${used}MiB"
}

run_step() {  # name logfile cmd...
  local name="$1"; local logf="$2"; shift 2
  log_status "START  $name"
  if "$@" >"$logf" 2>&1; then
    log_status "OK     $name"
    return 0
  else
    log_status "FAIL   $name (exit $?) -- see $logf"
    return 1
  fi
}

run_gpu_step() {  # like run_step, but sweeps stale GPU state first (and after a failure)
  gpu_cleanup
  if ! run_step "$@"; then
    gpu_cleanup
    return 1
  fi
}

newest_solver_file() {  # dir dataset modelshort
  ls -t "$1/$2_$3"_*.json 2>/dev/null | grep -v '_scores\.json' | grep -v '_prompts\.json' | head -1
}

log_status "================ ${TIER} PIPELINE START ================"
log_status "models: ${MODELS[*]}"

DATASETS=("countbench:data/countbench:vlm/score_results.py"
          "charxiv:data/charxiv:vlm/score_charxiv.py")

for entry in "${DATASETS[@]}"; do
  IFS=":" read -r DS DATADIR SCORER <<<"$entry"
  RDIR="vlm/result/${DS}_${TIER}"
  mkdir -p "$RDIR"
  log_status "########## DATASET: $DS (data=$DATADIR -> $RDIR) ##########"

  SCORED_FILES=()

  for M in "${MODELS[@]}"; do
    MS=$(short_for "$M")
    MSLUG=$(slug_for "$MS")
    MML=$(mml_for "$M")

    SF=$(newest_solver_file "$RDIR" "$DS" "$MSLUG")
    if [[ -z "$SF" ]]; then
      run_gpu_step "solve  $DS/$MS" "$LOGDIR/solve_${DS}_${TIER}_${MSLUG}.log" \
        $PY vlm/vlm_inference.py \
          --solver_model_name "$M" --data_dir "$DATADIR" \
          --backend "$BACKEND" $MML $(rp_solve_for "$M") $SUBSET_ARG --output_dir "$RDIR"
      SF=$(newest_solver_file "$RDIR" "$DS" "$MSLUG")
    else
      log_status "SKIP   solve $DS/$MS -- already done ($SF)"
    fi
    if [[ -z "$SF" ]]; then
      log_status "SKIP   score $DS/$MS -- no solver output found"
      continue
    fi

    SCORES="${SF%.json}_scores.json"
    if [[ -f "$SCORES" ]]; then
      log_status "SKIP   score $DS/$MS -- already done"
      SCORED_FILES+=("$SCORES")
      continue
    fi
    if run_step "score  $DS/$MS" "$LOGDIR/score_${DS}_${TIER}_${MSLUG}.log" \
        $PY "$SCORER" "$SF"; then
      [[ -f "$SCORES" ]] && SCORED_FILES+=("$SCORES")
    fi
  done

  log_status "scored solver runs for $DS: ${#SCORED_FILES[@]} -> ${SCORED_FILES[*]:-none}"
  if [[ ${#SCORED_FILES[@]} -eq 0 ]]; then
    log_status "SKIP   verification for $DS -- no scored solver runs"
    continue
  fi

  MRC=$(mrc_for "$DS")
  for V in "${MODELS[@]}"; do
    VS=$(short_for "$V")
    VSLUG=$(slug_for "$VS")
    VMML=$(mml_for "$V")

    NVF=$(ls "$RDIR"/verify_*_verifier-"$VSLUG"_*.json 2>/dev/null | wc -l)
    if [[ "$NVF" -ge ${#SCORED_FILES[@]} ]]; then
      log_status "SKIP   verify $DS/verifier=$VS -- $NVF verify files already present"
      continue
    fi

    run_gpu_step "verify $DS/verifier=$VS" "$LOGDIR/verify_${DS}_${TIER}_${VSLUG}.log" \
      $PY vlm/vlm_verify.py \
        --verifier_model_name "$V" --backend "$BACKEND" $VMML $MRC $(rp_verify_for "$V") \
        --output_dir "$RDIR" \
        --solver_run_files "${SCORED_FILES[@]}"
  done
done

log_status "================ ${TIER} PIPELINE DONE ================"
