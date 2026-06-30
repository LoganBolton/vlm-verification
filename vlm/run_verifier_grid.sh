#!/usr/bin/env bash
# Solver x Verifier grid for the "When Does Verification Pay Off?" replication (VLM version).
# CharXiv only for now. Each VERIFIER model judges every SOLVER's existing base scored run
# (no new solver generation needed), producing one confusion matrix per (solver, verifier)
# pair via vlm/vlm_verify.py. Verifier gain is computed afterwards from these matrices.
#
# The grid is square: every model in GRID_MODELS is used as both a solver (its base run is
# judged) and a verifier (it judges everyone). self = (m,m); intra-family = same family,
# different size; cross-family = different family.
#
# One verifier is loaded once and judges ALL solver runs (cheap). Resumable at verifier
# granularity (skips a verifier whose pair files already all exist).
# Usage:  bash vlm/run_verifier_grid.sh        (override set via GRID_MODELS="...")
set -u
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
DS=charxiv
OUTDIR=vlm/result/verifier_grid/$DS; mkdir -p "$OUTDIR"
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_verifier_grid.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

# Default square grid: 2 sizes x {Qwen-VL, InternVL3.5, gemma-4} + llava weak baseline.
GRID_MODELS="${GRID_MODELS:-\
Qwen/Qwen3-VL-2B-Instruct \
Qwen/Qwen3-VL-8B-Instruct \
OpenGVLab/InternVL3_5-2B \
OpenGVLab/InternVL3_5-8B \
google/gemma-4-E4B-it \
google/gemma-4-12B-it \
llava-hf/llava-1.5-7b-hf}"

short() { local s="${1##*/}"; echo "${s//_/-}"; }     # HF id -> base-run short name (_ -> -)

base_file() {  # echo the charxiv base scored run for a model short-name (first match)
  ls vlm/result/${DS}*/${DS}_"$1"_*_scores.json 2>/dev/null | grep -v '/verify_' | head -1
}

extra_for() {  # per-family vLLM quirk flags for the VERIFIER load
  case "$1" in
    OpenGVLab/InternVL3_5-*) echo "--max_model_len 32768 --verifier_repetition_penalty 1.1 --disable_chunked_mm" ;;
    google/gemma-4-*|Qwen/Qwen3-VL-*) echo "--max_model_len 32768" ;;
    # LLaVA-1.5 hard-caps at 4096 ctx; the CharXiv verifier prompt overflows unless the solver
    # response is trimmed, so cap both model_len and response chars (per the known gotcha).
    llava-hf/llava-1.5-*) echo "--max_model_len 4096 --max_response_chars 2500" ;;
    *) echo "--max_model_len 32768" ;;
  esac
}

# Resolve every solver's base scored file once (keep the short name alongside, for resume).
SOLVER_FILES=(); SOLVER_SHORTS=(); MISSING=()
for M in $GRID_MODELS; do
  SS="$(short "$M")"
  f=$(base_file "$SS")
  if [[ -n "$f" ]]; then SOLVER_FILES+=("$f"); SOLVER_SHORTS+=("$SS"); else MISSING+=("$M"); fi
done
NSOLVERS=${#SOLVER_FILES[@]}

# Solver files per vlm_verify call. ALL 7 at once (7000 charxiv image prompts) OOM-kills the
# vLLM engine on 61GB RAM; 2 files (~2000 prompts) is the proven-safe batch. Override via CHUNK=.
CHUNK="${CHUNK:-2}"
log "grid=$NSOLVERS solvers x $(echo $GRID_MODELS | wc -w) verifiers on $DS"
[[ ${#MISSING[@]} -gt 0 ]] && log "WARN no base run for: ${MISSING[*]} (excluded as solver)"

LIMIT="${LIMIT:-}"; LIMARG=""; [[ -n "$LIMIT" ]] && LIMARG="--limit $LIMIT"

wait_gpu() {
  for _ in $(seq 1 360); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0; sleep 10
  done; log "WARN wait_gpu timeout (used=${used:-?})"; return 1
}

NDONE=0; NFAIL=0; NSKIP=0
for V in $GRID_MODELS; do
  VS=$(short "$V")
  # Only judge solvers whose pair file doesn't already exist (resumable per pair).
  TODO_FILES=(); TODO_SHORTS=()
  for i in "${!SOLVER_FILES[@]}"; do
    SS="${SOLVER_SHORTS[$i]}"
    if ls "$OUTDIR"/verify_${DS}_solver-"${SS}"_verifier-"${VS}"_*.json >/dev/null 2>&1; then continue; fi
    TODO_FILES+=("${SOLVER_FILES[$i]}"); TODO_SHORTS+=("$SS")
  done
  NTODO=${#TODO_FILES[@]}
  if [[ "$NTODO" -eq 0 ]]; then log "SKIP verifier=$VS (all $NSOLVERS pairs exist)"; NSKIP=$((NSKIP+1)); continue; fi
  LOGF="$LOGDIR/vg_${DS}_verifier-${VS}.log"
  log "verifier=$VS: $NTODO/$NSOLVERS solvers to judge, CHUNK=$CHUNK -> ${TODO_SHORTS[*]}"
  ci=0
  while [[ "$ci" -lt "$NTODO" ]]; do
    CHUNK_FILES=("${TODO_FILES[@]:$ci:$CHUNK}")
    wait_gpu || true
    log "START verifier=$VS chunk@$ci (${#CHUNK_FILES[@]} solvers)"
    if $PY vlm/vlm_verify.py --verifier_model_name "$V" $(extra_for "$V") $LIMARG \
          --solver_run_files "${CHUNK_FILES[@]}" --output_dir "$OUTDIR" >>"$LOGF" 2>&1; then
      log "OK    verifier=$VS chunk@$ci"; NDONE=$((NDONE+1))
    else
      log "FAIL  verifier=$VS chunk@$ci (exit $?) -- see $LOGF"; NFAIL=$((NFAIL+1))
      pkill -9 -f "vlm_verify.py" 2>/dev/null
      pkill -9 -f "EngineCore|multiproc_executor" 2>/dev/null
      sleep 15
    fi
    ci=$((ci+CHUNK))
  done
done
log "===== VERIFIER GRID DONE: $NDONE ok, $NFAIL failed, $NSKIP skipped ====="
