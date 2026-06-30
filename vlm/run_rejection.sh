#!/usr/bin/env bash
# Rejection-sampling queue: runs vlm/rejection_sampling.py for a list of
# (dataset, solver, verifier|oracle) combos, sequentially. Waits for any running
# scale-pipeline to finish first, so it can be queued while the GPUs are busy.
#
# Pair selection rationale (from the static grids):
#   - Qwen3-VL-8B + gemma-4-12B   : best cross-model verifier cell (countbench 0.90)
#   - gemma-4-12B self            : best solver judging itself (self-leniency case)
#   - InternVL 2B/4B + 14B        : small-solver + big-verifier compute-efficiency story
#   - oracle rows                 : upper bound on what verification could ever buy
#
# Usage:  bash vlm/run_rejection.sh
set -u
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
LOGDIR=vlm/result/_run_logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_rejection.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_status() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

# ---- wait for any scale pipeline + GPUs to free up ----
log_status "waiting for GPUs (scale pipeline still running?)"
while pgrep -f "run_scale_pipeline.sh" >/dev/null 2>&1; do sleep 60; done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  [[ "${used:-0}" -lt 1500 ]] && break
  sleep 10
done
log_status "GPUs free -- starting rejection queue"

mml_for() {
  # Cap context at 32768 for families whose default context OOMs (Qwen3-VL 262k, gemma-4,
  # InternVL3.5). llava-1.5 (4096 native) is left uncapped on purpose -> "".
  case "$1" in
    Qwen/Qwen3-VL-*|google/gemma-4-*|OpenGVLab/InternVL3_5-*) echo 32768 ;;
    *) echo "" ;;
  esac
}
extra_for() {  # role-prefixed quirk flags
  local role="$1" model="$2" out=""
  local mml; mml=$(mml_for "$model")
  [[ -n "$mml" ]] && out+=" --${role}_max_model_len $mml"
  case "$model" in
    OpenGVLab/InternVL3_5-*) out+=" --${role}_repetition_penalty 1.1 --${role}_disable_chunked_mm" ;;
  esac
  echo "$out"
}
short_for() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

# dataset : solver : verifier (or "oracle")
# Runs already producing metrics.json are skipped automatically, so this is the
# full target set; only the missing rungs actually execute.
#
# PHASE 1 (locked 2026-06-25): §5.1 realized rejection sampling for ALL 49 cells of
# the CharXiv 7-model verifier grid, k=5. The RUNS list is auto-generated below from
# GRID_MODELS x GRID_MODELS, ordered cheap-first (small solver, then small verifier),
# so the GPU-cheapest cells complete first and a crash resumes mid-grid. The legacy
# curated list (CountBench + InternVL ladder/oracle) is preserved under RUNS_LEGACY
# and can be appended via APPEND_LEGACY=1 once Phase 1 is done.
#
# 7 grid models, listed cheap->expensive by (active) param count for the ordering:
GRID_MODELS="${GRID_MODELS:-\
Qwen/Qwen3-VL-2B-Instruct \
OpenGVLab/InternVL3_5-2B \
google/gemma-4-E4B-it \
llava-hf/llava-1.5-7b-hf \
OpenGVLab/InternVL3_5-8B \
Qwen/Qwen3-VL-8B-Instruct \
google/gemma-4-12B-it}"
GRID_DS="${GRID_DS:-charxiv}"

RUNS=()
for S in $GRID_MODELS; do
  for V in $GRID_MODELS; do
    RUNS+=("${GRID_DS}:${S}:${V}")
  done
done

RUNS_LEGACY=(
  # --- CountBench: InternVL solver ladder @ fixed 14B verifier + oracle (fast first) ---
  "countbench:OpenGVLab/InternVL3_5-1B:OpenGVLab/InternVL3_5-14B"
  "countbench:OpenGVLab/InternVL3_5-1B:oracle"
  "countbench:OpenGVLab/InternVL3_5-2B:OpenGVLab/InternVL3_5-14B"
  "countbench:OpenGVLab/InternVL3_5-2B:oracle"
  "countbench:OpenGVLab/InternVL3_5-4B:OpenGVLab/InternVL3_5-14B"
  "countbench:OpenGVLab/InternVL3_5-4B:oracle"
  "countbench:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-14B"
  "countbench:OpenGVLab/InternVL3_5-8B:oracle"
  # --- CharXiv: InternVL solver ladder @ fixed 14B verifier + oracle ---
  "charxiv:OpenGVLab/InternVL3_5-1B:OpenGVLab/InternVL3_5-14B"
  "charxiv:OpenGVLab/InternVL3_5-1B:oracle"
  "charxiv:OpenGVLab/InternVL3_5-2B:OpenGVLab/InternVL3_5-14B"
  "charxiv:OpenGVLab/InternVL3_5-2B:oracle"
  "charxiv:OpenGVLab/InternVL3_5-4B:OpenGVLab/InternVL3_5-14B"
  "charxiv:OpenGVLab/InternVL3_5-4B:oracle"
  "charxiv:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-14B"
  "charxiv:OpenGVLab/InternVL3_5-8B:oracle"
  # --- prior cross-family / self runs (already done; kept for completeness) ---
  "countbench:Qwen/Qwen3-VL-8B-Instruct:google/gemma-4-12B-it"
  "countbench:google/gemma-4-12B-it:google/gemma-4-12B-it"
  "countbench:Qwen/Qwen3-VL-8B-Instruct:oracle"
  "countbench:google/gemma-4-12B-it:oracle"
  "charxiv:Qwen/Qwen3-VL-8B-Instruct:google/gemma-4-12B-it"
  "charxiv:Qwen/Qwen3-VL-8B-Instruct:oracle"
  # --- verifier-scaling extra: fixed InternVL-8B solver, vary verifier (runs if time permits) ---
  "countbench:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-2B"
  "countbench:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-8B"
  "charxiv:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-2B"
  "charxiv:OpenGVLab/InternVL3_5-8B:OpenGVLab/InternVL3_5-8B"
)
[[ "${APPEND_LEGACY:-0}" == "1" ]] && RUNS+=("${RUNS_LEGACY[@]}")

for entry in "${RUNS[@]}"; do
  IFS=":" read -r DS SOLVER VERIFIER <<<"$entry"
  SS=$(short_for "$SOLVER")
  MRC=""
  [[ "$DS" == "charxiv" ]] && MRC="--max_response_chars 2500"
  if [[ "$VERIFIER" == "oracle" ]]; then
    VS="oracle"; VARGS="--oracle_verifier"
  else
    VS=$(short_for "$VERIFIER"); VARGS="--verifier_model_name $VERIFIER $(extra_for verifier "$VERIFIER")"
  fi
  OUT="vlm/result/rejection/${DS}/${SS}__${VS}"
  if [[ -f "$OUT/metrics.json" ]]; then
    log_status "SKIP   $DS $SS vs $VS -- metrics.json exists"
    continue
  fi
  mkdir -p "$OUT"
  LOGF="$LOGDIR/reject_${DS}_${SS}__${VS}.log"
  log_status "START  $DS solver=$SS verifier=$VS"
  if $PY vlm/rejection_sampling.py \
      --solver_model_name "$SOLVER" $(extra_for solver "$SOLVER") \
      $VARGS $MRC \
      --data_dir "data/$DS" --max_attempts 5 \
      --output_dir "$OUT" >"$LOGF" 2>&1; then
    log_status "OK     $DS solver=$SS verifier=$VS"
  else
    log_status "FAIL   $DS solver=$SS verifier=$VS (exit $?) -- see $LOGF"
    pkill -f "VLLM::EngineCore|VLLM::Worker" 2>/dev/null
    sleep 10
  fi
done
log_status "================ REJECTION QUEUE DONE ================"
