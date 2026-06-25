#!/usr/bin/env bash
# Self-consistency / majority-vote@N queue. Waits for the GPUs to free up (so it can be
# queued behind another job), then runs vlm/self_consistency.py for each combo.
# Skips combos whose metrics.json already exists.  Usage: bash vlm/run_self_consistency.sh
set -u
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
N=16
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_selfconsistency.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

log "waiting for GPUs to free up..."
for _ in $(seq 1 4320); do  # up to ~12h
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  [[ "${used:-9999}" -lt 1500 ]] && break
  sleep 10
done
log "GPUs free -- starting self-consistency queue (N=$N)"

wait_gpu() {  # block until BOTH GPUs are essentially empty (prev run fully released VRAM)
  for _ in $(seq 1 180); do  # up to ~30 min
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0
    sleep 10
  done
  log "WARN wait_gpu timed out (used=${used:-?} MiB)"; return 1
}

extra_for() {  # InternVL quirk flags
  case "$1" in
    OpenGVLab/InternVL3_5-*) echo "--solver_max_model_len 32768 --solver_repetition_penalty 1.1 --solver_disable_chunked_mm" ;;
    Qwen/Qwen3-VL-*|google/gemma-4-*) echo "--solver_max_model_len 32768" ;;
    *) echo "" ;;
  esac
}
short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

# dataset : solver
RUNS=(
  # CountBench (491, fast): full ladder + cross-family
  "countbench:OpenGVLab/InternVL3_5-1B"
  "countbench:OpenGVLab/InternVL3_5-2B"
  "countbench:OpenGVLab/InternVL3_5-4B"
  "countbench:OpenGVLab/InternVL3_5-8B"
  "countbench:Qwen/Qwen3-VL-8B-Instruct"
  "countbench:google/gemma-4-12B-it"
  # CharXiv (1000, heavier): the solver we have full rejection+oracle for, plus best solver
  "charxiv:OpenGVLab/InternVL3_5-8B"
  "charxiv:google/gemma-4-12B-it"
)
for entry in "${RUNS[@]}"; do
  IFS=":" read -r DS SOLVER <<<"$entry"
  SS=$(short "$SOLVER")
  MRC=""; [[ "$DS" == "charxiv" ]] && MRC="--solver_max_new_tokens 8192"
  OUT="vlm/result/self_consistency/${DS}/${SS}"
  if [[ -f "$OUT/metrics.json" ]]; then log "SKIP  $DS $SS (exists)"; continue; fi
  mkdir -p "$OUT"
  LOGF="$LOGDIR/sc_${DS}_${SS}.log"
  wait_gpu || true   # ensure prior run fully released VRAM before TP init (avoids NCCL stall)
  log "START $DS solver=$SS"
  if $PY vlm/self_consistency.py --solver_model_name "$SOLVER" $(extra_for "$SOLVER") $MRC \
        --data_dir "data/$DS" --n_samples "$N" --output_dir "$OUT" >"$LOGF" 2>&1; then
    log "OK    $DS solver=$SS"
  else
    log "FAIL  $DS solver=$SS (exit $?) -- see $LOGF"
    pkill -9 -f "self_consistency.py" 2>/dev/null
    pkill -9 -f "from multiprocessing.spawn|multiproc_executor|EngineCore" 2>/dev/null
    sleep 15
  fi
done
log "============ SELF-CONSISTENCY QUEUE DONE ============"
