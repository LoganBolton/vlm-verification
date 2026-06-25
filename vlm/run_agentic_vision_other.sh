#!/usr/bin/env bash
# Agentic-vision (zoom-tool) queue for the NON-Qwen families -- run after the Qwen sweep.
# The zoom protocol uses the Qwen/Hermes <tool_call> format; other models are instructed to
# emit it too, but compliance is not guaranteed, so this is exploratory. llava-1.5 is
# excluded: it accepts only ONE image per prompt and the zoom loop adds images.
#
# Each family gets its known vLLM quirk flags (see vlm/run_self_consistency.sh extra_for):
#   InternVL3.5 -> 32k ctx, repetition_penalty 1.1, disable chunked mm
#   gemma-4     -> 32k ctx
#
# Budget fixed at c=4 (the primary), both datasets. RESUMABLE (skips existing metrics.json).
# Usage:  bash vlm/run_agentic_vision_other.sh
set -u
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
CROPS="${CROPS:-4}"
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_agentic_other.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

DATASETS="${DATASETS:-countbench charxiv}"
MODELS="${MODELS:-OpenGVLab/InternVL3_5-2B OpenGVLab/InternVL3_5-4B OpenGVLab/InternVL3_5-8B google/gemma-4-E4B-it google/gemma-4-12B-it}"

extra_for() {  # per-family vLLM quirk flags
  case "$1" in
    OpenGVLab/InternVL3_5-*) echo "--solver_max_model_len 32768 --solver_repetition_penalty 1.1 --solver_disable_chunked_mm" ;;
    google/gemma-4-*|Qwen/Qwen3-VL-*) echo "--solver_max_model_len 32768" ;;
    *) echo "--solver_max_model_len 32768" ;;
  esac
}
short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

wait_gpu() {
  for _ in $(seq 1 180); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0
    sleep 10
  done
  log "WARN wait_gpu timed out (used=${used:-?} MiB)"; return 1
}

log "waiting for GPUs to free up..."
for _ in $(seq 1 4320); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  [[ "${used:-9999}" -lt 1500 ]] && break
  sleep 10
done
log "GPUs free -- starting NON-Qwen agentic sweep (c=$CROPS, datasets=[$DATASETS])"

NDONE=0; NFAIL=0; NSKIP=0
for DS in $DATASETS; do
  MNT=2048; [[ "$DS" == "charxiv" ]] && MNT=4096
  for SOLVER in $MODELS; do
    SS=$(short "$SOLVER")
    OUT="vlm/result/agentic_vision/${DS}_c${CROPS}/${SS}"
    if [[ -f "$OUT/metrics.json" ]]; then log "SKIP  $DS $SS (exists)"; NSKIP=$((NSKIP+1)); continue; fi
    mkdir -p "$OUT"
    LOGF="$LOGDIR/av_${DS}_c${CROPS}_${SS}.log"
    wait_gpu || true
    log "START $DS c=$CROPS solver=$SS"
    if $PY vlm/agentic_vision.py --solver_model_name "$SOLVER" $(extra_for "$SOLVER") \
          --data_dir "data/$DS" --max_crops "$CROPS" --solver_max_new_tokens "$MNT" \
          --output_dir "$OUT" >"$LOGF" 2>&1; then
      log "OK    $DS c=$CROPS solver=$SS"; NDONE=$((NDONE+1))
    else
      log "FAIL  $DS c=$CROPS solver=$SS (exit $?) -- see $LOGF"; NFAIL=$((NFAIL+1))
      pkill -9 -f "agentic_vision.py" 2>/dev/null
      pkill -9 -f "multiproc_executor|EngineCore" 2>/dev/null
      sleep 15
    fi
  done
done
log "===== NON-Qwen AGENTIC SWEEP DONE: $NDONE ok, $NFAIL failed, $NSKIP skipped ====="
