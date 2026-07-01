#!/usr/bin/env bash
# One-off: gemma-4-12B-it agentic zoom on CharXiv for all 3 budgets (c2/c4/c8), now that the
# audio-tower profiler crash is fixed (limit_mm audio=0 in agentic_vision.py). Resumable
# (skips a budget whose metrics.json exists). When done: rebuild the report and RESUME the
# paused CountBench verifier grid (phase 4 supervisor + watchdog).
set -u
cd /home/log/Github/vlm-verification || exit 1
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
SOLVER=google/gemma-4-12B-it
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_gemma12b_zoom.txt"
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

wait_gpu() {
  for _ in $(seq 1 360); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [ "${u:-9999}" -lt 1500 ] && return 0; sleep 10
  done; log "WARN wait_gpu timeout (used=${u:-?})"; return 1
}

for C in 2 4 8; do
  OUT="vlm/result/agentic_vision/charxiv_c${C}/gemma-4-12B-it"
  if [ -f "$OUT/metrics.json" ]; then log "SKIP charxiv c=$C (exists)"; continue; fi
  mkdir -p "$OUT"
  wait_gpu || true
  log "START charxiv c=$C solver=gemma-4-12B-it"
  if $PY vlm/agentic_vision.py --solver_model_name "$SOLVER" --solver_max_model_len 32768 \
        --data_dir data/charxiv --max_crops "$C" --solver_max_new_tokens 4096 \
        --output_dir "$OUT" >"$LOGDIR/av_charxiv_c${C}_gemma-4-12B-it.log" 2>&1; then
    log "OK    charxiv c=$C"
  else
    log "FAIL  charxiv c=$C (exit $?) -- see av_charxiv_c${C}_gemma-4-12B-it.log"
    pkill -9 -f "agentic_vision.py" 2>/dev/null
    pkill -9 -f "EngineCore|multiproc_executor" 2>/dev/null
    sleep 15
  fi
done

log "gemma-12B CharXiv zoom finished -- rebuilding report"
.venv/bin/python vlm/build_charxiv_report.py >>"$LOGDIR/gemma12b_report.out" 2>&1 && log "report rebuilt"

log "resuming paused CountBench grid (phase 4 supervisor + watchdog)"
setsid bash vlm/run_phase4_countbench.sh >>"$LOGDIR/phase4_supervisor.out" 2>&1 </dev/null &
setsid bash vlm/result/_run_logs/_phase4_watch.sh >/dev/null 2>&1 </dev/null &
log "===== GEMMA-12B ZOOM SUPERVISOR DONE ====="
