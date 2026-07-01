#!/usr/bin/env bash
# All gemma-4 models on CharXiv agentic zoom using Gemma's NATIVE tool protocol (tools= +
# native <tool_call|> stop + parse), for a fair, apples-to-apples gemma family comparison.
# 3 models x 3 budgets (c2/c4/c8), resumable (skips a run whose metrics.json exists).
# When done: rebuild the report, then RESUME the paused CountBench verifier grid.
# NOTE: old Hermes-format gemma charxiv zoom dirs are cleared by the launcher BEFORE this runs;
# this script only skips-if-exists so a watchdog relaunch never wipes completed native results.
set -u
cd /home/log/Github/vlm-verification || exit 1
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
MODELS="google/gemma-4-12B-it google/gemma-4-E4B-it google/gemma-4-E2B-it"
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_gemma_native.txt"
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }
short() { local s="${1##*/}"; echo "${s//_/-}"; }

wait_gpu() {
  for _ in $(seq 1 360); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [ "${u:-9999}" -lt 1500 ] && return 0; sleep 10
  done; log "WARN wait_gpu timeout (used=${u:-?})"; return 1
}

for SOLVER in $MODELS; do
  SS=$(short "$SOLVER")
  for C in 2 4 8; do
    OUT="vlm/result/agentic_vision/charxiv_c${C}/${SS}"
    if [ -f "$OUT/metrics.json" ]; then log "SKIP charxiv c=$C $SS (exists)"; continue; fi
    mkdir -p "$OUT"
    wait_gpu || true
    log "START charxiv c=$C solver=$SS (gemma-native)"
    if $PY vlm/agentic_vision.py --solver_model_name "$SOLVER" --solver_max_model_len 32768 \
          --data_dir data/charxiv --max_crops "$C" --solver_max_new_tokens 4096 \
          --output_dir "$OUT" >"$LOGDIR/av_charxiv_c${C}_${SS}.log" 2>&1; then
      log "OK    charxiv c=$C $SS"
    else
      log "FAIL  charxiv c=$C $SS (exit $?) -- see av_charxiv_c${C}_${SS}.log"
      pkill -9 -f "agentic_vision.py" 2>/dev/null
      pkill -9 -f "EngineCore|multiproc_executor" 2>/dev/null
      sleep 15
    fi
  done
done

log "all gemma-native charxiv zoom done -- rebuilding report"
.venv/bin/python vlm/build_charxiv_report.py >>"$LOGDIR/gemma_native_report.out" 2>&1 && log "report rebuilt"

log "resuming paused CountBench grid (phase 4 supervisor + watchdog)"
setsid bash vlm/run_phase4_countbench.sh >>"$LOGDIR/phase4_supervisor.out" 2>&1 </dev/null &
setsid bash vlm/result/_run_logs/_phase4_watch.sh >/dev/null 2>&1 </dev/null &
log "===== GEMMA-NATIVE ZOOM SUPERVISOR DONE ====="
