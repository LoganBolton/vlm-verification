#!/usr/bin/env bash
# One-off queue job: fill the single missing agentic-zoom cell -- InternVL3.5-2B, CountBench, c8.
# It originally crashed on a context-length overflow (8 crops -> a 32851-token prompt > the old
# hardcoded 32768 cap). run_agentic_vision.sh now uses InternVL's native 40960 context (mml_for),
# so this completes. Gated to start only AFTER the overnight rejection queue finishes (they share
# the same 2 GPUs), and fully resumable -- run_agentic_vision.sh skips any cell with a metrics.json.
set -u
cd /home/log/Github/vlm-verification || exit 1
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_zoom_fill.txt"
DONE_MARK="$LOGDIR/zoom_fill.DONE"
REJ_DONE="$LOGDIR/overnight_rejection.DONE"
CELL="vlm/result/agentic_vision/countbench_c8/InternVL3-5-2B/metrics.json"
ts(){ date "+%F %T"; }
log(){ echo "[$(ts)] $*" | tee -a "$STATUS"; }
[[ -f "$DONE_MARK" ]] && { log "already done -- nothing to do"; exit 0; }
[[ -f "$CELL" ]] && { log "cell already present -- marking done"; touch "$DONE_MARK"; exit 0; }

log "zoom-fill queued -- waiting for the rejection queue to finish (marker $REJ_DONE)"
while [[ ! -f "$REJ_DONE" ]]; do sleep 300; done
# belt-and-suspenders: don't start until no rejection process is still holding the GPUs
while pgrep -f "rejection_sampling.py|run_overnight_rejection.sh" >/dev/null 2>&1; do sleep 60; done
log "rejection queue done + GPUs releasing -- running the missing zoom cell (c8 InternVL-2B, 40960 ctx)"

env DATASETS=countbench BUDGETS=8 MODELS="OpenGVLab/InternVL3_5-2B" \
    bash vlm/run_agentic_vision.sh >>"$LOGDIR/zoom_fill_run.out" 2>&1
log "run_agentic_vision returned"

if [[ -f "$CELL" ]]; then
  log "cell filled -- rebuilding report"
  .venv/bin/python vlm/build_charxiv_report.py >>"$LOGDIR/zoom_fill_report.out" 2>&1 \
    && log "report rebuilt" || log "report rebuild failed (non-fatal)"
  touch "$DONE_MARK"
  log "===== ZOOM FILL DONE ====="
else
  log "WARN cell still missing after run -- leaving marker unset so the watchdog retries"
fi
