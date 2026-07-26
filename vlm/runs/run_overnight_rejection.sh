#!/usr/bin/env bash
# Overnight §5.1 REALIZED rejection-sampling queue -- the "does verification pay off in PRACTICE"
# data (actually resample with the judge, k=5, then compare realized accuracy to the gain the
# static grid predicted). This is the last major replication gap: CountBench has almost no
# rejection grid, and CharXiv only has the 7x7 core.
#
# Fills cheap-first + fully resumable (run_rejection.sh skips any cell whose metrics.json exists):
#   STEP 1: CountBench 7x7 core grid (49 cells, k=5)  -- symmetric with the CharXiv Phase-1 core
#   STEP 2: CountBench 13x13 expansion               -- skips the 49; fills the rest
#   STEP 3: CharXiv  13x13 expansion                 -- skips the ~53 done; completes Phase-2 plan
# Gated to start only after the in-flight zoom fill-in releases the GPUs (run_rejection.sh's own
# top-of-script GPU wait is only ~10 min, too short to reliably outlast the zoom runs).
set -u
cd /home/log/Github/vlm-verification || exit 1
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_overnight_rejection.txt"
DONE_MARK="$LOGDIR/overnight_rejection.DONE"
ts(){ date "+%F %T"; }
log(){ echo "[$(ts)] $*" | tee -a "$STATUS"; }
[[ -f "$DONE_MARK" ]] && { log "DONE marker present -- nothing to do"; exit 0; }

# 13 grid models, cheap -> expensive by active params (drives run_rejection.sh cheap-first order).
M13="Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-1B google/gemma-4-E2B-it \
OpenGVLab/InternVL3_5-2B google/gemma-4-E4B-it OpenGVLab/InternVL3_5-4B Qwen/Qwen3-VL-4B-Instruct \
llava-hf/llava-1.5-7b-hf OpenGVLab/InternVL3_5-8B Qwen/Qwen3-VL-8B-Instruct \
llava-hf/llava-1.5-13b-hf google/gemma-4-12B-it OpenGVLab/InternVL3_5-14B"

# ---- gate: wait out the in-flight zoom/agentic run so we never share the GPUs ----
log "overnight-rejection supervisor up -- waiting for any zoom/agentic run to release GPUs"
while pgrep -f "agentic_vision.py|run_agentic_vision.sh" >/dev/null 2>&1; do sleep 60; done
log "GPUs clear of zoom -- starting rejection queue"

log "STEP 1/3: CountBench 7x7 core rejection grid (k=5, cheap-first, resumable)"
GRID_DS=countbench bash vlm/runs/run_rejection.sh >>"$LOGDIR/overnight_rej_cb7.out" 2>&1
log "STEP 1/3 returned"

log "STEP 2/3: CountBench 13x13 rejection expansion (fills the remaining cells)"
GRID_DS=countbench GRID_MODELS="$M13" bash vlm/runs/run_rejection.sh >>"$LOGDIR/overnight_rej_cb13.out" 2>&1
log "STEP 2/3 returned"

log "STEP 3/3: CharXiv 13x13 rejection expansion (skips the 7x7 core already done)"
GRID_DS=charxiv GRID_MODELS="$M13" bash vlm/runs/run_rejection.sh >>"$LOGDIR/overnight_rej_cx13.out" 2>&1
log "STEP 3/3 returned"

# refresh the §5.1 predicted-vs-realized figures/report off the new rejection data
.venv/bin/python vlm/analysis/build_charxiv_report.py >>"$LOGDIR/overnight_rej_report.out" 2>&1 \
  && log "report rebuilt" || log "report rebuild skipped/failed"

touch "$DONE_MARK"
log "===== OVERNIGHT REJECTION QUEUE DONE ====="
