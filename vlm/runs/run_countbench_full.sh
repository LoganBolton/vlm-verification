#!/usr/bin/env bash
# CountBench FULL replication -- brings CountBench up to CharXiv parity.
# Gated to start AFTER the Phase-4 verifier grid finishes (countbench_gain.csv exists), so it
# never contends with the grid or the running gemma-native CharXiv zoom. Then, cheap-first:
#   A. maj@k self-consistency (N=5) for the 7 CountBench solvers still missing it
#   B. agentic zoom sweep (c2/c4/c8) for the 11-model set (Qwen x3, InternVL x5, gemma x3),
#      resumable -- skips the 13 combos already done; gemma auto-uses its NATIVE tool protocol.
# Reuses run_self_consistency.sh / run_agentic_vision.sh (GPU-wait, per-family flags, resume,
# crash-recovery all built in). Fully resumable: a watchdog relaunch re-skips finished work.
set -u
cd /home/log/Github/vlm-verification || exit 1
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_countbench_full.txt"
DONE_MARK="$LOGDIR/countbench_full.DONE"
GAIN_CSV=vlm/result/verifier_grid/countbench_gain.csv
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

[[ -f "$DONE_MARK" ]] && { log "DONE marker present -- nothing to do"; exit 0; }

# ---- 0. gate: block until the Phase-4 CountBench verifier grid + gain finished ----
log "CountBench-full supervisor up -- waiting for Phase-4 grid (countbench_gain.csv)"
for _ in $(seq 1 240); do   # up to ~20h of polling (300s each)
  [[ -f "$GAIN_CSV" ]] && break
  sleep 300
done
if [[ ! -f "$GAIN_CSV" ]]; then
  log "WARN gate timed out -- $GAIN_CSV still absent; proceeding anyway (GPU-wait guards us)"
else
  log "Phase-4 grid complete ($GAIN_CSV present) -- starting CountBench-full"
fi

# ---- A. maj@k self-consistency (N=5) for the 7 missing solvers, cheap-first ----
SC_MISSING="llava-hf/llava-1.5-7b-hf Qwen/Qwen3-VL-2B-Instruct google/gemma-4-E2B-it \
Qwen/Qwen3-VL-4B-Instruct google/gemma-4-E4B-it llava-hf/llava-1.5-13b-hf OpenGVLab/InternVL3_5-14B"
log "PHASE A: CountBench maj@k (N=5) for missing solvers -> $SC_MISSING"
N=5 DATASETS=countbench MODELS="$SC_MISSING" bash vlm/runs/run_self_consistency.sh \
  >>"$LOGDIR/countbench_full_sc.out" 2>&1
log "PHASE A returned"

# ---- B. agentic zoom sweep c2/c4/c8 for the 11-model set (skips the 13 already done) ----
ZOOM_MODELS="Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct \
OpenGVLab/InternVL3_5-1B OpenGVLab/InternVL3_5-2B OpenGVLab/InternVL3_5-4B \
OpenGVLab/InternVL3_5-8B OpenGVLab/InternVL3_5-14B \
google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-12B-it"
log "PHASE B: CountBench zoom sweep (budgets 2/4/8) for 11 models (gemma native auto)"
DATASETS=countbench BUDGETS="2 4 8" MODELS="$ZOOM_MODELS" bash vlm/runs/run_agentic_vision.sh \
  >>"$LOGDIR/countbench_full_zoom.out" 2>&1
log "PHASE B returned"

# ---- C. refresh CountBench gain (grid unchanged, but harmless) + report ----
.venv/bin/python vlm/verifier_gain.py --dataset countbench >>"$LOGDIR/countbench_full_gain.out" 2>&1 \
  && log "verifier_gain countbench refreshed" || log "verifier_gain countbench refresh FAILED"
.venv/bin/python vlm/build_charxiv_report.py >>"$LOGDIR/countbench_full_report.out" 2>&1 \
  && log "report rebuilt" || log "report rebuild skipped/failed"

touch "$DONE_MARK"
log "===== COUNTBENCH FULL REPLICATION DONE ====="
