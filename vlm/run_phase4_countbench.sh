#!/usr/bin/env bash
# Phase 4 supervisor: CountBench static verifier grid + gain.
# Queued to run AFTER the n=5 CharXiv maj@k self-consistency finishes (user wants maj@k first).
# Order:
#   1. block until all 13 CharXiv self_consistency metrics.json exist (maj@k done)
#   2. 13x13 CountBench verifier grid  (DS=countbench run_verifier_grid.sh) -- verifiers judge
#      the 13 existing countbench base runs; no new solver generation; resumable per pair.
#   3. verifier_gain --dataset countbench  -> gain-by-regime + matrices + countbench_gain.csv
# Fully resumable: the grid skips finished pairs, so a relaunch picks up where it died.
set -u
cd /home/log/Github/vlm-verification || exit 1
PYV=.venv/bin/python
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_phase4.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

M13="OpenGVLab/InternVL3_5-1B Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-2B \
google/gemma-4-E2B-it Qwen/Qwen3-VL-4B-Instruct OpenGVLab/InternVL3_5-4B google/gemma-4-E4B-it \
llava-hf/llava-1.5-7b-hf OpenGVLab/InternVL3_5-8B Qwen/Qwen3-VL-8B-Instruct google/gemma-4-12B-it \
llava-hf/llava-1.5-13b-hf OpenGVLab/InternVL3_5-14B"
short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

sc_done() {  # 13/13 CharXiv self-consistency metrics present?
  local n=0 m
  for m in $M13; do
    [[ -f "vlm/result/self_consistency/charxiv/$(short "$m")/metrics.json" ]] && n=$((n+1))
  done
  echo "$n"
}

log "Phase 4 supervisor up -- waiting for CharXiv maj@k (13 self-consistency runs) to finish first"
while true; do
  d=$(sc_done)
  [[ "$d" -ge 13 ]] && { log "maj@k complete ($d/13) -- starting CountBench grid"; break; }
  sleep 300
done

# ---- 2. CountBench 13x13 verifier grid (blocks; resumable) ----
log "launching CountBench 13x13 verifier grid"
DS=countbench GRID_MODELS="$M13" bash vlm/run_verifier_grid.sh >>"$LOGDIR/phase4_grid.out" 2>&1
log "CountBench verifier grid returned"

# ---- 3. gain analysis ----
log "computing CountBench verifier gain"
$PYV vlm/verifier_gain.py --dataset countbench >>"$LOGDIR/phase4_gain.out" 2>&1 \
  && log "verifier_gain countbench OK -> vlm/result/verifier_grid/countbench/countbench_gain.csv" \
  || log "verifier_gain countbench FAILED -- see phase4_gain.out"

log "===== PHASE 4 (CountBench grid + gain) DONE ====="
