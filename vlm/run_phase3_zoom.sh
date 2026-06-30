#!/usr/bin/env bash
# Phase 3 (agentic-zoom stragglers on CharXiv). Waits for Phase 2a (13-model static grid) to
# fully finish so we never run two vLLM engines at once, then runs the missing zoom cells,
# cheap-first, resumable + babysat (relaunch on death, stall-abort a permanently-failing run).
#
# Scope (zoomable = NOT llava [single-image] and NOT gemma-12B [vLLM build bug]):
#   PRIMARY  c=4 row completion -> the 3 newly-added grid models with no zoom yet:
#            InternVL3.5-1B, InternVL3.5-14B, gemma-4-E2B-it
#   FILL     budget curves for the non-Qwen families at c=2 and c=8 (Qwen 2/4/8 already have
#            c2/c4/c8): InternVL3.5 {1,2,4,8,14}B + gemma-4 {E2B,E4B}
# All via vlm/run_agentic_vision_other.sh (MODELS/DATASETS/CROPS env). Launch detached:
#   setsid bash vlm/run_phase3_zoom.sh >vlm/result/_run_logs/phase3_zoom.out 2>&1 &
set -u
cd /home/log/Github/vlm-verification || exit 1
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
ST="$LOGDIR/STATUS_phase3.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$ST"; }
short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

NEW_C4="OpenGVLab/InternVL3_5-1B google/gemma-4-E2B-it OpenGVLab/InternVL3_5-14B"
NONQWEN="OpenGVLab/InternVL3_5-1B OpenGVLab/InternVL3_5-2B google/gemma-4-E2B-it \
OpenGVLab/InternVL3_5-4B google/gemma-4-E4B-it OpenGVLab/InternVL3_5-8B OpenGVLab/InternVL3_5-14B"

zoom_done() {  # $1=budget  $2=model list -> count metrics.json present
  local b="$1" n=0 m
  for m in $2; do
    [[ -f "vlm/result/agentic_vision/charxiv_c${b}/$(short "$m")/metrics.json" ]] && n=$((n+1))
  done; echo "$n"
}
sweep_gpu() {
  pkill -9 -f "agentic_vision.py|VLLM::EngineCore|VLLM::Worker|EngineCore|multiproc_executor" 2>/dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0; sleep 5
  done
}

# run one (budget, models) zoom group to completion, resumable; stall-abort after 4 no-progress
run_zoom() {  # $1=budget  $2=models  $3=label
  local b="$1" models="$2" label="$3" target stall=0 prev=-1 d
  target=$(echo $models | wc -w)
  while true; do
    d=$(zoom_done "$b" "$models")
    if [[ "$d" -ge "$target" ]]; then log "$label DONE ($d/$target)"; return 0; fi
    if [[ "$d" -le "$prev" ]]; then stall=$((stall+1)); else stall=0; fi
    prev="$d"
    if [[ "$stall" -ge 4 ]]; then log "$label STUCK $d/$target after 4 no-progress -- skipping group"; return 1; fi
    log "$label $d/$target -- (re)launching zoom (c=$b)"
    sweep_gpu
    DATASETS=charxiv MODELS="$models" CROPS="$b" bash vlm/run_agentic_vision_other.sh \
      >>"$LOGDIR/phase3_c${b}.out" 2>&1 || log "$label launcher exited nonzero (re-checking)"
  done
}

log "===== PHASE 3 START -- waiting for Phase 2a (static grid 169 pairs) to finish ====="
while true; do
  gp=$(ls vlm/result/verifier_grid/charxiv/verify_charxiv_solver-*_verifier-*.json 2>/dev/null \
       | sed -E 's/.*solver-(.+)_verifier-(.+)_[0-9]{8}-[0-9]{6}\.json/\1|\2/' | sort -u | wc -l)
  if [[ "$gp" -ge 169 ]] && ! pgrep -f "vlm_verify.py|run_verifier_grid.sh" >/dev/null 2>&1; then break; fi
  sleep 60
done
log "Phase 2a complete -- GPUs clear, starting zoom"
sweep_gpu

run_zoom 4 "$NEW_C4"  "zoom-c4-new"    # PRIMARY deliverable: complete the c4 row
run_zoom 2 "$NONQWEN" "zoom-c2-nonqwen"  # FILL: budget curve low
run_zoom 8 "$NONQWEN" "zoom-c8-nonqwen"  # FILL: budget curve high

log "===== PHASE 3 DONE (whatever completed; STUCK groups noted above) ====="
log "Rollout viewers + report regenerate via: bash vlm/build_report.sh"
