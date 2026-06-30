#!/usr/bin/env bash
# Walk-away supervisor: wait for Phase 1 (49-cell CharXiv rejection grid) to finish, then
# AUTO-ADVANCE to Phase 2 (widen static grid + rejection to 13 models), regenerating the
# §5.1 figures along the way. Every sub-step is resumable; this script relaunches a step if
# it dies, with stall-detection so a permanently-failing cell can't loop forever.
# Launch detached:
#   setsid bash vlm/run_phase_chain.sh >vlm/result/_run_logs/phase_chain.out 2>&1 &
set -u
cd /home/log/Github/vlm-verification || exit 1
PYV=.venv/bin/python            # no-GPU analysis venv
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
ST="$LOGDIR/STATUS_phase_chain.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$ST"; }

short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

MODELS7="Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-2B google/gemma-4-E4B-it \
llava-hf/llava-1.5-7b-hf OpenGVLab/InternVL3_5-8B Qwen/Qwen3-VL-8B-Instruct google/gemma-4-12B-it"
# 13 models, cheap-first by (active) param size
MODELS13="OpenGVLab/InternVL3_5-1B Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-2B \
google/gemma-4-E2B-it Qwen/Qwen3-VL-4B-Instruct OpenGVLab/InternVL3_5-4B google/gemma-4-E4B-it \
llava-hf/llava-1.5-7b-hf OpenGVLab/InternVL3_5-8B Qwen/Qwen3-VL-8B-Instruct google/gemma-4-12B-it \
llava-hf/llava-1.5-13b-hf OpenGVLab/InternVL3_5-14B"

rej_done() {  # $* = model HF ids; count metrics.json among model x model cells
  local n=0
  for S in $1; do for V in $1; do
    [[ -f "vlm/result/rejection/charxiv/$(short "$S")__$(short "$V")/metrics.json" ]] && n=$((n+1))
  done; done
  echo "$n"
}
grid_pairs_done() {  # distinct (solver,verifier) verify files in the static grid
  ls vlm/result/verifier_grid/charxiv/verify_charxiv_solver-*_verifier-*.json 2>/dev/null \
    | sed -E 's/.*solver-(.+)_verifier-(.+)_[0-9]{8}-[0-9]{6}\.json/\1|\2/' | sort -u | wc -l
}
sweep_gpu() {
  pkill -9 -f "VLLM::EngineCore|VLLM::Worker|EngineCore|multiproc_executor" 2>/dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0; sleep 5
  done
}
figures() {
  log "regenerating verifier_gain (F1/FNR + by-regime) + §5.1 scatter"
  $PYV vlm/verifier_gain.py --dataset charxiv >"$LOGDIR/verifier_gain.out" 2>&1 || true
  $PYV vlm/plot_gain_scatter.py --dataset charxiv --k 5 >"$LOGDIR/gain_scatter.out" 2>&1 || true
}

# run a resumable launcher until its done-count reaches target; stall-abort after 4 no-progress tries
run_until() {  # $1 target  $2 countfn  $3 launch-cmd  $4 label
  local target="$1" countfn="$2" launch="$3" label="$4" stall=0 prev=-1 d
  while true; do
    d=$(eval "$countfn")
    if [[ "$d" -ge "$target" ]]; then log "$label DONE ($d/$target)"; return 0; fi
    if [[ "$d" -le "$prev" ]]; then stall=$((stall+1)); else stall=0; fi
    prev="$d"
    if [[ "$stall" -ge 4 ]]; then
      log "$label STUCK at $d/$target after 4 no-progress relaunches -- aborting chain"; return 1
    fi
    log "$label $d/$target -- (re)launching"
    sweep_gpu
    eval "$launch" || log "$label launcher exited nonzero (will re-check + maybe relaunch)"
  done
}

# ---------------------------------------------------------------------------
log "===== PHASE CHAIN START ====="

# ---- wait for Phase 1 (the standalone queue + _reject_watch own its resilience) ----
log "waiting for Phase 1: 49 CharXiv rejection cells"
while [[ "$(rej_done "$MODELS7")" -lt 49 ]]; do sleep 120; done
log "Phase 1 COMPLETE (49/49)"
# make sure no Phase-1 vLLM is still resident before we grab the GPUs
while pgrep -f "rejection_sampling.py|run_rejection.sh|vlm_verify.py" >/dev/null 2>&1; do sleep 30; done
sweep_gpu
figures

# ---- Phase 2a: widen static grid to 13 models (169 pairs) ----
log "PHASE 2a: static verifier grid, 13 models (target 169 pairs)"
run_until 169 grid_pairs_done \
  "GRID_MODELS=\"$MODELS13\" bash vlm/run_verifier_grid.sh" "grid13" || exit 1
figures

# ---- Phase 2b: SKIPPED per user decision 2026-06-28 (value-for-time) ----
# §5.1 is already validated by Phase 1's 49 rejection cells; the 13-model rejection re-run
# (~5.5 days) only densifies the scatter, so we stop after 2a. To run it later:
#   GRID_MODELS="$MODELS13" GRID_DS=charxiv bash vlm/run_rejection.sh   (resumable, skips done)
log "PHASE 2b SKIPPED (user decision) -- stopping after the 13-model static grid"

log "===== PHASE CHAIN DONE: Phase 1 + Phase 2a (13-model static gain grid) complete ====="
log "Final figures regenerated. Next (needs human judgment): Phase 2b rejection if wanted,"
log "then Phase 3 agentic-zoom stragglers, then Phase 4 CountBench."
