#!/usr/bin/env bash
# Agentic-vision (zoom-tool) overnight queue -- the active-perception counterpart to
# vlm/run_self_consistency.sh (pass@N) and the rejection/verify runs (VLM judge).
#
# Sweeps the full Qwen3-VL ladder (2B/4B/8B) x both datasets x a zoom budget {2,4,8}, so
# you get an accuracy-vs-zoom-budget curve to compare against pass@k and the verifier.
# Ordered cheap-first (countbench before charxiv, small budget/model first) so a partial
# overnight run still yields the most complete combos. RESUMABLE: any combo whose
# metrics.json already exists is skipped, so it is safe to re-run.  Usage:
#   bash vlm/run_agentic_vision.sh
# Override the sweep:  MODELS="..." DATASETS="..." BUDGETS="..." bash vlm/run_agentic_vision.sh
set -u
PY=.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
LOGDIR=vlm/result/_run_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS_agentic.txt"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

# Sweep axes (override via env). Order matters: cheap-first for best partial-overnight yield.
DATASETS="${DATASETS:-countbench charxiv}"
BUDGETS="${BUDGETS:-2 4 8}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct}"

short() { local s="${1##*/}"; echo "$s" | sed -E 's/[^A-Za-z0-9.-]+/-/g'; }

wait_gpu() {  # block until BOTH GPUs are essentially empty (prev run fully released VRAM)
  for _ in $(seq 1 180); do  # up to ~30 min
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
    [[ "${used:-9999}" -lt 1500 ]] && return 0
    sleep 10
  done
  log "WARN wait_gpu timed out (used=${used:-?} MiB)"; return 1
}

log "waiting for GPUs to free up..."
for _ in $(seq 1 4320); do  # up to ~12h
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  [[ "${used:-9999}" -lt 1500 ]] && break
  sleep 10
done
log "GPUs free -- starting agentic-vision sweep (datasets=[$DATASETS] budgets=[$BUDGETS])"

NDONE=0; NFAIL=0; NSKIP=0
for DS in $DATASETS; do
  # CharXiv chart reasoning is longer; give the solver more room per turn.
  MNT=2048; [[ "$DS" == "charxiv" ]] && MNT=4096
  for CROPS in $BUDGETS; do
    for SOLVER in $MODELS; do
      SS=$(short "$SOLVER")
      OUT="vlm/result/agentic_vision/${DS}_c${CROPS}/${SS}"
      if [[ -f "$OUT/metrics.json" ]]; then
        log "SKIP  $DS c=$CROPS $SS (exists)"; NSKIP=$((NSKIP+1)); continue
      fi
      mkdir -p "$OUT"
      LOGF="$LOGDIR/av_${DS}_c${CROPS}_${SS}.log"
      wait_gpu || true   # ensure prior run fully released VRAM before TP init
      log "START $DS c=$CROPS solver=$SS"
      if $PY vlm/agentic_vision.py --solver_model_name "$SOLVER" \
            --data_dir "data/$DS" --max_crops "$CROPS" \
            --solver_max_model_len 32768 --solver_max_new_tokens "$MNT" \
            --output_dir "$OUT" >"$LOGF" 2>&1; then
        log "OK    $DS c=$CROPS solver=$SS"; NDONE=$((NDONE+1))
      else
        log "FAIL  $DS c=$CROPS solver=$SS (exit $?) -- see $LOGF"; NFAIL=$((NFAIL+1))
        pkill -9 -f "agentic_vision.py" 2>/dev/null
        pkill -9 -f "from multiprocessing.spawn|multiproc_executor|EngineCore" 2>/dev/null
        sleep 15
      fi
    done
  done
done
log "===== AGENTIC-VISION SWEEP DONE: $NDONE ok, $NFAIL failed, $NSKIP skipped ====="
