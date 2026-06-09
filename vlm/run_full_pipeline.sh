#!/usr/bin/env bash
# Full VLM solver+verifier pipeline over the FULL CountBench + CharXiv datasets.
#
# For each dataset: every model solves it (vLLM), each run is scored, then every model
# verifies every solver run (the 3x3 grid). Each step is logged separately and failures
# are non-fatal -- one bad model must not sink the whole overnight run. A live STATUS file
# tracks progress. Re-running is safe-ish: it appends new timestamped result files.
#
# Usage:  bash vlm/run_full_pipeline.sh
set -u

PY=.venv-vllm/bin/python
BACKEND=vllm
# Optional: set SUBSET=0.01 to solve only a fraction (for a fast end-to-end smoke test).
SUBSET="${SUBSET:-}"
SUBSET_ARG=""
[[ -n "$SUBSET" ]] && SUBSET_ARG="--dataset_subset_ratio $SUBSET"
LOGDIR=vlm/result/_run_logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.txt"

MODELS=("Qwen/Qwen3-VL-2B-Instruct" "google/gemma-4-E2B-it" "llava-hf/llava-1.5-7b-hf")

# Models with a huge declared context need a vLLM context cap to fit 2x24GB.
mml_for() {
  case "$1" in
    Qwen/Qwen3-VL-2B-Instruct) echo "--max_model_len 32768" ;;
    *) echo "" ;;
  esac
}

short_for() { echo "${1##*/}"; }  # llava-hf/llava-1.5-7b-hf -> llava-1.5-7b-hf

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_status() { echo "[$(ts)] $*" | tee -a "$STATUS"; }

run_step() {  # name logfile cmd...
  local name="$1"; local logf="$2"; shift 2
  log_status "START  $name"
  if "$@" >"$logf" 2>&1; then
    log_status "OK     $name"
    return 0
  else
    log_status "FAIL   $name (exit $?) -- see $logf"
    return 1
  fi
}

# newest solver run file for a (dataset,model) in a dir, excluding sidecars
newest_solver_file() {  # dir dataset modelshort
  ls -t "$1/$2_$3"_*.json 2>/dev/null | grep -v '_scores\.json' | grep -v '_prompts\.json' | head -1
}

log_status "================ PIPELINE START ================"
log_status "models: ${MODELS[*]}"

# DATASET   DATADIR            SCORER
DATASETS=("countbench:data/countbench:vlm/score_results.py"
          "charxiv:data/charxiv:vlm/score_charxiv.py")

for entry in "${DATASETS[@]}"; do
  IFS=":" read -r DS DATADIR SCORER <<<"$entry"
  RDIR="vlm/result/$DS"
  mkdir -p "$RDIR"
  log_status "########## DATASET: $DS (data=$DATADIR -> $RDIR) ##########"

  SCORED_FILES=()

  # ---- Solve + score each model ----
  for M in "${MODELS[@]}"; do
    MS=$(short_for "$M")
    MML=$(mml_for "$M")
    run_step "solve  $DS/$MS" "$LOGDIR/solve_${DS}_${MS}.log" \
      $PY vlm/vlm_inference.py \
        --solver_model_name "$M" --data_dir "$DATADIR" \
        --backend "$BACKEND" $MML $SUBSET_ARG --output_dir "$RDIR"

    SF=$(newest_solver_file "$RDIR" "$DS" "$MS")
    if [[ -z "$SF" ]]; then
      log_status "SKIP   score $DS/$MS -- no solver output found"
      continue
    fi
    if run_step "score  $DS/$MS" "$LOGDIR/score_${DS}_${MS}.log" \
        $PY "$SCORER" "$SF"; then
      SCORES="${SF%.json}_scores.json"
      [[ -f "$SCORES" ]] && SCORED_FILES+=("$SCORES")
    fi
  done

  log_status "scored solver runs for $DS: ${#SCORED_FILES[@]} -> ${SCORED_FILES[*]:-none}"
  if [[ ${#SCORED_FILES[@]} -eq 0 ]]; then
    log_status "SKIP   verification for $DS -- no scored solver runs"
    continue
  fi

  # ---- Verify: every model verifies all scored solver runs ----
  for V in "${MODELS[@]}"; do
    VS=$(short_for "$V")
    VMML=$(mml_for "$V")
    run_step "verify $DS/verifier=$VS" "$LOGDIR/verify_${DS}_${VS}.log" \
      $PY vlm/vlm_verify.py \
        --verifier_model_name "$V" --backend "$BACKEND" $VMML \
        --output_dir "$RDIR" \
        --solver_run_files "${SCORED_FILES[@]}"
  done
done

log_status "================ PIPELINE DONE ================"
