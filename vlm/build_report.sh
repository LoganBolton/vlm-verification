#!/usr/bin/env bash
# Build all figures, per-example HTML viewers, and the combined HTML report from the
# result JSON produced by run_full_pipeline.sh / run_scale_pipeline.sh. Datasets whose
# result dir has no verify_*.json yet are skipped, so this is safe to re-run anytime.
#
# Usage:  bash vlm/build_report.sh
set -u
PY=.venv/bin/python   # plotting/report deps (matplotlib) live in the base venv
VIEW_LIMIT="${VIEW_LIMIT:-200}"   # cap examples per HTML viewer so files stay openable

# key : result_dir : dataset_label_for_titles
DATASETS=("countbench:vlm/result/countbench:CountBenchQA"
          "charxiv:vlm/result/charxiv:CharXiv"
          "countbench_tier2:vlm/result/countbench_tier2:CountBenchQA tier2 (4B-13B)"
          "charxiv_tier2:vlm/result/charxiv_tier2:CharXiv tier2 (4B-13B)"
          "countbench_tier3:vlm/result/countbench_tier3:CountBenchQA tier3 (8B-12B)"
          "charxiv_tier3:vlm/result/charxiv_tier3:CharXiv tier3 (8B-12B)"
          "countbench_intern:vlm/result/countbench_intern:CountBenchQA InternVL3.5 (1B-14B)"
          "charxiv_intern:vlm/result/charxiv_intern:CharXiv InternVL3.5 (1B-14B)")

REPORT_ARGS=()
for entry in "${DATASETS[@]}"; do
  IFS=":" read -r DS RDIR LABEL <<<"$entry"
  FIGDIR="vlm/viz/figures/$DS"
  VIEWDIR="vlm/viz/views/$DS"
  if ! ls "$RDIR"/verify_*.json >/dev/null 2>&1; then
    echo "skip $DS -- no verify_*.json in $RDIR"; continue
  fi
  echo "=== figures: $DS ==="
  $PY vlm/viz/plot_results.py --result_dir "$RDIR" --fig_dir "$FIGDIR" \
      --out_md "$FIGDIR/RESULTS_VISUAL.md" --dataset_label "$LABEL"
  echo "=== viewers: $DS (limit $VIEW_LIMIT) ==="
  $PY vlm/viz/view_examples.py --result_dir "$RDIR" --out_dir "$VIEWDIR" --limit "$VIEW_LIMIT"
  REPORT_ARGS+=(--dataset "$LABEL:$RDIR:$FIGDIR:views/$DS")
done

if [[ ${#REPORT_ARGS[@]} -eq 0 ]]; then
  echo "no datasets with results -- nothing to report"; exit 0
fi
echo "=== combined report ==="
$PY vlm/viz/make_report.py "${REPORT_ARGS[@]}" --out vlm/viz/REPORT.html
echo "DONE -> vlm/viz/REPORT.html"
