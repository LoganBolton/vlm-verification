#!/usr/bin/env bash
# Build all figures, per-example HTML viewers, and the combined HTML report from the
# result JSON produced by run_full_pipeline.sh. Safe to re-run anytime (reads results only).
#
# Usage:  bash vlm/build_report.sh
set -u
PY=.venv/bin/python   # plotting/report deps (matplotlib) live in the base venv
VIEW_LIMIT="${VIEW_LIMIT:-200}"   # cap examples per HTML viewer so files stay openable

# label : result_dir : dataset_label_for_titles
DATASETS=("countbench:vlm/result/countbench:CountBenchQA"
          "charxiv:vlm/result/charxiv:CharXiv")

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
done

echo "=== combined report ==="
$PY vlm/viz/make_report.py \
  --dataset "CountBenchQA:vlm/result/countbench:vlm/viz/figures/countbench:views/countbench" \
  --dataset "CharXiv:vlm/result/charxiv:vlm/viz/figures/charxiv:views/charxiv" \
  --out vlm/viz/REPORT.html
echo "DONE -> vlm/viz/REPORT.html"
