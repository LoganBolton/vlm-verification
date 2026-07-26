#!/usr/bin/env python3
"""Update the summarised blog (report/index.html) in place.

report/index.html is HAND-EDITED prose. This script surgically swaps ONLY the per-model summary
table for the AVERAGE-across-both-datasets, PERCENT-ONLY version (base acc kept as the anchor,
comparison columns shown as relative % change). All prose is preserved; a .bak is written first.

Re-run whenever the underlying data changes:  .venv/bin/python vlm/analysis/update_index_tables.py
"""
import re, shutil, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import build_charxiv_report as R

INDEX = "report/index.html"


def main():
    d = R.load_all()
    # Average-across-both-datasets, percent-only summary table (base acc retained as anchor).
    models = R.models_of(d["grid"]["avg"])
    tbl = R.render_summary(models, d["base"]["avg"], d["maj5"]["avg"],
                           d["cross"]["avg"], d["zoom"]["avg"], pct_only=True)

    html = open(INDEX).read()
    pat = re.compile(r"<table class='mx sum'.*?</table>", re.S)
    if not pat.search(html):
        sys.exit("!! could not find the active per-model summary table in index.html")
    shutil.copyfile(INDEX, INDEX + ".bak")
    new_html, n = pat.subn(tbl, html)
    # ensure the footer-row style exists in index.html's own <style> block
    avgrow_css = (".mx tr.avgrow td,.mx tr.avgrow th{border-top:2px solid #2c3e50;"
                  "background:#eef2f5;font-weight:700}")
    if "tr.avgrow" not in new_html:
        new_html = new_html.replace("</style>", "    " + avgrow_css + "\n    </style>", 1)
    open(INDEX, "w").write(new_html)
    print(f"updated {n} summary table(s) in {INDEX} (Average, percent-only). Backup: {INDEX}.bak")


if __name__ == "__main__":
    main()
