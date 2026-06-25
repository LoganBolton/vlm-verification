# What we have right now

VLM port of "When Does Verification Pay Off?" — solvers and verifiers on **CountBench** and
**CharXiv**, 13 models across 4 families (Qwen3-VL, InternVL3.5, gemma-4, llava-1.5).

- **Solver accuracy** — done for all 13 models, both datasets (`base` runs, scored).
- **Three test-time-compute legs** (see `vlm/collate_results.py` → `ALL_RESULTS.csv` coverage matrix):
  - `maj`  — majority vote / self-consistency (`self_consistency.py`)
  - `judge` — VLM-as-verifier rejection sampling (`rejection_sampling.py`)
  - `zoom` — agentic-vision active perception (`agentic_vision.py`)
- **Key finding so far**: real verifiers rubber-stamp (precision ≈ base rate); only the oracle
  captures the resampling headroom.

## Missing (the current task)
- The full **N×N solver×verifier grid** needed for the paper's self / intra / cross-family
  verifier-gain split. Tooling is ready (`run_verifier_grid.sh` + `verifier_gain.py`); running
  it for **CharXiv** now. See `vlm/CLAUDE.md`.
