# vlm-verification

Analysis of which test-time compute methods help small VLMs.

**Full write-up: https://loganbolton.github.io/blog/vlm-verification/**


- `vlm/` — all experiment code: solving (`vlm_inference.py`), verification (`vlm_verify.py`), rejection sampling, self-consistency, agentic zoom (`agentic_vision.py`), scoring, and plotting
- `vlm/runs/` — the shell scripts that drove the actual runs
- `prompts/` — solver / verifier / zoom-agent prompts
- `scripts/`, `data/` — dataset download scripts and metadata
- `report/` — source for the write-up and its figures
- `src/` — unmodified text-LLM code from the upstream repo

Raw solver/verifier logs from all runs: [loganbolton/vlm-verification-logs](https://huggingface.co/datasets/loganbolton/vlm-verification-logs) on Hugging Face.

Forked from [agentic-learning-ai-lab/llm-verification](https://github.com/agentic-learning-ai-lab/llm-verification) ([Lu et al., 2025](https://arxiv.org/abs/2512.02304)).
