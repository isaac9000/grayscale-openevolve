# Advisor-Worker Autoresearch

An advisor-worker agent pair that iteratively optimizes CUDA kernels. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on a cloud GPU via Modal, and logs the result.

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

---

## Problem: Grayscale (A100)

Convert an RGB image to grayscale using `Y = 0.2989 R + 0.5870 G + 0.1140 B`.

- **Input:** `(H, W, 3)` float32 tensor on CUDA
- **Output:** `(H, W)` float32 tensor
- **Benchmark sizes:** 512, 1024, 2048, 4096, 8192, 16384 (square images)
- **Target GPU:** A100 via Modal
- **Metric:** Geometric mean latency across all 6 sizes

### Deploy the evaluator (once)

```bash
uv run modal deploy eval_modal_grayscale.py
```

### Run the agent

```bash
uv run grayscale/agent.py --iterations 20
```

Start from a specific baseline:

```bash
uv run grayscale/agent.py --baseline grayscale/submission.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run grayscale/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

In tmux (recommended for long runs):

```bash
tmux new-session -d -s agent "set -a && source .env && set +a && uv run grayscale/agent.py --iterations 25 2>&1 | tee grayscale/agent_run.log"
tmux attach -t agent
```

Evaluate a kernel file without running the agent:

```bash
uv run grayscale/run_eval.py grayscale/submission.py -o results.json
```

### Structure

```
grayscale/
├── agent.py            — advisor-worker agentic loop
├── advisor_prompt.md   — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md    — worker system prompt: mandatory sequence, rules
├── submission.py       — the kernel file the worker edits each iteration
├── run_eval.py         — submits submission.py to the Modal A100 evaluator
├── tools.py            — log_experiment and get_experiment_history tools
└── runs/               — one directory per run: history, TSV log, plots, best submission
eval_modal_grayscale.py — Modal app deployed on A100
```

---

## Problem: NVfp4 GEMV (B200)

Batched NVfp4 matrix-vector multiplication with fp8 block scale factors, producing fp16 output. Ranked by geometric mean latency across three shapes.

- **Target GPU:** B200 via Modal
- **Benchmark shapes:** (M=7168, K=16384, L=1), (M=4096, K=7168, L=8), (M=7168, K=2048, L=4)

### Deploy the evaluator (once)

```bash
uv run modal deploy eval_modal_nvfp4_gemv.py
```

### Run the agent

```bash
uv run nvfp4_gemv/agent.py --iterations 20
uv run nvfp4_gemv/agent.py --baseline nvfp4_gemv/baseline37.py --iterations 20
uv run nvfp4_gemv/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

### Structure

```
nvfp4_gemv/
├── agent.py            — advisor-worker agentic loop
├── advisor_prompt.md   — advisor system prompt
├── worker_prompt.md    — worker system prompt
├── submission.py       — the kernel file the worker edits each iteration
├── run_eval.py         — submits submission.py to the Modal B200 evaluator
├── tools.py            — log_experiment and get_experiment_history tools
├── baseline37.py       — custom CUDA kernel baseline (~21 µs geomean)
└── runs/               — one directory per run
eval_modal_nvfp4_gemv.py — Modal app deployed on B200
```

---

## Run directory contents

Each run directory under `*/runs/` contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency plot updated each iteration
- `iterations.png` — best-per-advisor-iteration plot
- `best_submission.py` — snapshot of the fastest kernel found
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it
