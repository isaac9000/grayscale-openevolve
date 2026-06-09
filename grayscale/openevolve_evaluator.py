"""
OpenEvolve evaluator for the grayscale kernel.

Stage 1: quick correctness check (--mode test) — gates the expensive benchmark.
Stage 2: full H100 benchmark (--mode leaderboard) — returns 1/geomean_us so
         higher score = lower latency (OpenEvolve maximizes).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def _run_eval(program_path: str, mode: str = "leaderboard"):
    """Run run_eval.py on program_path. Returns (markdown_str, returncode, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        result = subprocess.run(
            [PYTHON, "run_eval.py", os.path.abspath(program_path), "-o", out_path, "--mode", mode],
            capture_output=True,
            text=True,
            timeout=420,
            cwd=SCRIPT_DIR,
            env={**os.environ, "PYTHONPATH": SCRIPT_DIR},
        )
        if not os.path.exists(out_path):
            return None, result.returncode, result.stderr
        with open(out_path) as f:
            md = json.load(f)
        return md, result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        return None, -1, "eval timed out"
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def evaluate_stage1(program_path: str) -> dict:
    """Quick correctness check — gates expensive benchmarking."""
    md, rc, stderr = _run_eval(program_path, mode="test")
    if md is None or rc != 0:
        error = stderr[:500] if stderr else f"run_eval exited {rc}"
        return {"score": 0.0, "error": error}
    return {"score": 1.0}


def evaluate(program_path: str) -> dict:
    """OpenEvolve entry point — gates stage2 behind a correctness check."""
    stage1 = evaluate_stage1(program_path)
    if stage1["score"] == 0.0:
        return stage1
    return evaluate_stage2(program_path)


def evaluate_stage2(program_path: str) -> dict:
    """Full H100 benchmark — score = 1e6 / geomean_us (higher is faster)."""
    md, rc, stderr = _run_eval(program_path, mode="leaderboard")
    if md is None or rc != 0:
        error = stderr[:500] if stderr else f"run_eval exited {rc}"
        return {"score": 0.0, "geomean_us": 0.0, "error": error}
    m = re.search(r"Geometric mean: ⏱ ([\d.]+)", md)
    if not m:
        return {"score": 0.0, "geomean_us": 0.0, "error": "could not parse geomean from results"}
    geomean_us = float(m.group(1))
    return {
        "score": 1e6 / geomean_us,
        "geomean_us": geomean_us,
    }
