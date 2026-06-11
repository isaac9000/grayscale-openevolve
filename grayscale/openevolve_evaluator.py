"""
OpenEvolve evaluator for the grayscale kernel.

Single leaderboard-mode Modal call per candidate: correctness is checked
inside the evaluator before benchmarking, so broken kernels still fail fast
without a separate round-trip. Using one call avoids JIT-state variance that
occurred when stage1 (test) and stage2 (leaderboard) hit different containers.
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


def _run_eval(program_path: str):
    """Run run_eval.py in leaderboard mode. Returns (markdown_str, returncode, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        result = subprocess.run(
            [PYTHON, "run_eval.py", os.path.abspath(program_path), "-o", out_path, "--mode", "leaderboard"],
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


def evaluate(program_path: str) -> dict:
    """OpenEvolve entry point — single Modal call, correctness + benchmark."""
    md, rc, stderr = _run_eval(program_path)
    if md is None:
        error = stderr[:500] if stderr else f"run_eval exited {rc}"
        return {"score": 0.0, "error": error}

    m_tests = re.search(r"Passed (\d+)/(\d+) tests", md)
    tests_passed = int(m_tests.group(1)) if m_tests else 0
    tests_total = int(m_tests.group(2)) if m_tests else 1

    if tests_passed < tests_total:
        return {"score": 0.0, "pass_rate": tests_passed / tests_total, "error": f"correctness failed ({tests_passed}/{tests_total})"}

    m_geo = re.search(r"Geometric mean: ⏱ ([\d.]+)", md)
    if not m_geo:
        error = stderr[:500] if stderr else "benchmark not available"
        return {"score": 0.0, "pass_rate": 1.0, "error": error}

    geomean_us = float(m_geo.group(1))
    return {
        "score": 1e6 / geomean_us,
        "geomean_us": geomean_us,
        "pass_rate": 1.0,
    }
