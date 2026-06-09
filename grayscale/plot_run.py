"""
Plot OpenEvolve run progress from checkpoint data.

Usage:
    python plot_run.py <openevolve_output_dir>
    python plot_run.py grayscale/openevolve_runs/run1
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_programs(output_dir: str) -> list[dict]:
    """Collect all evaluated programs across all checkpoints, deduped by id."""
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        sys.exit(f"No checkpoints found in {output_dir}")

    programs: dict[str, dict] = {}
    for ckpt in sorted(checkpoints_dir.iterdir()):
        programs_dir = ckpt / "programs"
        if not programs_dir.exists():
            continue
        for f in programs_dir.glob("*.json"):
            with open(f) as fp:
                p = json.load(fp)
            programs[p["id"]] = p

    return list(programs.values())


def count_llm_calls(programs: list[dict]) -> int:
    """Each program with a parent_id required one LLM call to generate."""
    return sum(1 for p in programs if p.get("parent_id") is not None)


def plot_progress(output_dir: str) -> None:
    programs = load_programs(output_dir)
    if not programs:
        sys.exit("No programs found.")

    llm_calls = count_llm_calls(programs)

    # Sort by iteration_found so step line is correct
    programs.sort(key=lambda p: p.get("iteration_found", 0))

    iterations, geomeans, kinds = [], [], []
    best_so_far = float("inf")
    best_per_iter: list[tuple[int, float]] = []

    for p in programs:
        it = p.get("iteration_found", 0)
        score = p.get("metrics", {}).get("score", 0.0)
        gm = p.get("metrics", {}).get("geomean_us", 0.0)

        if score == 0.0 or gm == 0.0:
            iterations.append(it)
            geomeans.append(None)
            kinds.append("crash")
        else:
            iterations.append(it)
            geomeans.append(gm)
            if gm < best_so_far:
                best_so_far = gm
                kinds.append("keep")
            else:
                kinds.append("discard")

        best_per_iter.append((it, best_so_far if best_so_far < float("inf") else None))

    # --- Progress plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    all_valid_gm = [g for g in geomeans if g]
    y_lo = -(max(all_valid_gm) * 1.15) if all_valid_gm else -100
    y_hi = -(min(all_valid_gm) * 0.85) if all_valid_gm else 0

    keep_x = [it for it, k in zip(iterations, kinds) if k == "keep"]
    keep_y = [-geomeans[i] for i, k in enumerate(kinds) if k == "keep"]
    discard_x = [it for it, k in zip(iterations, kinds) if k == "discard"]
    discard_y = [-geomeans[i] for i, k in enumerate(kinds) if k == "discard"]
    crash_x = [it for it, k in zip(iterations, kinds) if k == "crash"]

    if keep_x:
        ax.scatter(keep_x, keep_y, c="#22c55e", s=60, zorder=5, label="keep",
                   edgecolors="white", linewidths=0.5)
    if discard_x:
        ax.scatter(discard_x, discard_y, c="#ef4444", s=40, zorder=4, label="discard",
                   edgecolors="white", linewidths=0.5, alpha=0.7)
    if crash_x:
        ax.scatter(crash_x, [y_lo] * len(crash_x), c="#fbbf24", s=25, zorder=3,
                   label=f"crash ({len(crash_x)})", marker="x", alpha=0.6)

    valid_best = [(it, -gm) for it, gm in best_per_iter if gm is not None]
    if valid_best:
        bx, by = zip(*valid_best)
        ax.step(bx, by, where="post", color="#3b82f6", linewidth=2, label="best time", zorder=6)

    ax.set_ylim(y_lo * 1.05, y_hi)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Iteration #", fontsize=12)
    ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
    ax.set_title("OpenEvolve grayscale — Evolution Progress", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if best_so_far < float("inf"):
        ax.annotate(
            f"Best: {best_so_far:.2f} μs",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
        )

    ax.annotate(
        f"LLM calls: {llm_calls}",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=10, color="#6b7280",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
    )

    fig.tight_layout()
    progress_path = os.path.join(output_dir, "progress.png")
    fig.savefig(progress_path, dpi=150)
    plt.close(fig)
    print(f"Saved {progress_path}")

    # --- Best-per-iteration plot ---
    iter_best: dict[int, float] = {}
    running_best = float("inf")
    for p in programs:
        it = p.get("iteration_found", 0)
        gm = p.get("metrics", {}).get("geomean_us", 0.0)
        if gm > 0 and gm < running_best:
            running_best = gm
        iter_best[it] = running_best

    if iter_best and min(iter_best.values()) < float("inf"):
        iters = sorted(iter_best)
        bests = [-iter_best[i] for i in iters]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.step(iters, bests, where="post", color="#3b82f6", linewidth=2)
        ax.scatter(iters, bests, c="#3b82f6", s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel("Iteration #", fontsize=12)
        ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
        ax.set_title("OpenEvolve grayscale — Best per Iteration", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        best_overall = min(iter_best.values())
        ax.annotate(
            f"Best: {best_overall:.2f} μs",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
        )

        ax.annotate(
            f"LLM calls: {llm_calls}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=10, color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
        )

        fig.tight_layout()
        iter_path = os.path.join(output_dir, "iterations.png")
        fig.savefig(iter_path, dpi=150)
        plt.close(fig)
        print(f"Saved {iter_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <openevolve_output_dir>")
        sys.exit(1)
    plot_progress(sys.argv[1])
