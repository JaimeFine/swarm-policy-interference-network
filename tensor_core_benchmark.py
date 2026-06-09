from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.quantum import LocalQuantumState
from src.tensor import (
    build_clique_mps,
    compute_reduced_density_matrices_by_enumeration,
    compute_reduced_density_matrices_from_mps,
)

OUTDIR = Path(__file__).resolve().parent / "tensor_benchmark_results"
LOCAL_DIM = 5
CLIQUE_SIZES = (2, 3, 4, 5, 6)
MODES = ("tracking", "dispersion", "multi_goal")
TRIALS = 20
REPEATS = 12
SEED = 7


def pretty_mode_name(mode: str) -> str:
    if mode == "multi_goal":
        return "Multi-Goal"
    if mode == "dispersion":
        return "Dispersion"
    return "Tracking"


def make_random_states(rng: np.random.Generator, clique_size: int) -> list[LocalQuantumState]:
    states: list[LocalQuantumState] = []
    for _ in range(clique_size):
        state = LocalQuantumState(LOCAL_DIM)
        amplitudes = rng.normal(size=LOCAL_DIM) + 1j * rng.normal(size=LOCAL_DIM)
        norm = np.linalg.norm(amplitudes)
        if norm <= 1e-12:
            amplitudes = np.full(LOCAL_DIM, 1.0 / np.sqrt(LOCAL_DIM), dtype=np.complex128)
        else:
            amplitudes = amplitudes / norm
        state.amplitudes = amplitudes.astype(np.complex128)
        state.density_matrix = np.outer(state.amplitudes, np.conj(state.amplitudes))
        states.append(state)
    return states


def make_positions(rng: np.random.Generator, clique_size: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, clique_size, endpoint=False)
    radius = 8.0 + 0.25 * clique_size
    center = np.array([50.0, 50.0], dtype=float)
    base = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    jitter = rng.normal(scale=0.6, size=(clique_size, 2))
    return center + base + jitter


def time_callable(fn, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / repeats


def run_benchmark() -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []

    total_jobs = len(MODES) * len(CLIQUE_SIZES) * TRIALS
    completed_jobs = 0

    for mode in MODES:
        for clique_size in CLIQUE_SIZES:
            clique = list(range(clique_size))
            for trial in range(TRIALS):
                completed_jobs += 1
                print(
                    f"[{completed_jobs}/{total_jobs}] tensor benchmark: "
                    f"mode={mode}, clique={clique_size}, trial={trial + 1}/{TRIALS}"
                )
                states = make_random_states(rng, clique_size)
                positions = make_positions(rng, clique_size)
                cores = build_clique_mps(states, clique, positions, mode=mode)

                mps_ms = time_callable(
                    lambda: compute_reduced_density_matrices_from_mps(cores),
                    repeats=REPEATS,
                )
                enum_ms = time_callable(
                    lambda: compute_reduced_density_matrices_by_enumeration(cores),
                    repeats=REPEATS,
                )

                rows.append(
                    {
                        "mode": mode,
                        "clique_size": clique_size,
                        "trial": trial,
                        "mps_reduction_ms": mps_ms,
                        "enumeration_reduction_ms": enum_ms,
                        "speedup": enum_ms / max(mps_ms, 1e-12),
                    }
                )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["clique_size"]))].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (mode, clique_size), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary.append(
            {
                "mode": mode,
                "clique_size": clique_size,
                "trials": len(group),
                "mps_reduction_ms_mean": float(np.mean([float(r["mps_reduction_ms"]) for r in group])),
                "mps_reduction_ms_std": float(np.std([float(r["mps_reduction_ms"]) for r in group])),
                "enumeration_reduction_ms_mean": float(np.mean([float(r["enumeration_reduction_ms"]) for r in group])),
                "enumeration_reduction_ms_std": float(np.std([float(r["enumeration_reduction_ms"]) for r in group])),
                "speedup_mean": float(np.mean([float(r["speedup"]) for r in group])),
                "speedup_std": float(np.std([float(r["speedup"]) for r in group])),
            }
        )
    return summary


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)

    colors = {
        "tracking": "#1d4ed8",
        "dispersion": "#059669",
        "multi_goal": "#dc2626",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), constrained_layout=True)

    for col_idx, mode in enumerate(MODES):
        runtime_ax = axes[0, col_idx]
        speedup_ax = axes[1, col_idx]

        mode_rows = sorted(grouped[mode], key=lambda row: int(row["clique_size"]))
        clique_sizes = [int(row["clique_size"]) for row in mode_rows]
        mps_mean = [float(row["mps_reduction_ms_mean"]) for row in mode_rows]
        mps_std = [float(row["mps_reduction_ms_std"]) for row in mode_rows]
        enum_mean = [float(row["enumeration_reduction_ms_mean"]) for row in mode_rows]
        enum_std = [float(row["enumeration_reduction_ms_std"]) for row in mode_rows]
        speedup_mean = [float(row["speedup_mean"]) for row in mode_rows]
        speedup_std = [float(row["speedup_std"]) for row in mode_rows]

        runtime_ax.errorbar(
            clique_sizes,
            mps_mean,
            yerr=mps_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=colors[mode],
            label="MPS contraction",
        )
        runtime_ax.errorbar(
            clique_sizes,
            enum_mean,
            yerr=enum_std,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            capsize=3,
            color="#111827",
            alpha=0.9,
            label="Explicit enumeration",
        )
        runtime_ax.set_title(f"{pretty_mode_name(mode)}: Reduction Runtime")
        runtime_ax.set_xlabel("Clique size")
        runtime_ax.set_ylabel("Mean reduction time (ms)")
        runtime_ax.set_yscale("log")
        runtime_ax.grid(alpha=0.25, which="both")
        runtime_ax.legend(frameon=True, fontsize=8.5)

        speedup_ax.errorbar(
            clique_sizes,
            speedup_mean,
            yerr=speedup_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=colors[mode],
        )
        speedup_ax.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle=":")
        speedup_ax.set_title(f"{pretty_mode_name(mode)}: Enumeration / MPS Speedup")
        speedup_ax.set_xlabel("Clique size")
        speedup_ax.set_ylabel("Mean speedup factor")
        speedup_ax.grid(alpha=0.25)

    fig.suptitle("SPIN Tensor-Core Microbenchmark", fontsize=16, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = run_benchmark()
    summary_rows = summarize(rows)

    raw_csv = OUTDIR / "tensor_core_raw.csv"
    summary_csv = OUTDIR / "tensor_core_summary.csv"
    figure_path = OUTDIR / "tensor_core_benchmark.pdf"

    write_csv(rows, raw_csv)
    write_csv(summary_rows, summary_csv)
    plot_summary(summary_rows, figure_path)

    print(f"Saved tensor-core raw CSV to: {raw_csv}")
    print(f"Saved tensor-core summary CSV to: {summary_csv}")
    print(f"Saved tensor-core figure to: {figure_path}")


if __name__ == "__main__":
    main()
