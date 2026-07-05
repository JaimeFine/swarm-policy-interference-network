from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "noise_robustness_results" / "noise_robustness_raw.csv"
OUTPUT_PDF = ROOT / "noise_robustness_results" / "noise_degradation_comparison.pdf"
OUTPUT_PNG = ROOT / "noise_robustness_results" / "noise_degradation_comparison.png"

METHOD_ORDER = ("spin", "apf_velocity", "cbba", "mappo")
MODE_ORDER = ("tracking", "multi_goal")
METHOD_LABELS = {
    "spin": "SPIN",
    "apf_velocity": "APF-Velocity",
    "cbba": "CBBA",
    "mappo": "MAPPO",
}
MODE_LABELS = {
    "tracking": "Tracking",
    "multi_goal": "Multi-Goal",
}
COLORS = {
    "spin": "#1d4ed8",
    "apf_velocity": "#059669",
    "cbba": "#dc2626",
    "mappo": "#7c3aed",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str, float], dict[str, float]]:
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["mode"], float(row["sigma"]))].append(row)

    stats = {}
    for key, group in grouped.items():
        final_tasks = np.array([float(row["final_task"]) for row in group], dtype=float)
        entropies = np.array([float(row["final_entropy"]) for row in group], dtype=float)
        stats[key] = {
            "task_mean": float(np.mean(final_tasks)),
            "task_std": float(np.std(final_tasks)),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
        }
    return stats


def normalized_task_degradation(
    stats: dict[tuple[str, str, float], dict[str, float]],
    method: str,
    mode: str,
    sigma: float,
) -> float:
    baseline = max(stats[(method, mode, 0.0)]["task_mean"], 1e-9)
    current = stats[(method, mode, sigma)]["task_mean"]
    return (current - baseline) / baseline


def entropy_delta(
    stats: dict[tuple[str, str, float], dict[str, float]],
    method: str,
    mode: str,
    sigma: float,
) -> float:
    baseline = stats[(method, mode, 0.0)]["entropy_mean"]
    current = stats[(method, mode, sigma)]["entropy_mean"]
    return current - baseline


def plot(rows: list[dict[str, str]], pdf_path: Path, png_path: Path) -> None:
    stats = grouped_stats(rows)
    sigmas = sorted({float(row["sigma"]) for row in rows})

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)

    for col_idx, mode in enumerate(MODE_ORDER):
        task_ax = axes[0, col_idx]
        entropy_ax = axes[1, col_idx]

        for method in METHOD_ORDER:
            task_values = [
                normalized_task_degradation(stats, method, mode, sigma)
                for sigma in sigmas
            ]
            entropy_values = [
                entropy_delta(stats, method, mode, sigma)
                for sigma in sigmas
            ]

            task_ax.plot(
                sigmas,
                task_values,
                marker="o",
                linewidth=2.2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            entropy_ax.plot(
                sigmas,
                entropy_values,
                marker="o",
                linewidth=2.2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )

        task_ax.axhline(0.0, color="#475569", linewidth=1.0, alpha=0.6)
        task_ax.set_title(f"{MODE_LABELS[mode]}: Normalized Task Degradation")
        task_ax.set_xlabel("Noise sigma")
        task_ax.set_ylabel(r"$(J_\sigma - J_0) / J_0$")
        task_ax.grid(alpha=0.25)
        task_ax.legend(frameon=True)

        entropy_ax.axhline(0.0, color="#475569", linewidth=1.0, alpha=0.6)
        entropy_ax.set_title(f"{MODE_LABELS[mode]}: Entropy Change")
        entropy_ax.set_xlabel("Noise sigma")
        entropy_ax.set_ylabel(r"$H_\sigma - H_0$")
        entropy_ax.grid(alpha=0.25)
        entropy_ax.legend(frameon=True)

    fig.suptitle(
        "Noise Robustness: Normalized Task Degradation and Entropy Shift",
        fontsize=15,
        fontweight="bold",
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows(INPUT_CSV)
    required = {"method", "mode", "sigma", "final_task", "final_entropy"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"{INPUT_CSV} must contain columns: {', '.join(sorted(required))}"
        )
    plot(rows, OUTPUT_PDF, OUTPUT_PNG)
    print(f"Saved noise degradation PDF to: {OUTPUT_PDF}")
    print(f"Saved noise degradation PNG to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
