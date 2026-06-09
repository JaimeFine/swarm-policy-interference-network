from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.pettingzoo_env import (
    compute_task_metric,
    run_internal_episode,
)
from src.simulation import pretrain_mlp

DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results"
DISTANCE_STAT_FIELDS = (
    "initial_distance_mean",
    "initial_distance_variance",
    "initial_distance_p90",
    "initial_distance_p90_tail_mean",
    "initial_distance_mad",
    "final_distance_mean",
    "final_distance_variance",
    "final_distance_p90",
    "final_distance_p90_tail_mean",
    "final_distance_mad",
)


def simulate_trial(
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, float | int | str]:
    rollout = run_internal_episode(
        phi_omega=phi_omega,
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        seed=seed,
    )
    initial_task = float(rollout["initial_task"])
    final_task = float(rollout["final_task"])
    final_entropy = float(rollout["final_entropy"])
    final_clique_count = float(rollout["final_clique_count"])
    final_mean_clique_size = float(rollout["final_mean_clique_size"])
    final_spatial_entropy = float(rollout["final_spatial_entropy"])
    final_voronoi_area_variance = float(rollout["final_voronoi_area_variance"])
    mean_trajectory_length = float(rollout["mean_trajectory_length"])
    task_label = str(rollout["task_label"])

    improvement = (
        initial_task - final_task if mode in {"tracking", "multi_goal"} else final_task - initial_task
    )
    return {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "agents": n_agents,
        "metric_label": task_label,
        "initial_task": float(initial_task),
        "final_task": float(final_task),
        **{field: float(rollout[field]) for field in DISTANCE_STAT_FIELDS},
        "improvement": float(improvement),
        "final_entropy": float(final_entropy),
        "final_clique_count": float(final_clique_count),
        "final_mean_clique_size": float(final_mean_clique_size),
        "final_spatial_entropy": float(final_spatial_entropy),
        "final_voronoi_area_variance": float(final_voronoi_area_variance),
        "mean_trajectory_length": float(mean_trajectory_length),
    }


def write_trial_csv(rows: list[dict[str, float | int | str]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "trial_results.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_summary_csv(rows: list[dict[str, float | int | str]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "trial_summary.csv"
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)

    fieldnames = [
        "mode",
        "metric_label",
        "trials",
        "initial_task_mean",
        "final_task_mean",
        "initial_distance_mean_mean",
        "initial_distance_variance_mean",
        "initial_distance_p90_mean",
        "initial_distance_p90_tail_mean_mean",
        "initial_distance_mad_mean",
        "final_distance_mean_mean",
        "final_distance_variance_mean",
        "final_distance_p90_mean",
        "final_distance_p90_tail_mean_mean",
        "final_distance_mad_mean",
        "improvement_mean",
        "improvement_std",
        "final_entropy_mean",
        "final_clique_count_mean",
        "final_mean_clique_size_mean",
        "final_spatial_entropy_mean",
        "final_voronoi_area_variance_mean",
        "mean_trajectory_length_mean",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, mode_rows in grouped.items():
            writer.writerow(
                {
                    "mode": mode,
                    "metric_label": mode_rows[0]["metric_label"],
                    "trials": len(mode_rows),
                    "initial_task_mean": f"{np.mean([float(r['initial_task']) for r in mode_rows]):.3f}",
                    "final_task_mean": f"{np.mean([float(r['final_task']) for r in mode_rows]):.3f}",
                    **{
                        f"{field}_mean": f"{np.mean([float(r[field]) for r in mode_rows]):.3f}"
                        for field in DISTANCE_STAT_FIELDS
                    },
                    "improvement_mean": f"{np.mean([float(r['improvement']) for r in mode_rows]):.3f}",
                    "improvement_std": f"{np.std([float(r['improvement']) for r in mode_rows]):.3f}",
                    "final_entropy_mean": f"{np.mean([float(r['final_entropy']) for r in mode_rows]):.3f}",
                    "final_clique_count_mean": f"{np.mean([float(r['final_clique_count']) for r in mode_rows]):.3f}",
                    "final_mean_clique_size_mean": f"{np.mean([float(r['final_mean_clique_size']) for r in mode_rows]):.3f}",
                    "final_spatial_entropy_mean": f"{np.mean([float(r['final_spatial_entropy']) for r in mode_rows]):.3f}",
                    "final_voronoi_area_variance_mean": f"{np.mean([float(r['final_voronoi_area_variance']) for r in mode_rows]):.3f}",
                    "mean_trajectory_length_mean": f"{np.mean([float(r['mean_trajectory_length']) for r in mode_rows]):.3f}",
                }
            )
    return summary_path


def main() -> None:
    trials = 5
    n_agents = 10
    steps = 120
    base_seed = 7

    np.random.seed(base_seed)
    phi_omega = pretrain_mlp(epochs=5000)

    rows = []
    for trial_idx in range(trials):
        for mode_idx, mode in enumerate(("tracking", "dispersion", "multi_goal")):
            seed = base_seed + trial_idx * 1000 + mode_idx * 101
            print(f"Running {mode} trial {trial_idx + 1}/{trials} with seed {seed}...")
            rows.append(
                simulate_trial(
                    phi_omega=phi_omega,
                    mode=mode,
                    n_agents=n_agents,
                    steps=steps,
                    seed=seed,
                )
            )

    trial_csv = write_trial_csv(rows, DEFAULT_OUTDIR)
    summary_csv = write_summary_csv(rows, DEFAULT_OUTDIR)
    print(f"Saved trial-level CSV to: {trial_csv}")
    print(f"Saved summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
