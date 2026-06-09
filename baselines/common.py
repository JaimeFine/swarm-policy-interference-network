from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from src.coverage import (
    compute_spatial_entropy,
    compute_voronoi_area_variance,
    directional_action_distribution,
)
from src.pettingzoo_env import (
    ARENA_SIZE,
    _make_env,
    compute_mean_trajectory_length,
    compute_task_metric,
    compute_target_distance_statistics,
    scenario_targets,
)
from src.spin_policy import R_SENSE
from src.topology import compute_adjacency_matrix, extract_overlapping_maximal_cliques

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


def _prefix_metric_keys(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


class BaselineController(Protocol):
    def reset(self, seed: int | None = None) -> None:
        ...

    def act(
        self,
        mode: str,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
        step_idx: int,
        agent_velocities: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        ...


def pretty_mode_name(mode: str) -> str:
    if mode == "dispersion":
        return "Dispersion"
    return mode.replace("_", " ").title()


def pretty_baseline_name(name: str) -> str:
    tokens = {
        "apf_velocity": "APF-Velocity",
        "cbba": "Distributed Auction-CBBA",
        "mappo": "MAPPO",
    }
    return tokens.get(name, name.replace("_", " ").title())


def policy_entropy(probs: np.ndarray) -> float:
    safe = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    safe /= np.sum(safe)
    return float(-np.sum(safe * np.log(safe)))


def vector_to_action_distribution(
    vector: np.ndarray,
    *,
    pinpoint_mass: float = 0.08,
) -> np.ndarray:
    directional = directional_action_distribution(np.asarray(vector, dtype=float))
    probs = np.zeros(5, dtype=float)
    probs[:4] = (1.0 - pinpoint_mass) * directional
    probs[4] = pinpoint_mass
    probs /= np.sum(probs)
    return probs


def action_from_distribution(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    action = np.zeros(5, dtype=np.float32)
    action[0] = float(np.clip(probs[4], 0.0, 1.0))
    action[1] = float(np.clip(probs[3], 0.0, 1.0))
    action[2] = float(np.clip(probs[2], 0.0, 1.0))
    action[3] = float(np.clip(probs[1], 0.0, 1.0))
    action[4] = float(np.clip(probs[0], 0.0, 1.0))
    return action


def vector_to_action(vector: np.ndarray, *, pinpoint_mass: float = 0.08) -> tuple[np.ndarray, float]:
    probs = vector_to_action_distribution(vector, pinpoint_mass=pinpoint_mass)
    return action_from_distribution(probs), policy_entropy(probs)


def rollout_controller(
    *,
    baseline_name: str,
    controller: BaselineController,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray | str | float | list[list[int]]]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    controller.reset(seed=seed)

    positions = env.agent_positions.copy()
    targets = scenario_targets(mode, env.landmark_positions)
    initial_task, task_label = compute_task_metric(mode, positions, targets)
    initial_distance_stats = compute_target_distance_statistics(mode, positions, targets)

    trajectories = np.empty((steps + 1, n_agents, 2), dtype=float)
    trajectories[0] = positions
    task_curve = np.empty(steps, dtype=float)
    clique_counts = np.empty(steps, dtype=float)
    mean_clique_sizes = np.empty(steps, dtype=float)
    entropy_curve = np.empty(steps, dtype=float)
    spatial_entropy_curve = np.empty(steps, dtype=float)
    voronoi_variance_curve = np.empty(steps, dtype=float)
    last_cliques: list[list[int]] = []

    for step in range(steps):
        positions = env.agent_positions.copy()
        landmark_positions = env.landmark_positions.copy()
        actions, diagnostics = controller.act(
            mode,
            positions,
            landmark_positions,
            step,
            agent_velocities=env.agent_velocities.copy(),
        )
        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        task_value, task_label = compute_task_metric(mode, positions, targets)

        adj = compute_adjacency_matrix(positions, R_SENSE)
        cliques = extract_overlapping_maximal_cliques(adj)
        clique_counts[step] = len(cliques)
        mean_clique_sizes[step] = float(np.mean([len(clique) for clique in cliques]))
        task_curve[step] = task_value
        entropy_curve[step] = float(diagnostics.get("entropy", 0.0))
        spatial_entropy_curve[step] = compute_spatial_entropy(positions, ARENA_SIZE)
        voronoi_variance_curve[step] = compute_voronoi_area_variance(positions, ARENA_SIZE)
        trajectories[step + 1] = positions
        last_cliques = cliques

        if all(terminations.values()) or all(truncations.values()):
            if step + 1 < steps:
                task_curve[step + 1 :] = task_curve[step]
                clique_counts[step + 1 :] = clique_counts[step]
                mean_clique_sizes[step + 1 :] = mean_clique_sizes[step]
                entropy_curve[step + 1 :] = entropy_curve[step]
                spatial_entropy_curve[step + 1 :] = spatial_entropy_curve[step]
                voronoi_variance_curve[step + 1 :] = voronoi_variance_curve[step]
                trajectories[step + 2 :] = positions
            break

    env.close()

    mean_trajectory_length = compute_mean_trajectory_length(trajectories)
    final_distance_stats = compute_target_distance_statistics(mode, positions, targets)

    return {
        "baseline": baseline_name,
        "mode": mode,
        "targets": targets,
        "trajectories": trajectories,
        "task_curve": task_curve,
        "task_label": task_label,
        "clique_counts": clique_counts,
        "mean_clique_sizes": mean_clique_sizes,
        "entropy_curve": entropy_curve,
        "spatial_entropy_curve": spatial_entropy_curve,
        "voronoi_variance_curve": voronoi_variance_curve,
        "final_cliques": last_cliques,
        "initial_task": float(initial_task),
        "final_task": float(task_curve[-1]),
        **_prefix_metric_keys("initial", initial_distance_stats),
        **_prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(entropy_curve[-1]),
        "final_clique_count": float(clique_counts[-1]),
        "final_mean_clique_size": float(mean_clique_sizes[-1]),
        "final_spatial_entropy": float(spatial_entropy_curve[-1]),
        "final_voronoi_area_variance": float(voronoi_variance_curve[-1]),
        "mean_trajectory_length": mean_trajectory_length,
    }


def summarize_rollout(result: dict[str, np.ndarray | str | float | list[list[int]]], seed: int, n_agents: int, steps: int) -> dict[str, float | int | str]:
    mode = str(result["mode"])
    initial_task = float(result["initial_task"])
    final_task = float(result["final_task"])
    improvement = initial_task - final_task if mode in {"tracking", "multi_goal"} else final_task - initial_task
    return {
        "baseline": str(result["baseline"]),
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "agents": n_agents,
        "metric_label": str(result["task_label"]),
        "initial_task": initial_task,
        "final_task": final_task,
        **{field: float(result[field]) for field in DISTANCE_STAT_FIELDS},
        "improvement": float(improvement),
        "final_entropy": float(result["final_entropy"]),
        "final_clique_count": float(result["final_clique_count"]),
        "final_mean_clique_size": float(result["final_mean_clique_size"]),
        "final_spatial_entropy": float(result["final_spatial_entropy"]),
        "final_voronoi_area_variance": float(result["final_voronoi_area_variance"]),
        "mean_trajectory_length": float(result["mean_trajectory_length"]),
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
        "baseline",
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
                    "baseline": mode_rows[0]["baseline"],
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


def save_metrics(result: dict[str, np.ndarray | str | float | list[list[int]]], outdir: Path) -> Path:
    mode = str(result["mode"])
    output_path = outdir / f"{mode}_metrics.npz"
    np.savez(
        output_path,
        trajectories=np.asarray(result["trajectories"]),
        targets=np.asarray(result["targets"]),
        task_curve=np.asarray(result["task_curve"]),
        clique_counts=np.asarray(result["clique_counts"]),
        mean_clique_sizes=np.asarray(result["mean_clique_sizes"]),
        entropy_curve=np.asarray(result["entropy_curve"]),
        spatial_entropy_curve=np.asarray(result["spatial_entropy_curve"]),
        voronoi_variance_curve=np.asarray(result["voronoi_variance_curve"]),
        mean_trajectory_length=np.asarray(result["mean_trajectory_length"]),
        **{field: np.asarray(result[field]) for field in DISTANCE_STAT_FIELDS},
    )
    return output_path


def plot_trajectories(results: list[dict[str, np.ndarray | str | float | list[list[int]]]], outdir: Path, baseline_name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    for ax, result in zip(axes, results):
        mode = str(result["mode"])
        trajectories = np.asarray(result["trajectories"])
        targets = np.asarray(result["targets"])

        for agent_idx in range(trajectories.shape[1]):
            xy = trajectories[:, agent_idx, :]
            ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9, alpha=0.45)

        ax.scatter(
            trajectories[-1, :, 0],
            trajectories[-1, :, 1],
            s=34,
            c="#0f766e",
            edgecolors="white",
            linewidths=0.5,
        )
        if len(targets) > 0:
            ax.scatter(
                targets[:, 0],
                targets[:, 1],
                s=150,
                c="#dc2626",
                marker="*",
                edgecolors="black",
                linewidths=0.6,
            )

        ax.set_xlim(0, ARENA_SIZE)
        ax.set_ylim(0, ARENA_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(pretty_mode_name(mode))
        ax.set_xlabel("x-position")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("y-position")
    fig.suptitle(f"{pretty_baseline_name(baseline_name)} Scenario Overview", fontsize=15, fontweight="bold")

    output_path = outdir / "trajectories.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_diagnostics(results: list[dict[str, np.ndarray | str | float | list[list[int]]]], outdir: Path, baseline_name: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)

    for col, result in enumerate(results):
        mode = str(result["mode"])
        task_curve = np.asarray(result["task_curve"])
        clique_counts = np.asarray(result["clique_counts"])
        mean_clique_sizes = np.asarray(result["mean_clique_sizes"])
        entropy_curve = np.asarray(result["entropy_curve"])
        task_label = str(result["task_label"])
        ticks = np.arange(1, len(task_curve) + 1)
        mode_label = pretty_mode_name(mode)

        metric_ax = axes[0, col]
        topology_ax = axes[1, col]

        metric_ax.plot(ticks, task_curve, color="#1d4ed8", linewidth=2.2)
        metric_ax.set_title(f"{mode_label}: {task_label}")
        metric_ax.set_xlabel("Control step")
        metric_ax.set_ylabel(task_label)
        metric_ax.grid(alpha=0.25)

        topology_ax.plot(ticks, clique_counts, color="#7c3aed", linewidth=2.0, label="Clique count")
        topology_ax.plot(ticks, mean_clique_sizes, color="#ea580c", linewidth=2.0, label="Mean clique size")
        entropy_ax = topology_ax.twinx()
        entropy_ax.plot(ticks, entropy_curve, color="#059669", linewidth=1.8, linestyle="--", label="Mean policy entropy")

        topology_ax.set_title(f"{mode_label}: Topology and Policy Diagnostics")
        topology_ax.set_xlabel("Control step")
        topology_ax.set_ylabel("Topology metrics")
        entropy_ax.set_ylabel("Entropy")
        topology_ax.grid(alpha=0.25)

        left_handles, left_labels = topology_ax.get_legend_handles_labels()
        right_handles, right_labels = entropy_ax.get_legend_handles_labels()
        topology_ax.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="upper right",
            fontsize=8.5,
            frameon=True,
        )

    fig.suptitle(
        f"{pretty_baseline_name(baseline_name)} Comparative Diagnostics Across Collective Regimes",
        fontsize=15,
        fontweight="bold",
    )
    output_path = outdir / "diagnostics.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
