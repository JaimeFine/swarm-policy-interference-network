from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.pettingzoo_env import (
    run_internal_episode,
)
from src.simulation import pretrain_mlp

ARENA_SIZE = 100.0


def pretty_mode_name(mode: str) -> str:
    if mode == "dispersion":
        return "Dispersion"
    return mode.replace("_", " ").title()

def simulate_scenario(
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray | str | list[list[int]]]:
    return run_internal_episode(
        phi_omega=phi_omega,
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        seed=seed,
    )

def plot_overview(results: list[dict[str, np.ndarray | str | list[list[int]]]], outdir: Path) -> Path:
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
    fig.suptitle("SPIN-Exact Scenario Overview", fontsize=15, fontweight="bold")

    output_path = outdir / "trajectories.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

def plot_diagnostics_grid(
    results: list[dict[str, np.ndarray | str | list[list[int]]]], outdir: Path
) -> Path:
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

        topology_ax.plot(
            ticks,
            clique_counts,
            color="#7c3aed",
            linewidth=2.0,
            label="Clique count",
        )
        topology_ax.plot(
            ticks,
            mean_clique_sizes,
            color="#ea580c",
            linewidth=2.0,
            label="Mean clique size",
        )
        entropy_ax = topology_ax.twinx()
        entropy_ax.plot(
            ticks,
            entropy_curve,
            color="#059669",
            linewidth=1.8,
            linestyle="--",
            label="Mean policy entropy",
        )

        topology_ax.set_title(
            f"{mode_label}: Topology and Policy Diagnostics"
        )
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
        "SPIN-Exact Comparative Diagnostics Across Collective Regimes",
        fontsize=15,
        fontweight="bold",
    )
    output_path = outdir / "diagnostics.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

def save_metrics(result: dict[str, np.ndarray | str | list[list[int]]], outdir: Path) -> Path:
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
    )
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-friendly visualizations for the SPIN-Exact prototype."
    )
    parser.add_argument("--agents", type=int, default=10, help="Number of agents to simulate.")
    parser.add_argument("--steps", type=int, default=120, help="Number of control steps.")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Global random seed used for MLP pretraining and scenario initialization.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5000,
        help="Offline MLP pretraining epochs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory where figures and metric archives will be saved.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    phi_omega = pretrain_mlp(epochs=args.pretrain_epochs)

    results = []
    for offset, mode in enumerate(("tracking", "dispersion", "multi_goal")):
        result = simulate_scenario(
            phi_omega=phi_omega,
            mode=mode,
            n_agents=args.agents,
            steps=args.steps,
            seed=args.seed + offset * 101,
        )
        results.append(result)
        metric_path = save_metrics(result, args.outdir)
        print(f"Saved {mode} metrics to: {metric_path}")

    overview_path = plot_overview(results, args.outdir)
    diagnostics_path = plot_diagnostics_grid(results, args.outdir)
    print(f"Saved overview figure to: {overview_path}")
    print(f"Saved diagnostics figure to: {diagnostics_path}")


if __name__ == "__main__":
    main()
