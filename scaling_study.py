from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pettingzoo_env import (
    _make_env,
    compute_target_distance_statistics,
    compute_task_metric,
    scenario_targets,
)
from src.simulation import pretrain_mlp
from src.spin_policy import SpinPolicyController

DEFAULT_AGENT_COUNTS = (4, 8, 10, 16, 25, 32)
DEFAULT_MODES = ("tracking", "dispersion", "multi_goal")
DEFAULT_BASE_SEED = 7
DISTANCE_STAT_FIELDS = (
    "final_distance_mean",
    "final_distance_variance",
    "final_distance_p90",
    "final_distance_p90_tail_mean",
    "final_distance_mad",
)


def prefix_metric_keys(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def trials_for_agent_count(n_agents: int) -> int:
    return 5 if n_agents <= 16 else 3


def pretty_mode_name(mode: str) -> str:
    if mode == "multi_goal":
        return "Multi-Goal"
    if mode == "dispersion":
        return "Dispersion"
    return "Tracking"


def metric_label_for_mode(mode: str) -> str:
    if mode == "dispersion":
        return "Final spatial entropy"
    return "Final mean target distance"


def run_timed_rollout(
    *,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, float | int | str]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    controller = SpinPolicyController(phi_omega=phi_omega, n_agents=n_agents)
    controller.reset()

    controller_step_times: list[float] = []
    full_step_times: list[float] = []
    final_task = None
    final_distance_stats = compute_target_distance_statistics(
        mode, env.agent_positions.copy(), scenario_targets(mode, env.landmark_positions)
    )

    rollout_start = time.perf_counter()
    for step in range(steps):
        step_start = time.perf_counter()

        positions = env.agent_positions.copy()
        landmark_positions = env.landmark_positions.copy()

        control_start = time.perf_counter()
        actions, _ = controller.act(mode, positions, landmark_positions)
        controller_step_times.append(time.perf_counter() - control_start)

        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        final_task, _ = compute_task_metric(mode, positions, targets)
        final_distance_stats = compute_target_distance_statistics(mode, positions, targets)
        full_step_times.append(time.perf_counter() - step_start)

        if all(terminations.values()) or all(truncations.values()):
            break

    rollout_time = time.perf_counter() - rollout_start
    env.close()

    realized_steps = len(full_step_times)
    if final_task is None:
        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        final_task, _ = compute_task_metric(mode, positions, targets)
        final_distance_stats = compute_target_distance_statistics(mode, positions, targets)

    return {
        "mode": mode,
        "agents": n_agents,
        "steps": realized_steps,
        "seed": seed,
        "controller_step_ms": 1000.0 * float(np.mean(controller_step_times)),
        "full_step_ms": 1000.0 * float(np.mean(full_step_times)),
        "rollout_seconds": float(rollout_time),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "metric_label": metric_label_for_mode(mode),
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["agents"]))].append(row)

    summary = []
    for (mode, agents), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary.append(
            {
                "mode": mode,
                "agents": agents,
                "trials": len(group),
                "controller_step_ms_mean": float(np.mean([float(r["controller_step_ms"]) for r in group])),
                "controller_step_ms_std": float(np.std([float(r["controller_step_ms"]) for r in group])),
                "full_step_ms_mean": float(np.mean([float(r["full_step_ms"]) for r in group])),
                "full_step_ms_std": float(np.std([float(r["full_step_ms"]) for r in group])),
                "rollout_seconds_mean": float(np.mean([float(r["rollout_seconds"]) for r in group])),
                "rollout_seconds_std": float(np.std([float(r["rollout_seconds"]) for r in group])),
                "final_task_mean": float(np.mean([float(r["final_task"]) for r in group])),
                "final_task_std": float(np.std([float(r["final_task"]) for r in group])),
                **{
                    f"{field}_mean": float(np.mean([float(r[field]) for r in group]))
                    for field in DISTANCE_STAT_FIELDS
                },
                **{
                    f"{field}_std": float(np.std([float(r[field]) for r in group]))
                    for field in DISTANCE_STAT_FIELDS
                },
                "metric_label": group[0]["metric_label"],
            }
        )
    return summary


def plot_scaling(summary_rows: list[dict[str, float | int | str]], path: Path) -> None:
    colors = {
        "tracking": "#1d4ed8",
        "dispersion": "#059669",
        "multi_goal": "#dc2626",
    }
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["mode"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)

    runtime_ax = axes[0, 0]
    full_ax = axes[0, 1]
    target_ax = axes[1, 0]
    dispersion_ax = axes[1, 1]

    for mode in DEFAULT_MODES:
        rows = sorted(grouped[mode], key=lambda row: int(row["agents"]))
        agent_counts = [int(row["agents"]) for row in rows]
        controller_means = [float(row["controller_step_ms_mean"]) for row in rows]
        controller_stds = [float(row["controller_step_ms_std"]) for row in rows]
        full_means = [float(row["full_step_ms_mean"]) for row in rows]
        full_stds = [float(row["full_step_ms_std"]) for row in rows]
        final_means = [float(row["final_task_mean"]) for row in rows]
        final_stds = [float(row["final_task_std"]) for row in rows]

        runtime_ax.errorbar(
            agent_counts,
            controller_means,
            yerr=controller_stds,
            marker="o",
            linewidth=2.0,
            capsize=3,
            color=colors[mode],
            label=pretty_mode_name(mode),
        )
        full_ax.errorbar(
            agent_counts,
            full_means,
            yerr=full_stds,
            marker="o",
            linewidth=2.0,
            capsize=3,
            color=colors[mode],
            label=pretty_mode_name(mode),
        )

        if mode == "dispersion":
            dispersion_ax.errorbar(
                agent_counts,
                final_means,
                yerr=final_stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[mode],
            )
        else:
            target_ax.errorbar(
                agent_counts,
                final_means,
                yerr=final_stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[mode],
                label=pretty_mode_name(mode),
            )

    runtime_ax.set_title("Controller Runtime Scaling")
    runtime_ax.set_xlabel("Number of agents")
    runtime_ax.set_ylabel("Mean controller time per step (ms)")
    runtime_ax.grid(alpha=0.25)
    runtime_ax.legend(frameon=True)

    full_ax.set_title("Full Loop Runtime Scaling")
    full_ax.set_xlabel("Number of agents")
    full_ax.set_ylabel("Mean full step time (ms)")
    full_ax.grid(alpha=0.25)
    full_ax.legend(frameon=True)

    target_ax.set_title("Task Quality Scaling: Tracking and Multi-Goal")
    target_ax.set_xlabel("Number of agents")
    target_ax.set_ylabel("Final mean target distance")
    target_ax.grid(alpha=0.25)
    target_ax.legend(frameon=True)

    dispersion_ax.set_title("Task Quality Scaling: Dispersion")
    dispersion_ax.set_xlabel("Number of agents")
    dispersion_ax.set_ylabel("Final spatial entropy")
    dispersion_ax.grid(alpha=0.25)

    fig.suptitle("SPIN Agent-Scaling Study", fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent-scaling study for the SPIN controller."
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_AGENT_COUNTS),
        help="Agent counts to evaluate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Control steps per rollout.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5000,
        help="Offline MLP pretraining epochs.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base seed for the study.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "scaling_results",
        help="Directory for the scaling-study outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.base_seed)
    phi_omega = pretrain_mlp(epochs=args.pretrain_epochs)

    raw_rows: list[dict[str, float | int | str]] = []
    total_jobs = sum(trials_for_agent_count(n_agents) * len(DEFAULT_MODES) for n_agents in args.agent_counts)
    completed_jobs = 0
    study_start = time.perf_counter()

    for n_agents in args.agent_counts:
        trials = trials_for_agent_count(n_agents)
        for trial_idx in range(trials):
            for mode_idx, mode in enumerate(DEFAULT_MODES):
                seed = args.base_seed + trial_idx * 1000 + mode_idx * 101 + n_agents * 10000
                print(
                    f"[{completed_jobs + 1}/{total_jobs}] scaling run: "
                    f"agents={n_agents}, mode={mode}, trial={trial_idx + 1}/{trials}, seed={seed}"
                )
                job_start = time.perf_counter()
                raw_rows.append(
                    run_timed_rollout(
                        phi_omega=phi_omega,
                        mode=mode,
                        n_agents=n_agents,
                        steps=args.steps,
                        seed=seed,
                    )
                )
                completed_jobs += 1
                elapsed = time.perf_counter() - study_start
                job_seconds = time.perf_counter() - job_start
                average_job_seconds = elapsed / completed_jobs
                remaining_jobs = total_jobs - completed_jobs
                eta_seconds = average_job_seconds * remaining_jobs
                percent = 100.0 * completed_jobs / total_jobs
                print(
                    f"    completed {percent:5.1f}% | "
                    f"last={format_duration(job_seconds)} | "
                    f"elapsed={format_duration(elapsed)} | "
                    f"ETA={format_duration(eta_seconds)}"
                )

    summary_rows = summarize_rows(raw_rows)

    raw_csv = args.outdir / "scaling_raw.csv"
    summary_csv = args.outdir / "scaling_summary.csv"
    figure_path = args.outdir / "scaling_plot.pdf"

    write_csv(raw_rows, raw_csv)
    write_csv(summary_rows, summary_csv)
    plot_scaling(summary_rows, figure_path)

    print(f"Saved raw scaling CSV to: {raw_csv}")
    print(f"Saved summary scaling CSV to: {summary_csv}")
    print(f"Saved scaling figure to: {figure_path}")


if __name__ == "__main__":
    main()
