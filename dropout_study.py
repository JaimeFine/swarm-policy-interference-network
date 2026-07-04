from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.apf_velocity import APFVelocityController
from baselines.cbba import DistributedAuctionCBBAController
from src.network import forward_pass
from src.pettingzoo_env import (
    _make_env,
    compute_target_distance_statistics,
    compute_task_metric,
    scenario_targets,
)
from src.quantum import (
    apply_radon_nikodym_filter,
    compute_clique_reduced_densities,
    evaluate_born_probabilities,
    reconcile_overlapping_densities,
)
from src.simulation import pretrain_mlp
from src.spin_policy import (
    GAMMA_CLAMPING,
    MAX_CLIQUE_NEIGHBORS,
    SpinPolicyController,
    compute_policy_signal,
    mean_policy_entropy,
)
from src.topology import compute_adjacency_matrix, extract_overlapping_maximal_cliques

OUTDIR = Path(__file__).resolve().parent / "dropout_robustness_results"
METHODS = ("spin", "apf_velocity", "cbba")
MODES = ("tracking", "multi_goal")
DROP_RATES = (0.0, 0.1, 0.25, 0.4, 0.6)
TRIALS = 5
STEPS = 120
AGENTS = 10
PRETRAIN_EPOCHS = 5000
BASE_SEED = 7
MODE_SEED_OFFSET = {"tracking": 0, "multi_goal": 202}
DISTANCE_STAT_FIELDS = (
    "final_distance_mean",
    "final_distance_variance",
    "final_distance_p90",
    "final_distance_p90_tail_mean",
    "final_distance_mad",
)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def prefix_metric_keys(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def pretty_mode_name(mode: str) -> str:
    return "Multi-Goal" if mode == "multi_goal" else "Tracking"


def pretty_method_name(method: str) -> str:
    if method == "apf_velocity":
        return "APF-Velocity"
    if method == "cbba":
        return "CBBA"
    return "SPIN"


class DropoutSpinPolicyController(SpinPolicyController):
    def __init__(self, phi_omega, n_agents: int, drop_rate: float, rng: np.random.Generator) -> None:
        super().__init__(phi_omega=phi_omega, n_agents=n_agents)
        self.drop_rate = float(drop_rate)
        self.rng = rng

    def act(
        self,
        mode: str,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | list[list[int]]]]:
        adj_matrix = compute_adjacency_matrix(
            agent_positions,
            self._sense_radius(),
            max_neighbors=MAX_CLIQUE_NEIGHBORS,
        )
        cliques = extract_overlapping_maximal_cliques(adj_matrix)

        clean_signals = [
            compute_policy_signal(mode, idx, agent_positions, landmark_positions)
            for idx in range(self.n_agents)
        ]
        dropped_signals = [
            _apply_coordinate_dropout(self.rng, signal, self.drop_rate)
            for signal in clean_signals
        ]

        for idx, signal in enumerate(dropped_signals):
            measure_output, _ = forward_pass(self.phi_omega, signal)
            apply_radon_nikodym_filter(
                self.states[idx], measure_output, GAMMA_CLAMPING
            )

        density_contributions: dict[int, list[np.ndarray]] = {
            idx: [] for idx in range(self.n_agents)
        }
        for clique in cliques:
            if len(clique) > 1:
                clique_rhos = compute_clique_reduced_densities(
                    self.states, clique, agent_positions, mode=mode
                )
                for local_idx, agent_idx in enumerate(clique):
                    density_contributions[agent_idx].append(clique_rhos[local_idx])

        reconcile_overlapping_densities(self.states, density_contributions)

        actions: dict[str, np.ndarray] = {}
        for idx, signal in enumerate(dropped_signals):
            probs = evaluate_born_probabilities(self.states[idx])
            probs = np.clip(probs, 1e-6, None)
            probs /= np.sum(probs)
            actions[f"agent_{idx}"] = self._to_mpe_action(
                probs=probs, signal=signal, mode=mode
            )

        diagnostics = {
            "cliques": [clique[:] for clique in cliques],
            "entropy": mean_policy_entropy(self.states),
        }
        return actions, diagnostics

    @staticmethod
    def _sense_radius() -> float:
        from src.spin_policy import R_SENSE

        return R_SENSE


def _apply_coordinate_dropout(
    rng: np.random.Generator,
    values: np.ndarray,
    drop_rate: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if drop_rate <= 0.0:
        return values.copy()
    keep_mask = rng.random(size=values.shape) >= float(drop_rate)
    return values * keep_mask.astype(float)


def _build_baseline_controller(method: str, n_agents: int):
    if method == "apf_velocity":
        return APFVelocityController(n_agents=n_agents)
    if method == "cbba":
        return DistributedAuctionCBBAController(n_agents=n_agents)
    raise ValueError(f"Unsupported baseline method: {method}")


def run_dropout_spin_episode(
    *,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    drop_rate: float,
) -> dict[str, float | int | str]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    controller = DropoutSpinPolicyController(
        phi_omega=phi_omega,
        n_agents=n_agents,
        drop_rate=drop_rate,
        rng=rng,
    )
    controller.reset()

    final_task = None
    final_entropy = None
    final_distance_stats = compute_target_distance_statistics(
        mode, env.agent_positions.copy(), scenario_targets(mode, env.landmark_positions)
    )
    for _ in range(steps):
        positions = env.agent_positions.copy()
        landmark_positions = env.landmark_positions.copy()
        actions, diagnostics = controller.act(mode, positions, landmark_positions)
        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        final_task, _ = compute_task_metric(mode, positions, targets)
        final_distance_stats = compute_target_distance_statistics(mode, positions, targets)
        final_entropy = float(diagnostics["entropy"])

        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
    return {
        "method": "spin",
        "mode": mode,
        "drop_rate": float(drop_rate),
        "bandwidth_ratio": float(1.0 - drop_rate),
        "seed": int(seed),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(final_entropy),
    }


def run_dropout_baseline_episode(
    *,
    method: str,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    drop_rate: float,
) -> dict[str, float | int | str]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    controller = _build_baseline_controller(method, n_agents)
    controller.reset(seed=seed)

    final_task = None
    final_entropy = None
    final_distance_stats = compute_target_distance_statistics(
        mode, env.agent_positions.copy(), scenario_targets(mode, env.landmark_positions)
    )
    for step_idx in range(steps):
        true_positions = env.agent_positions.copy()
        true_landmarks = env.landmark_positions.copy()
        perceived_positions = _apply_coordinate_dropout(rng, true_positions, drop_rate)
        perceived_landmarks = _apply_coordinate_dropout(rng, true_landmarks, drop_rate)

        actions, diagnostics = controller.act(
            mode,
            perceived_positions,
            perceived_landmarks,
            step_idx,
            agent_velocities=env.agent_velocities.copy(),
        )
        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        final_task, _ = compute_task_metric(mode, positions, targets)
        final_distance_stats = compute_target_distance_statistics(mode, positions, targets)
        final_entropy = float(diagnostics.get("entropy", 0.0))

        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
    return {
        "method": method,
        "mode": mode,
        "drop_rate": float(drop_rate),
        "bandwidth_ratio": float(1.0 - drop_rate),
        "seed": int(seed),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(final_entropy),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["mode"]), float(row["drop_rate"]))].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (method, mode, drop_rate), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        final_tasks = [float(r["final_task"]) for r in group]
        entropies = [float(r["final_entropy"]) for r in group]
        summary_rows.append(
            {
                "method": method,
                "mode": mode,
                "drop_rate": drop_rate,
                "bandwidth_ratio": float(1.0 - drop_rate),
                "trials": len(group),
                "final_task_mean": float(np.mean(final_tasks)),
                "final_task_std": float(np.std(final_tasks)),
                **{
                    f"{field}_mean": float(np.mean([float(r[field]) for r in group]))
                    for field in DISTANCE_STAT_FIELDS
                },
                **{
                    f"{field}_std": float(np.std([float(r[field]) for r in group]))
                    for field in DISTANCE_STAT_FIELDS
                },
                "final_entropy_mean": float(np.mean(entropies)),
                "final_entropy_std": float(np.std(entropies)),
            }
        )
    return summary_rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict[str, float | int | str]], path: Path) -> Path:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["method"]), str(row["mode"]))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    colors = {
        "spin": "#1d4ed8",
        "apf_velocity": "#059669",
        "cbba": "#dc2626",
    }

    for ax, mode in zip(axes, MODES):
        for method in METHODS:
            rows = sorted(grouped[(method, mode)], key=lambda row: float(row["drop_rate"]))
            bandwidth = [float(r["bandwidth_ratio"]) for r in rows]
            means = [float(r["final_task_mean"]) for r in rows]
            stds = [float(r["final_task_std"]) for r in rows]

            ax.errorbar(
                bandwidth,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[method],
                label=pretty_method_name(method),
            )
        ax.set_title(f"{pretty_mode_name(mode)} Robustness")
        ax.set_xlabel("Retained bandwidth ratio (1 - dropout rate)")
        ax.set_ylabel("Final mean target distance")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
        ax.invert_xaxis()

    fig.suptitle("Reduced-Bandwidth / Dropout Robustness Study", fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except PermissionError:
        output_path = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reduced-bandwidth dropout robustness study.")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--agents", type=int, default=AGENTS)
    parser.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(BASE_SEED)
    phi_omega = pretrain_mlp(epochs=args.epochs)

    raw_rows: list[dict[str, float | int | str]] = []
    total_jobs = len(METHODS) * len(MODES) * len(DROP_RATES) * args.trials
    completed = 0
    study_start = time.perf_counter()

    for method_idx, method in enumerate(METHODS):
        for mode in MODES:
            for drop_rate in DROP_RATES:
                for trial_idx in range(args.trials):
                    seed = (
                        BASE_SEED
                        + trial_idx * 1000
                        + MODE_SEED_OFFSET[mode]
                        + int(drop_rate * 100) * 10000
                        + method_idx * 10_000_000
                    )
                    print(
                        f"[{completed + 1}/{total_jobs}] dropout study: "
                        f"method={method}, mode={mode}, drop_rate={drop_rate}, "
                        f"bandwidth_ratio={1.0 - drop_rate:.2f}, "
                        f"trial={trial_idx + 1}/{args.trials}, seed={seed}"
                    )
                    job_start = time.perf_counter()
                    if method == "spin":
                        row = run_dropout_spin_episode(
                            phi_omega=phi_omega,
                            mode=mode,
                            n_agents=args.agents,
                            steps=args.steps,
                            seed=seed,
                            drop_rate=drop_rate,
                        )
                    else:
                        row = run_dropout_baseline_episode(
                            method=method,
                            mode=mode,
                            n_agents=args.agents,
                            steps=args.steps,
                            seed=seed,
                            drop_rate=drop_rate,
                        )
                    raw_rows.append(row)
                    completed += 1
                    elapsed = time.perf_counter() - study_start
                    job_seconds = time.perf_counter() - job_start
                    eta_seconds = (elapsed / completed) * (total_jobs - completed)
                    percent = 100.0 * completed / total_jobs
                    print(
                        f"    completed {percent:5.1f}% | "
                        f"last={format_duration(job_seconds)} | "
                        f"elapsed={format_duration(elapsed)} | "
                        f"ETA={format_duration(eta_seconds)}"
                    )

    summary_rows = summarize(raw_rows)
    raw_csv = OUTDIR / "dropout_robustness_raw.csv"
    summary_csv = OUTDIR / "dropout_robustness_summary.csv"
    figure_path = OUTDIR / "dropout_robustness_plot.pdf"

    write_csv(raw_rows, raw_csv)
    write_csv(summary_rows, summary_csv)
    saved_figure_path = plot_summary(summary_rows, figure_path)

    print(f"Saved dropout raw CSV to: {raw_csv}")
    print(f"Saved dropout summary CSV to: {summary_csv}")
    print(f"Saved dropout robustness figure to: {saved_figure_path}")


if __name__ == "__main__":
    main()
