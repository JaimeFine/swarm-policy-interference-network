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
from baselines.mappo import build_reference_mappo_controller_from_checkpoint
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

OUTDIR = Path(__file__).resolve().parent / "noise_robustness_results"
MAPPO_RUNS_ROOT = Path(__file__).resolve().parent / "baselines" / "results" / "mappo_convergence_runs"
METHODS = ("spin", "apf_velocity", "cbba", "mappo")
MODES = ("tracking", "multi_goal")
SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
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
    if method == "mappo":
        return "MAPPO"
    return "SPIN"


class NoisySpinPolicyController(SpinPolicyController):
    def __init__(self, phi_omega, n_agents: int, noise_sigma: float, rng: np.random.Generator) -> None:
        super().__init__(phi_omega=phi_omega, n_agents=n_agents)
        self.noise_sigma = float(noise_sigma)
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
        noisy_signals = []
        for signal in clean_signals:
            if self.noise_sigma > 0.0:
                signal = np.asarray(signal, dtype=float) + self.rng.normal(
                    loc=0.0, scale=self.noise_sigma, size=2
                )
            noisy_signals.append(np.asarray(signal, dtype=float))

        for idx, signal in enumerate(noisy_signals):
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
        for idx, signal in enumerate(noisy_signals):
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


def _noisy_observation(
    rng: np.random.Generator,
    values: np.ndarray,
    sigma: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sigma <= 0.0:
        return values.copy()
    return values + rng.normal(loc=0.0, scale=float(sigma), size=values.shape)


def _latest_completed_mappo_run_dir(explicit_run_dir: Path | None = None) -> Path:
    if explicit_run_dir is not None:
        run_dir = Path(explicit_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"MAPPO run directory does not exist: {run_dir}")
        return run_dir

    if not MAPPO_RUNS_ROOT.exists():
        raise FileNotFoundError(
            f"No MAPPO convergence runs found under: {MAPPO_RUNS_ROOT}"
        )

    candidates = sorted(
        [path for path in MAPPO_RUNS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if all(
            (run_dir / "training" / "mappo" / mode / "convergence" / "converged_model" / "actor.pt").exists()
            for mode in MODES
        ):
            return run_dir
    raise FileNotFoundError(
        f"No completed MAPPO convergence run with converged checkpoints for modes {MODES} was found."
    )


def _build_baseline_controller(method: str, n_agents: int, mappo_run_dir: Path | None = None, *, mode: str | None = None):
    if method == "apf_velocity":
        return APFVelocityController(n_agents=n_agents)
    if method == "cbba":
        return DistributedAuctionCBBAController(n_agents=n_agents)
    if method == "mappo":
        if mode is None:
            raise ValueError("MAPPO controller construction requires a mode.")
        run_dir = _latest_completed_mappo_run_dir(mappo_run_dir)
        model_dir = run_dir / "training" / "mappo" / mode / "convergence" / "converged_model"
        return build_reference_mappo_controller_from_checkpoint(
            mode=mode,
            n_agents=n_agents,
            steps=STEPS,
            model_dir=model_dir,
            seed=BASE_SEED + MODE_SEED_OFFSET[mode],
        )
    raise ValueError(f"Unsupported baseline method: {method}")


def run_noisy_spin_episode(
    *,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    noise_sigma: float,
) -> dict[str, float | int | str]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    controller = NoisySpinPolicyController(
        phi_omega=phi_omega,
        n_agents=n_agents,
        noise_sigma=noise_sigma,
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
        "sigma": float(noise_sigma),
        "seed": int(seed),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(final_entropy),
    }


def run_noisy_baseline_episode(
    *,
    method: str,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    noise_sigma: float,
    controller=None,
) -> dict[str, float | int | str]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    controller = controller or _build_baseline_controller(method, n_agents, mode=mode)
    controller.reset(seed=seed)

    final_task = None
    final_entropy = None
    final_distance_stats = compute_target_distance_statistics(
        mode, env.agent_positions.copy(), scenario_targets(mode, env.landmark_positions)
    )
    for step_idx in range(steps):
        true_positions = env.agent_positions.copy()
        true_landmarks = env.landmark_positions.copy()
        true_velocities = env.agent_velocities.copy()
        perceived_positions = _noisy_observation(rng, true_positions, noise_sigma)
        perceived_landmarks = _noisy_observation(rng, true_landmarks, noise_sigma)
        perceived_velocities = _noisy_observation(rng, true_velocities, noise_sigma)

        actions, diagnostics = controller.act(
            mode,
            perceived_positions,
            perceived_landmarks,
            step_idx,
            agent_velocities=perceived_velocities,
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
        "sigma": float(noise_sigma),
        "seed": int(seed),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(final_entropy),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["mode"]), float(row["sigma"]))].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (method, mode, sigma), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        final_tasks = [float(r["final_task"]) for r in group]
        entropies = [float(r["final_entropy"]) for r in group]
        summary_rows.append(
            {
                "method": method,
                "mode": mode,
                "sigma": sigma,
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


def plot_summary(summary_rows: list[dict[str, float | int | str]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["method"]), str(row["mode"]))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    colors = {
        "spin": "#1d4ed8",
        "apf_velocity": "#059669",
        "cbba": "#dc2626",
        "mappo": "#7c3aed",
    }

    for ax, mode in zip(axes, MODES):
        for method in METHODS:
            rows = sorted(grouped[(method, mode)], key=lambda row: float(row["sigma"]))
            sigmas = [float(r["sigma"]) for r in rows]
            means = [float(r["final_task_mean"]) for r in rows]
            stds = [float(r["final_task_std"]) for r in rows]

            ax.errorbar(
                sigmas,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[method],
                label=pretty_method_name(method),
            )
        ax.set_title(f"{pretty_mode_name(mode)} Robustness")
        ax.set_xlabel("Gaussian observation / descriptor noise sigma")
        ax.set_ylabel("Final mean target distance")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)

    fig.suptitle("Descriptor / Observation-Noise Robustness Study", fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gaussian observation-noise robustness study.")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--agents", type=int, default=AGENTS)
    parser.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument(
        "--mappo-run-dir",
        type=Path,
        default=None,
        help="Optional path to a completed MAPPO convergence run directory. Defaults to the most recent completed run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(BASE_SEED)
    phi_omega = pretrain_mlp(epochs=args.epochs)
    mappo_run_dir = _latest_completed_mappo_run_dir(args.mappo_run_dir)
    mappo_controllers = {
        mode: _build_baseline_controller("mappo", args.agents, mappo_run_dir, mode=mode)
        for mode in MODES
    }
    print(
        "Including MAPPO from converged checkpoints at: "
        f"{mappo_run_dir}\n"
        "Note: MAPPO was trained on clean observations, so the noise study is an out-of-distribution robustness evaluation for MAPPO."
    )

    raw_rows: list[dict[str, float | int | str]] = []
    total_jobs = len(METHODS) * len(MODES) * len(SIGMAS) * args.trials
    completed = 0
    study_start = time.perf_counter()

    for method_idx, method in enumerate(METHODS):
        for mode in MODES:
            for sigma in SIGMAS:
                for trial_idx in range(args.trials):
                    seed = (
                        BASE_SEED
                        + trial_idx * 1000
                        + MODE_SEED_OFFSET[mode]
                        + int(sigma * 100) * 10000
                        + method_idx * 10_000_000
                    )
                    print(
                        f"[{completed + 1}/{total_jobs}] noise study: "
                        f"method={method}, mode={mode}, sigma={sigma}, "
                        f"trial={trial_idx + 1}/{args.trials}, seed={seed}"
                    )
                    job_start = time.perf_counter()
                    if method == "spin":
                        row = run_noisy_spin_episode(
                            phi_omega=phi_omega,
                            mode=mode,
                            n_agents=args.agents,
                            steps=args.steps,
                            seed=seed,
                            noise_sigma=sigma,
                        )
                    else:
                        row = run_noisy_baseline_episode(
                            method=method,
                            mode=mode,
                            n_agents=args.agents,
                            steps=args.steps,
                            seed=seed,
                            noise_sigma=sigma,
                            controller=mappo_controllers.get(mode) if method == "mappo" else None,
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
    raw_csv = OUTDIR / "noise_robustness_raw.csv"
    summary_csv = OUTDIR / "noise_robustness_summary.csv"
    figure_path = OUTDIR / "noise_robustness_plot.pdf"

    write_csv(raw_rows, raw_csv)
    write_csv(summary_rows, summary_csv)
    plot_summary(summary_rows, figure_path)

    print(f"Saved noise raw CSV to: {raw_csv}")
    print(f"Saved noise summary CSV to: {summary_csv}")
    print(f"Saved noise robustness figure to: {figure_path}")
    for controller in mappo_controllers.values():
        if hasattr(controller, "runner") and hasattr(controller.runner, "envs"):
            controller.runner.envs.close()
        if hasattr(controller, "runner") and hasattr(controller.runner, "writter"):
            controller.runner.writter.close()


if __name__ == "__main__":
    main()
