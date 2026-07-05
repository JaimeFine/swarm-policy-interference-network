from __future__ import annotations

import argparse
import csv
import gc
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.apf_velocity import APFVelocityController
from baselines.cbba import DistributedAuctionCBBAController
from baselines.mappo import build_reference_mappo_controller_from_checkpoint
from src.pettingzoo_env import (
    _make_env,
    compute_target_distance_statistics,
    compute_task_metric,
    scenario_targets,
)
from src.simulation import pretrain_mlp
from src.spin_policy import SpinPolicyController

OUTDIR = Path(__file__).resolve().parent / "memory_latency_results"
MAPPO_RUNS_ROOT = Path(__file__).resolve().parent / "baselines" / "results" / "mappo_convergence_runs"
METHODS = ("spin", "apf_velocity", "cbba", "mappo")
MODES = ("tracking", "multi_goal")
TRIALS = 5
STEPS = 120
WARMUP_STEPS = 10
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


def _build_controller(
    method: str,
    *,
    phi_omega,
    n_agents: int,
    steps: int,
    mode: str,
    mappo_run_dir: Path | None = None,
):
    if method == "spin":
        controller = SpinPolicyController(phi_omega=phi_omega, n_agents=n_agents)
        controller.reset()
        return controller
    if method == "apf_velocity":
        return APFVelocityController(n_agents=n_agents)
    if method == "cbba":
        return DistributedAuctionCBBAController(n_agents=n_agents)
    if method == "mappo":
        run_dir = _latest_completed_mappo_run_dir(mappo_run_dir)
        model_dir = run_dir / "training" / "mappo" / mode / "convergence" / "converged_model"
        return build_reference_mappo_controller_from_checkpoint(
            mode=mode,
            n_agents=n_agents,
            steps=steps,
            model_dir=model_dir,
            seed=BASE_SEED + MODE_SEED_OFFSET[mode],
        )
    raise ValueError(f"Unsupported method: {method}")


def _reset_controller(controller, method: str, seed: int) -> None:
    if method == "spin":
        controller.reset()
    else:
        controller.reset(seed=seed)


def _controller_act(controller, method: str, mode: str, env, step_idx: int):
    positions = env.agent_positions.copy()
    landmarks = env.landmark_positions.copy()
    if method == "spin":
        return controller.act(mode, positions, landmarks)
    return controller.act(
        mode,
        positions,
        landmarks,
        step_idx,
        agent_velocities=env.agent_velocities.copy(),
    )


def _estimate_numpy_state_bytes(obj, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        try:
            return int(obj.numel() * obj.element_size())
        except Exception:
            pass
    if isinstance(obj, dict):
        return sum(
            _estimate_numpy_state_bytes(key, seen) + _estimate_numpy_state_bytes(value, seen)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_estimate_numpy_state_bytes(item, seen) for item in obj)
    if hasattr(obj, "__dict__"):
        return sum(_estimate_numpy_state_bytes(value, seen) for value in vars(obj).values())
    return 0


def _close_controller(controller) -> None:
    if hasattr(controller, "runner") and hasattr(controller.runner, "envs"):
        controller.runner.envs.close()
    if hasattr(controller, "runner") and hasattr(controller.runner, "writter"):
        controller.runner.writter.close()


def _finalize_measured_values(values: list[float], fallback: list[float]) -> np.ndarray:
    chosen = values if values else fallback
    return np.asarray(chosen, dtype=float)


def run_latency_episode(
    *,
    method: str,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    mappo_run_dir: Path | None = None,
) -> dict[str, float | int | str]:
    gc.collect()
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    controller = _build_controller(
        method,
        phi_omega=phi_omega,
        n_agents=n_agents,
        steps=steps,
        mode=mode,
        mappo_run_dir=mappo_run_dir,
    )
    _reset_controller(controller, method, seed)

    controller_state_bytes = _estimate_numpy_state_bytes(controller)
    all_latencies_ms: list[float] = []
    measured_latencies_ms: list[float] = []
    final_task = None
    final_entropy = None
    final_distance_stats = compute_target_distance_statistics(
        mode, env.agent_positions.copy(), scenario_targets(mode, env.landmark_positions)
    )

    for step_idx in range(steps):
        start = time.perf_counter()
        actions, diagnostics = _controller_act(controller, method, mode, env, step_idx)
        latency_ms = 1000.0 * (time.perf_counter() - start)
        all_latencies_ms.append(latency_ms)
        if step_idx >= warmup_steps:
            measured_latencies_ms.append(latency_ms)

        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        final_task, _ = compute_task_metric(mode, positions, targets)
        final_distance_stats = compute_target_distance_statistics(mode, positions, targets)
        final_entropy = float(diagnostics.get("entropy", 0.0))

        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
    _close_controller(controller)
    latencies_ms = _finalize_measured_values(measured_latencies_ms, all_latencies_ms)
    return {
        "method": method,
        "mode": mode,
        "seed": int(seed),
        "steps_executed": int(len(all_latencies_ms)),
        "steps_measured": int(len(latencies_ms)),
        "controller_state_kib": float(controller_state_bytes / 1024.0),
        "act_latency_ms_mean": float(np.mean(latencies_ms)),
        "act_latency_ms_std": float(np.std(latencies_ms)),
        "act_latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "final_task": float(final_task),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(final_entropy),
    }


def run_memory_episode(
    *,
    method: str,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    mappo_run_dir: Path | None = None,
) -> dict[str, float | int | str]:
    gc.collect()
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    controller = _build_controller(
        method,
        phi_omega=phi_omega,
        n_agents=n_agents,
        steps=steps,
        mode=mode,
        mappo_run_dir=mappo_run_dir,
    )
    _reset_controller(controller, method, seed)

    controller_state_bytes = _estimate_numpy_state_bytes(controller)
    all_peaks_kib: list[float] = []
    measured_peaks_kib: list[float] = []

    tracemalloc.start()
    try:
        for step_idx in range(steps):
            tracemalloc.reset_peak()
            actions, _ = _controller_act(controller, method, mode, env, step_idx)
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_kib = peak_bytes / 1024.0
            all_peaks_kib.append(peak_kib)
            if step_idx >= warmup_steps:
                measured_peaks_kib.append(peak_kib)

            _, _, terminations, truncations, _ = env.step(actions)
            if all(terminations.values()) or all(truncations.values()):
                break
    finally:
        tracemalloc.stop()
        env.close()
        _close_controller(controller)

    peaks_kib = _finalize_measured_values(measured_peaks_kib, all_peaks_kib)
    return {
        "method": method,
        "mode": mode,
        "seed": int(seed),
        "memory_steps_measured": int(len(peaks_kib)),
        "controller_state_kib_memory_pass": float(controller_state_bytes / 1024.0),
        "act_peak_py_kib_mean": float(np.mean(peaks_kib)),
        "act_peak_py_kib_std": float(np.std(peaks_kib)),
        "act_peak_py_kib_max": float(np.max(peaks_kib)),
    }


def run_benchmark_trial(
    *,
    method: str,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    mappo_run_dir: Path | None = None,
) -> dict[str, float | int | str]:
    latency_row = run_latency_episode(
        method=method,
        phi_omega=phi_omega,
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        warmup_steps=warmup_steps,
        seed=seed,
        mappo_run_dir=mappo_run_dir,
    )
    memory_row = run_memory_episode(
        method=method,
        phi_omega=phi_omega,
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        warmup_steps=warmup_steps,
        seed=seed,
        mappo_run_dir=mappo_run_dir,
    )
    return {
        **latency_row,
        "controller_state_kib_memory_pass": memory_row["controller_state_kib_memory_pass"],
        "act_peak_py_kib_mean": memory_row["act_peak_py_kib_mean"],
        "act_peak_py_kib_std": memory_row["act_peak_py_kib_std"],
        "act_peak_py_kib_max": memory_row["act_peak_py_kib_max"],
        "memory_steps_measured": memory_row["memory_steps_measured"],
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["mode"]))].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    metrics = (
        "controller_state_kib",
        "controller_state_kib_memory_pass",
        "act_latency_ms_mean",
        "act_latency_ms_std",
        "act_latency_ms_p95",
        "act_peak_py_kib_mean",
        "act_peak_py_kib_std",
        "act_peak_py_kib_max",
        "final_task",
        "final_entropy",
    )
    for method, mode in sorted(grouped.keys()):
        group = grouped[(method, mode)]
        summary_row: dict[str, float | int | str] = {
            "method": method,
            "mode": mode,
            "trials": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values))
        for field in DISTANCE_STAT_FIELDS:
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            summary_row[f"{field}_mean"] = float(np.mean(values))
            summary_row[f"{field}_std"] = float(np.std(values))
        summary_rows.append(summary_row)
    return summary_rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict[str, float | int | str]], path: Path) -> None:
    grouped = {
        (str(row["method"]), str(row["mode"])): row
        for row in summary_rows
    }
    colors = {
        "spin": "#1d4ed8",
        "apf_velocity": "#059669",
        "cbba": "#dc2626",
        "mappo": "#7c3aed",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.34

    for col_idx, mode in enumerate(MODES):
        latency_ax = axes[0, col_idx]
        memory_ax = axes[1, col_idx]

        latency_means = [
            float(grouped[(method, mode)]["act_latency_ms_mean_mean"])
            for method in METHODS
        ]
        latency_stds = [
            float(grouped[(method, mode)]["act_latency_ms_mean_std"])
            for method in METHODS
        ]
        latency_ax.bar(
            x,
            latency_means,
            yerr=latency_stds,
            color=[colors[method] for method in METHODS],
            capsize=4,
        )
        latency_ax.set_title(f"{pretty_mode_name(mode)} Latency")
        latency_ax.set_xticks(x, [pretty_method_name(method) for method in METHODS], rotation=15)
        latency_ax.set_ylabel("Controller act time (ms)")
        latency_ax.grid(axis="y", alpha=0.25)

        state_means = [
            float(grouped[(method, mode)]["controller_state_kib_mean"])
            for method in METHODS
        ]
        peak_means = [
            float(grouped[(method, mode)]["act_peak_py_kib_max_mean"])
            for method in METHODS
        ]
        state_stds = [
            float(grouped[(method, mode)]["controller_state_kib_std"])
            for method in METHODS
        ]
        peak_stds = [
            float(grouped[(method, mode)]["act_peak_py_kib_max_std"])
            for method in METHODS
        ]
        memory_ax.bar(
            x - width / 2,
            state_means,
            width=width,
            yerr=state_stds,
            color="#94a3b8",
            capsize=4,
            label="Static controller state",
        )
        memory_ax.bar(
            x + width / 2,
            peak_means,
            width=width,
            yerr=peak_stds,
            color="#f59e0b",
            capsize=4,
            label="Peak Python step alloc",
        )
        memory_ax.set_title(f"{pretty_mode_name(mode)} Memory")
        memory_ax.set_xticks(x, [pretty_method_name(method) for method in METHODS], rotation=15)
        memory_ax.set_ylabel("Memory (KiB)")
        memory_ax.grid(axis="y", alpha=0.25)
        memory_ax.legend(frameon=True)

    fig.suptitle("Controller Memory Footprint and Latency Study", fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controller memory-footprint and latency study.")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
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
    print(
        "Including MAPPO from converged checkpoints at: "
        f"{mappo_run_dir}\n"
        "Note: this memory/latency study matches MAPPO's clean-observation evaluation setting only when the benchmark uses the same agent count and rollout length as training/test selection."
    )

    raw_rows: list[dict[str, float | int | str]] = []
    total_jobs = len(METHODS) * len(MODES) * args.trials
    completed = 0
    study_start = time.perf_counter()

    for method_idx, method in enumerate(METHODS):
        for mode in MODES:
            for trial_idx in range(args.trials):
                seed = (
                    BASE_SEED
                    + trial_idx * 1000
                    + MODE_SEED_OFFSET[mode]
                    + method_idx * 10_000_000
                )
                print(
                    f"[{completed + 1}/{total_jobs}] memory/latency study: "
                    f"method={method}, mode={mode}, "
                    f"trial={trial_idx + 1}/{args.trials}, seed={seed}"
                )
                job_start = time.perf_counter()
                row = run_benchmark_trial(
                    method=method,
                    phi_omega=phi_omega,
                    mode=mode,
                    n_agents=args.agents,
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                    seed=seed,
                    mappo_run_dir=mappo_run_dir,
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
    raw_csv = OUTDIR / "memory_latency_raw.csv"
    summary_csv = OUTDIR / "memory_latency_summary.csv"
    figure_path = OUTDIR / "memory_latency_plot.pdf"

    write_csv(raw_rows, raw_csv)
    write_csv(summary_rows, summary_csv)
    plot_summary(summary_rows, figure_path)

    print(f"Saved memory/latency raw CSV to: {raw_csv}")
    print(f"Saved memory/latency summary CSV to: {summary_csv}")
    print(f"Saved memory/latency figure to: {figure_path}")


if __name__ == "__main__":
    main()
