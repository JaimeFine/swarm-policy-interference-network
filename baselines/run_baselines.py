from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.apf_velocity import APFVelocityController
from baselines.cbba import DistributedAuctionCBBAController
from baselines.common import (
    plot_diagnostics,
    plot_trajectories,
    rollout_controller,
    save_metrics,
    summarize_rollout,
    write_summary_csv,
    write_trial_csv,
)
from baselines.mappo import run_reference_mappo_baseline


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPIN-comparable baseline evaluations.")
    parser.add_argument(
        "--baseline",
        choices=("apf_velocity", "cbba", "mappo", "all"),
        default="all",
        help="Baseline to run.",
    )
    parser.add_argument("--agents", type=int, default=10, help="Number of agents.")
    parser.add_argument("--steps", type=int, default=120, help="Number of control steps.")
    parser.add_argument("--trials", type=int, default=5, help="Number of repeated trials.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Baseline package output root.",
    )
    parser.add_argument(
        "--mode",
        choices=("tracking", "dispersion", "multi_goal"),
        default="tracking",
        help="MAPPO training mode.",
    )
    parser.add_argument(
        "--timesteps-total",
        type=int,
        default=50_000,
        help="MAPPO total training timesteps.",
    )
    parser.add_argument(
        "--rollout-threads",
        type=int,
        default=4,
        help="MAPPO CPU rollout worker count. Default keeps headroom on a 10-core machine.",
    )
    parser.add_argument(
        "--train-threads",
        type=int,
        default=2,
        help="MAPPO CPU training thread count.",
    )
    parser.add_argument(
        "--mappo-auto-converge",
        action="store_true",
        help="Enable periodic deterministic evaluation and early-stop MAPPO after convergence plateaus.",
    )
    parser.add_argument(
        "--mappo-convergence-eval-interval",
        type=int,
        default=100_000,
        help="Evaluate MAPPO convergence every N training env steps.",
    )
    parser.add_argument(
        "--mappo-convergence-min-steps",
        type=int,
        default=500_000,
        help="Do not allow MAPPO early stopping before this many training env steps.",
    )
    parser.add_argument(
        "--mappo-convergence-min-evals",
        type=int,
        default=3,
        help="Minimum number of convergence evaluations before early stopping is allowed.",
    )
    parser.add_argument(
        "--mappo-convergence-patience",
        type=int,
        default=4,
        help="Number of stale convergence evaluations tolerated before stopping MAPPO training.",
    )
    parser.add_argument(
        "--mappo-convergence-tol",
        type=float,
        default=0.01,
        help="Relative improvement threshold required to reset MAPPO convergence patience.",
    )
    parser.add_argument(
        "--mappo-selection-seed-offset",
        type=int,
        default=424242,
        help="Offset added to each MAPPO mode training seed to build the held-out checkpoint-selection seed.",
    )
    parser.add_argument(
        "--mappo-run-tag",
        type=str,
        default="",
        help="Optional folder tag for convergence-aware MAPPO runs.",
    )
    return parser.parse_args()


def build_mappo_convergence_config(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    return {
        "enabled": bool(args.mappo_auto_converge),
        "eval_interval_steps": int(args.mappo_convergence_eval_interval),
        "min_steps": int(args.mappo_convergence_min_steps),
        "min_evals": int(args.mappo_convergence_min_evals),
        "patience_evals": int(args.mappo_convergence_patience),
        "improvement_tol": float(args.mappo_convergence_tol),
        "selection_seed_offset": int(args.mappo_selection_seed_offset),
        "run_tag": str(args.mappo_run_tag),
    }


def run_deterministic_baseline(
    *,
    baseline_name: str,
    controller,
    n_agents: int,
    steps: int,
    trials: int,
    base_seed: int,
    outdir: Path,
) -> None:
    results_dir = outdir / "results" / baseline_name
    figures_dir = outdir / "figures" / baseline_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    representative: list[dict[str, np.ndarray | str | float | list[list[int]]]] = []
    total_jobs = trials * 3
    completed_jobs = 0
    baseline_start = time.perf_counter()

    for trial_idx in range(trials):
        for mode_idx, mode in enumerate(("tracking", "dispersion", "multi_goal")):
            seed = base_seed + trial_idx * 1000 + mode_idx * 101
            print(
                f"[{baseline_name}] [{completed_jobs + 1}/{total_jobs}] "
                f"running {mode} trial {trial_idx + 1}/{trials} with seed {seed}..."
            )
            job_start = time.perf_counter()
            result = rollout_controller(
                baseline_name=baseline_name,
                controller=controller,
                mode=mode,
                n_agents=n_agents,
                steps=steps,
                seed=seed,
            )
            rows.append(summarize_rollout(result, seed=seed, n_agents=n_agents, steps=steps))
            if trial_idx == 0:
                representative.append(result)
                metric_path = save_metrics(result, figures_dir)
                print(f"Saved {baseline_name} {mode} metrics to: {metric_path}")
            completed_jobs += 1
            elapsed = time.perf_counter() - baseline_start
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

    trial_csv = write_trial_csv(rows, results_dir)
    summary_csv = write_summary_csv(rows, results_dir)
    trajectory_pdf = plot_trajectories(representative, figures_dir, baseline_name)
    diagnostics_pdf = plot_diagnostics(representative, figures_dir, baseline_name)

    print(f"Saved {baseline_name} trial-level CSV to: {trial_csv}")
    print(f"Saved {baseline_name} summary CSV to: {summary_csv}")
    print(f"Saved {baseline_name} trajectories PDF to: {trajectory_pdf}")
    print(f"Saved {baseline_name} diagnostics PDF to: {diagnostics_pdf}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if args.baseline in {"apf_velocity", "all"}:
        run_deterministic_baseline(
            baseline_name="apf_velocity",
            controller=APFVelocityController(n_agents=args.agents),
            n_agents=args.agents,
            steps=args.steps,
            trials=args.trials,
            base_seed=args.seed,
            outdir=args.outdir,
        )

    if args.baseline in {"cbba", "all"}:
        run_deterministic_baseline(
            baseline_name="cbba",
            controller=DistributedAuctionCBBAController(n_agents=args.agents),
            n_agents=args.agents,
            steps=args.steps,
            trials=args.trials,
            base_seed=args.seed,
            outdir=args.outdir,
        )

    if args.baseline in {"mappo", "all"}:
        try:
            run_reference_mappo_baseline(
                n_agents=args.agents,
                steps=args.steps,
                trials=args.trials,
                base_seed=args.seed,
                num_env_steps=args.timesteps_total,
                outdir=args.outdir,
                rollout_threads=args.rollout_threads,
                training_threads=args.train_threads,
                convergence_config=build_mappo_convergence_config(args),
            )
        except RuntimeError as exc:
            print(f"MAPPO run skipped: {exc}")
            print(
                f"Install torch, tensorboardX, and the reference onpolicy stack, then rerun with:\n"
                f"python baselines\\run_baselines.py --baseline mappo --timesteps-total {args.timesteps_total} --rollout-threads {args.rollout_threads} --train-threads {args.train_threads} --outdir {args.outdir}"
            )


if __name__ == "__main__":
    main()
