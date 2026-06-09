from __future__ import annotations

import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
from gymnasium import spaces

from baselines.common import (
    plot_diagnostics,
    plot_trajectories,
    rollout_controller,
    save_metrics,
    summarize_rollout,
    write_summary_csv,
    write_trial_csv,
)
from src.coverage import compute_spatial_entropy
from src.pettingzoo_env import AGENT_BODY_RADIUS, _make_env, scenario_targets

REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "reference"
)


def _ensure_reference_path() -> None:
    ref_str = str(REFERENCE_ROOT)
    if ref_str not in sys.path:
        sys.path.append(ref_str)
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.SimpleNamespace(
            init=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
            finish=lambda *args, **kwargs: None,
            run=None,
        )


def ensure_reference_mappo_dependencies() -> None:
    missing: list[str] = []
    for name in ("torch", "tensorboardX"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Reference MAPPO baseline requires the following packages in the active environment: "
            + ", ".join(missing)
        )


def _decode_one_hot_actions(actions: np.ndarray) -> dict[str, np.ndarray]:
    decoded: dict[str, np.ndarray] = {}
    for agent_idx, vector in enumerate(np.asarray(actions)):
        action_idx = int(np.argmax(vector))
        action = np.zeros(5, dtype=np.float32)
        action[action_idx] = 1.0
        decoded[f"agent_{agent_idx}"] = action
    return decoded


def _build_observation(
    position: np.ndarray,
    velocity: np.ndarray,
    all_positions: np.ndarray,
    landmark_positions: np.ndarray,
    agent_idx: int,
) -> np.ndarray:
    rel_landmarks = (landmark_positions - position).reshape(-1)
    rel_agents = np.delete(all_positions, agent_idx, axis=0) - position
    rel_agents = rel_agents.reshape(-1)
    return np.concatenate([position, velocity, rel_landmarks, rel_agents]).astype(np.float32)


class SpinMAPPOEnv:
    def __init__(self, mode: str, n_agents: int, steps: int, seed: int) -> None:
        self.mode = mode
        self.n_agents = n_agents
        self.max_cycles = steps
        self.seed_value = seed
        self.env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
        self.current_step = 0
        self.last_task = 0.0
        self.observation_space: list[spaces.Box] = []
        self.share_observation_space: list[spaces.Box] = []
        self.action_space: list[spaces.Discrete] = [spaces.Discrete(5) for _ in range(n_agents)]
        self.reset()

    def seed(self, seed: int) -> None:
        self.seed_value = seed
        self.env.reset(seed=seed)

    def reset(self):
        self.env.reset(seed=self.seed_value)
        self.current_step = 0
        obs = self._obs_matrix()
        obs_dim = obs.shape[1]
        share_dim = obs_dim * self.n_agents
        self.observation_space = [
            spaces.Box(low=-100.0, high=100.0, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-100.0, high=100.0, shape=(share_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.last_task = self._task_value()
        return obs

    def close(self) -> None:
        self.env.close()

    def step(self, actions):
        action_dict = _decode_one_hot_actions(actions)
        prev_positions = self.env.agent_positions.copy()
        prev_task = self.last_task
        _, _, _, truncations, _ = self.env.step(action_dict)
        self.current_step += 1
        obs = self._obs_matrix()
        rewards = self._training_rewards(prev_positions, prev_task)
        dones = np.full(self.n_agents, bool(all(truncations.values())), dtype=bool)
        infos = [
            {"individual_reward": float(rewards[agent_idx, 0])}
            for agent_idx in range(self.n_agents)
        ]
        return obs, rewards, dones, infos

    def _obs_matrix(self) -> np.ndarray:
        positions = self.env.agent_positions.copy()
        velocities = self.env.agent_velocities.copy()
        landmarks = self.env.landmark_positions.copy()
        return np.stack(
            [
                _build_observation(
                    positions[agent_idx],
                    velocities[agent_idx],
                    positions,
                    landmarks,
                    agent_idx,
                )
                for agent_idx in range(self.n_agents)
            ],
            axis=0,
        )

    def _task_value(self) -> float:
        positions = self.env.agent_positions.copy()
        targets = scenario_targets(self.mode, self.env.landmark_positions)
        if self.mode in {"tracking", "multi_goal"}:
            distances = np.linalg.norm(
                positions[:, np.newaxis, :] - targets[np.newaxis, :, :],
                axis=2,
            )
            return float(np.mean(np.min(distances, axis=1)))
        return float(compute_spatial_entropy(positions, 100.0))

    def _training_rewards(self, prev_positions: np.ndarray, prev_task: float) -> np.ndarray:
        positions = self.env.agent_positions.copy()
        current_task = self._task_value()
        self.last_task = current_task

        pairwise = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(pairwise, axis=2)
        dist += np.eye(self.n_agents) * 1e6
        nearest = np.min(dist, axis=1)
        body_clearance = 2.0 * AGENT_BODY_RADIUS
        crowd_penalty = np.maximum(body_clearance + 2.0 - nearest, 0.0)
        overlap_depth = np.maximum(body_clearance - nearest, 0.0)
        overlap_penalty = overlap_depth ** 2
        boundary_margin = np.minimum.reduce(
            np.stack(
                [
                    positions[:, 0],
                    positions[:, 1],
                    100.0 - positions[:, 0],
                    100.0 - positions[:, 1],
                ],
                axis=1,
            ),
            axis=1,
        )
        boundary_penalty = np.maximum(6.0 - boundary_margin, 0.0) / 6.0

        if self.mode == "tracking":
            targets = self.env.landmark_positions[[0]]
            prev_dist = np.linalg.norm(prev_positions - targets[0], axis=1)
            new_dist = np.linalg.norm(positions - targets[0], axis=1)
            reward = (prev_dist - new_dist)
            reward -= 0.14 * crowd_penalty
            reward -= 1.20 * overlap_penalty
            reward -= 0.06 * boundary_penalty
        elif self.mode == "multi_goal":
            targets = self.env.landmark_positions.copy()
            prev_dist = np.min(
                np.linalg.norm(prev_positions[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2),
                axis=1,
            )
            new_dist = np.min(
                np.linalg.norm(positions[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2),
                axis=1,
            )
            reward = (prev_dist - new_dist)
            reward -= 0.12 * crowd_penalty
            reward -= 1.00 * overlap_penalty
            reward -= 0.05 * boundary_penalty
        else:
            entropy_gain = current_task - prev_task
            reward = np.full(self.n_agents, entropy_gain, dtype=float)
            reward += 0.02 * np.clip(nearest - 4.0, 0.0, 10.0)
            reward -= 0.10 * crowd_penalty
            reward -= 1.10 * overlap_penalty
            reward -= 0.08 * boundary_penalty

        return reward[:, np.newaxis].astype(np.float32)


def _build_args(
    *,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    num_env_steps: int,
    run_dir: Path,
    rollout_threads: int,
    training_threads: int,
):
    _ensure_reference_path()
    from onpolicy.config import get_config

    parser = get_config()
    args = parser.parse_args([])
    args.algorithm_name = "mappo"
    args.experiment_name = f"spin_{mode}"
    args.seed = seed
    args.cuda = False
    args.cuda_deterministic = True
    args.n_training_threads = int(training_threads)
    args.n_rollout_threads = int(rollout_threads)
    args.n_eval_rollout_threads = 1
    args.n_render_rollout_threads = 1
    args.num_env_steps = int(num_env_steps)
    args.user_name = "spin"
    args.use_wandb = False
    args.env_name = "MPE"
    args.episode_length = steps
    args.share_policy = True
    args.use_centralized_V = True
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_eval = False
    args.save_interval = max(1, num_env_steps // max(steps, 1))
    args.log_interval = 1
    args.eval_interval = 1000
    args.model_dir = None
    args.scenario_name = mode
    args.num_agents = n_agents
    args.run_dir = run_dir
    return args


def train_reference_mappo(
    *,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
    num_env_steps: int,
    run_dir: Path,
    rollout_threads: int,
    training_threads: int,
):
    ensure_reference_mappo_dependencies()
    _ensure_reference_path()
    import torch
    from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
    from onpolicy.runner.shared.mpe_runner import MPERunner as Runner

    class ProgressMPERunner(Runner):
        def run(self):
            self.warmup()

            start = time.time()
            episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
            if episodes <= 0:
                raise RuntimeError("MAPPO training budget is too small for even one episode.")

            progress_marks = {0.10, 0.25, 0.50, 0.75, 0.90, 1.00}
            reported_marks: set[float] = set()

            for episode in range(episodes):
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                for step in range(self.episode_length):
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    obs, rewards, dones, infos = self.envs.step(actions_env)
                    data = (
                        obs,
                        rewards,
                        dones,
                        infos,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                    self.insert(data)

                self.compute()
                train_infos = self.train()
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save()

                progress = (episode + 1) / episodes
                for mark in sorted(progress_marks):
                    if progress >= mark and mark not in reported_marks:
                        elapsed = time.time() - start
                        fps = total_num_steps / max(elapsed, 1e-6)
                        eta_seconds = max(self.num_env_steps - total_num_steps, 0) / max(fps, 1e-6)
                        print(
                            f"[mappo-train] {self.all_args.scenario_name}: "
                            f"{int(mark * 100)}% "
                            f"({episode + 1}/{episodes} episodes, "
                            f"{total_num_steps}/{self.num_env_steps} steps, "
                            f"ETA {eta_seconds / 60.0:.1f} min)"
                        )
                        reported_marks.add(mark)

                if episode % self.log_interval == 0 or episode == episodes - 1:
                    if self.env_name == "MPE":
                        env_infos = {}
                        for agent_id in range(self.num_agents):
                            idv_rews = []
                            for info in infos:
                                if "individual_reward" in info[agent_id].keys():
                                    idv_rews.append(info[agent_id]["individual_reward"])
                            agent_k = "agent%i/individual_rewards" % agent_id
                            env_infos[agent_k] = idv_rews

                    train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    self.log_train(train_infos, total_num_steps)
                    self.log_env(env_infos, total_num_steps)

    args = _build_args(
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        seed=seed,
        num_env_steps=num_env_steps,
        run_dir=run_dir,
        rollout_threads=rollout_threads,
        training_threads=training_threads,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(training_threads)))

    env_fns = [
        (lambda env_seed=seed + worker_idx * 9973: SpinMAPPOEnv(mode, n_agents, steps, env_seed))
        for worker_idx in range(int(rollout_threads))
    ]
    envs = (
        DummyVecEnv(env_fns)
        if int(rollout_threads) == 1
        else SubprocVecEnv(env_fns)
    )
    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": n_agents,
        "device": torch.device("cpu"),
        "run_dir": run_dir,
    }
    runner = ProgressMPERunner(config)
    runner.run()
    return runner


class ReferenceMAPPOController:
    def __init__(self, runner, n_agents: int) -> None:
        self.runner = runner
        self.n_agents = n_agents
        self.policy = runner.policy
        self.recurrent_N = int(runner.recurrent_N)
        self.hidden_size = int(runner.hidden_size)
        self.rnn_states = np.zeros((n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.masks = np.ones((n_agents, 1), dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.rnn_states = np.zeros_like(self.rnn_states)
        self.masks = np.ones_like(self.masks)

    def act(
        self,
        mode: str,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
        step_idx: int,
        agent_velocities: np.ndarray | None = None,
    ):
        del mode, step_idx
        ensure_reference_mappo_dependencies()
        import torch

        if agent_velocities is None:
            agent_velocities = np.zeros_like(agent_positions)

        obs = np.stack(
            [
                _build_observation(
                    agent_positions[agent_idx],
                    agent_velocities[agent_idx],
                    agent_positions,
                    landmark_positions,
                    agent_idx,
                )
                for agent_idx in range(self.n_agents)
            ],
            axis=0,
        )

        actions, next_rnn_states = self.policy.act(
            obs,
            self.rnn_states,
            self.masks,
            deterministic=True,
        )
        actions_np = actions.detach().cpu().numpy()
        self.rnn_states = next_rnn_states.detach().cpu().numpy()

        obs_tensor = torch.from_numpy(obs.astype(np.float32))
        features = self.policy.actor.base(obs_tensor)
        probs = self.policy.actor.act.get_probs(features).detach().cpu().numpy()
        entropy = float(
            np.mean(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
        )

        action_dict: dict[str, np.ndarray] = {}
        for agent_idx in range(self.n_agents):
            action_idx = int(actions_np[agent_idx, 0]) if actions_np.ndim == 2 else int(actions_np[agent_idx])
            action = np.zeros(5, dtype=np.float32)
            action[action_idx] = 1.0
            action_dict[f"agent_{agent_idx}"] = action

        return action_dict, {"entropy": entropy}


def run_reference_mappo_baseline(
    *,
    n_agents: int,
    steps: int,
    trials: int,
    base_seed: int,
    num_env_steps: int,
    outdir: Path,
    rollout_threads: int,
    training_threads: int,
) -> None:
    figures_dir = outdir / "figures" / "mappo"
    results_dir = outdir / "results" / "mappo"
    train_root = outdir / "results" / "mappo_training"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    representative = []
    rows: list[dict[str, float | int | str]] = []

    for mode_idx, mode in enumerate(("tracking", "dispersion", "multi_goal")):
        train_seed = base_seed + mode_idx * 101
        mode_run_dir = train_root / mode
        mode_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[mappo] training {mode} with seed {train_seed} for {num_env_steps} env steps...")
        runner = train_reference_mappo(
            mode=mode,
            n_agents=n_agents,
            steps=steps,
            seed=train_seed,
            num_env_steps=num_env_steps,
            run_dir=mode_run_dir,
            rollout_threads=rollout_threads,
            training_threads=training_threads,
        )
        controller = ReferenceMAPPOController(runner, n_agents=n_agents)

        for trial_idx in range(trials):
            seed = base_seed + trial_idx * 1000 + mode_idx * 101
            print(f"[mappo] evaluating {mode} trial {trial_idx + 1}/{trials} with seed {seed}...")
            result = rollout_controller(
                baseline_name="mappo",
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
                print(f"Saved mappo {mode} metrics to: {metric_path}")

        runner.envs.close()
        if hasattr(runner, "writter"):
            runner.writter.close()

    trial_csv = write_trial_csv(rows, results_dir)
    summary_csv = write_summary_csv(rows, results_dir)
    trajectory_pdf = plot_trajectories(representative, figures_dir, "mappo")
    diagnostics_pdf = plot_diagnostics(representative, figures_dir, "mappo")

    print(f"Saved mappo trial-level CSV to: {trial_csv}")
    print(f"Saved mappo summary CSV to: {summary_csv}")
    print(f"Saved mappo trajectories PDF to: {trajectory_pdf}")
    print(f"Saved mappo diagnostics PDF to: {diagnostics_pdf}")
