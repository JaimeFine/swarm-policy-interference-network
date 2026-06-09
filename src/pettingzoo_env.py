from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .coverage import compute_spatial_entropy, compute_voronoi_area_variance
from .spin_policy import SpinPolicyController

ARENA_SIZE = 100.0
ACTION_SCALE = 1.35
MAX_SPEED = 2.6
VELOCITY_DAMPING = 0.72
AGENT_BODY_RADIUS = 2.0


def _is_dispersion_mode(mode: str) -> bool:
    return mode in {"dispersion", "exploration"}


@dataclass(frozen=True)
class ScenarioConfig:
    mode: str
    n_agents: int
    max_cycles: int
    n_landmarks: int


def _sample_points(
    rng: np.random.Generator,
    count: int,
    *,
    low: float,
    high: float,
    min_dist: float,
) -> np.ndarray:
    span = float(high - low)
    return _structured_random_points(
        rng,
        count,
        low=np.array([low, low], dtype=float),
        high=np.array([high, high], dtype=float),
        min_dist=min_dist,
        region_span=np.array([span, span], dtype=float),
    )


def _structured_random_points(
    rng: np.random.Generator,
    count: int,
    *,
    low: np.ndarray,
    high: np.ndarray,
    min_dist: float,
    region_span: np.ndarray,
) -> np.ndarray:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    region_span = np.asarray(region_span, dtype=float)

    lower_limit = low
    upper_limit = high - region_span
    if np.any(upper_limit < lower_limit):
        raise ValueError("Requested region span does not fit inside the arena bounds.")

    region_origin = rng.uniform(lower_limit, upper_limit)
    region_high = region_origin + region_span
    width = float(region_high[0] - region_origin[0])
    height = float(region_high[1] - region_origin[1])

    best_layout: tuple[int, int] | None = None
    best_score = float("inf")
    for rows in range(1, count + 1):
        cols = int(np.ceil(count / rows))
        cell_w = width / cols
        cell_h = height / rows
        if cell_w + 1e-9 < min_dist or cell_h + 1e-9 < min_dist:
            continue
        score = abs(rows - cols) + 0.01 * (rows * cols - count)
        if score < best_score:
            best_score = score
            best_layout = (rows, cols)

    if best_layout is None:
        raise ValueError(
            f"Cannot place {count} points with minimum distance {min_dist} in region {region_span}."
        )

    rows, cols = best_layout
    cell_w = width / cols
    cell_h = height / rows
    jitter_x = 0.5 * max(cell_w - min_dist, 0.0)
    jitter_y = 0.5 * max(cell_h - min_dist, 0.0)

    cells = [(row, col) for row in range(rows) for col in range(cols)]
    selected = rng.choice(len(cells), size=count, replace=False)

    points = []
    for cell_idx in selected:
        row, col = cells[int(cell_idx)]
        center = np.array(
            [
                region_origin[0] + (col + 0.5) * cell_w,
                region_origin[1] + (row + 0.5) * cell_h,
            ],
            dtype=float,
        )
        jitter = np.array(
            [
                rng.uniform(-jitter_x, jitter_x) if jitter_x > 0.0 else 0.0,
                rng.uniform(-jitter_y, jitter_y) if jitter_y > 0.0 else 0.0,
            ],
            dtype=float,
        )
        points.append(center + jitter)

    return np.asarray(points, dtype=float)


def _random_agent_positions(
    rng: np.random.Generator,
    n_agents: int,
    mode: str,
) -> np.ndarray:
    del mode

    positions: list[np.ndarray] = []
    min_center_distance = 2.0 * AGENT_BODY_RADIUS

    while len(positions) < n_agents:
        candidate = rng.uniform(
            AGENT_BODY_RADIUS,
            ARENA_SIZE - AGENT_BODY_RADIUS,
            size=2,
        )
        if all(
            np.linalg.norm(candidate - existing) >= min_center_distance
            for existing in positions
        ):
            positions.append(candidate)

    return np.asarray(positions, dtype=float)


def _random_static_landmarks(
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "multi_goal":
        return _sample_points(rng, 3, low=16.0, high=84.0, min_dist=24.0)
    return _sample_points(rng, 9, low=14.0, high=86.0, min_dist=12.0)


def compute_mean_trajectory_length(trajectories: np.ndarray) -> float:
    path_segments = np.diff(np.asarray(trajectories, dtype=float), axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=2).sum(axis=0)
    return float(np.mean(path_lengths))


class SpinParallelEnv(ParallelEnv):
    metadata = {"name": "spin_parallel_v0", "render_modes": []}

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.possible_agents = [f"agent_{idx}" for idx in range(config.n_agents)]
        self.agents = self.possible_agents[:]
        self._action_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        obs_dim = 4 + 2 * config.n_landmarks + 2 * (config.n_agents - 1)
        self._observation_space = spaces.Box(
            low=-ARENA_SIZE,
            high=ARENA_SIZE,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.np_random = np.random.default_rng()
        self.step_count = 0
        self.agent_positions = np.zeros((config.n_agents, 2), dtype=float)
        self.agent_velocities = np.zeros((config.n_agents, 2), dtype=float)
        self.landmark_positions = np.zeros((config.n_landmarks, 2), dtype=float)
        self._tracking_center = np.zeros(2, dtype=float)
        self._tracking_swing = np.zeros(2, dtype=float)
        self._tracking_phase = 0.0
        self._static_landmarks = np.zeros((config.n_landmarks, 2), dtype=float)

    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent: str):
        return self._action_space

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.agent_positions = _random_agent_positions(
            self.np_random, self.config.n_agents, self.config.mode
        )
        self.agent_velocities = np.zeros_like(self.agent_positions)
        self._sample_scenario_parameters()
        self.landmark_positions = self._landmarks_for_step(self.step_count)
        observations = {
            agent: self._build_observation(idx)
            for idx, agent in enumerate(self.agents)
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        for idx, agent in enumerate(self.agents):
            action = np.asarray(actions.get(agent, np.zeros(5, dtype=np.float32)))
            acceleration = np.array(
                [action[2] - action[1], action[4] - action[3]], dtype=float
            )
            self.agent_velocities[idx] = (
                VELOCITY_DAMPING * self.agent_velocities[idx]
                + ACTION_SCALE * acceleration
            )
            speed = np.linalg.norm(self.agent_velocities[idx])
            if speed > MAX_SPEED:
                self.agent_velocities[idx] *= MAX_SPEED / speed
            self.agent_positions[idx] += self.agent_velocities[idx]
            self._enforce_bounds(idx)

        self._resolve_agent_overlaps()

        self.step_count += 1
        self.landmark_positions = self._landmarks_for_step(self.step_count)

        observations = {
            agent: self._build_observation(idx)
            for idx, agent in enumerate(self.agents)
        }
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncated = self.step_count >= self.config.max_cycles
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self):
        return None

    def _sample_scenario_parameters(self) -> None:
        mode = self.config.mode
        if mode == "tracking":
            self._tracking_center = _sample_points(
                self.np_random, 1, low=35.0, high=65.0, min_dist=0.0
            )[0]
            self._tracking_swing = self.np_random.uniform(8.0, 16.0, size=2)
            self._tracking_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            self._static_landmarks = np.zeros((1, 2), dtype=float)
        else:
            self._static_landmarks = _random_static_landmarks(mode, self.np_random)

    def _landmarks_for_step(self, step_idx: int) -> np.ndarray:
        if self.config.mode == "tracking":
            oscillation = np.array(
                [
                    self._tracking_swing[0]
                    * np.sin(0.050 * step_idx + self._tracking_phase),
                    self._tracking_swing[1]
                    * np.cos(0.037 * step_idx + 0.5 * self._tracking_phase),
                ],
                dtype=float,
            )
            target = np.clip(
                self._tracking_center + oscillation,
                10.0,
                ARENA_SIZE - 10.0,
            )
            return target[np.newaxis, :]
        return self._static_landmarks.copy()

    def _build_observation(self, idx: int) -> np.ndarray:
        position = self.agent_positions[idx]
        velocity = self.agent_velocities[idx]
        rel_landmarks = (self.landmark_positions - position).reshape(-1)
        rel_agents = np.delete(self.agent_positions, idx, axis=0) - position
        rel_agents = rel_agents.reshape(-1)
        return np.concatenate([position, velocity, rel_landmarks, rel_agents]).astype(
            np.float32
        )

    def _enforce_bounds(self, idx: int) -> None:
        for axis in range(2):
            if self.agent_positions[idx, axis] < 0.0:
                self.agent_positions[idx, axis] = 0.0
                self.agent_velocities[idx, axis] *= -0.35
            elif self.agent_positions[idx, axis] > ARENA_SIZE:
                self.agent_positions[idx, axis] = ARENA_SIZE
                self.agent_velocities[idx, axis] *= -0.35

    def _resolve_agent_overlaps(self) -> None:
        min_distance = 2.0 * AGENT_BODY_RADIUS
        for _ in range(3):
            moved = False
            for i in range(self.config.n_agents):
                for j in range(i + 1, self.config.n_agents):
                    delta = self.agent_positions[j] - self.agent_positions[i]
                    distance = float(np.linalg.norm(delta))
                    if distance >= min_distance:
                        continue

                    if distance < 1e-8:
                        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                        normal = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                    else:
                        normal = delta / distance

                    penetration = min_distance - distance
                    correction = 0.5 * (penetration + 1e-6) * normal
                    self.agent_positions[i] -= correction
                    self.agent_positions[j] += correction
                    self.agent_velocities[i] *= 0.60
                    self.agent_velocities[j] *= 0.60
                    self._enforce_bounds(i)
                    self._enforce_bounds(j)
                    moved = True

            if not moved:
                break


def compute_task_metric(
    mode: str, positions: np.ndarray, targets: np.ndarray
) -> tuple[float, str]:
    if mode in {"tracking", "multi_goal"}:
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2
        )
        return float(np.mean(np.min(distances, axis=1))), "Mean target distance"

    return compute_spatial_entropy(positions, ARENA_SIZE), "Spatial entropy"


def compute_target_distance_statistics(
    mode: str, positions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    if mode not in {"tracking", "multi_goal"} or targets.size == 0:
        return {
            "distance_mean": float("nan"),
            "distance_variance": float("nan"),
            "distance_p90": float("nan"),
            "distance_p90_tail_mean": float("nan"),
            "distance_mad": float("nan"),
        }

    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2
    )
    nearest_distances = np.min(distances, axis=1)
    mean_distance = float(np.mean(nearest_distances))
    p90_distance = float(np.percentile(nearest_distances, 90.0))
    tail = nearest_distances[nearest_distances >= p90_distance]
    return {
        "distance_mean": mean_distance,
        "distance_variance": float(np.var(nearest_distances)),
        "distance_p90": p90_distance,
        "distance_p90_tail_mean": float(np.mean(tail)),
        "distance_mad": float(np.mean(np.abs(nearest_distances - mean_distance))),
    }


def prefix_metric_keys(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def scenario_targets(mode: str, landmark_positions: np.ndarray) -> np.ndarray:
    if mode == "tracking":
        return landmark_positions[[0]].copy()
    if mode == "multi_goal":
        return landmark_positions.copy()
    return np.empty((0, 2), dtype=float)


def _make_env(mode: str, n_agents: int, steps: int) -> SpinParallelEnv:
    if mode == "tracking":
        n_landmarks = 1
    elif mode == "multi_goal":
        n_landmarks = 3
    elif _is_dispersion_mode(mode):
        n_landmarks = 9
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SpinParallelEnv(
        ScenarioConfig(
            mode=mode,
            n_agents=n_agents,
            max_cycles=steps,
            n_landmarks=n_landmarks,
        )
    )


def run_internal_episode(
    *,
    phi_omega,
    mode: str,
    n_agents: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray | str | list[list[int]] | float]:
    env = _make_env(mode=mode, n_agents=n_agents, steps=steps)
    env.reset(seed=seed)
    controller = SpinPolicyController(phi_omega=phi_omega, n_agents=n_agents)
    controller.reset()

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
        actions, diagnostics = controller.act(mode, positions, landmark_positions)
        _, _, terminations, truncations, _ = env.step(actions)

        positions = env.agent_positions.copy()
        targets = scenario_targets(mode, env.landmark_positions)
        task_value, task_label = compute_task_metric(mode, positions, targets)

        cliques = diagnostics["cliques"]
        clique_counts[step] = len(cliques)
        mean_clique_sizes[step] = float(np.mean([len(clique) for clique in cliques]))
        task_curve[step] = task_value
        entropy_curve[step] = float(diagnostics["entropy"])
        spatial_entropy_curve[step] = compute_spatial_entropy(positions, ARENA_SIZE)
        voronoi_variance_curve[step] = compute_voronoi_area_variance(
            positions, ARENA_SIZE
        )
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
        **prefix_metric_keys("initial", initial_distance_stats),
        **prefix_metric_keys("final", final_distance_stats),
        "final_entropy": float(entropy_curve[-1]),
        "final_clique_count": float(clique_counts[-1]),
        "final_mean_clique_size": float(mean_clique_sizes[-1]),
        "final_spatial_entropy": float(spatial_entropy_curve[-1]),
        "final_voronoi_area_variance": float(voronoi_variance_curve[-1]),
        "mean_trajectory_length": mean_trajectory_length,
    }
