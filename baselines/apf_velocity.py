from __future__ import annotations

import numpy as np

from baselines.common import policy_entropy, vector_to_action_distribution
from src.coverage import _soft_boundary_inward_field


def _formation_slot(center: np.ndarray, slot_idx: int, slot_count: int, radius: float) -> np.ndarray:
    angle = (2.0 * np.pi * slot_idx) / max(slot_count, 1)
    return center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)


class APFVelocityController:
    def __init__(self, n_agents: int, sensing_radius: float = 15.0) -> None:
        self.n_agents = n_agents
        self.sensing_radius = sensing_radius
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(
        self,
        mode: str,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
        step_idx: int,
        agent_velocities: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        del step_idx, agent_velocities
        actions: dict[str, np.ndarray] = {}
        entropies: list[float] = []

        for agent_idx in range(self.n_agents):
            vector = self._control_vector(mode, agent_idx, agent_positions, landmark_positions)
            probs = vector_to_action_distribution(vector, pinpoint_mass=0.05 if mode != "dispersion" else 0.08)
            action = np.zeros(5, dtype=np.float32)
            action[0] = float(probs[4])
            action[1] = float(np.clip(probs[3], 0.0, 1.0))
            action[2] = float(np.clip(probs[2], 0.0, 1.0))
            action[3] = float(np.clip(probs[1], 0.0, 1.0))
            action[4] = float(np.clip(probs[0], 0.0, 1.0))
            actions[f"agent_{agent_idx}"] = action
            entropies.append(policy_entropy(probs))

        return actions, {"entropy": float(np.mean(entropies))}

    def _control_vector(
        self,
        mode: str,
        agent_idx: int,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
    ) -> np.ndarray:
        position = agent_positions[agent_idx]
        repulsion = self._inverse_cube_repulsion(agent_idx, agent_positions)
        boundary = _soft_boundary_inward_field(position, 100.0)

        if mode == "tracking":
            goal = landmark_positions[0]
            desired = _formation_slot(
                goal,
                slot_idx=agent_idx,
                slot_count=self.n_agents,
                radius=9.0,
            )
            return 0.95 * (desired - position) + 16.0 * repulsion + 0.55 * boundary

        if mode == "multi_goal":
            distances = np.linalg.norm(landmark_positions - position, axis=1)
            goal_idx = int(np.argmin(distances))
            peers_to_goal = np.flatnonzero(
                np.argmin(
                    np.linalg.norm(
                        agent_positions[:, np.newaxis, :] - landmark_positions[np.newaxis, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                == goal_idx
            )
            slot_idx = int(np.where(peers_to_goal == agent_idx)[0][0]) if agent_idx in peers_to_goal else 0
            desired = _formation_slot(landmark_positions[goal_idx], slot_idx, len(peers_to_goal), 6.5)
            return 0.92 * (desired - position) + 14.0 * repulsion + 0.40 * boundary

        landmark_dist = np.linalg.norm(
            landmark_positions[np.newaxis, :, :] - agent_positions[:, np.newaxis, :],
            axis=2,
        )
        coverage_gaps = np.min(landmark_dist, axis=0)
        ranked = np.argsort(-coverage_gaps)
        anchor_idx = int(ranked[agent_idx % len(ranked)])
        anchor_vector = landmark_positions[anchor_idx] - position
        return 0.55 * anchor_vector + 18.0 * repulsion + 0.85 * boundary

    def _inverse_cube_repulsion(self, agent_idx: int, agent_positions: np.ndarray) -> np.ndarray:
        position = agent_positions[agent_idx]
        others = np.delete(agent_positions, agent_idx, axis=0)
        if others.size == 0:
            return np.zeros(2, dtype=float)

        deltas = position - others
        distances = np.linalg.norm(deltas, axis=1)
        distances = np.maximum(distances, 1.0)
        weights = np.where(distances <= self.sensing_radius, 1.0 / np.power(distances, 3), 0.0)
        return np.sum(deltas * weights[:, np.newaxis], axis=0)
