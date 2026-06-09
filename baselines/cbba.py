from __future__ import annotations

import math

import numpy as np

from baselines.common import policy_entropy, vector_to_action_distribution
from src.coverage import _soft_boundary_inward_field


def _ring_slots(center: np.ndarray, count: int, radius: float) -> list[np.ndarray]:
    slots: list[np.ndarray] = []
    for slot_idx in range(max(count, 1)):
        angle = (2.0 * np.pi * slot_idx) / max(count, 1)
        slots.append(center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float))
    return slots


class DistributedAuctionCBBAController:
    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
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
        slots = self._build_slots(mode, landmark_positions)
        assignments, bid_distributions = self._auction_assignments(agent_positions, slots)

        actions: dict[str, np.ndarray] = {}
        entropies: list[float] = []
        for agent_idx in range(self.n_agents):
            slot = slots[assignments[agent_idx]]
            repulsion = self._repulsion(agent_idx, agent_positions)
            boundary = _soft_boundary_inward_field(agent_positions[agent_idx], 100.0)
            vector = 0.92 * (slot - agent_positions[agent_idx]) + 9.5 * repulsion + 0.50 * boundary
            probs = vector_to_action_distribution(vector, pinpoint_mass=0.04 if mode != "dispersion" else 0.06)
            action = np.zeros(5, dtype=np.float32)
            action[0] = float(probs[4])
            action[1] = float(np.clip(probs[3], 0.0, 1.0))
            action[2] = float(np.clip(probs[2], 0.0, 1.0))
            action[3] = float(np.clip(probs[1], 0.0, 1.0))
            action[4] = float(np.clip(probs[0], 0.0, 1.0))
            actions[f"agent_{agent_idx}"] = action
            entropies.append(policy_entropy(bid_distributions[agent_idx]))

        return actions, {"entropy": float(np.mean(entropies))}

    def _build_slots(self, mode: str, landmark_positions: np.ndarray) -> list[np.ndarray]:
        if mode == "tracking":
            return _ring_slots(landmark_positions[0], self.n_agents, 9.0)

        if mode == "multi_goal":
            slots_per_goal = max(1, math.ceil(self.n_agents / max(len(landmark_positions), 1)))
            slots: list[np.ndarray] = []
            for goal in landmark_positions:
                slots.extend(_ring_slots(goal, slots_per_goal, 6.5))
            return slots

        slots_per_anchor = max(1, math.ceil(self.n_agents / max(len(landmark_positions), 1)))
        slots = []
        for anchor in landmark_positions:
            slots.extend(_ring_slots(anchor, slots_per_anchor, 2.5))
        return slots

    def _auction_assignments(
        self,
        agent_positions: np.ndarray,
        slots: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        slot_array = np.asarray(slots, dtype=float)
        utility = -np.linalg.norm(
            agent_positions[:, np.newaxis, :] - slot_array[np.newaxis, :, :],
            axis=2,
        )

        bid_distributions = np.empty((self.n_agents, len(slots) + 1), dtype=float)
        for agent_idx in range(self.n_agents):
            raw = utility[agent_idx]
            stabilized = raw - np.max(raw)
            weights = np.exp(stabilized)
            weights /= np.sum(weights)
            bid_distributions[agent_idx, : len(slots)] = 0.96 * weights
            bid_distributions[agent_idx, -1] = 0.04

        assignments = np.full(self.n_agents, -1, dtype=int)
        available_agents = set(range(self.n_agents))
        available_slots = set(range(len(slots)))

        while available_agents and available_slots:
            proposals: dict[int, list[tuple[int, float]]] = {}
            for agent_idx in available_agents:
                best_slot = max(available_slots, key=lambda slot_idx: utility[agent_idx, slot_idx])
                proposals.setdefault(best_slot, []).append((agent_idx, utility[agent_idx, best_slot]))

            winners: list[tuple[int, int]] = []
            for slot_idx, bidders in proposals.items():
                agent_idx, _ = max(
                    bidders,
                    key=lambda item: (item[1], -item[0]),
                )
                winners.append((agent_idx, slot_idx))

            for agent_idx, slot_idx in winners:
                if agent_idx in available_agents and slot_idx in available_slots:
                    assignments[agent_idx] = slot_idx
                    available_agents.remove(agent_idx)
                    available_slots.remove(slot_idx)

        if np.any(assignments < 0):
            for agent_idx in np.flatnonzero(assignments < 0):
                assignments[agent_idx] = int(np.argmax(utility[agent_idx]))

        return assignments, bid_distributions

    def _repulsion(self, agent_idx: int, agent_positions: np.ndarray) -> np.ndarray:
        position = agent_positions[agent_idx]
        others = np.delete(agent_positions, agent_idx, axis=0)
        if others.size == 0:
            return np.zeros(2, dtype=float)
        deltas = position - others
        distances = np.maximum(np.linalg.norm(deltas, axis=1), 1.0)
        weights = 1.0 / np.power(distances, 2)
        return np.sum(deltas * weights[:, np.newaxis], axis=0)
