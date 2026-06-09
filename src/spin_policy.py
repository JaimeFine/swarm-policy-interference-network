from __future__ import annotations

import numpy as np

from .coverage import construct_exploration_measure, modulate_exploration_probabilities
from .network import forward_pass
from .quantum import (
    LocalQuantumState,
    apply_radon_nikodym_filter,
    compute_clique_reduced_densities,
    evaluate_born_probabilities,
    reconcile_overlapping_densities,
)
from .topology import compute_adjacency_matrix, extract_overlapping_maximal_cliques

GAMMA_CLAMPING = 5.0
N_BEHAVIORS = 5
R_SENSE = 15.0
MAX_CLIQUE_NEIGHBORS = 8
TRACKING_RING_RADIUS = 9.0
MULTI_GOAL_RING_RADIUS = 6.5


def _is_dispersion_mode(mode: str) -> bool:
    return mode in {"dispersion", "exploration"}


def mean_policy_entropy(states: list[LocalQuantumState]) -> float:
    entropies = []
    for state in states:
        probs = evaluate_born_probabilities(state)
        entropies.append(-np.sum(probs * np.log(probs + 1e-12)))
    return float(np.mean(entropies))


def _assigned_landmark_vector(
    mode: str,
    agent_idx: int,
    agent_positions: np.ndarray,
    landmark_positions: np.ndarray,
) -> np.ndarray:
    if mode == "tracking":
        return landmark_positions[0] - agent_positions[agent_idx]

    if mode == "multi_goal":
        distances = np.linalg.norm(
            landmark_positions - agent_positions[agent_idx], axis=1
        )
        assigned_idx = int(np.argmin(distances))
        return landmark_positions[assigned_idx] - agent_positions[agent_idx]

    landmark_dist = np.linalg.norm(
        landmark_positions[np.newaxis, :, :] - agent_positions[:, np.newaxis, :],
        axis=2,
    )
    coverage_gaps = np.min(landmark_dist, axis=0)
    ranked_landmarks = np.argsort(-coverage_gaps)
    assigned_idx = ranked_landmarks[agent_idx % len(ranked_landmarks)]
    return landmark_positions[assigned_idx] - agent_positions[agent_idx]


def _formation_slot_vector(
    center: np.ndarray,
    agent_idx: int,
    slot_count: int,
    radius: float,
) -> np.ndarray:
    angle = (2.0 * np.pi * agent_idx) / max(slot_count, 1)
    return center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _tracking_vector(
    agent_idx: int,
    agent_positions: np.ndarray,
    landmark_positions: np.ndarray,
) -> np.ndarray:
    target = landmark_positions[0]
    desired_point = _formation_slot_vector(
        target,
        agent_idx=agent_idx,
        slot_count=agent_positions.shape[0],
        radius=TRACKING_RING_RADIUS,
    )
    return desired_point - agent_positions[agent_idx]


def _multi_goal_vector(
    agent_idx: int,
    agent_positions: np.ndarray,
    landmark_positions: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        agent_positions[:, np.newaxis, :] - landmark_positions[np.newaxis, :, :],
        axis=2,
    )
    goal_assignments = np.argmin(distances, axis=1)
    goal_idx = int(goal_assignments[agent_idx])
    target = landmark_positions[goal_idx]
    slot_indices = np.flatnonzero(goal_assignments == goal_idx).tolist()
    local_slot = slot_indices.index(agent_idx)
    desired_point = _formation_slot_vector(
        target,
        agent_idx=local_slot,
        slot_count=len(slot_indices),
        radius=MULTI_GOAL_RING_RADIUS,
    )
    return desired_point - agent_positions[agent_idx]


def _repulsion_vector(agent_idx: int, agent_positions: np.ndarray) -> np.ndarray:
    position = agent_positions[agent_idx]
    others = np.delete(agent_positions, agent_idx, axis=0)
    if others.size == 0:
        return np.zeros(2, dtype=float)

    deltas = position - others
    distances = np.linalg.norm(deltas, axis=1)
    weights = np.exp(-distances / max(R_SENSE, 1e-6))
    weights /= np.maximum(distances, 1.0)
    return np.sum(deltas * weights[:, np.newaxis], axis=0)


def compute_policy_signal(
    mode: str,
    agent_idx: int,
    agent_positions: np.ndarray,
    landmark_positions: np.ndarray,
) -> np.ndarray:
    repulsion = _repulsion_vector(agent_idx, agent_positions)

    if mode == "tracking":
        tracking_drive = _tracking_vector(agent_idx, agent_positions, landmark_positions)
        return 0.90 * tracking_drive + 0.18 * repulsion
    if mode == "multi_goal":
        goal_drive = _multi_goal_vector(agent_idx, agent_positions, landmark_positions)
        return 0.88 * goal_drive + 0.24 * repulsion
    landmark_drive = _assigned_landmark_vector(
        mode, agent_idx, agent_positions, landmark_positions
    )
    return 0.55 * landmark_drive + 0.70 * repulsion


class SpinPolicyController:
    def __init__(self, phi_omega, n_agents: int) -> None:
        self.phi_omega = phi_omega
        self.n_agents = n_agents
        self.states = [LocalQuantumState(N_BEHAVIORS) for _ in range(n_agents)]

    def reset(self) -> None:
        self.states = [LocalQuantumState(N_BEHAVIORS) for _ in range(self.n_agents)]

    def act(
        self,
        mode: str,
        agent_positions: np.ndarray,
        landmark_positions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | list[list[int]]]]:
        adj_matrix = compute_adjacency_matrix(
            agent_positions, R_SENSE, max_neighbors=MAX_CLIQUE_NEIGHBORS
        )
        cliques = extract_overlapping_maximal_cliques(adj_matrix)

        signals = [
            compute_policy_signal(mode, idx, agent_positions, landmark_positions)
            for idx in range(self.n_agents)
        ]

        for idx, signal in enumerate(signals):
            measure_output, _ = forward_pass(self.phi_omega, signal)
            if _is_dispersion_mode(mode):
                measure_output = construct_exploration_measure(measure_output, signal)
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
        for idx, signal in enumerate(signals):
            probs = evaluate_born_probabilities(self.states[idx])
            if _is_dispersion_mode(mode):
                probs = modulate_exploration_probabilities(
                    probs,
                    self.states[idx].density_matrix,
                    signal,
                    int(np.sum(adj_matrix[idx])),
                )
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

    def _to_mpe_action(
        self, probs: np.ndarray, signal: np.ndarray, mode: str
    ) -> np.ndarray:
        """
        Map the SPIN local behavior distribution onto the MPE continuous action
        convention: [noop, left, right, down, up].
        """
        signal = np.asarray(signal, dtype=float)
        norm = np.linalg.norm(signal)
        if norm > 1e-8:
            direction = signal / norm
        else:
            direction = np.zeros(2, dtype=float)

        lateral = probs[2] - probs[3]
        vertical = probs[0] - probs[1]

        if mode == "tracking":
            mix = 0.75
        elif mode == "multi_goal":
            mix = 0.65
        else:
            mix = 0.55

        motion_x = mix * lateral + (1.0 - mix) * direction[0]
        motion_y = mix * vertical + (1.0 - mix) * direction[1]

        confidence = np.clip(
            0.25 + 0.55 * np.max(probs[:4]) + 0.20 * min(norm / 40.0, 1.0),
            0.20,
            0.95,
        )
        motion_x = float(np.tanh(motion_x)) * confidence
        motion_y = float(np.tanh(motion_y)) * confidence

        action = np.zeros(5, dtype=np.float32)
        action[0] = float(np.clip(probs[4], 0.0, 1.0))
        action[1] = float(np.clip(-motion_x, 0.0, 1.0))
        action[2] = float(np.clip(motion_x, 0.0, 1.0))
        action[3] = float(np.clip(-motion_y, 0.0, 1.0))
        action[4] = float(np.clip(motion_y, 0.0, 1.0))
        return action
