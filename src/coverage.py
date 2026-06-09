from __future__ import annotations

import numpy as np


def _interior_anchor_centers(arena_size: float, grid_size: int = 6) -> np.ndarray:
    axis = np.linspace(0.0, arena_size, grid_size + 1)
    centers = 0.5 * (axis[:-1] + axis[1:])
    grid_x, grid_y = np.meshgrid(centers[1:-1], centers[1:-1], indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _agent_anchor(agent_idx: int, arena_size: float, grid_size: int = 6) -> np.ndarray:
    anchors = _interior_anchor_centers(arena_size, grid_size=grid_size)
    return anchors[agent_idx % len(anchors)]


def _soft_boundary_inward_field(position: np.ndarray, arena_size: float) -> np.ndarray:
    """
    Apply a smooth inward potential near the walls so agents do not exploit the
    clipping boundary as a fake coverage solution.
    """
    inward = np.zeros(2, dtype=float)
    margin = 0.18 * arena_size
    gain = 6.0

    if position[0] < margin:
        inward[0] += gain * ((margin - position[0]) / margin) ** 2
    elif position[0] > arena_size - margin:
        inward[0] -= gain * (
            (position[0] - (arena_size - margin)) / margin
        ) ** 2

    if position[1] < margin:
        inward[1] += gain * ((margin - position[1]) / margin) ** 2
    elif position[1] > arena_size - margin:
        inward[1] -= gain * (
            (position[1] - (arena_size - margin)) / margin
        ) ** 2

    return inward


def compute_exploration_observation(
    agent_idx: int,
    positions: np.ndarray,
    adj_matrix: np.ndarray,
    arena_size: float,
    r_sense: float,
) -> np.ndarray:
    """
    Build a closed-loop coverage observation for the exploration regime.
    Nearby neighbors induce local repulsion, sparse interior regions induce
    vacancy attraction, and a soft inward boundary field prevents wall-hugging.
    """
    position = positions[agent_idx]
    neighbors = np.flatnonzero(adj_matrix[agent_idx])
    anchor_vector = _agent_anchor(agent_idx, arena_size) - position
    boundary_field = _soft_boundary_inward_field(position, arena_size)

    if neighbors.size > 0:
        offsets = position - positions[neighbors]
        distances = np.linalg.norm(offsets, axis=1)
        weights = np.clip((r_sense - distances) / max(r_sense, 1e-6), 0.0, None)
        weights /= np.maximum(distances, 1.0)
        repulsion = np.sum(offsets * weights[:, np.newaxis], axis=0)
        center_of_mass = np.mean(positions[neighbors], axis=0)
        local_escape = position - center_of_mass
        observation = (
            1.25 * repulsion
            + 0.18 * anchor_vector
            + 0.12 * local_escape
        )
    else:
        observation = 0.90 * anchor_vector

    return observation + 1.20 * boundary_field


def directional_action_distribution(vector: np.ndarray) -> np.ndarray:
    """
    Convert a 2D control vector into a distribution over
    [north, south, east, west].
    """
    dx, dy = vector
    weights = np.array(
        [max(dy, 0.0), max(-dy, 0.0), max(dx, 0.0), max(-dx, 0.0)],
        dtype=float,
    )
    total = np.sum(weights)
    if total <= 1e-12:
        return np.full(4, 0.25, dtype=float)
    return weights / total


def construct_exploration_measure(
    base_measure: np.ndarray,
    obs_vector: np.ndarray,
) -> np.ndarray:
    """
    Reuse the frozen perceptual front-end while replacing the static zero-input
    exploration bias with a crowd-aware, interior-covering cardinal drive.
    """
    cardinal_base = np.asarray(base_measure[:4], dtype=float).copy()
    base_total = np.sum(cardinal_base)
    if base_total <= 1e-12:
        cardinal_base[:] = 0.25
    else:
        cardinal_base /= base_total

    directional = directional_action_distribution(obs_vector)
    cardinal_mix = 0.25 * cardinal_base + 0.75 * directional
    cardinal_mix /= np.sum(cardinal_mix)

    magnitude = float(np.linalg.norm(obs_vector))
    pinpoint_mass = 0.07 + 0.10 * np.exp(-magnitude / 10.0)
    pinpoint_mass = float(np.clip(pinpoint_mass, 0.07, 0.17))

    measure = np.zeros(5, dtype=float)
    measure[:4] = (1.0 - pinpoint_mass) * cardinal_mix
    measure[4] = pinpoint_mass
    return measure


def density_coherence_strength(rho: np.ndarray) -> float:
    """
    Normalize the magnitude of off-diagonal coherence terms into [0, 1].
    """
    off_diagonal = rho - np.diag(np.diag(rho))
    scale = rho.shape[0] * (rho.shape[0] - 1)
    if scale <= 0:
        return 0.0
    strength = np.sum(np.abs(off_diagonal)) / scale
    return float(np.clip(strength, 0.0, 1.0))


def modulate_exploration_probabilities(
    probs: np.ndarray,
    rho: np.ndarray,
    obs_vector: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    """
    Use topology for direction and density-matrix coherence for strength,
    shifting exploration away from stagnant pinpoint behavior.
    """
    adjusted = np.asarray(probs, dtype=float).copy()
    directional = directional_action_distribution(obs_vector)

    cardinal = adjusted[:4].copy()
    cardinal_total = np.sum(cardinal)
    if cardinal_total <= 1e-12:
        cardinal[:] = 0.25
    else:
        cardinal /= cardinal_total

    coherence = density_coherence_strength(rho)
    crowd_factor = 1.0 - np.exp(-max(neighbor_count, 0) / 2.0)
    blend = np.clip(0.24 + 0.34 * coherence + 0.22 * crowd_factor, 0.24, 0.82)
    cardinal = (1.0 - blend) * cardinal + blend * directional
    cardinal /= np.sum(cardinal)

    suppression = np.clip(
        0.08 + 0.24 * coherence + 0.22 * crowd_factor,
        0.0,
        0.65,
    )
    adjusted[4] *= 1.0 - suppression
    adjusted[:4] = (1.0 - adjusted[4]) * cardinal
    return adjusted


def compute_spatial_entropy(
    positions: np.ndarray,
    arena_size: float,
    grid_size: int = 10,
) -> float:
    """
    Normalized Shannon entropy of spatial occupancy across a fixed grid.
    """
    edges = np.linspace(0.0, arena_size, grid_size + 1)
    hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=(edges, edges))
    probs = hist.ravel() / max(np.sum(hist), 1.0)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(grid_size * grid_size))


def compute_voronoi_area_variance(
    positions: np.ndarray,
    arena_size: float,
    resolution: int = 32,
) -> float:
    """
    Approximate Voronoi-cell area variance by rasterizing the arena and
    assigning each sample point to its nearest agent.
    """
    axis = np.linspace(0.0, arena_size, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    sample_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    deltas = sample_points[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(deltas ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    cell_counts = np.bincount(nearest, minlength=positions.shape[0]).astype(float)

    cell_area = (arena_size / max(resolution - 1, 1)) ** 2
    return float(np.var(cell_counts * cell_area))
