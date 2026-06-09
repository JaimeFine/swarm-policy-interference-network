import numpy as np
from .tensor import (
    build_clique_mps,
    compute_reduced_density_matrices_from_mps,
    project_density_to_amplitudes,
)


class LocalQuantumState:
    def __init__(self, n_behaviors):
        self.amplitudes = np.full(
            n_behaviors, 1.0 / np.sqrt(n_behaviors) + 0.0j, dtype=np.complex128
        )
        self.density_matrix = np.eye(
            n_behaviors, dtype=np.complex128
        ) / n_behaviors


def evaluate_born_probabilities(state):
    probs = np.abs(state.amplitudes) ** 2
    total = np.sum(probs)
    if total > 0.0:
        probs /= total
    return probs


def apply_radon_nikodym_filter(state, target_measure, gamma, delta_t=0.15):
    prior_probs = evaluate_born_probabilities(state)
    safe_prior = np.maximum(prior_probs, 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.sqrt(target_measure / safe_prior)

    g_clamped = np.minimum(gain, gamma)
    smooth_drive = np.exp((g_clamped - 1.0) * delta_t)
    state.amplitudes *= smooth_drive

    norm_val = np.linalg.norm(state.amplitudes)
    if norm_val > 1e-8:
        state.amplitudes /= norm_val
    else:
        n = len(state.amplitudes)
        state.amplitudes = np.full(
            n, 1.0 / np.sqrt(n) + 0.0j, dtype=np.complex128
        )


def compute_clique_reduced_densities(states, clique, positions, mode):
    """
    SPIN-Exact clique evolution:
    1. Build a clique-level MPS over the full overlapping clique.
    2. Contract it exactly into the joint wavefunction.
    3. Compute one reduced density matrix per clique member by partial trace.
    """
    cores = build_clique_mps(states, clique, positions, mode)
    return compute_reduced_density_matrices_from_mps(cores)


def density_matrix_from_amplitudes(amplitudes):
    rho = np.outer(amplitudes, np.conj(amplitudes))
    rho = 0.5 * (rho + np.conj(rho.T))
    trace = np.trace(rho)
    if np.abs(trace) > 1e-12:
        rho = rho / trace
    return rho


def _project_physical_density(rho):
    rho = 0.5 * (rho + np.conj(rho.T))
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals.real, 0.0, None)
    total = np.sum(eigvals)
    if total <= 1e-12:
        n = rho.shape[0]
        return np.eye(n, dtype=np.complex128) / n
    projected = eigvecs @ np.diag(eigvals / total) @ np.conj(eigvecs.T)
    return 0.5 * (projected + np.conj(projected.T))


def _trace_distance(rho_a, rho_b):
    diff = 0.5 * ((rho_a - rho_b) + np.conj((rho_a - rho_b).T))
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def _optimize_trace_distance_consensus(contributions, prior_rho, max_iters=24, tol=1e-6):
    """
    Approximate the trace-distance Fréchet median over overlapping clique
    density contributions. This replaces the old arithmetic mean shortcut with
    an iterative consensus update that explicitly minimizes synchronization
    discrepancy under the trace norm.
    """
    rho = _project_physical_density(prior_rho)
    projected = [_project_physical_density(contribution) for contribution in contributions]
    if len(projected) == 1:
        return projected[0]

    for _ in range(max_iters):
        distances = np.array(
            [_trace_distance(rho, contribution) for contribution in projected],
            dtype=float,
        )
        if np.max(distances) <= tol:
            break

        weights = 1.0 / np.maximum(distances, 1e-6)
        candidate = sum(
            weight * contribution
            for weight, contribution in zip(weights, projected)
        ) / np.sum(weights)
        candidate = _project_physical_density(candidate)

        if _trace_distance(candidate, rho) <= tol:
            rho = candidate
            break
        rho = candidate

    return rho


def reconcile_overlapping_densities(states, density_contributions):
    """
    Merge all reduced density contributions for each agent. Overlapping clique
    memberships are reconciled at the density-matrix level before projection
    back to the executable local amplitude state.
    """
    for agent_idx, contributions in density_contributions.items():
        if contributions:
            prior_rho = density_matrix_from_amplitudes(states[agent_idx].amplitudes)
            rho = _optimize_trace_distance_consensus(contributions, prior_rho)
        else:
            rho = density_matrix_from_amplitudes(states[agent_idx].amplitudes)

        states[agent_idx].density_matrix = rho
        states[agent_idx].amplitudes = project_density_to_amplitudes(rho, states[agent_idx])
