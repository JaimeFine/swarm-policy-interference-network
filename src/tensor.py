import numpy as np


def _normalize_vector(values):
    norm = np.linalg.norm(values)
    if norm > 1e-12:
        return values / norm
    return np.full(values.shape, 1.0 / np.sqrt(values.size), dtype=np.complex128)


def _destructive_relative_indices(clique_positions, local_idx):
    rel = clique_positions - clique_positions[local_idx]
    rel[local_idx] = 0.0
    indices = []
    for peer_idx, (dx, dy) in enumerate(rel):
        if peer_idx == local_idx:
            continue
        if abs(dy) > abs(dx):
            indices.append(0 if dy > 0 else 1)
        else:
            indices.append(2 if dx > 0 else 3)
    return indices


def local_phase_profile(state, clique_positions, local_idx, mode):
    amplitudes = np.asarray(state.amplitudes, dtype=np.complex128).copy()
    n_behaviors = amplitudes.size
    clique_size = max(len(clique_positions), 1)

    if mode == "tracking":
        amplitudes[-1] *= np.exp(1j * np.pi / clique_size)
    elif mode == "exploration":
        for idx in _destructive_relative_indices(clique_positions, local_idx):
            amplitudes[idx] *= -1.0
    elif mode == "multi_goal":
        amplitudes[:-1] *= np.exp(1j * local_idx * np.pi / clique_size)
        amplitudes[-1] *= -1.0

    return _normalize_vector(amplitudes)


def build_clique_mps(states, clique, positions, mode, bond_dim=2):
    """
    Build a clique-level MPS whose bond channels encode the mode-dependent
    phase-coupled interaction structure without enumerating the full joint
    tensor.
    """
    clique_positions = positions[np.asarray(clique)]
    profiles = [
        local_phase_profile(states[agent_idx], clique_positions, local_idx, mode)
        for local_idx, agent_idx in enumerate(clique)
    ]

    n_behaviors = profiles[0].size
    cores = []
    for local_idx, profile in enumerate(profiles):
        if len(clique) == 1:
            core = profile.reshape(1, n_behaviors, 1)
            cores.append(core)
            continue

        if local_idx == 0:
            core = np.zeros((1, n_behaviors, bond_dim), dtype=np.complex128)
            core[0, :, 0] = profile
            core[0, :, 1] = profile
        elif local_idx == len(clique) - 1:
            core = np.zeros((bond_dim, n_behaviors, 1), dtype=np.complex128)
            core[0, :, 0] = profile
            core[1, :, 0] = profile
        else:
            core = np.zeros((bond_dim, n_behaviors, bond_dim), dtype=np.complex128)
            core[0, :, 0] = profile
            core[1, :, 1] = profile

        if mode == "tracking":
            core[..., -1] *= np.exp(1j * np.pi / max(len(clique), 1))
        elif mode == "exploration":
            for idx in _destructive_relative_indices(clique_positions, local_idx):
                core[:, idx, :] *= -1.0
        elif mode == "multi_goal":
            core[:, -1, :] *= -1.0

        cores.append(core)

    return cores


def compute_reduced_density_matrices_from_mps(cores):
    """
    Compute one-site reduced density matrices directly from the clique MPS via
    left/right environment contractions.
    """
    n_sites = len(cores)
    local_dim = cores[0].shape[1]

    left_envs = [np.ones((1, 1), dtype=np.complex128)]
    for core in cores[:-1]:
        left_env = np.einsum(
            "ab, asr, bsu -> ru",
            left_envs[-1],
            core,
            np.conj(core),
        )
        left_envs.append(left_env)

    right_envs = [None] * n_sites
    right_env = np.ones((1, 1), dtype=np.complex128)
    for site in range(n_sites - 1, -1, -1):
        right_envs[site] = right_env
        core = cores[site]
        right_env = np.einsum(
            "rt, lsr, mst -> lm",
            right_env,
            core,
            np.conj(core),
        )

    reduced = []
    for site, core in enumerate(cores):
        rho = np.einsum(
            "ab, asr, btu, ru -> st",
            left_envs[site],
            core,
            np.conj(core),
            right_envs[site],
        )
        rho = 0.5 * (rho + np.conj(rho.T))
        trace = np.trace(rho)
        if np.abs(trace) > 1e-12:
            rho = rho / trace
        else:
            rho = np.eye(local_dim, dtype=np.complex128) / local_dim
        reduced.append(rho)

    return reduced


def contract_mps_to_joint_state(cores):
    """
    Explicitly contract the MPS into the full joint state tensor.

    This is intended for small-clique benchmarking and validation only.
    """
    if not cores:
        raise ValueError("At least one MPS core is required.")

    joint = np.squeeze(cores[0], axis=0)
    for core in cores[1:]:
        joint = np.einsum("...a, asb -> ...sb", joint, core)
    return np.squeeze(joint, axis=-1)


def compute_reduced_density_matrices_by_enumeration(cores):
    """
    Compute one-site reduced density matrices by explicitly materializing the
    full joint state tensor and marginalizing each site.

    This path is exponentially more expensive in clique size and is meant only
    as a benchmark reference for the compressed MPS contraction routine.
    """
    joint = contract_mps_to_joint_state(cores)
    wave = np.asarray(joint, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(wave)
    if norm > 1e-12:
        joint = joint / norm
    else:
        local_dim = cores[0].shape[1]
        n_sites = len(cores)
        joint = np.full((local_dim,) * n_sites, 1.0 / np.sqrt(local_dim**n_sites), dtype=np.complex128)

    n_sites = joint.ndim
    local_dim = joint.shape[0]
    reduced = []
    for site in range(n_sites):
        psi_site_first = np.moveaxis(joint, site, 0).reshape(local_dim, -1)
        rho = psi_site_first @ np.conj(psi_site_first.T)
        rho = 0.5 * (rho + np.conj(rho.T))
        trace = np.trace(rho)
        if np.abs(trace) > 1e-12:
            rho = rho / trace
        else:
            rho = np.eye(local_dim, dtype=np.complex128) / local_dim
        reduced.append(rho)
    return reduced


def project_density_to_amplitudes(rho, reference_state):
    diag_probs = np.clip(np.real(np.diag(rho)), 0.0, None)
    total = np.sum(diag_probs)
    if total > 1e-12:
        diag_probs /= total
    else:
        diag_probs[:] = 1.0 / diag_probs.size

    phases = np.angle(reference_state.amplitudes)
    return np.sqrt(diag_probs) * np.exp(1j * phases)
