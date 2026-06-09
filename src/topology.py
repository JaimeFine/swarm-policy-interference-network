import numpy as np

def compute_adjacency_matrix(positions, r_sense, max_neighbors=None):
    """
    Generates a boolean adjacency mask using vectorized pairwise distance
    checks. Edges represent ad-hoc communication boundaries. If max_neighbors
    is provided, the radius graph is pruned to a mutual k-nearest-neighbor graph
    to bound crowded local interaction degree.
    """
    # Vectorized broadcasting to get N x N x 2 differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    # Contiguous boolean mask
    adj = dist_sq <= (r_sense ** 2)
    np.fill_diagonal(adj, False)    # Remove self-loops
    if max_neighbors is not None:
        max_neighbors = int(max_neighbors)
        if max_neighbors < 1:
            return np.zeros_like(adj, dtype=bool)

        directed_keep = np.zeros_like(adj, dtype=bool)
        for idx in range(adj.shape[0]):
            neighbors = np.flatnonzero(adj[idx])
            if neighbors.size == 0:
                continue
            ranked = neighbors[np.argsort(dist_sq[idx, neighbors])]
            directed_keep[idx, ranked[:max_neighbors]] = True

        adj = directed_keep & directed_keep.T
    return adj

def extract_overlapping_maximal_cliques(adj):
    """
    Enumerates overlapping maximal cliques using Bron-Kerbosch recursion.
    This matches the SPIN-Exact assumption that agents may belong to multiple
    local communication structures simultaneously.
    """
    n = adj.shape[0]
    neighbor_sets = [set(np.flatnonzero(adj[i])) for i in range(n)]
    maximal_cliques = []

    def bron_kerbosch(r, p, x):
        if not p and not x:
            maximal_cliques.append(sorted(r))
            return

        pivot_pool = p | x
        pivot = max(pivot_pool, key=lambda node: len(neighbor_sets[node]), default=None)
        pivot_neighbors = neighbor_sets[pivot] if pivot is not None else set()

        for node in list(p - pivot_neighbors):
            bron_kerbosch(
                r | {node},
                p & neighbor_sets[node],
                x & neighbor_sets[node],
            )
            p.remove(node)
            x.add(node)

    bron_kerbosch(set(), set(range(n)), set())

    covered = {node for clique in maximal_cliques for node in clique}
    for node in range(n):
        if node not in covered:
            maximal_cliques.append([node])

    maximal_cliques.sort(key=lambda clique: (clique[0], len(clique), clique))
    return maximal_cliques


def extract_local_cliques(adj):
    """
    Backwards-compatible alias retained for callers that still import the old
    name. The returned cliques are overlapping maximal cliques.
    """
    return extract_overlapping_maximal_cliques(adj)
