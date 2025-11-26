import numpy as np


def squared_euclidean_distance_matrix(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    """
    Computes the local distance matrix between two sequences
    using squared Euclidean distance.

    seq1: (T1, D)
    seq2: (T2, D)

    Return: (T1, T2) matrix with d_ij = ||seq1[i] - seq2[j]||^2
    """
    if seq1.ndim != 2 or seq2.ndim != 2:
        raise ValueError("seq1 and seq2 must be 2D arrays (T, D).")

    # We use the identity:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    # This allows everything to be computed with matrix operations.
    a = seq1.astype(np.float32)
    b = seq2.astype(np.float32)

    # Squared norms of individual vectors
    a_sq = np.sum(a ** 2, axis=1).reshape(-1, 1)   # (T1, 1)
    b_sq = np.sum(b ** 2, axis=1).reshape(1, -1)   # (1, T2)

    # Dot products
    ab = np.dot(a, b.T)                            # (T1, T2)

    dist = a_sq + b_sq - 2.0 * ab
    # Numerical stability: clamp negative rounding errors to 0
    dist = np.maximum(dist, 0.0)

    return dist


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int | None = None,
    return_cost_matrix: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Computes the DTW distance between two feature sequences,
    optionally with a Sakoe-Chiba band.

    seq1: (T1, D)
    seq2: (T2, D)
    window: maximum deviation from the diagonal path (in "time" steps).
            If None: full DTW (no restriction).
    return_cost_matrix: if True, return the accumulated cost matrix as well.

    Return:
        if return_cost_matrix=False: float (DTW distance)
        if return_cost_matrix=True: (dist, cost_matrix)
    """
    if seq1.size == 0 or seq2.size == 0:
        # One of the sequences is empty -> define as "infinitely far apart"
        return float("inf") if not return_cost_matrix else (float("inf"), np.empty((0, 0)))

    T1 = seq1.shape[0]
    T2 = seq2.shape[0]

    # Local distance matrix d(i,j)
    local_dist = squared_euclidean_distance_matrix(seq1, seq2)  # Shape (T1, T2)

    # Accumulated cost matrix (T1+1, T2+1), borders = +inf
    # cost[i, j] corresponds to path cost up to seq1[i-1], seq2[j-1]
    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Sakoe-Chiba band
    if window is None:
        window = max(T1, T2)  # effectively "no restriction"

    # DP
    for i in range(1, T1 + 1):
        # Limit j by the band
        j_start = max(1, i - window)
        j_end   = min(T2, i + window)

        for j in range(j_start, j_end + 1):
            # local cost d(i-1, j-1)
            d_ij = local_dist[i - 1, j - 1]

            # Three possible predecessors:
            #  - (i-1, j)   : vertical (insertion into seq2)
            #  - (i,   j-1) : horizontal (insertion into seq1)
            #  - (i-1, j-1) : diagonal (match)
            prev_min = min(
                cost[i - 1, j],     # insertion
                cost[i, j - 1],     # deletion
                cost[i - 1, j - 1]  # match
            )
            cost[i, j] = d_ij + prev_min

    dist = float(cost[T1, T2])

    if return_cost_matrix:
        return dist, cost
    else:
        return dist


def dtw_alignment_path(cost: np.ndarray) -> list[tuple[int, int]]:
    """
    Optional: Reconstructs the optimal path (alignment) from the
    accumulated cost matrix.

    cost: (T1+1, T2+1) matrix as in dtw_distance.

    Return: List of (i, j) index pairs in data-space notation:
            i in [0, T1-1], j in [0, T2-1]
    """
    if cost.size == 0:
        return []

    i = cost.shape[0] - 1
    j = cost.shape[1] - 1

    path: list[tuple[int, int]] = []

    # Backtracking until (0,0)
    while i > 0 or j > 0:
        # current index in data coordinates is (i-1, j-1)
        path.append((i - 1, j - 1))

        # choose predecessor with smallest cost
        candidates = []

        if i > 0 and j > 0:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))  # diagonal
        if i > 0:
            candidates.append((cost[i - 1, j], i - 1, j))          # vertical
        if j > 0:
            candidates.append((cost[i, j - 1], i, j - 1))          # horizontal

        if not candidates:
            break

        # choose the smallest cost
        _, i, j = min(candidates, key=lambda x: x[0])

    # Add start point
    path.append((0, 0))

    # Path is collected backward -> reverse
    path.reverse()
    return path
