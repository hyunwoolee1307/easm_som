import numpy as np
from numpy.linalg import norm


def edis(x1: np.ndarray, x2: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def BestMatchingUnit(w: np.ndarray, x: np.ndarray, m1: int, m2: int):
    """
    Find BMU indices (i0, j0) for vector x.
    w: shape (nd, m1, m2)
    """
    dismin = np.inf
    i0, j0 = 0, 0  # Python 0-based
    for j in range(m2):
        for i in range(m1):
            dis = edis(w[:, i, j], x)
            if dis < dismin:
                dismin = dis
                i0, j0 = i, j
    return i0, j0


# ---------------------------
# SOM training
# ---------------------------


def Som(
    x: np.ndarray,
    nd: int,
    nt: int,
    m1: int,
    m2: int,
    nter: int,
    alpha0: float,
    taua: float,
    alphamin: float,
    sigma0: float,
    taus: float,
    normalize_weights: bool = True,
    seed: int = 1234,
):
    """
    Train a SOM for input matrix x (nd x nt).

    Parameters
    ----------
    x : np.ndarray
        Input data matrix of shape (nd, nt).
    nd : int
        Number of dimensions (features).
    nt : int
        Number of samples (time points).
    m1 : int
        Number of neurons in the first dimension.
    m2 : int
        Number of neurons in the second dimension.
    nter : int
        Number of training iterations.
    alpha0 : float
        Initial learning rate.
    taua : float
        Time constant for learning rate decay.
    alphamin : float
        Minimum learning rate.
    sigma0 : float
        Initial neighborhood radius.
    taus : float
        Time constant for neighborhood radius decay.
    normalize_weights : bool, optional
        Whether to normalize weight vectors after updates (default is True).
    seed : int, optional
        Random seed for weight initialization (default is 1234).

    Returns
    -------
    w : np.ndarray
        Trained weight matrix of shape (nd, m1, m2).
    """

    print("Training SOM...")
    # Normalize each sample vector (column)
    for t in range(nt):
        v = x[:, t]
        n = norm(v)
        if n > 1e-12:
            x[:, t] = v / n
    print("Input data normalized")

    # Initialize weights
    rng = np.random.default_rng(seed)
    w = rng.random((nd, m1, m2))
    if normalize_weights:
        for j in range(m2):
            for i in range(m1):
                v = w[:, i, j]
                n = norm(v)
                if n > 1e-12:
                    w[:, i, j] = v / n

    alpha = alpha0
    sigma = sigma0

    for iter in range(1, nter + 1):
        t = iter - 1
        it = iter - 1 if iter <= nt else rng.integers(0, nt)  # 0-based index

        i0, j0 = BestMatchingUnit(w, x[:, it], m1, m2)

        alpha = max(alpha0 * np.exp(-t / taua), alphamin)
        sigma = max(sigma0 * np.exp(-t / taus), 0.5)  # keep radius reasonable

        for j in range(m2):
            for i in range(m1):
                d = np.sqrt((i0 - i) ** 2 + (j0 - j) ** 2)
                h = np.exp(-0.5 * (d / sigma) ** 2)
                if h > 1e-6:
                    w[:, i, j] += alpha * h * (x[:, it] - w[:, i, j])
                    if normalize_weights:
                        v = w[:, i, j]
                        n = norm(v)
                        if n > 1e-12:
                            w[:, i, j] = v / n

        if iter % 50 == 0:
            print(f"Iteration {iter}, alpha={alpha:.4f}, sigma={sigma:.4f}")

    print("Final learning rate:", alpha)
    print("Final influence radius:", sigma)
    return w


# ---------------------------
# Clustering and QE metrics
# ---------------------------


def cluster(x: np.ndarray, w: np.ndarray, nd: int, nt: int, m1: int, m2: int):
    """
    Assign each sample to BMU, compute per-neuron average QE,
    total quantization error, and counts.
    """
    print("Clustering...")
    qerror = 0.0
    qerr = np.zeros(m1 * m2, dtype=float)
    count = np.zeros(m1 * m2, dtype=int)
    cluster1 = np.zeros(nt, dtype=int)
    cluster2 = np.zeros(nt, dtype=int)

    for it in range(nt):
        dismin = np.inf
        m = 0
        m0 = 0
        i0 = 0
        j0 = 0
        for j in range(m2):
            for i in range(m1):
                dis = edis(w[:, i, j], x[:, it])
                if dis < dismin:
                    dismin = dis
                    i0, j0 = i, j
        m0 = i0 + j0 * m1
        count[m0] += 1
        qerr[m0] += dismin
        qerror += dismin
        # Store 1-based indices to mirror Julia output if desired
        cluster1[it] = i0 + 1
        cluster2[it] = j0 + 1

    qerror /= nt

    print("Neurons (i,j) counts quantization_error:")
    for m in range(m1 * m2):
        avgqe = (qerr[m] / count[m]) if count[m] > 0 else 0.0
        j = m // m1
        i = m % m1
        print(f"Neuron {m + 1} ({i + 1}, {j + 1}) = {count[m]} ({avgqe:.4f})")
        qerr[m] = avgqe

    print(f"Quantization Error = {qerror:.4f}")

    return cluster1, cluster2, qerr, qerror, count
