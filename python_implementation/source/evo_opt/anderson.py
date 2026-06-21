from numpy import array, eye, ones, trace
from numpy.linalg import solve, LinAlgError


def anderson_extrapolate(history, depth):
    """
    Type-I Anderson extrapolation over a raw (unaccelerated) iterate history.
    Returns None if there isn't enough history yet.

    history: list of ndarray, oldest first.
    depth:   number of residual vectors (steps back) to mix.
    """
    if depth < 1 or len(history) < depth + 1:
        return None

    X = array(history[-(depth + 1):])   # depth+1 points
    F = X[1:] - X[:-1]                  # depth residuals f_i = x_{i+1} - x_i

    # minimize ||F^T @ alpha||^2 s.t. sum(alpha) = 1, via Lagrange multipliers:
    # alpha = G^-1 1 / (1^T G^-1 1), G = F F^T, regularized since residuals
    # from a stochastic inner optimizer can be nearly collinear.
    G   = F @ F.T
    reg = 1e-10 * (trace(G) / depth if trace(G) > 0 else 1.0)
    try:
        w = solve(G + reg * eye(depth), ones(depth))
    except LinAlgError:
        return None

    if not w.any():
        return None

    alpha = w / w.sum()
    return alpha @ X[1:]
