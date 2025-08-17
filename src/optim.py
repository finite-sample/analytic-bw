"""Optimisation utilities."""

from typing import Callable, Tuple

import numpy as np


def newton_armijo(
    objective: Callable[[np.ndarray, float, str], Tuple[float, float, float]],
    x: np.ndarray,
    h0: float,
    kernel: str = "gauss",
    tol: float = 1e-5,
    max_iter: int = 12,
) -> Tuple[float, int]:
    """Run Newton–Armijo iterations for a generic objective.

    Parameters
    ----------
    objective:
        Callable returning ``(score, grad, hess)`` for given ``(x, h, kernel)``.
    x:
        Sample locations passed to ``objective``.
    h0:
        Initial bandwidth guess.
    kernel:
        Kernel name forwarded to ``objective``.
    tol:
        Tolerance for gradient magnitude to stop optimisation.
    max_iter:
        Maximum number of Newton updates.

    Returns
    -------
    Tuple[float, int]
        Optimised bandwidth and number of objective evaluations.
    """

    h = float(h0)
    evals = 0
    for _ in range(max_iter):
        f, g, H = objective(x, h, kernel)
        evals += 1
        if abs(g) < tol:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if objective(x, h_new, kernel)[0] < f:
                h = h_new
                break
            step *= 0.5
    return h, evals
