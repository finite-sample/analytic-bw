"""Newton–Armijo bandwidth selection for univariate KDE."""

import argparse
from typing import Callable, Tuple

import numpy as np

from derivatives import KERNELS


def lscv_generic(x: np.ndarray, h: float, kernel: str) -> Tuple[float, float, float]:
    """Return LSCV score, gradient and Hessian for bandwidth *h*.

    Parameters
    ----------
    x : np.ndarray
        Sample data.
    h : float
        Bandwidth.
    kernel : str
        Kernel name: ``"gauss"`` or ``"epan"``.
    """
    K, Kp, Kpp, K2, K2p, K2pp = KERNELS[kernel]
    n = len(x)
    u = (x[:, None] - x[None, :]) / h

    # score
    term1 = K2(u).sum() / (n ** 2 * h)
    Ku = K(u)
    term2 = (Ku.sum() - np.sum(np.diag(Ku))) / (n * (n - 1) * h)
    score = term1 - 2 * term2

    # gradient
    S_F = (K2(u) + u * K2p(u)).sum()
    S_K_matrix = Ku + u * Kp(u)
    S_K = S_K_matrix.sum() - np.sum(np.diag(S_K_matrix))
    grad = -S_F / (n ** 2 * h ** 2) + 2 * S_K / (n * (n - 1) * h ** 2)

    # Hessian
    S_F2 = (2 * K2p(u) + u * K2pp(u)).sum()
    Kp_u = Kp(u)
    Kpp_u = Kpp(u)
    S_K2_matrix = 2 * Kp_u + u * Kpp_u
    S_K2 = S_K2_matrix.sum() - np.sum(np.diag(S_K2_matrix))
    hess = 2 * S_F / (n ** 2 * h ** 3) - S_F2 / (n ** 2 * h ** 2)
    hess += -4 * S_K / (n * (n - 1) * h ** 3) + 2 * S_K2 / (n * (n - 1) * h ** 2)
    return score, grad, hess


def newton_armijo(
    x: np.ndarray,
    h0: float,
    kernel: str = "gauss",
    tol: float = 1e-5,
    max_iter: int = 12,
) -> Tuple[float, int]:
    """Run Newton–Armijo to minimise LSCV and return (h_opt, evaluations)."""
    h = float(h0)
    evals = 0
    for _ in range(max_iter):
        f, g, H = lscv_generic(x, h, kernel)
        evals += 1
        if abs(g) < tol:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if lscv_generic(x, h_new, kernel)[0] < f:
                h = h_new
                break
            step *= 0.5
    return h, evals


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic-Hessian KDE bandwidth selection")
    parser.add_argument("data", nargs="?", help="Path to 1D data (one value per line)")
    parser.add_argument("--kernel", choices=["gauss", "epan"], default="gauss")
    parser.add_argument("--h0", type=float, default=1.0, help="Initial bandwidth guess")
    args = parser.parse_args()

    if args.data:
        x = np.loadtxt(args.data, ndmin=1)
    else:
        x = np.random.randn(200)
    h, evals = newton_armijo(x, args.h0, kernel=args.kernel)
    print(f"Optimal h={h:.5f} after {evals} evaluations")


if __name__ == "__main__":
    main()
