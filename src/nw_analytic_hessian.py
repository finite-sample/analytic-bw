"""Newton–Armijo bandwidth selection for Nadaraya–Watson regression using :func:`optim.newton_armijo`."""

import argparse
from typing import Tuple

import numpy as np

from optim import newton_armijo

SQRT_2PI = np.sqrt(2 * np.pi)


def _weights(u: np.ndarray, h: float, kernel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weights w, w', w'' for given kernel."""
    if kernel == "gauss":
        base = np.exp(-0.5 * u * u) / (h * SQRT_2PI)
        w1 = base * (u * u - 1) / h
        w2 = base * (u**4 - 3 * u * u + 1) / (h * h)
        return base, w1, w2
    elif kernel == "epan":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        uu = u * u
        w[mask] = 0.75 * (1 - uu[mask]) / h
        w1[mask] = 0.75 * (-1 + 3 * uu[mask]) / (h * h)
        w2[mask] = 1.5 * (1 - 6 * uu[mask]) / (h ** 3)
        return w, w1, w2
    else:
        raise ValueError("Unknown kernel")


def loocv_mse(x: np.ndarray, y: np.ndarray, h: float, kernel: str) -> Tuple[float, float, float]:
    """Return LOOCV MSE, gradient and Hessian for bandwidth ``h``."""
    n = len(x)
    u = (x[:, None] - x[None, :]) / h
    w, w1, w2 = _weights(u, h, kernel)
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(w1, 0.0)
    np.fill_diagonal(w2, 0.0)

    num = w @ y
    den = w.sum(axis=1)
    den_safe = np.where(den == 0, np.finfo(float).eps, den)  # Guard against division by zero
    m = num / den_safe

    num1 = w1 @ y
    den1 = w1.sum(axis=1)
    m1 = (num1 * den_safe - num * den1) / (den_safe ** 2)

    num2 = w2 @ y
    den2 = w2.sum(axis=1)
    m2 = (num2 * den_safe - num * den2) / (den_safe ** 2) - 2 * m1 * den1 / den_safe

    resid = y - m
    loss = np.mean(resid**2)
    grad = (-2.0 / n) * np.sum(resid * m1)
    hess = (2.0 / n) * np.sum(m1 * m1 - resid * m2)
    return loss, grad, hess


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic-Hessian NW bandwidth selection")
    parser.add_argument("data", nargs="?", help="Path to data with two columns x,y")
    parser.add_argument("--kernel", choices=["gauss", "epan"], default="gauss")
    parser.add_argument("--h0", type=float, default=1.0, help="Initial bandwidth guess")
    args = parser.parse_args()

    if args.data:
        arr = np.loadtxt(args.data)
        x, y = arr[:, 0], arr[:, 1]
    else:
        x = np.linspace(-2, 2, 200)
        y = np.sin(x) + 0.1 * np.random.randn(len(x))
    objective = lambda x_, h, k: loocv_mse(x_, y, h, k)
    h, evals = newton_armijo(objective, x, args.h0, kernel=args.kernel)
    print(f"Optimal h={h:.5f} after {evals} evaluations")


if __name__ == "__main__":
    main()
