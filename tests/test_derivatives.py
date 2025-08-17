import random
import math
from .utils import lscv_generic


def finite_diff(f, h, eps=1e-5):
    f_plus = f(h + eps)
    f_minus = f(h - eps)
    grad = (f_plus - f_minus) / (2 * eps)
    hess = (f_plus - 2 * f(h) + f_minus) / (eps ** 2)
    return grad, hess


def test_lscv_derivatives_against_finite_diff():
    rng = random.Random(0)
    x = [rng.gauss(0, 1) for _ in range(15)]
    for kernel in ["gauss", "epan"]:
        for h in [0.5, 1.0, 1.5]:
            score, grad, _ = lscv_generic(x, h, kernel)
            num_grad, _ = finite_diff(lambda hh: lscv_generic(x, hh, kernel)[0], h)
            assert math.isclose(grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)
