import math

SQRT_2PI = math.sqrt(2 * math.pi)


def K_gauss(u):
    return math.exp(-0.5 * u * u) / SQRT_2PI

def K_gauss_p(u):
    return -u * K_gauss(u)

def K_gauss_pp(u):
    return (u * u - 1) * K_gauss(u)

def K2_gauss(u):
    return math.exp(-0.25 * u * u) / math.sqrt(4 * math.pi)

def K2_gauss_p(u):
    return -0.5 * u * K2_gauss(u)

def K2_gauss_pp(u):
    return (0.25 * u * u - 0.5) * K2_gauss(u)

def K_epan(u):
    a = abs(u)
    return 0.75 * (1 - u * u) if a <= 1 else 0.0

def K_epan_p(u):
    a = abs(u)
    return -1.5 * u if a <= 1 else 0.0

def K_epan_pp(u):
    a = abs(u)
    return -1.5 if a <= 1 else 0.0

def K2_epan(u):
    a = abs(u)
    if a <= 2:
        return 0.6 - 0.75 * a ** 2 + 0.375 * a ** 3 - 0.01875 * a ** 5
    return 0.0

def K2_epan_p(u):
    a = abs(u)
    if a <= 2:
        sign = 1 if u >= 0 else -1
        return sign * (-0.09375 * a ** 4 + 1.125 * a ** 2 - 1.5 * a)
    return 0.0

def K2_epan_pp(u):
    a = abs(u)
    if a <= 2:
        return -0.375 * a ** 3 + 2.25 * a - 1.5
    return 0.0

KERNELS = {
    "gauss": (K_gauss, K_gauss_p, K_gauss_pp, K2_gauss, K2_gauss_p, K2_gauss_pp),
    "epan": (K_epan, K_epan_p, K_epan_pp, K2_epan, K2_epan_p, K2_epan_pp),
}

def lscv_generic(x, h, kernel):
    K, Kp, Kpp, K2, K2p, K2pp = KERNELS[kernel]
    n = len(x)
    score_term1 = 0.0
    score_term2 = 0.0
    S_F = 0.0
    S_K = 0.0
    S_F2 = 0.0
    S_K2 = 0.0
    for i in range(n):
        for j in range(n):
            u = (x[i] - x[j]) / h
            k2 = K2(u)
            k2p = K2p(u)
            k2pp = K2pp(u)
            score_term1 += k2
            S_F += k2 + u * k2p
            S_F2 += 2 * k2p + u * k2pp
            if i != j:
                k = K(u)
                kp = Kp(u)
                kpp = Kpp(u)
                score_term2 += k
                S_K += k + u * kp
                S_K2 += 2 * kp + u * kpp
    score = score_term1 / (n ** 2 * h) - 2 * (score_term2 / (n * (n - 1) * h))
    grad = -S_F / (n ** 2 * h ** 2) + 2 * S_K / (n * (n - 1) * h ** 2)
    hess = 2 * S_F / (n ** 2 * h ** 3) - S_F2 / (n ** 2 * h ** 2)
    hess += -4 * S_K / (n * (n - 1) * h ** 3) + 2 * S_K2 / (n * (n - 1) * h ** 2)
    return score, grad, hess

def newton_opt(x, h0, score_grad_hess, tol=1e-5, max_iter=12):
    """Newton–Armijo optimisation using a numeric Hessian estimate."""
    h, evals = h0, 0
    for _ in range(max_iter):
        f, g, _ = score_grad_hess(x, h)
        evals += 1
        if abs(g) < tol:
            break
        eps = max(1e-4 * h, 1e-6)
        _, g_plus, _ = score_grad_hess(x, h + eps)
        H = (g_plus - g) / eps
        step = -g / H if (H > 0 and math.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if score_grad_hess(x, h_new)[0] < f:
                h = h_new
                break
            step *= 0.5
    return h, evals
