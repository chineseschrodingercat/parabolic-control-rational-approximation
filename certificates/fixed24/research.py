"""Exact rational verification of the fixed 24-pole Robin subrange."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parent
TAU = Q(1, 320)
SCALAR_DATA = json.loads((ROOT / "scalar_certificate.json").read_text())
POLES = [Q(s) for s in SCALAR_DATA["cases"][0]["poles_exact"]]
assert all(row["poles_exact"] == list(map(str, POLES)) for row in SCALAR_DATA["cases"])
assert len(POLES) == 24 and all(0 < mu < 1 for mu in POLES)
RATES = [(1 - mu) / (TAU * mu) for mu in POLES]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polynomial_value(p, x):
    return (p[2] * x + p[1]) * x + p[0]


def bernstein(p, lo, hi):
    a = polynomial_value(p, lo)
    return (a, a + (hi - lo) * (p[1] + 2 * p[2] * lo) / 2,
            polynomial_value(p, hi))


def ratio_upper(num, den, lo, hi):
    nn, dd = bernstein(num, lo, hi), bernstein(den, lo, hi)
    if min(dd) <= 0:
        mid = (lo + hi) / 2
        return max(ratio_upper(num, den, lo, mid), ratio_upper(num, den, mid, hi))
    return max(n / d for n, d in zip(nn, dd))


def compactify(p):
    """(1-t)^2 p(t/(1-t)), all coefficients exact."""
    return (p[0], p[1] - 2 * p[0], p[0] - p[1] + p[2])


def product_cell(curves, lo, hi):
    v = Q(1)
    for num, den in curves:
        upper = ratio_upper(num, den, lo, hi)
        assert upper >= 0
        v *= upper
    return v


def rounded_root_upper(x, digits=12):
    scale = 10 ** digits
    n = math.ceil(math.sqrt(float(x)) * scale) + 1
    while Q(n, scale) ** 2 < x:
        n += 1
    assert Q(n, scale) ** 2 >= x
    return Q(n, scale)


def sinc_cos_lower(lam, beta, odd_order=5):
    """Alternating Taylor lower bound cos(sqrt(lam))+beta*sinc(sqrt(lam))."""
    assert odd_order % 2 == 1 and 0 < lam <= 4
    return sum(((-lam) ** j / math.factorial(2 * j)
                + beta * (-lam) ** j / math.factorial(2 * j + 1)
                for j in range(odd_order + 1)), Q(0))


def certificate(name, alpha, beta, lam, sqrt_lam_upper, a_numerator, target,
                circle_cells=3000, cells_per_decade=500):
    start = time.perf_counter()
    eta = 1 / (1 + beta)
    sqrt_tau_upper = Q(9, 160)
    inv_sqrt_lam_upper = Q(123, 200) if lam == Q(53, 20) else Q(1, 2)
    assert sqrt_tau_upper ** 2 >= TAU
    assert inv_sqrt_lam_upper ** 2 * lam >= 1
    assert sqrt_lam_upper ** 2 >= lam
    analytic_a_numerator = eta / 2 * (sqrt_tau_upper + inv_sqrt_lam_upper / (1 + TAU * lam))
    assert analytic_a_numerator <= a_numerator
    eigenvalue_lower_test = sinc_cos_lower(lam, beta)
    assert eigenvalue_lower_test > 0
    a = a_numerator / alpha
    m = lam - a * sqrt_lam_upper
    join = lam - a * a / 2
    assert 0 < m < join
    # Exact numerical-range boundary: an initial circle, then a parabola.
    # The circle starts at the true irrational tangent point >= m.
    # A harmless tiny extension to rational m is also bounded below.
    circle = []
    parabola = []
    c = a ** 4 / 4
    for rate in RATES:
        p = 1 + TAU * rate
        circle.append(((rate * rate + a * a * lam - lam * lam,
                        2 * (lam - rate), Q(0)),
                       (rate * rate + p * p * (a * a * lam - lam * lam),
                        2 * p * (p * lam + rate), Q(0))))
        num = (rate * rate + c, a * a - 2 * rate, Q(1))
        den = (rate * rate + p * p * c, 2 * rate * p + p * p * a * a, p * p)
        parabola.append((compactify(num), compactify(den)))
    maximum = Q(0)
    worst = None
    for i in range(circle_cells):
        lo = m + (join - m) * Q(i, circle_cells)
        hi = m + (join - m) * Q(i + 1, circle_cells)
        bound = product_cell(circle, lo, hi)
        if bound > maximum:
            maximum, worst = bound, ["circle", str(lo), str(hi)]
    # Floating point proposes partition points only; the exact sorted rational
    # intervals themselves cover [join/(1+join), 1] without a gap.
    count = math.ceil((12 - math.log10(float(join))) * cells_per_decade)
    xx = [join] + [Q(math.exp(math.log(float(join)) +
                   (math.log(1e12) - math.log(float(join))) * i / count))
                   for i in range(1, count + 1)]
    tt = [x / (1 + x) for x in xx] + [Q(1)]
    assert tt[0] == join / (1 + join) and tt[-1] == 1
    assert all(lo < hi for lo, hi in zip(tt, tt[1:]))
    for lo, hi in zip(tt, tt[1:]):
        bound = product_cell(parabola, lo, hi)
        if bound > maximum:
            maximum, worst = bound, ["parabola_compactified", str(lo), str(hi)]
    infinity_squared = math.prod(mu * mu for mu in POLES)
    maximum = max(maximum, infinity_squared)
    scalar_upper = rounded_root_upper(maximum)
    t0 = 1 + eta / (4 * alpha * lam)
    both_upper = Q(5, 2) * t0 * t0 * scalar_upper
    assert both_upper <= target
    return {
        "name": name, "alpha_lower": str(alpha), "beta_lower": str(beta),
        "beta_upper": "10", "lambda_lower": str(lam), "eta_upper": str(eta),
        "robin_eigenvalue_taylor_lower": str(eigenvalue_lower_test),
        "a_numerator_upper": str(a_numerator),
        "analytic_a_numerator_rational_upper": str(analytic_a_numerator),
        "a_upper": str(a), "rational_real_lower": str(m), "join_real": str(join),
        "circle_cells": circle_cells, "parabola_cells": len(tt) - 1,
        "scalar_upper": float(scalar_upper), "scalar_upper_rational": str(scalar_upper),
        "both_mass_operator_upper": float(both_upper),
        "both_mass_operator_upper_rational": str(both_upper),
        "target": str(target), "target_exactly_verified": bool(both_upper <= target),
        "squared_exact_upper_sha256": hashlib.sha256(str(maximum).encode()).hexdigest(),
        "worst_cell": worst, "seconds": time.perf_counter() - start,
        "scope": "Every conforming Galerkin mesh, fixed tau=1/320, L=kappa=1; any observation interval. The alpha and beta lower bounds listed here are essential.",
    }

