"""Interval-arithmetic verification for Lemma 3.6, range 1/20 <= a <= 10.

Statement verified.  With tau = -9/25 and, for a > 0,

    n(a) = (tau u(a) + v(a))^2 / (1 - tau^2) + sum_{j>=1} t_{2j}(a)^2 / omega_{2j},
    m(a) = 1 - (1 + a) e^{-a},

where  omega_k = (k + 1/2) pi,  u(a) = 1 - e^{-a},  v(a) = t_0(a) / sqrt(omega_0),
       t_k(a) = sqrt2 a (omega_k - (-1)^k a e^{-a}) / (a^2 + omega_k^2),

one has  n(a) <= (2/5) m(a)  for every a in [1/20, 10].

Method.  Ball arithmetic (Arb, through python-flint).  The interval [1/20, 10] is covered by
closed subintervals; on each of them every quantity is evaluated on a ball containing the
subinterval, so the computed ball for (2/5) m - (upper bound for n) contains its range.
The terms j = 1, ..., j_* = 80 of the sum are evaluated; the remainder is bounded as in the
proof of Lemma 3.6:

    sum_{j > j_*} t_{2j}^2 / omega_{2j}  <=  a^2 / (2 pi (a^2 + omega_{2 j_*}^2)),

which requires omega_{2 j_*} >= a / sqrt3 (true here, since omega_{160} > 10).

The ranges a <= 1/20 and a >= 10 are treated in the proof of Lemma 3.6 without a computer;
the decimal constants quoted there are enclosed at the end of this program.

Requires Python 3 with python-flint (an interface to Arb); run:  python lemma_3_6_interval_check.py
The accompanying output file was produced with python-flint 0.9.0.
"""
from flint import arb, ctx

ctx.prec = 120
PI = arb.pi()
TAU = arb(-9) / 25
J_STAR = 80  # terms j = 1, ..., J_STAR are evaluated


def omega(k):
    return (arb(k) + arb(1) / 2) * PI


def quantities(a):
    """Balls for the first term of n(a), an upper bound for the sum, and m(a)."""
    ea = (-a).exp()
    u = 1 - ea
    om0 = omega(0)
    v = (arb(2) / om0).sqrt() * a * (om0 - a * ea) / (a * a + om0 * om0)
    y = TAU * u + v
    first = y * y / (1 - TAU * TAU)
    total = arb(0)
    for j in range(1, J_STAR + 1):
        om = omega(2 * j)
        total += 2 * a * a * (om - a * ea) ** 2 / (om * (a * a + om * om) ** 2)
    om_last = omega(2 * J_STAR)
    remainder = a * a / (2 * PI * (a * a + om_last * om_last))
    m = 1 - (1 + a) * ea
    return first, total + remainder, m


if __name__ == "__main__":
    lo, hi, ratio = 0.05, 10.0, 1.0015
    edges = [lo]
    while edges[-1] < hi:
        edges.append(min(edges[-1] * ratio, hi))
    smallest = None
    for i in range(len(edges) - 1):
        a_left, a_right = edges[i], edges[i + 1]
        mid = (a_left + a_right) / 2
        rad = (a_right - a_left) / 2 * (1 + 1e-9) + 1e-15
        a = arb(mid, rad)
        if not (a.contains(arb(a_left)) and a.contains(arb(a_right))):
            raise SystemExit("a ball does not contain its subinterval")
        if i == 0 and not a.contains(arb(1) / 20):
            raise SystemExit("left endpoint 1/20 not covered")
        if i == len(edges) - 2 and not a.contains(arb(10)):
            raise SystemExit("right endpoint 10 not covered")
        if i > 0 and not a.overlaps(previous):
            raise SystemExit("gap between consecutive balls")
        previous = a
        first, rest, m = quantities(a)
        margin = arb(2) / 5 * m - (first + rest)
        if not (margin > 0):
            raise SystemExit(f"not verified on [{a_left}, {a_right}]: (2/5) m - n = {margin}")
        rel = margin / m
        lower = float(rel.mid()) - float(rel.rad())
        if smallest is None or lower < smallest[0]:
            smallest = (lower, a_left, a_right)
        if i % 500 == 0:
            print(f"   [{a_left:.5f}, {a_right:.5f}]: (2/5) - n/m >= {lower:.5f}", flush=True)
    print(f"verified: n(a) <= (2/5) m(a) on [1/20, 10] using {len(edges) - 1} subintervals, "
          f"j_* = {J_STAR}, precision {ctx.prec} bits")
    print(f"smallest lower bound for (2/5) - n/m: {smallest[0]:.5f} on "
          f"[{smallest[1]:.5f}, {smallest[2]:.5f}]")

    # Constants quoted in the proof of Lemma 3.6 for the ranges a >= 10 and a <= 1/20.
    om0 = omega(0)
    c_star = arb(2).sqrt() / om0 ** arb("1.5")
    print("\nconstants quoted in the proof of Lemma 3.6:")
    print("   c_* = sqrt2 * omega_0^(-3/2)             =", c_star)
    sigma3 = arb(0)
    terms = 2000
    for j in range(1, terms + 1):
        sigma3 += 1 / omega(2 * j) ** 3
    # sum_{j > terms} omega_{2j}^{-3} <= (1/(2 pi)) int_{omega_{2 terms}}^infty x^{-3} dx
    sigma3_upper = sigma3 + 1 / (4 * PI * omega(2 * terms) ** 2)
    print("   Sigma_3 = sum_{j>=1} omega_{2j}^(-3)  in  [", sigma3, ",", sigma3_upper, "]")
    print("   tau^2/(1-tau^2)                          =", TAU * TAU / (1 - TAU * TAU))
    print("   1/(2 pi)                                 =", 1 / (2 * PI))
    print("   9/(8 sqrt3)                              =", arb(9) / (8 * arb(3).sqrt()))
    a = arb(10)
    print("   range a >= 10: bound for n/m             =",
          (TAU * TAU / (1 - TAU * TAU) + 1 / (2 * PI) + arb(9) / (8 * arb(3).sqrt()) / a) / (1 - 11 * (-a).exp()))
    a = arb(1) / 20
    numerator = (c_star + TAU + arb(9) / 50 * a) ** 2 / (1 - TAU * TAU) + 2 * sigma3_upper
    denominator = arb(1) / 2 - a / 3
    print("   range a <= 1/20: bound for n/m           =", numerator / denominator)
