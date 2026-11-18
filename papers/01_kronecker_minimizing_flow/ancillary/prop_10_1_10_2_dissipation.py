"""Interval-arithmetic enclosures for Proposition 10.2 and for the example after Proposition 10.1.

Notation of the paper: the dissipation is  calD = calD_V + calD_U + calD_W,  where for each block

    calD_block = tr( |Omega| (Phi - varphi) sin(Phi - varphi) ),

Omega is the block of the endomorphism, Phi its polar phase and varphi = arg Z the argument of the total central charge.
The phase potential satisfies  sigma dK/dt = -calD  (Proposition 7.2), so calD < 0 means that the
potential increases along the flow at the given metrics.

(A) Proposition 10.2.  Ordered charges, stable cohomology, r = 2, n_0 = n_1 = 1, k = 2:
        z_0 = 10^-6 (-1 + i),  z_1 = 1 + i/500,
        V = C^3, U = C^2, W = C with the standard metrics,
        D = [0, diag(sqrt(l_1), sqrt(l_2))],  l_1 = 1/20,  l_2 = 8,
        A_1 = (1/5, 0, 0),  A_2 = (0, a_1, a_2),
        a_1 = sqrt(eps / l_1),  a_2 = (7/10) sqrt(eps l_2),
        eps = (3/10) / ( (1 - e^{-l_1}) / l_1^2 + (49/100) (1 - e^{-l_2}) ).
    Result: calD < 0.

    Consistency check: for the same complex and metrics and the symmetric charges
    z_0 = -1 + i, z_1 = 1 + i, Theorem 7.6 gives calD > 0; the program confirms this.

(B) Example after Proposition 10.1.  Reversed charges, no arrows:
        z_0 = 1 + 2i,  z_1 = -3 + 2i,  V = C^2, U = W = C,  D = (0, 1/2),  A = 0.
    The enclosure of calD is compared with the formula of Proposition 10.1,
        calD = c_0 [ n_0 (arg z_0 - arg z_1) + pi tr e^{-L} ],  c_0 = |z_0| sin(arg z_0 - varphi).

All blocks have size 1 or 2, so the polar decomposition is explicit: for a 2 x 2 matrix T,
    |T| = (T T^*)^{1/2} = (M + sqrt(det M)) / sqrt(tr M + 2 sqrt(det M)),   M = T T^*,
the unitary factor is |T|^{-1} T, and its eigenvalues are the roots of its characteristic
polynomial.

Enclosures are computed with the ball arithmetic of Arb (through python-flint) at 300 bits.

Requires Python 3 with python-flint (an interface to Arb); run:  python prop_10_1_10_2_dissipation.py
The accompanying output file was produced with python-flint 0.9.0.
"""
from flint import arb, acb, ctx

ctx.prec = 300
PI = arb.pi()


def R(x):
    """R(x) = (1 - e^{-x}) / x."""
    return (1 - (-x).exp()) / x


def first_difference(x, y):
    """First divided difference of e^{-t} at x, y, with the opposite sign."""
    return ((-y).exp() - (-x).exp()) / (x - y)


def second_difference(x, y):
    """j(x, y): second divided difference of e^{-t} at x, 0, y  (x != y)."""
    return (R(x) - R(y)) / (y - x)


def second_difference_diagonal(x):
    """j(x, x) = (1 - e^{-x} - x e^{-x}) / x^2."""
    return (1 - (-x).exp() - x * (-x).exp()) / (x * x)


def conj(z):
    return acb(z.real, -z.imag)


def matmul(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def adjoint(A):
    return [[conj(A[0][0]), conj(A[1][0])], [conj(A[0][1]), conj(A[1][1])]]


def block2(T, shift, varphi):
    """T is 2 x 2 with positive definite imaginary part.  Returns an enclosure of
    tr(|T| (Phi - varphi) sin(Phi - varphi)),  where Phi = (argument of the unitary polar
    factor of T) - shift, together with the eigenvalues of Phi."""
    M = matmul(T, adjoint(T))
    trM = (M[0][0] + M[1][1]).real
    detM = (M[0][0] * M[1][1] - M[0][1] * M[1][0]).real
    if not (detM > 0):
        raise SystemExit("determinant not enclosed away from zero")
    sd = detM.sqrt()
    den = (trM + 2 * sd).sqrt()
    P = [[(M[0][0] + sd) / den, M[0][1] / den], [M[1][0] / den, (M[1][1] + sd) / den]]
    dP = P[0][0] * P[1][1] - P[0][1] * P[1][0]
    Pinv = [[P[1][1] / dP, -P[0][1] / dP], [-P[1][0] / dP, P[0][0] / dP]]
    W = matmul(Pinv, T)
    tr = W[0][0] + W[1][1]
    det = W[0][0] * W[1][1] - W[0][1] * W[1][0]
    disc = (tr * tr - 4 * det).sqrt()
    eig = [(tr + disc) / 2, (tr - disc) / 2]
    total = arb(0)
    angles = []
    for k in (0, 1):
        this, other = eig[k], eig[1 - k]
        theta = this.arg()
        if not (theta > 0 and theta < PI):
            raise SystemExit("argument not enclosed in (0, pi)")
        proj = [[(W[0][0] - other) / (this - other), W[0][1] / (this - other)],
                [W[1][0] / (this - other), (W[1][1] - other) / (this - other)]]
        weight = (P[0][0] * proj[0][0] + P[0][1] * proj[1][0]
                  + P[1][0] * proj[0][1] + P[1][1] * proj[1][1]).real
        ph = theta - shift - varphi
        total += weight * ph * ph.sin()
        angles.append(theta - shift)
    return total, angles


def block1(z, shift, varphi):
    ph = z.arg() - shift - varphi
    return abs(z) * ph * ph.sin()


def dissipation(z0, z1, l1, l2, h, a1, a2):
    """n_0 = n_1 = 1, r = 2, A_1 = (h, 0, 0), A_2 = (0, a1, a2), D = [0, diag(sqrt l1, sqrt l2)]."""
    Z = z0 + z1
    varphi = Z.arg()
    e1, e2 = (-l1).exp(), (-l2).exp()
    k12 = a1 * a2 * first_difference(l1, l2)
    # block of Omega_V on the orthogonal complement of ker D
    omega_V_perp = [[z0 * e1 + a1 * a1 * e1, acb(k12)], [acb(k12), z0 * e2 + a2 * a2 * e2]]
    j11 = l1 * a1 * a1 * second_difference_diagonal(l1)
    j22 = l2 * a2 * a2 * second_difference_diagonal(l2)
    j12 = (l1 * l2).sqrt() * a1 * a2 * second_difference(l1, l2)
    minus_omega_U = [[z0 * e1 - j11, acb(-j12)], [acb(-j12), z0 * e2 - j22]]
    trY = a1 * a1 * R(l1) + a2 * a2 * R(l2)
    omega_W = z1 - h * h - trY
    omega_V_ker = z0 + h * h
    # trace identity: tr Omega_V + tr Omega_U + tr Omega_W = Z
    trace = (omega_V_ker + omega_V_perp[0][0] + omega_V_perp[1][1]
             - minus_omega_U[0][0] - minus_omega_U[1][1] + omega_W)
    if not ((trace - Z).real.contains(0) and (trace - Z).imag.contains(0)):
        raise SystemExit("trace identity not confirmed")
    d_V_perp, _ = block2(omega_V_perp, arb(0), varphi)
    d_U, _ = block2(minus_omega_U, PI, varphi)
    d_V_ker = block1(omega_V_ker, arb(0), varphi)
    d_W = block1(omega_W, arb(0), varphi)
    trJ, detJ = j11 + j22, j11 * j22 - j12 * j12
    return d_V_ker + d_V_perp + d_U + d_W, (d_V_ker, d_V_perp, d_U, d_W), varphi, (trJ, detJ, trY)


def show(name, x):
    print(f"   {name} in {x.str(25)}")


if __name__ == "__main__":
    l1, l2 = arb(1) / 20, arb(8)
    eps = (arb(3) / 10) / ((1 - (-l1).exp()) / (l1 * l1) + arb(49) / 100 * (1 - (-l2).exp()))
    a1 = (eps / l1).sqrt()
    a2 = arb(7) / 10 * (eps * l2).sqrt()
    h = arb(1) / 5

    print("(A) Proposition 10.2: z_0 = 10^-6 (-1 + i), z_1 = 1 + i/500, r = 2")
    z0 = acb(-1, 1) / 10 ** 6
    z1 = acb(1, arb(1) / 500)
    if not (z0.arg() > z1.arg()):
        raise SystemExit("charges not ordered")
    d, parts, varphi, (trJ, detJ, trY) = dissipation(z0, z1, l1, l2, h, a1, a2)
    show("arg z_0", z0.arg()); show("arg z_1", z1.arg()); show("varphi", varphi)
    show("a_1", a1); show("a_2", a2); show("tr Y", trY)
    show("tr J", trJ); show("det J", detJ)
    for name, value in zip(["calD_V, block on ker D", "calD_V, block on (ker D)^perp", "calD_U", "calD_W"], parts):
        show(name, value)
    show("calD", d)
    if not (d < 0):
        raise SystemExit("calD < 0 not confirmed")
    print("   verified: calD < 0")

    print("consistency check: same complex and metrics, symmetric charges z_0 = -1 + i, z_1 = 1 + i")
    d_sym, _, varphi_sym, _ = dissipation(acb(-1, 1), acb(1, 1), l1, l2, h, a1, a2)
    show("varphi", varphi_sym); show("calD", d_sym)
    if not (d_sym > 0):
        raise SystemExit("calD > 0 not confirmed")
    print("   verified: calD > 0, as Theorem 7.6 states")

    print("(B) example after Proposition 10.1: z_0 = 1 + 2i, z_1 = -3 + 2i, D = (0, 1/2), A = 0, n_0 = n_1 = r = 1")
    z0, z1 = acb(1, 2), acb(-3, 2)
    if not (z0.arg() < z1.arg()):
        raise SystemExit("charges not reversed")
    Z = z0 + z1
    varphi = Z.arg()
    L = arb(1) / 4
    d_V = block1(z0, arb(0), varphi) + block1(z0 * (-L).exp(), arb(0), varphi)
    d_U = block1(z0 * (-L).exp(), PI, varphi)        # -Omega_U = z_0 e^{-L}
    d_W = block1(z1, arb(0), varphi)
    d = d_V + d_U + d_W
    formula = abs(z0) * (z0.arg() - varphi).sin() * ((z0.arg() - z1.arg()) + PI * (-L).exp())
    show("calD_V", d_V); show("calD_U", d_U); show("calD_W", d_W); show("calD", d)
    show("formula of Proposition 10.1", formula)
    if not (d < 0 and (d - formula).contains(0)):
        raise SystemExit("calD < 0 or the agreement with the formula not confirmed")
    print("   verified: calD < 0, and the enclosure contains the value of the formula")
