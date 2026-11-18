"""Interval-arithmetic enclosures for Theorem 6.4, and a comparison with the definition (1).

Data of Theorem 6.4.  Quiver K_m (one nonzero arrow), charges z_0 = -1 + i, z_1 = 1 + i,
standard metrics, three-term complex in degrees -1, 0, 1:

    (0 -> U_1) --D_1--> (V --A_1--> W) --D_0--> (U_0 -> 0),
    V = C^3,  W = C^2,  U_0 = C^2,  U_1 = C,
    D_0 = [[0, 1, 0], [0, 0, 4]],   D_1 = (0, 3)^T,   A_1 = [[1, 0, 0], [0, 4, 100]].

The complex is the orthogonal direct sum of the representation C --1--> C, whose two blocks
z_0 + 1 and z_1 - 1 equal i and contribute nothing to the sum calD (phi = pi/2), and of
the acyclic complex  (0 -> U_1) -> (C^2 --rho--> C) -> (U_0 -> 0),  rho = (4, 100).

Part 1 (ball arithmetic, Arb through python-flint, 400 bits).  With (l_1, l_2) = (1, 16) the
eigenvalues of D_0^dagger D_0 on C^2, p = 9 the eigenvalue of D_1 D_1^dagger, and f[...] the divided
differences of e^{-x}, the blocks of Omega on the acyclic complex are (proof of Theorem 6.4)

    (Omega_V)_ik    = z_0 e^{-l_i} delta_ik + rho_i rho_k ( -f[l_i, l_k] - p f[l_i, p, l_k] ),
    Omega_W         = (z_1 - |rho|^2) e^{-p} + sum_i rho_i^2 l_i f[p, l_i, p],
    (-Omega_U0)_ik  = z_0 e^{-l_i} delta_ik - sqrt(l_i l_k) rho_i rho_k f[l_i, p, l_k],
    -Omega_U1       = z_1 e^{-p} + p sum_i rho_i^2 f[p, l_i, p].

The program encloses the four contributions tr(|Omega| (Phi - phi) sin(Phi - phi)) and their
sum calD.  The polar decomposition of a 2 x 2 matrix T is explicit:
|T| = (M + sqrt(det M)) / sqrt(tr M + 2 sqrt(det M)),  M = T T^dagger.  It also checks that the trace
of Omega on the acyclic complex is zero.

Part 2 (floating point, numpy/scipy).  The endomorphism Omega of the whole complex is evaluated
from the definition (1): the algebra A(K_1) (x) End(E) is represented on A(K_1) (x) E by
(x (x) T)(y (x) v) = (-1)^{|T||y|} xy (x) Tv, the exponential is a matrix exponential, and the
component along f_0, f_1 is read off.  The blocks are compared with the midpoints of Part 1,
and calD is recomputed from the polar decompositions.

Run:  uv run --with numpy --with scipy --with python-flint python theorem_6_4_dissipation.py
"""
from flint import arb, acb, ctx

ctx.prec = 400
PI = arb.pi()
z0, z1 = acb(-1, 1), acb(1, 1)


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def f_first(u, w):
    """f[u, w] for f(x) = e^{-x}, u != w."""
    return ((-u).exp() - (-w).exp()) / (u - w)


def f_second(u, q, w):
    """f[u, q, w], three distinct points."""
    return ((-u).exp() / ((u - q) * (u - w)) + (-q).exp() / ((q - u) * (q - w))
            + (-w).exp() / ((w - u) * (w - q)))


def f_second_double(u, q):
    """f[u, u, q] = f[u, q, u], u != q."""
    return (-(-u).exp() - f_first(u, q)) / (u - q)


def conj(z):
    return acb(z.real, -z.imag)


def matmul(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def adjoint(A):
    return [[conj(A[0][0]), conj(A[1][0])], [conj(A[0][1]), conj(A[1][1])]]


def block2(T, shift, varphi):
    """T is 2 x 2 with positive definite imaginary part.  Encloses
    tr(|T| (Phi - varphi) sin(Phi - varphi)),  Phi = (argument of the unitary polar factor) - shift."""
    M = matmul(T, adjoint(T))
    trM = (M[0][0] + M[1][1]).real
    detM = (M[0][0] * M[1][1] - M[0][1] * M[1][0]).real
    require(detM > 0, "determinant not enclosed away from zero")
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
        require(theta > 0 and theta < PI, "argument not enclosed in (0, pi)")
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


def acyclic_blocks(l, p, rho):
    """Blocks of Omega on the acyclic complex; returns Omega_V, Omega_W, -Omega_U0, -Omega_U1."""
    S = [[f_second_double(l[0], p), f_second(l[0], p, l[1])],
         [f_second(l[1], p, l[0]), f_second_double(l[1], p)]]
    minus_F = [[(-l[0]).exp(), -f_first(l[0], l[1])], [-f_first(l[1], l[0]), (-l[1]).exp()]]
    omega_V = [[(z0 * (-l[i]).exp() if i == k else acb(0)) + rho[i] * rho[k] * (minus_F[i][k] - p * S[i][k])
                for k in (0, 1)] for i in (0, 1)]
    s_p = [f_second_double(p, l[0]), f_second_double(p, l[1])]          # f[p, l_i, p]
    norm2 = rho[0] * rho[0] + rho[1] * rho[1]
    omega_W = (z1 - norm2) * (-p).exp() + rho[0] * rho[0] * l[0] * s_p[0] + rho[1] * rho[1] * l[1] * s_p[1]
    minus_omega_U0 = [[(z0 * (-l[i]).exp() if i == k else acb(0)) - (l[i] * l[k]).sqrt() * rho[i] * rho[k] * S[i][k]
                       for k in (0, 1)] for i in (0, 1)]
    minus_omega_U1 = z1 * (-p).exp() + p * (rho[0] * rho[0] * s_p[0] + rho[1] * rho[1] * s_p[1])
    trace = omega_V[0][0] + omega_V[1][1] + omega_W - minus_omega_U0[0][0] - minus_omega_U0[1][1] - minus_omega_U1
    require(trace.real.contains(0) and trace.imag.contains(0) and abs(trace) < arb("1e-80"),
            "trace of Omega on the acyclic complex not enclosed as zero")
    return omega_V, omega_W, minus_omega_U0, minus_omega_U1


def enclose_calD(l, p, rho):
    varphi = PI / 2
    omega_V, omega_W, minus_omega_U0, minus_omega_U1 = acyclic_blocks(l, p, rho)
    d_V, _ = block2(omega_V, arb(0), varphi)
    d_W = block1(omega_W, arb(0), varphi)
    d_U0, angles_U0 = block2(minus_omega_U0, PI, varphi)       # phases in (-pi, 0)
    d_U1 = block1(minus_omega_U1, -PI, varphi)                  # phase in (pi, 2 pi)
    return d_V + d_W + d_U0 + d_U1, (d_V, d_W, d_U0, d_U1), (omega_V, omega_W, minus_omega_U0, minus_omega_U1), angles_U0


def definition_check(blocks, D_enclosure):
    import numpy as np
    import scipy.linalg as sla

    # spaces of the whole complex, in the order (vertex, degree): (1,-1)=U_1, (0,0)=V, (1,0)=W, (0,1)=U_0
    dims = {(1, -1): 1, (0, 0): 3, (1, 0): 2, (0, 1): 2}
    spots = sorted(dims)
    offset, n = {}, 0
    for s in spots:
        offset[s] = n
        n += dims[s]
    sl = lambda s: slice(offset[s], offset[s] + dims[s])
    d = np.zeros((n, n), dtype=complex)
    rho = np.zeros((n, n), dtype=complex)
    d[sl((1, 0)), sl((1, -1))] = np.array([[0], [3]])
    d[sl((0, 1)), sl((0, 0))] = np.array([[0, 1, 0], [0, 0, 4]])
    rho[sl((1, 0)), sl((0, 0))] = np.array([[1, 0, 0], [0, 4, 100]])

    def projection(pred):
        P = np.zeros((n, n))
        for s in spots:
            if pred(s):
                P[sl(s), sl(s)] = np.eye(dims[s])
        return P

    P0, P1 = projection(lambda s: s[0] == 0), projection(lambda s: s[0] == 1)
    grading = sum((-1) ** s[1] * projection(lambda t, s=s: t == s) for s in spots)

    # A(K_1): basis e0, e1, f0, f1, a, a^*; left multiplication matrices
    names = ["e0", "e1", "f0", "f1", "a", "a*"]
    degree = {"e0": 0, "e1": 0, "f0": 0, "f1": 0, "a": 1, "a*": -1}
    products = {("e0", "e0"): (1, "e0"), ("e1", "e1"): (1, "e1"),
                ("e0", "f0"): (1, "f0"), ("f0", "e0"): (1, "f0"), ("e1", "f1"): (1, "f1"), ("f1", "e1"): (1, "f1"),
                ("e1", "a"): (1, "a"), ("a", "e0"): (1, "a"), ("e0", "a*"): (1, "a*"), ("a*", "e1"): (1, "a*"),
                ("a", "a*"): (1j, "f1"), ("a*", "a"): (-1j, "f0")}
    index = {x: i for i, x in enumerate(names)}
    left = {}
    for x in names:
        M = np.zeros((6, 6), dtype=complex)
        for y in names:
            if (x, y) in products:
                c, w = products[(x, y)]
                M[index[w], index[y]] = c
        left[x] = M
    parity = np.diag([(-1) ** degree[x] for x in names]).astype(complex)

    def element(x, T, odd):
        return np.kron(left[x] @ (parity if odd else np.eye(6)), T)

    dag = lambda T: T.conj().T
    A = element("e0", P0 @ d @ P0, True) + element("e1", P1 @ d @ P1, True) + element("a", rho, False)
    A_star = element("e0", 1j * dag(P0 @ d @ P0), True) + element("e1", 1j * dag(P1 @ d @ P1), True) + element("a*", dag(rho), False)
    require(np.allclose(A @ A, 0, atol=1e-9), "A^2 = 0 not confirmed")
    F = (A - A_star) @ (A - A_star) + 1j * (complex(z0) * element("f0", P0, False) + complex(z1) * element("f1", P1, False))
    X = sla.expm(-1j * F)
    unit = np.zeros(6)
    unit[index["e0"]] = unit[index["e1"]] = 1
    omega = np.zeros((n, n), dtype=complex)
    for col in range(n):
        v = np.zeros(n)
        v[col] = 1
        out = (X @ np.kron(unit, v)).reshape(6, n)
        omega[:, col] = out[index["f0"]] + out[index["f1"]]
    omega = grading @ omega

    omega_V, omega_W, minus_omega_U0, minus_omega_U1 = blocks
    c = lambda x: complex(x)
    V = omega[sl((0, 0)), sl((0, 0))]
    W = omega[sl((1, 0)), sl((1, 0))]
    U0 = omega[sl((0, 1)), sl((0, 1))]
    U1 = omega[sl((1, -1)), sl((1, -1))]
    err = max(abs(c(omega_V[i][k]) - V[1 + i, 1 + k]) for i in (0, 1) for k in (0, 1))
    err = max(err, abs(c(omega_W) - W[1, 1]))
    err = max(err, max(abs(-c(minus_omega_U0[i][k]) - U0[i, k]) for i in (0, 1) for k in (0, 1)))
    err = max(err, abs(-c(minus_omega_U1) - U1[0, 0]))
    off = max(np.abs(V[0, 1:]).max(), np.abs(V[1:, 0]).max(), abs(W[0, 1]), abs(W[1, 0]))
    print(f"   largest difference between the blocks of Part 1 and the definition: {err:.2e}")
    print(f"   blocks of the representation C -> C: |Omega - i| = {abs(V[0, 0] - 1j):.1e}, {abs(W[0, 0] - 1j):.1e};"
          f" coupling to the acyclic complex: {off:.1e}")

    varphi = np.pi / 2
    total = 0.0
    for s in spots:
        T = omega[sl(s), sl(s)]
        modulus = sla.sqrtm(T @ dag(T))
        unitary = np.linalg.solve(modulus, T)
        w, R = np.linalg.eig(unitary)
        # branch of degree n: (-pi n, pi (1 - n))
        angles = np.angle(w * np.exp(1j * np.pi * s[1])) - np.pi * s[1]
        require(bool(np.all((angles > -np.pi * s[1]) & (angles < np.pi * (1 - s[1])))),
                "a phase lies outside its branch")
        Rinv = np.linalg.inv(R)
        for k in range(len(w)):
            proj = np.outer(R[:, k], Rinv[k, :])
            weight = np.trace(modulus @ proj).real
            total += weight * (angles[k] - varphi) * np.sin(angles[k] - varphi)
    print(f"   calD from the definition (floating point): {total:.12f};  midpoint of the enclosure: {float(D_enclosure.mid()):.12f}")
    require(err < 1e-9 and off < 1e-9 and abs(total - float(D_enclosure.mid())) < 1e-8,
            "comparison with the definition failed")
    print("   the blocks and calD agree with the definition")


if __name__ == "__main__":
    l, p, rho = [arb(1), arb(16)], arb(9), [arb(4), arb(100)]
    D, parts, blocks, angles_U0 = enclose_calD(l, p, rho)
    print("Part 1: Theorem 6.4, enclosures")
    for name, value in zip(("V  (degree 0, vertex 0)", "W  (degree 0, vertex 1)",
                            "U_0 (degree 1, vertex 0)", "U_1 (degree -1, vertex 1)"), parts):
        print(f"   contribution of {name:26s} in {value.str(20)}")
    print(f"   phases on U_0: {angles_U0[0].str(12)}, {angles_U0[1].str(12)}")
    print(f"   calD in {D.str(22)}")
    require(D < 0, "calD < 0 not confirmed")
    print("   verified: calD < 0, that is, the phase potential increases along the flow at these metrics")
    print("Part 2: comparison with the definition (1)")
    definition_check(blocks, D)
