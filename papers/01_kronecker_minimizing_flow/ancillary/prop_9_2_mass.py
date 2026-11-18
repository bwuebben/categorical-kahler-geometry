"""Interval-arithmetic enclosures for Proposition 9.2: metrics at which the mass increases.

Data of Proposition 9.2.  Charges z_0 = -1 + i, z_1 = 1 + i; generalized Kronecker quiver with
k = 512 arrows; V = C^2, U = C, W = C^512 with the standard metrics;

    D = (0, f),  f^2 = 383/100,
    A_j = the matrix whose only nonzero row is the j-th, equal to a_j,
    a_j = (-0.1978, -0.6072) for 480 values of j,  a_j = (-0.5038, 0.8392) for the other 32.

The cohomology representation C -> C^512 is stable (every a_j has a nonzero first entry), and
r = 1.

Formulas (proof of Proposition 9.2).  With v = D^*, l = |v|^2, C = sum_j a_j^* a_j and
R(l) = (1 - e^{-l}) / l,

    Omega_V = z_0 (1 - R(l) v v^*) + C - ((1 - R(l)) / l) (v v^* C + C v v^*)
              + ((1 - 2 R(l) + e^{-l}) / l^2) (v^* C v) v v^*,
    Omega_U = (v^* C v) (1 - e^{-l} - l e^{-l}) / l^2 - z_0 e^{-l},
    Omega_W = diag_j( z_1 - |a_j|^2 + ((1 - R(l)) / l) |a_j v|^2 ).

The flow preserves this form, with

    dv/dt = (Phi_V - Phi_U) v / 2,      da_j/dt = (a_j Phi_V - (Phi_W)_jj a_j) / 2.

The mass is  m = tr|Omega_V| + |Omega_U| + sum_j |(Omega_W)_jj|,  and for a matrix T with unitary
polar factor W_T,  d tr|T| / dt = Re tr( (dT/dt) W_T^* ).  The polar decomposition of a 2 x 2
matrix is explicit:  |T| = (M + sqrt(det M)) / sqrt(tr M + 2 sqrt(det M)),  M = T T^*.

The program encloses m, dm/dt and the dissipation calD of the phase potential (Section 7) with
the ball arithmetic of Arb (through python-flint) at 500 bits.  It also makes three consistency
checks:
  (1) the trace identity  tr Omega_V + Omega_U + tr Omega_W = z_0 + k z_1  (k the number of arrows);
  (2) the enclosure of dm/dt is compared with a difference quotient of the enclosed mass along
      the velocity of the flow (step 10^-30);
  (3) for the same data with all rows a_j divided by 20, the same computation gives dm/dt < 0.

Finally it encloses dm/dt for the same two rows in the proportion 15 : 1 with fewer arrows:
dm/dt > 0 for k = 400 (375 + 25) and dm/dt < 0 for k = 368 (345 + 23).

Requires Python 3 with python-flint (an interface to Arb); run:  python prop_9_2_mass.py
The accompanying output file was produced with python-flint 0.9.0.
"""
from flint import arb, acb, ctx

ctx.prec = 500
PI = arb.pi()
z0, z1 = acb(-1, 1), acb(1, 1)


def cj(z): return acb(z.real, -z.imag)
def matmul(A, B): return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                          [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
def adjoint(A): return [[cj(A[0][0]), cj(A[1][0])], [cj(A[0][1]), cj(A[1][1])]]
def add(A, B, c=1): return [[A[i][j] + c * B[i][j] for j in (0, 1)] for i in (0, 1)]
def scal(c, A): return [[c * A[i][j] for j in (0, 1)] for i in (0, 1)]
def tr(A): return A[0][0] + A[1][1]
def outer(u, w): return [[u[i] * cj(w[j]) for j in (0, 1)] for i in (0, 1)]      # u w^*
def apply(A, u): return [A[0][0]*u[0] + A[0][1]*u[1], A[1][0]*u[0] + A[1][1]*u[1]]
def inner(u, w): return cj(u[0]) * w[0] + cj(u[1]) * w[1]                         # u^* w
def scal_vec(c, u): return [c * u[0], c * u[1]]
def sub_vec(u, w): return [u[0] - w[0], u[1] - w[1]]
ID = [[acb(1), acb(0)], [acb(0), acb(1)]]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def polar2(T):
    """T is 2 x 2 with positive definite imaginary part.  Returns the left modulus |T|, the
    unitary factor W = |T|^{-1} T and the polar phase Phi (W = e^{i Phi}, spectrum in (0, pi))."""
    M = matmul(T, adjoint(T))
    trM = tr(M).real
    detM = (M[0][0] * M[1][1] - M[0][1] * M[1][0]).real
    require(detM > 0, "determinant not enclosed away from zero")
    sd = detM.sqrt(); den = (trM + 2 * sd).sqrt()
    P = scal(1 / den, add(M, scal(sd, ID)))
    dP = P[0][0] * P[1][1] - P[0][1] * P[1][0]
    Pinv = scal(1 / dP, [[P[1][1], -P[0][1]], [-P[1][0], P[0][0]]])
    W = matmul(Pinv, T)
    t, d = tr(W), W[0][0] * W[1][1] - W[0][1] * W[1][0]
    disc = (t * t - 4 * d).sqrt()
    eig = [(t + disc) / 2, (t - disc) / 2]
    Phi = [[acb(0), acb(0)], [acb(0), acb(0)]]
    for k in (0, 1):
        this, other = eig[k], eig[1 - k]
        theta = this.arg()
        require(theta > 0 and theta < PI, "argument not enclosed in (0, pi)")
        Phi = add(Phi, scal(theta / (this - other), add(W, scal(other, ID), -1)))
    return P, W, Phi


def data(l, row_types):
    """row_types: list of (multiplicity, first entry, second entry), real.  v = (0, sqrt l)."""
    v = [acb(0), acb(l.sqrt())]
    return v, [(n, [acb(x), acb(y)]) for (n, x, y) in row_types]


def coefficient(v, rows):
    l = inner(v, v).real
    el = (-l).exp()
    R = (1 - el) / l
    c1 = (1 - R) / l                       # (1 - R(l)) / l
    c2 = (1 - 2 * R + el) / (l * l)        # (1 - 2 R(l) + e^{-l}) / l^2
    jd = (1 - el - l * el) / (l * l)       # (1 - e^{-l} - l e^{-l}) / l^2
    C = [[acb(0), acb(0)], [acb(0), acb(0)]]
    for (n, a) in rows:                    # a is a row vector; a^* a = outer(conj a, conj a)
        ac = [cj(a[0]), cj(a[1])]
        C = add(C, scal(n, outer(ac, ac)))
    vv = outer(v, v)
    vCv = inner(v, apply(C, v))
    omega_V = add(add(add(scal(z0, add(ID, scal(R, vv), -1)), C),
                      scal(c1, add(matmul(vv, C), matmul(C, vv))), -1), scal(c2 * vCv, vv))
    omega_U = vCv * jd - z0 * el
    omega_W = []
    for (n, a) in rows:
        av = a[0] * v[0] + a[1] * v[1]
        omega_W.append(z1 - (abs(a[0]) ** 2 + abs(a[1]) ** 2) + c1 * abs(av) ** 2)
    return omega_V, omega_U, omega_W, dict(l=l, el=el, R=R, c1=c1, c2=c2, jd=jd, C=C, vv=vv, vCv=vCv)


def mass_and_derivative(l, row_types):
    v, rows = data(l, row_types)
    omega_V, omega_U, omega_W, q = coefficient(v, rows)
    k = sum(n for n, _ in rows)
    residual = tr(omega_V) + omega_U + sum(n * w for (n, _), w in zip(rows, omega_W)) - (z0 + k * z1)
    require(residual.real.contains(0) and residual.imag.contains(0) and abs(residual) < arb("1e-100"),
            "trace identity not confirmed")
    mod_V, W_V, Phi_V = polar2(omega_V)
    phi_U = (-omega_U).arg() - PI
    require(phi_U < 0 and phi_U > -PI, "phase on U not enclosed in (-pi, 0)")
    phi_W = [w.arg() for w in omega_W]
    for x in phi_W:
        require(x > 0 and x < PI, "phase on W not enclosed in (0, pi)")
    m = tr(mod_V).real + abs(omega_U) + sum(n * abs(w) for (n, _), w in zip(rows, omega_W))
    # velocities of the flow
    v_dot = scal_vec(acb(1) / 2, sub_vec(apply(Phi_V, v), [phi_U * v[0], phi_U * v[1]]))
    rows_dot = []
    for (n, a), pw in zip(rows, phi_W):
        a_Phi = [a[0] * Phi_V[0][0] + a[1] * Phi_V[1][0], a[0] * Phi_V[0][1] + a[1] * Phi_V[1][1]]
        rows_dot.append([(a_Phi[0] - pw * a[0]) / 2, (a_Phi[1] - pw * a[1]) / 2])
    l_, el, R, c1, c2, jd, C, vv, vCv = (q[x] for x in ("l", "el", "R", "c1", "c2", "jd", "C", "vv", "vCv"))
    l_dot = (inner(v_dot, v) + inner(v, v_dot)).real
    R_dot = -jd * l_dot
    c1_dot = (jd / l_ - c1 / l_) * l_dot
    c2_dot = ((2 * jd - el) / (l_ * l_) - 2 * c2 / l_) * l_dot
    jd_dot = (el / l_ - 2 * jd / l_) * l_dot
    vv_dot = add(outer(v_dot, v), outer(v, v_dot))
    C_dot = [[acb(0), acb(0)], [acb(0), acb(0)]]
    for (n, a), ad in zip(rows, rows_dot):
        ac, adc = [cj(a[0]), cj(a[1])], [cj(ad[0]), cj(ad[1])]
        C_dot = add(C_dot, scal(n, add(outer(adc, ac), outer(ac, adc))))
    vCv_dot = inner(v_dot, apply(C, v)) + inner(v, apply(C_dot, v)) + inner(v, apply(C, v_dot))
    omega_V_dot = scal(z0, add(scal(-R_dot, vv), scal(R, vv_dot), -1))
    omega_V_dot = add(omega_V_dot, C_dot)
    omega_V_dot = add(omega_V_dot, scal(c1_dot, add(matmul(vv, C), matmul(C, vv))), -1)
    omega_V_dot = add(omega_V_dot, scal(c1, add(add(matmul(vv_dot, C), matmul(vv, C_dot)),
                                               add(matmul(C_dot, vv), matmul(C, vv_dot)))), -1)
    omega_V_dot = add(omega_V_dot, scal(c2_dot * vCv, vv))
    omega_V_dot = add(omega_V_dot, scal(c2, add(scal(vCv_dot, vv), scal(vCv, vv_dot))))
    omega_U_dot = vCv_dot * jd + vCv * jd_dot + z0 * el * l_dot
    m_dot = tr(matmul(omega_V_dot, adjoint(W_V))).real + (omega_U_dot * cj(omega_U) / abs(omega_U)).real
    for (n, a), ad, w in zip(rows, rows_dot, omega_W):
        av = a[0] * v[0] + a[1] * v[1]
        av_dot = ad[0] * v[0] + ad[1] * v[1] + a[0] * v_dot[0] + a[1] * v_dot[1]
        norm_dot = 2 * (cj(a[0]) * ad[0] + cj(a[1]) * ad[1]).real
        w_dot = -norm_dot + c1_dot * abs(av) ** 2 + c1 * 2 * (cj(av) * av_dot).real
        m_dot += n * (w_dot * cj(w) / abs(w)).real
    return m, m_dot, (v, rows, v_dot, rows_dot), (Phi_V, phi_U, phi_W)


def mass_only(v, rows):
    omega_V, omega_U, omega_W, _ = coefficient(v, rows)
    mod_V, _, _ = polar2(omega_V)
    return tr(mod_V).real + abs(omega_U) + sum(n * abs(w) for (n, _), w in zip(rows, omega_W))


if __name__ == "__main__":
    l = arb(383) / 100
    row_types = [(480, arb(-1978) / 10000, arb(-6072) / 10000), (32, arb(-5038) / 10000, arb(8392) / 10000)]
    m, m_dot, (v, rows, v_dot, rows_dot), (Phi_V, phi_U, phi_W) = mass_and_derivative(l, row_types)
    print("Proposition 9.2: f^2 = 383/100; a_j = (-0.1978, -0.6072) for 480 arrows, (-0.5038, 0.8392) for 32 arrows;"
          " z_0 = -1 + i, z_1 = 1 + i")
    print(f"   m in {m.str(25)}")
    print(f"   dm/dt in {m_dot.str(25)}")
    print(f"   Phi_U in {phi_U.str(25)}; the two values of Phi_W in {phi_W[0].str(25)}, {phi_W[1].str(25)}")
    require(m_dot > 0, "dm/dt > 0 not confirmed")
    print("   verified: dm/dt > 0")
    step = arb("1e-30")
    v_step = [v[0] + step * v_dot[0], v[1] + step * v_dot[1]]
    rows_step = [(n, [a[0] + step * ad[0], a[1] + step * ad[1]]) for (n, a), ad in zip(rows, rows_dot)]
    quotient = (mass_only(v_step, rows_step) - m) / step
    print(f"   consistency check: difference quotient of m along the velocity, step 10^-30: {quotient.str(20)}")
    require(abs(quotient - m_dot) < arb("1e-20"), "difference quotient does not agree with dm/dt")
    # dissipation of the phase potential for the same data; varphi = arg(z_0 + 512 z_1)
    Z = z0 + 512 * z1
    varphi = Z.arg()
    omega_V, omega_U, omega_W, _ = coefficient(v, rows)
    mod_V, W_V, _ = polar2(omega_V)
    t, d = tr(W_V), W_V[0][0] * W_V[1][1] - W_V[0][1] * W_V[1][0]
    disc = (t * t - 4 * d).sqrt()
    eig = [(t + disc) / 2, (t - disc) / 2]
    dissipation = arb(0)
    for i in (0, 1):
        this, other = eig[i], eig[1 - i]
        proj = scal(1 / (this - other), add(W_V, scal(other, ID), -1))
        weight = tr(matmul(mod_V, proj)).real
        x = this.arg() - varphi
        dissipation += weight * x * x.sin()
    x = phi_U - varphi
    dissipation += abs(omega_U) * x * x.sin()
    for (n, a), w, pw in zip(rows, omega_W, phi_W):
        x = pw - varphi
        dissipation += n * abs(w) * x * x.sin()
    print(f"   dissipation calD of the phase potential in {dissipation.str(25)}")
    require(dissipation > 0, "calD > 0 not confirmed")
    print("   verified: calD > 0")
    print("consistency check: the same data with all rows a_j divided by 20")
    small_rows = [(480, arb(-1978) / 200000, arb(-6072) / 200000), (32, arb(-5038) / 200000, arb(8392) / 200000)]
    m_small, m_dot_small, _, _ = mass_and_derivative(l, small_rows)
    print(f"   m in {m_small.str(20)}, dm/dt in {m_dot_small.str(20)}")
    require(m_dot_small < 0, "dm/dt < 0 not confirmed")
    print("   verified: dm/dt < 0")
    print("the same two rows in the proportion 15 : 1 with fewer arrows")
    for k, n_first, n_second, positive in [(400, 375, 25, True), (368, 345, 23, False)]:
        fewer = [(n_first, arb(-1978) / 10000, arb(-6072) / 10000), (n_second, arb(-5038) / 10000, arb(8392) / 10000)]
        m_k, m_dot_k, _, _ = mass_and_derivative(l, fewer)
        print(f"   k = {k} ({n_first} + {n_second} arrows): m in {m_k.str(20)}, dm/dt in {m_dot_k.str(20)}")
        if positive:
            require(m_dot_k > 0, "dm/dt > 0 not confirmed")
            print("   verified: dm/dt > 0")
        else:
            require(m_dot_k < 0, "dm/dt < 0 not confirmed")
            print("   verified: dm/dt < 0")
