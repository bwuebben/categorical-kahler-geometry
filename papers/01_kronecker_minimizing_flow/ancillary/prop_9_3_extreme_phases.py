"""Interval-arithmetic enclosures for Proposition 9.3: metrics at which an extreme phase moves
in the direction opposite to the one expected.

Charges z_0 = -1 + i, z_1 = 1 + i throughout; standard metrics; r = 1.

(i) The smallest phase decreases.  V = C^2, U = W = C, D = (0, f) with f^2 = l = 1/10,
    A_1 = (1, 0), A_2 = (0, 10).  All blocks of the endomorphism Omega are numbers:
        Omega_V = diag( z_0 + 1,  e^{-l} (z_0 + 100) ),
        Omega_U = x m(l) - z_0 e^{-l},        Omega_W = z_1 - 1 - x (1 - e^{-l}),
    with x = 100 / l and m(l) = 1 - (1 + l) e^{-l}.  The flow preserves this form, with
        dl/dt = l (Phi_A - Phi_U),      dx/dt = x (Phi_U - Phi_W),
    Phi_A = arg(z_0 + 100) the phase on the second line of V.  The smallest phase is
        phi_- = Phi_U = -arctan( 1 / (1 + y) ),      y = x m(l) e^{l},
    and  d phi_- / dt = (dy/dt) / ((1 + y)^2 + 1),
         (dy/dt) / y = (Phi_U - Phi_W) + (l e^{-l} / m(l) + 1) dl/dt.

(ii) The largest phase increases.  V = C^3, U = C, W = C^2, D = (0, 0, 9/2),
        A_1 = [[5.7, -1.7, 6], [6, 0, -5.6]],      A_2 = [[6, 0.1, -6], [-6, -1.8, -6]].
    With v = D^* (a column), l = |v|^2, C = A_1^* A_1 + A_2^* A_2 and R(l) = (1 - e^{-l}) / l:
        Omega_V = z_0 (1 - R(l) v v^*) + C - ((1 - R(l)) / l) (v v^* C + C v v^*)
                  + ((1 - 2 R(l) + e^{-l}) / l^2) (v^* C v) v v^*,
        Omega_U = (v^* C v) (1 - e^{-l} - l e^{-l}) / l^2 - z_0 e^{-l},
        Omega_W = z_1 - sum_j A_j (1 - ((1 - R(l)) / l) v v^*) A_j^*.
    The flow is  dv/dt = (Phi_V - Phi_U) v / 2,  dA_j/dt = (A_j Phi_V - Phi_W A_j) / 2.
    The program encloses the spectra of Phi_V and Phi_W, confirms that the largest eigenvalue
    of Phi_V is simple and exceeds the spectrum of Phi_W, and encloses its derivative along the
    flow by Lemma 4.4: with P~ = e^{-i Phi/2} |Omega_V| e^{i Phi/2} and
    X = e^{-i Phi/2} (d Omega_V/dt) e^{-i Phi/2}, the Hermitian solution Theta of
    P~ Theta + Theta P~ = (X - X^*) / (2i) gives d phi_+/dt = 2 tr(Pi Theta), Pi the spectral
    projection of the largest eigenvalue.  It also encloses the weight tr(|Omega_V| Pi) of this
    eigenvalue in the mass measure, and the mass.

    Consistency check: the derivative of Omega_V along the flow is compared with a difference
    quotient (step 10^-30).

Enclosures are computed with the ball arithmetic of Arb (through python-flint) at 400 bits.

Requires Python 3 with python-flint (an interface to Arb); run:  python prop_9_3_extreme_phases.py
The accompanying output file was produced with python-flint 0.9.0.
"""
from flint import arb, acb, acb_mat, ctx

ctx.prec = 400
PI = arb.pi()
IMAG = acb(0, 1)
z0, z1 = acb(-1, 1), acb(1, 1)


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def show(name, x, digits=22):
    print(f"   {name} in {x.str(digits)}")


def smallest_phase():
    print("(i) smallest phase: D = (0, f), f^2 = 1/10, A_1 = (1, 0), A_2 = (0, 10)")
    l, a2, h2 = arb(1) / 10, arb(100), arb(1)
    el = (-l).exp()
    m = 1 - (1 + l) * el
    x = a2 / l
    phi_H = (z0 + h2).arg()
    phi_A = (z0 + a2).arg()
    phi_U = (z0 * el - x * m).arg() - PI          # -Omega_U = z_0 e^{-l} - x m(l)
    phi_W = (z1 - h2 - x * (1 - el)).arg()
    require(phi_U < 0 and phi_U > -PI, "phase on U not enclosed in (-pi, 0)")
    y = x * m / el
    require((phi_U + (1 / (1 + y)).atan()).contains(0), "formula for Phi_U not confirmed")
    l_dot = l * (phi_A - phi_U)
    y_dot = y * ((phi_U - phi_W) + (l * el / m + 1) * l_dot)
    phi_U_dot = y_dot / ((1 + y) ** 2 + 1)
    show("phases on V", phi_H); show("           ", phi_A); show("phase on W", phi_W)
    show("phi_- = Phi_U", phi_U)
    show("d phi_-/dt", phi_U_dot)
    require(phi_U_dot < 0, "d phi_-/dt < 0 not confirmed")
    print("   verified: d phi_-/dt < 0")


def mat(rows):
    return acb_mat([[acb(x) for x in r] for r in rows])


def adjoint(M):
    return M.conjugate().transpose()


def diag(values):
    n = len(values)
    M = acb_mat(n, n)
    for i in range(n):
        M[i, i] = values[i]
    return M


def function_of(M, f):
    """f(M) for a normal matrix with simple eigenvalues, from enclosures of the eigensystem."""
    E, R = M.eig(right=True)
    return R * diag([f(e) for e in E]) * R.inv(), E


def polar_data(T):
    """T has positive definite imaginary part.  Returns the left modulus |T| = (T T^*)^{1/2},
    the polar phase Phi (spectrum in (0, pi)), its eigenvalues, its spectral projections, and
    the eigenvector matrices."""
    M = T * adjoint(T)
    modulus, EM = function_of(M, lambda e: acb(e.real.sqrt()))
    for e in EM:
        require(e.real > 0, "T T^* not enclosed as positive definite")
    W = modulus.inv() * T
    E, Q = W.eig(right=True)
    Qi = Q.inv()
    angles = [e.arg() for e in E]
    for t in angles:
        require(t > 0 and t < PI, "argument not enclosed in (0, pi)")
    n = len(angles)
    Phi = Q * diag([acb(t) for t in angles]) * Qi
    projections = []
    for k in range(n):
        ek = acb_mat(n, n)
        ek[k, k] = 1
        projections.append(Q * ek * Qi)
    return modulus, Phi, angles, projections, (Q, Qi)


def entry(M):
    return M[0, 0]


def largest_phase():
    print("(ii) largest phase: D = (0, 0, 9/2), A_1, A_2 as in Proposition 9.3")
    v = mat([[0], [0], [arb(9) / 2]])
    A = [mat([[arb(57) / 10, arb(-17) / 10, 6], [6, 0, arb(-56) / 10]]),
         mat([[6, arb(1) / 10, -6], [-6, arb(-18) / 10, -6]])]
    Id3 = diag([acb(1)] * 3)
    Id2 = diag([acb(1)] * 2)

    def coefficient(v, A):
        l = entry(adjoint(v) * v).real
        el = (-l).exp()
        R = (1 - el) / l
        c1 = (1 - R) / l
        c2 = (1 - 2 * R + el) / (l * l)
        vv = v * adjoint(v)
        C = adjoint(A[0]) * A[0] + adjoint(A[1]) * A[1]
        vCv = entry(adjoint(v) * C * v)
        K = C - (vv * C + C * vv) * acb(c1) + vv * (acb(c2) * vCv)
        omega_V = (Id3 - vv * acb(R)) * z0 + K
        J = vCv.real * (1 - el - l * el) / (l * l)
        minus_omega_U = z0 * el - J
        G = A[0] * adjoint(A[0]) + A[1] * adjoint(A[1])
        for a in A:
            av = a * v
            G = G - av * adjoint(av) * acb(c1)
        return omega_V, minus_omega_U, G, (l, el, R, c1, c2, vv, C, vCv)

    omega_V, minus_omega_U, G, (l, el, R, c1, c2, vv, C, vCv) = coefficient(v, A)
    modulus, Phi_V, angles_V, projections, (Q, Qi) = polar_data(omega_V)
    phi_U = minus_omega_U.arg() - PI
    Phi_W, EG = function_of(G, lambda e: acb((z1 - e.real).arg()))
    angles_W = [(z1 - e.real).arg() for e in EG]
    top = max(range(3), key=lambda k: float(angles_V[k].mid()))
    for k in range(3):
        if k != top:
            require(angles_V[top] > angles_V[k], "largest eigenvalue of Phi_V not enclosed as simple")
    for t in angles_W:
        require(angles_V[top] > t, "largest eigenvalue of Phi_V not above the spectrum of Phi_W")
    print("   spectrum of Phi_V:", ", ".join(t.str(12) for t in angles_V))
    print("   spectrum of Phi_W:", ", ".join(t.str(12) for t in angles_W))
    show("Phi_U", phi_U, 12)
    show("gap between phi_+ and the spectrum of Phi_W", angles_V[top] - max(angles_W, key=lambda t: float(t.mid())), 12)
    weight = (modulus * projections[top]).trace()
    require(weight.imag.contains(0), "weight not enclosed as real")
    mass = modulus.trace().real + abs(minus_omega_U) + sum(abs(z1 - e.real) for e in EG)
    show("weight of phi_+ in the mass measure", weight.real, 12)
    show("mass", mass, 12)

    # velocities of the flow
    v_dot = (Phi_V - Id3 * acb(phi_U)) * v * acb(arb(1) / 2)
    A_dot = [(a * Phi_V - Phi_W * a) * acb(arb(1) / 2) for a in A]
    l_dot = (entry(adjoint(v) * v_dot) * 2).real
    C_dot = (Phi_V * C + C * Phi_V) * acb(arb(1) / 2) - adjoint(A[0]) * Phi_W * A[0] - adjoint(A[1]) * Phi_W * A[1]
    vv_dot = v_dot * adjoint(v) + v * adjoint(v_dot)
    R_dot = (l * el - (1 - el)) / (l * l) * l_dot
    c1_dot = (-R_dot * l - (1 - R) * l_dot) / (l * l)
    c2_dot = ((-2 * R_dot - el * l_dot) * l * l - (1 - 2 * R + el) * 2 * l * l_dot) / (l ** 4)
    vCv_dot = entry(adjoint(v_dot) * C * v) + entry(adjoint(v) * C_dot * v) + entry(adjoint(v) * C * v_dot)
    K_dot = (C_dot - (vv * C + C * vv) * acb(c1_dot) - (vv_dot * C + vv * C_dot + C_dot * vv + C * vv_dot) * acb(c1)
             + vv * (acb(c2_dot) * vCv) + vv * (acb(c2) * vCv_dot) + vv_dot * (acb(c2) * vCv))
    omega_V_dot = (vv * acb(-R_dot) - vv_dot * acb(R)) * z0 + K_dot

    # consistency check: difference quotient of Omega_V along the velocity
    step = arb(10) ** (-30)
    v_step = v + v_dot * acb(step)
    A_step = [a + ad * acb(step) for a, ad in zip(A, A_dot)]
    quotient = (coefficient(v_step, A_step)[0] - omega_V) * acb(1 / step) - omega_V_dot
    worst = max(float(abs(quotient[i, j]).upper()) for i in range(3) for j in range(3))
    print(f"   consistency check: |difference quotient of Omega_V - derivative| <= {worst:.2e}")
    require(worst < 1e-20, "difference quotient does not agree with the derivative")

    # derivative of the largest eigenvalue of Phi_V (Lemma 4.4)
    half = Q * diag([(IMAG * acb(t) / 2).exp() for t in angles_V]) * Qi       # e^{i Phi/2}
    half_inv = Q * diag([(-IMAG * acb(t) / 2).exp() for t in angles_V]) * Qi
    P_tilde = half_inv * modulus * half
    X = half_inv * omega_V_dot * half_inv
    right = (X - adjoint(X)) * (1 / (2 * IMAG))
    system = acb_mat(9, 9)
    rhs = acb_mat(9, 1)
    for i in range(3):
        for j in range(3):
            rhs[3 * i + j, 0] = right[i, j]
            for k in range(3):
                system[3 * i + j, 3 * k + j] += P_tilde[i, k]      # (P~ Theta)_ij
                system[3 * i + j, 3 * i + k] += P_tilde[k, j]      # (Theta P~)_ij
    solution = system.solve(rhs)
    Theta = acb_mat(3, 3)
    for i in range(3):
        for j in range(3):
            Theta[i, j] = solution[3 * i + j, 0]
    derivative = 2 * (projections[top] * Theta).trace()
    require(derivative.imag.contains(0), "derivative not enclosed as real")
    show("d phi_+/dt", derivative.real)
    require(derivative.real > 0, "d phi_+/dt > 0 not confirmed")
    print("   verified: phi_+ is the largest eigenvalue of Phi_V near these metrics, and d phi_+/dt > 0")


if __name__ == "__main__":
    smallest_phase()
    largest_phase()
