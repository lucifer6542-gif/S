#!/usr/bin/env python3
"""Check of the m=3 closed form of Paper A (Theorem 9').

Claim: for 5/2 <= S0 <= 2*sqrt(2),  T_max(S0) = (9/2)x - 4x^3 + (3*sqrt(3)/2)*sqrt(1-x^2),
x = sqrt(2) cos(pi/3 + arccos(S0/(2 sqrt 2))/3), i.e. Omega_Q^(3)(S0) = 1 - T_max/4.

What this script does
  1. Enumerates the 72 variants and their orbits under the symmetry group (32 elements) that
     preserves T = E00 - E01 - E10 + E11  (expected orbit sizes 4,4,8,8,16,16,16).
  2. Exact dual certificates (file m3_certificates.json, rational data) for the six orbits other
     than the main one: Z = [[diag(d_u), -M],[-M^T, diag(d_v)]] with M = Tm + lam*C is checked to
     be positive definite by an exact LDL^T factorisation over the rationals, and the resulting
     bound  sum(d)/2 - lam*a  on T_max(S0), S0 >= a, is compared with the closed form.
  3. For the main orbit, the analytic certificate lam(eps), d(eps):  (i) the 12 stationarity
     identities hold exactly, (ii) d_v > 0, (iii) trace of the Schur complement equals an explicit
     rational function of t = tan(eps/2), which is positive on (0, tan(pi/24)).
  4. Optional (--sdp, needs cvxpy): compares with the SDP optimum over all 72 variants.
Requires: sympy (and mpmath, which comes with it).  Run: python3 verify_m3_curve.py [--sdp]
"""
import itertools, json, sys, os
from fractions import Fraction as Fr
import sympy as sp
import mpmath as mp
mp.mp.dps = 50

HERE = os.path.dirname(os.path.abspath(__file__))
pairs = list(itertools.combinations(range(3), 2))

def variants():
    out = []
    for xs in pairs:
        for ys in pairs:
            for k in range(4):
                for ov in (1, -1):
                    c = [1, 1, 1, 1]; c[k] = -1
                    C = [[0] * 3 for _ in range(3)]
                    for i, (x, y) in enumerate([(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])]):
                        C[x][y] = ov * c[i]
                    out.append(tuple(map(tuple, C)))
    return out

def act(C, a, b, f2u, f2v, tr):
    C = [list(r) for r in C]
    if f2u: C[2] = [-v for v in C[2]]
    if f2v:
        for r in C: r[2] = -r[2]
    if a: C = [[-v for v in C[1]], [-v for v in C[0]], C[2]]
    if b: C = [[-r[1], -r[0], r[2]] for r in C]
    if tr: C = [list(r) for r in zip(*C)]
    return tuple(map(tuple, C))

def orbits():
    seen, orbs = set(), []
    for C in variants():
        if C in seen: continue
        o = {act(C, *g) for g in itertools.product([0, 1], repeat=5)}
        seen |= o; orbs.append(sorted(o))
    return orbs

def f_closed(S0):
    S0 = mp.mpf(S0)
    x = mp.sqrt(2) * mp.cos(mp.pi / 3 + mp.acos(S0 / (2 * mp.sqrt(2))) / 3)
    return mp.mpf(9) / 2 * x - 4 * x ** 3 + 3 * mp.sqrt(3) / 2 * mp.sqrt(1 - x * x)

def ldl_pd(Z):
    """exact: True iff the symmetric rational matrix Z is positive definite"""
    n = len(Z); A = [row[:] for row in Z]
    for k in range(n):
        if A[k][k] <= 0: return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= f * A[k][j]
    return True

def check_orbits_and_certs():
    orbs = orbits()
    sizes = sorted(len(o) for o in orbs)
    assert sizes == [4, 4, 8, 8, 16, 16, 16], sizes
    print("orbit sizes:", sizes)
    main_C = ((0, -1, 1), (0, 0, 0), (0, 1, 1))     # S = -E01 + E02 + E21 + E22
    main = [o for o in orbs if main_C in o]; assert len(main) == 1 and len(main[0]) == 16
    certs = json.load(open(os.path.join(HERE, "m3_certificates.json")))
    Tm = [[1, -1, 0], [-1, 1, 0], [0, 0, 0]]
    tau = Fr(36048, 10000)   # < f(2*sqrt2) = (3*sqrt6 + 5*sqrt2)/4 = 3.604885...
    assert mp.mpf(tau.numerator) / tau.denominator < (3 * mp.sqrt(6) + 5 * mp.sqrt(2)) / 4
    covered = []
    for cert in certs:
        C = tuple(map(tuple, cert["C"])); a = Fr(cert["a"]); lam = Fr(cert["lam"]); d = [Fr(s) for s in cert["d"]]
        assert lam >= 0 and all(x > 0 for x in d)
        Z = [[Fr(0)] * 6 for _ in range(6)]
        for i in range(3):
            Z[i][i] = d[i]; Z[3 + i][3 + i] = d[3 + i]
            for j in range(3):
                m = Tm[i][j] + lam * C[i][j]
                Z[i][3 + j] = -m; Z[3 + j][i] = -m
        assert ldl_pd(Z), "Z not positive definite"
        bound = sum(d) / 2 - lam * a
        orb = [o for o in orbs if C in o][0]
        print(f"  orbit size {len(orb):2d}  a={a}  exact bound T <= {float(bound):.6f}")
        covered.append((len(orb), C, a, bound))
        # claim: T_max(S0) <= bound for all S0 >= a
        if a == Fr(5, 2) and len(orb) == 8 and bound > tau:
            hi = Fr(11, 4)
            assert mp.mpf(bound.numerator) / bound.denominator <= f_closed(mp.mpf(hi.numerator) / hi.denominator)
            print("     bound <= f(11/4): valid for S0 in [5/2, 11/4]")
        elif a == Fr(11, 4):
            assert bound <= tau
            print("     bound <= f(2*sqrt2): valid for S0 in [11/4, 2*sqrt2]")
        else:
            assert bound <= tau
            print("     bound <= f(2*sqrt2): valid for all S0 <= 2*sqrt2")
    got = {c[1] for c in covered}
    for o in orbs:
        if o is main[0]: continue
        assert any(C in o for C in got), "orbit without certificate"
    print("all 6 non-main orbits covered")

def check_main_certificate():
    e, t = sp.symbols('e t')
    pi = sp.pi
    ua = [e, pi - e, 2 * pi / 3 - e]; va = [0, pi - 2 * e, pi / 3]
    lam = 2 * sp.sin(e) * sp.cos(e) ** 2 / (sp.sin(pi / 3 - e) * sp.sin(pi / 6 - 2 * e))
    M = sp.Matrix([[1, -1 - lam, lam], [-1, 1, 0], [0, lam, lam]])
    E = sp.Matrix(3, 3, lambda i, j: sp.cos(ua[i] - va[j]))
    d = [sum(M[i, j] * E[i, j] for j in range(3)) for i in range(3)]
    dp = [sum(M[i, j] * E[i, j] for i in range(3)) for j in range(3)]
    sub = {sp.cos(e): (1 - t ** 2) / (1 + t ** 2), sp.sin(e): 2 * t / (1 + t ** 2)}
    def zero(x):
        y = sp.expand_trig(sp.expand(x)).subs(sub)
        return sp.cancel(sp.together(y), extension=sp.sqrt(3)) == 0
    res = []
    for x in range(3):
        for f in (sp.cos, sp.sin):
            res.append(sum(M[x, y] * f(va[y]) for y in range(3)) - d[x] * f(ua[x]))
    for y in range(3):
        for f in (sp.cos, sp.sin):
            res.append(sum(M[x, y] * f(ua[x]) for x in range(3)) - dp[y] * f(va[y]))
    assert all(zero(r) for r in res)
    print("main orbit: 12 stationarity identities hold exactly (Z annihilates the planar optimum)")
    # sign facts: d' = (2cos e, d0, d2); d0 = cos e + (1+lam)cos 3e + lam cos(e-pi/3); d2 = 2 lam cos(e-pi/3)
    assert zero(dp[0] - 2 * sp.cos(e)) and zero(d[0] - (sp.cos(e) + (1 + lam) * sp.cos(3 * e) + lam * sp.cos(e - pi / 3)))
    assert zero(d[2] - 2 * lam * sp.cos(e - pi / 3)) and zero(dp[2] - d[2])
    print("   d_v = (2cos e, d0, d2) > 0 on (0, pi/12): all terms are positive (cos 3e > 0, lam > 0)")
    # trace of the Schur complement
    tr = sum(d) - sum(sum(M[x, y] ** 2 for x in range(3)) / dp[y] for y in range(3))
    trs = sp.cancel(sp.together(sp.expand_trig(tr).subs(sub)), extension=sp.sqrt(3))
    r3 = sp.sqrt(3)
    N1 = t**4 - (2 + 2*r3/3)*t**3 + (6 + 4*r3)*t**2 + (2 + 2*r3/3)*t + 1
    N2 = t**4 + (2 - 2*r3/3)*t**3 + (6 - 4*r3)*t**2 + (-2 + 2*r3/3)*t + 1
    den = (t - r3/3)**2 * (t + r3)**2 * (t**2 + 1) * (t - 2 - r3) * (t - r3 + 2)
    claimed = -2 * N1 * N2 / den
    assert sp.cancel(sp.together(trs - claimed), extension=sp.sqrt(3)) == 0
    print("   trace(Schur complement) = -2*N1*N2/den  (identity verified)")
    t0 = mp.tan(mp.pi / 24)
    # elementary positivity on 0 < t <= 0.14 (note t0 = tan(pi/24) = 0.1317 < 0.14)
    tb = sp.Rational(14, 100)
    assert 1 - (2 + 2 * r3 / 3) * tb ** 3 > 0                      # N1 >= 1 - c3 t^3 (other terms >= 0)
    assert 1 - (2 - 2 * r3 / 3) * tb - (4 * r3 - 6) * tb ** 2 > 0   # N2 >= 1 - |c1| t - |c2| t^2
    assert (-2 - r3) < 0 and (2 - r3) > 0                                              # 2 - sqrt3 = 0.268 > 0, so (t - sqrt3 + 2) > 0 for t > 0
    print("   N1, N2 > 0 on (0, tan(pi/24)); den < 0 there  =>  trace > 0  =>  Z >= 0")

def sdp_check():
    import numpy as np, cvxpy as cp
    V = variants()
    def solve(S0, C):
        G = cp.Variable((6, 6), PSD=True)
        S = sum(C[x][y] * G[x, 3 + y] for x in range(3) for y in range(3))
        T = G[0, 3] - G[0, 4] - G[1, 3] + G[1, 4]
        pr = cp.Problem(cp.Maximize(T), [cp.diag(G) == 1, S >= S0]); pr.solve(solver=cp.CLARABEL)
        return pr.value
    worst = 0
    for S0 in np.linspace(2.5, 2 * 2 ** .5 - 1e-3, 12):
        t = max(solve(S0, C) for C in V)
        worst = max(worst, abs(t - float(f_closed(S0))))
    print("SDP over all 72 variants vs closed form, 12 points: max |diff| =", worst)

if __name__ == "__main__":
    check_orbits_and_certs()
    check_main_certificate()
    # endpoints and small-delta behaviour
    print("Omega(5/2) =", 1 - f_closed(mp.mpf(5) / 2) / 4)
    print("Omega(2 sqrt2) =", 1 - f_closed(2 * mp.sqrt(2)) / 4, " exact 1-(3 sqrt6+5 sqrt2)/16 =", 1 - (3 * mp.sqrt(6) + 5 * mp.sqrt(2)) / 16)
    if "--sdp" in sys.argv: sdp_check()
    print("ALL CHECKS PASSED")
