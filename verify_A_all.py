#!/usr/bin/env python3
"""
논문 A 수치 교차 확인 스크립트 (2자 이진: 정확한 환원, 양자 곡선, 입력 수 의존성)
필요: numpy, scipy.  모든 결과는 증명이 아니라 수치 교차 확인이다.

사용:
  python3 verify_A_all.py              # 전부 (--exact --curve --perturb --adv --thm8 --prop7 --sec7)
  python3 verify_A_all.py --exact      # 정리 1, 2, 3 (NS 상자, g 전수)
  python3 verify_A_all.py --curve      # 정리 4 (무작위 단위벡터), 정리 5 (명시적 곡선)
  python3 verify_A_all.py --perturb    # 정리 4: 정리 5 배치에 각도 잡음
  python3 verify_A_all.py --adv        # 정리 4: 적대적 최적화 (Smax = 2.1 ... 2.8)
  python3 verify_A_all.py --thm8       # 정리 8: m=3, Omega=0 이면 S_glob <= 5/2
  python3 verify_A_all.py --prop7      # 명제 7: m=4
  python3 verify_A_all.py --sec7       # 7절 표 + 정리 9 (상계/하계/쌍대 증인)
  옵션: --seed N (기본 12345)

7절은 각 변형 sigma 마다 Gram 행렬 G = V V^T (V: 6x6, 열 정규화) 위의 비볼록 최적화(SLSQP 다중 시작)이다.
rank 6 = 전체 차원이므로 Burer-Monteiro 의미에서 SDP와 같은 문제이고, 값은 T 의 하한(Omega 의 상한)이다.
"""
import sys, itertools, time
import numpy as np
from scipy.optimize import minimize, brentq

SQ2 = np.sqrt(2.0)
SIG = np.array([s for s in itertools.product([1, -1], repeat=4) if np.prod(s) == -1], float)   # 8개 CHSH 변형
TV = np.array([1, -1, -1, 1], float)
OK_ALL = []

def report(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)
    OK_ALL.append(bool(ok)); return bool(ok)

# ---------- 기본 함수 ----------
def Fc(c): return 6*c - 4*c**3
def c_of_S(s):
    """6c-4c^3 = s 의 [1/sqrt2, 1] 위 근 (F 는 거기서 감소)."""
    if s >= Fc(1/SQ2): return 1/SQ2          # 끝점의 부동소수점 오차 처리
    if s <= 2.0: return 1.0
    return brentq(lambda c: Fc(c) - s, 1/SQ2, 1.0, xtol=1e-14)

def subpairs(m):
    for xs in itertools.combinations(range(m), 2):
        for ys in itertools.combinations(range(m), 2):
            yield xs, ys

def sub_stats(E, m):
    """E: (N,m,m). 반환 (S_glob (N,), max|T_sub| (N,))."""
    Sg = np.full(E.shape[0], -np.inf); Tm = np.zeros(E.shape[0])
    for (xa, xb), (ya, yb) in subpairs(m):
        e = np.stack([E[:, xa, ya], E[:, xa, yb], E[:, xb, ya], E[:, xb, yb]], 1)
        Sg = np.maximum(Sg, (e @ SIG.T).max(1)); Tm = np.maximum(Tm, np.abs(e @ TV))
    return Sg, Tm

def smax_T(E):
    """2-2-2: E (N,2,2) -> (Smax, T)."""
    e = np.stack([E[:, 0, 0], E[:, 0, 1], E[:, 1, 0], E[:, 1, 1]], 1)
    return (e @ SIG.T).max(1), e @ TV

# ---------- 무작위 NS 상자 (국소 결정론 + 균등 주변분포 패리티 상자의 볼록 결합) ----------
def rand_ns(m, rng, ncomp=12):
    comps = []
    for _ in range(ncomp):
        P = np.zeros((2, 2, m, m))
        if rng.random() < 0.5:
            fa = rng.integers(0, 2, m); fb = rng.integers(0, 2, m)
            for x in range(m):
                for y in range(m): P[fa[x], fb[y], x, y] = 1
        else:
            f = rng.integers(0, 2, (m, m))
            for x in range(m):
                for y in range(m):
                    for a in range(2): P[a, a ^ f[x, y], x, y] = 0.5
        comps.append(P)
    w = rng.dirichlet(np.ones(ncomp))
    return np.tensordot(w, np.array(comps), 1)

def corr(P):
    sg = np.array([[1, -1], [-1, 1]], float)
    return np.einsum('ab,abxy->xy', sg, P)

def W_of(P, gA, gB):
    return sum(P[a, b, gA[b], gB[a]] for a in range(2) for b in range(2))

def omega_enum(P, m):
    best = np.inf
    for gA in itertools.product(range(m), repeat=2):
        for gB in itertools.product(range(m), repeat=2):
            best = min(best, W_of(P, gA, gB))
    return best

# ================= 정리 1, 2, 3 =================
def run_exact(rng):
    print("\n=== EXACT: 정리 1, 2, 3 ===")
    for m, n, name in [(2, 300, "정리 1 (NS 2-2-2, 16개 g 전수)"), (3, 300, "정리 2 (NS 3-3-2-2, 81개 g 전수)")]:
        err = 0.0
        for _ in range(n):
            P = rand_ns(m, rng); E = corr(P)
            _, Tm = sub_stats(E[None], m)
            err = max(err, abs(omega_enum(P, m) - (1 - Tm[0]/4)))
        report(f"{name}: Omega = 1 - max|T_sub|/4", err < 1e-12, f"{n}개, 최대 오차 {err:.1e}")
    # 정리 3
    n = 20000; Es = np.array([corr(rand_ns(2, rng)) for _ in range(n)])
    S, T = smax_T(Es)
    viol = int((np.abs(T) + S > 6 + 1e-12).sum())
    report("정리 3: |T| + Smax <= 6 (무작위 NS 2만 개)", viol == 0, f"위반 {viol}, 최대 |T|+Smax = {(np.abs(T)+S).max():.6f}")
    xs = np.linspace(-1, 1, 41); e = 0.0
    for x in xs:
        E = np.array([[[x, -1], [-1, 1]]]); Sx, Tx = smax_T(E)
        e = max(e, abs(Sx[0] - (3 - x)), abs(Tx[0] - (3 + x)))
    report("정리 3: 등호 족 s=(x,1,1,1): Smax=3-x, T=3+x", e < 1e-12, f"최대 오차 {e:.1e}")

# ================= 정리 4, 5 =================
def run_curve(rng_seed):
    print("\n=== CURVE: 정리 4 (무작위 단위벡터), 정리 5 (명시적 곡선) ===")
    rng = np.random.default_rng(rng_seed)
    n = 60000; d = 4
    u = rng.normal(size=(n, 2, d)); v = rng.normal(size=(n, 2, d))
    u /= np.linalg.norm(u, axis=2, keepdims=True); v /= np.linalg.norm(v, axis=2, keepdims=True)
    E = np.einsum('nxd,nyd->nxy', u, v)
    S, T = smax_T(E); sel = (S >= 2) & (S <= 2*SQ2)
    cs = np.array([c_of_S(s) for s in S[sel]])
    margin = (1 - np.abs(T[sel])/4) - (1 - cs**3)
    report("정리 4: 무작위 단위벡터에서 Omega >= 1 - c(Smax)^3",
           int((margin < -1e-12).sum()) == 0,
           f"{n}개 중 2<=Smax<=2sqrt2 {int(sel.sum())}개, 최소 여유 {margin.min():+.2e}, 위반 {int((margin < -1e-12).sum())}")
    t = np.linspace(pi_over4 + 1e-3, np.pi/2, 200)
    a0, a1, b0, b1 = 0*t, 2*t, 3*t - np.pi/2, t - np.pi/2
    E = np.stack([np.stack([np.cos(a0-b0), np.cos(a0-b1)], 1), np.stack([np.cos(a1-b0), np.cos(a1-b1)], 1)], 1)
    S, T = smax_T(E); c = np.sin(t)
    e = max(np.abs(S - Fc(c)).max(), np.abs(np.abs(T) - 4*c**3).max(), np.abs(E[:, 0, 0] - np.sin(3*t)).max())
    report("정리 5: Smax = 6c-4c^3, |T| = 4c^3, E00 = sin3t (명시적 곡선)", e < 1e-12, f"최대 오차 {e:.1e}")
    report("정리 5: Smax=2 에서 Omega=0, Smax=2sqrt2 에서 Omega=1-2^(-3/2)",
           abs(1 - c_of_S(2.0)**3) < 1e-9 and abs((1 - c_of_S(2*SQ2)**3) - (1 - 2**-1.5)) < 1e-9,
           f"{1-2**-1.5:.6f}")
pi_over4 = np.pi/4

def run_perturb(rng_seed):
    print("\n=== PERTURB: 정리 5 배치에 각도 가우스 잡음 ===")
    rng = np.random.default_rng(rng_seed + 1)
    for eps in (1e-3, 1e-2, 1e-1):
        n = 50000
        t = rng.uniform(pi_over4 + 0.02, np.pi/2 - 0.02, n)
        ang = np.stack([0*t, 2*t, 3*t - np.pi/2, t - np.pi/2], 1) + rng.normal(0, eps, (n, 4))
        a, b = ang[:, :2], ang[:, 2:]
        E = np.cos(a[:, :, None] - b[:, None, :])
        S, T = smax_T(E); sel = (S >= 2) & (S <= 2*SQ2)
        cs = np.array([c_of_S(s) for s in S[sel]])
        marg = (1 - np.abs(T[sel])/4) - (1 - cs**3)
        report(f"정리 4 타이트함, eps={eps:g}", int((marg < -1e-12).sum()) == 0,
               f"유효 {int(sel.sum())}개, 최소 여유 {marg.min():+.1e}")

# ---------- 작은 비볼록 최적화 도구 ----------
def unit_rows(X, k, d):
    V = X.reshape(k, d); return V / np.linalg.norm(V, axis=1, keepdims=True)

def tmax_gram(m, sig_pair, S0, rng, starts=12, extra_starts=()):
    """m 입력, 목표 변형 (xs, ys, sigma). 최대화: T = (u0-u1).(v0-v1) s.t. S_sigma >= S0.
    벡터 u_0..u_{m-1}, v_0..v_{m-1} in R^{2m} (Gram PSD, 대각 1)."""
    xs, ys, sg = sig_pair; k = 2*m; d = 2*m
    def split(X):
        V = unit_rows(X, k, d); return V[:m], V[m:]
    def T_of(X):
        U, Vv = split(X); return (U[0]-U[1]) @ (Vv[0]-Vv[1])
    def S_of(X):
        U, Vv = split(X)
        e = np.array([U[xs[0]] @ Vv[ys[0]], U[xs[0]] @ Vv[ys[1]], U[xs[1]] @ Vv[ys[0]], U[xs[1]] @ Vv[ys[1]]])
        return e @ sg
    best = -np.inf
    inits = list(extra_starts) + [rng.normal(size=k*d) for _ in range(starts)]
    for x0 in inits:
        r = minimize(lambda X: -T_of(X), x0, method="SLSQP",
                     constraints=[{"type": "ineq", "fun": lambda X: S_of(X) - S0}],
                     options={"maxiter": 300, "ftol": 1e-13})
        if S_of(r.x) >= S0 - 1e-9: best = max(best, T_of(r.x))
    return best

def run_adv(rng_seed):
    print("\n=== ADV: 적대적 최적화 (2-2-2, 변형 8개 x T 부호 2개, 다중 시작) ===")
    rng = np.random.default_rng(rng_seed + 2)
    worst = 0.0
    for s in (2.1, 2.3, 2.5, 2.7, 2.8):
        best = -np.inf
        for sg in SIG:   # u 전체 부호 반전이 (T, sigma) -> (-T, -sigma) 이므로 T 만 최대화하면 충분
            best = max(best, tmax_gram(2, ((0, 1), (0, 1), sg), s, rng, starts=6))
        om = 1 - best/4; ref = 1 - c_of_S(s)**3
        worst = max(worst, abs(om - ref))
        print(f"    Smax>={s}: Omega_min(수치)={om:.6f}  1-c^3={ref:.6f}  차이 {om-ref:+.2e}", flush=True)
    report("정리 4 타이트: 적대적 최적화가 1-c^3 와 소수 다섯째 자리까지 일치", worst < 5e-5, f"최대 차이 {worst:.1e}")

# ================= 정리 8, 명제 7 =================
def run_thm8(rng_seed):
    print("\n=== THM8: m=3, Omega=0 이면 S_glob <= 5/2 ===")
    rng = np.random.default_rng(rng_seed + 3)
    n = 400000
    be = rng.uniform(0, np.pi/2, n); ga = rng.uniform(0, np.pi/2, n)
    lo, hi = np.cos(be + ga), np.cos(be - ga); r = lo + (hi - lo)*rng.random(n)
    p, q = np.cos(ga), np.cos(be)
    E = np.zeros((n, 3, 3)); E[:, 0, 0] = 1; E[:, 0, 1] = -1; E[:, 1, 0] = -1; E[:, 1, 1] = 1
    E[:, 0, 2] = p; E[:, 1, 2] = -p; E[:, 2, 0] = q; E[:, 2, 1] = -q; E[:, 2, 2] = r
    S, Tm = sub_stats(E, 3)
    report("정리 8: 40만 개 무작위 배치에서 S_glob <= 5/2 (Omega=0 인 배치)", S.max() <= 2.5 + 1e-12 and (Tm >= 4 - 1e-12).all(),
           f"최댓값 {S.max():.4f}, 모든 배치 max|T|=4: {(Tm >= 4-1e-12).all()}")
    Ea = np.array([[[1, -1, .5], [-1, 1, -.5], [.5, -.5, -.5]]])
    Sa, Ta = sub_stats(Ea, 3)
    report("정리 8 달성: p=q=1/2, r=-1/2 에서 S_glob = 5/2 이고 Omega=0", abs(Sa[0] - 2.5) < 1e-12 and abs(Ta[0] - 4) < 1e-12, f"S_glob={Sa[0]}, max|T|={Ta[0]}")
    # 달성 배치가 실제 단위벡터로 실현되는지 (Gram 이 PSD 인지)
    e = np.array([1.0, 0, 0])
    ang = lambda th: np.array([np.cos(th), np.sin(th), 0])
    u2, v2 = ang(np.pi/3), ang(-np.pi/3)
    Em = np.array([[1, -1, e @ v2], [-1, 1, -(e @ v2)], [u2 @ e, -(u2 @ e), u2 @ v2]])
    report("정리 8 달성 배치: 단위벡터 실현 확인", np.abs(Em - Ea[0]).max() < 1e-12, f"최대 오차 {np.abs(Em-Ea[0]).max():.1e}")

def run_prop7():
    print("\n=== PROP7: m=4 ===")
    e1 = np.array([1.0, 0, 0]); pl = lambda th: np.array([0, np.cos(th), np.sin(th)])
    u = [e1, -e1, pl(0), pl(np.pi/2)]; v = [e1, -e1, pl(np.pi/4), pl(-np.pi/4)]
    E = np.array([[a @ b for b in v] for a in u])[None]
    S, Tm = sub_stats(E, 4)
    report("명제 7: S_glob = 2sqrt2 이고 {0,1}^2 에서 |T|=4 (Omega=0)", abs(S[0] - 2*SQ2) < 1e-12 and abs(Tm[0] - 4) < 1e-12,
           f"S_glob={S[0]:.12f}, max|T|={Tm[0]:.1f}")

# ================= 7절 표, 정리 9 =================
def all_sigma_m3():
    return [(xs, ys, sg) for xs, ys in subpairs(3) for sg in SIG]

SIG0 = ((0, 2), (0, 2), np.array([1, 1, 1, -1.0]))     # sigma_0: S = E00 + E02 + E20 - E22

def theorem9_witness(delta):
    lam = 16*delta/9; c = 3*lam**2/16; a = lam/2 + lam**2/(4*(lam + c))
    d0, d1, d2 = 1 + a, 1.0, lam/2 + c
    N = np.array([[1+lam, -1, lam], [-1, 1, 0], [lam, 0, -lam]])
    Dg = np.diag([d0, d1, d2]); Pm = 2*Dg - N; Pp = 2*Dg + N
    emin = min(np.linalg.eigvalsh(Pm).min(), np.linalg.eigvalsh(Pp).min())
    val = 4 + 0.0 - lam*delta + 9*lam**2*(4+lam)/(8*(16+3*lam))
    # tr D - lam*S0 = 2*(d0+d1+d2) - lam*(5/2+delta)
    val2 = 2*(d0 + d1 + d2) - lam*(2.5 + delta)
    return emin, val, val2

def run_sec7(rng_seed):
    print("\n=== SEC7: 7절 표 (72개 변형 전수) + 정리 9 ===")
    rng = np.random.default_rng(rng_seed + 4)
    sigmas = all_sigma_m3(); assert len(sigmas) == 72
    paper = {2.50: 4.0000, 2.52: 3.9996, 2.55: 3.9976, 2.60: 3.9894, 2.70: 3.9457, 2.80: 3.8053}
    t0 = time.time(); rows = {}
    for S0 in paper:
        vals = [tmax_gram(3, sg, S0, rng, starts=8) for sg in sigmas]
        rows[S0] = (max(vals), vals)
        print(f"    S0={S0:.2f}: max T = {rows[S0][0]:.4f}  Omega_min = {1-rows[S0][0]/4:.4f}  (논문 표 {paper[S0]:.4f})  [{time.time()-t0:.0f}s]", flush=True)
    ok = all(abs(rows[s][0] - paper[s]) < 2e-4 or rows[s][0] > paper[s] for s in paper)
    report("7절 표: max T 가 논문 표와 소수 넷째 자리까지 일치(또는 더 큼)", ok)
    # 지배 변형: delta = 0.07 에서 72개 변형의 T_max 분포
    S0 = 2.57
    vals = np.array([tmax_gram(3, sg, S0, rng, starts=8) for sg in sigmas])
    eta = 4 - vals; order = np.argsort(eta)
    eta0 = 4 - tmax_gram(3, SIG0, S0, rng, starts=12)
    others = [eta[i] for i in range(72) if abs(eta[i] - eta0) > 1e-4]
    print(f"    delta=0.07: sigma_0 의 eta = {eta0:.5f}; 서로 다른 eta 값(상위 5개 오름차순): "
          f"{np.round(sorted(set(np.round(eta, 5)))[:5], 5)}")
    report("A.6.1: sigma_0 와 다른 값을 주는 변형은 eta >= 1/72 (delta=0.07)", (min(others) if others else 1) >= 1/72 - 1e-9,
           f"최소 {min(others):.5f} vs 1/72={1/72:.5f}")
    # 정리 9: 하한 <= 수치 <= 상한, 쌍대 증인
    allok = True; wit_ok = True
    for delta in (0.005, 0.01, 0.02, 0.03, 0.05, 0.07):
        S0 = 2.5 + delta
        eta_num = 4 - tmax_gram(3, SIG0, S0, rng, starts=12)
        lower = (8/9)*delta**2*(1 - delta/9); upper = (8/9)*delta**2*(1 + delta)**2
        emin, v1, v2 = theorem9_witness(delta)
        wit_ok &= (emin > -1e-12) and abs(v1 - v2) < 1e-12
        good = lower - 1e-9 <= eta_num <= upper + 1e-9
        allok &= good
        print(f"    delta={delta:.3f}: 하한 {lower:.6f} <= eta(수치) {eta_num:.6f} <= 상한 {upper:.6f}  {'OK' if good else 'FAIL'}  | 증인 최소고유값 {emin:+.1e}, trD-lam*S0={v1:.6f}", flush=True)
    report("정리 9: 하한 <= SLSQP 값 <= 상한 (delta=0.005..0.07)", allok)
    report("정리 9: 쌍대 증인 P-, P+ 양반정부호, tr D - lam S0 식 일치", wit_ok)
    # 상계의 명시적 배치 (A.6.3)
    ub_ok = True
    for delta in (0.005, 0.02, 0.05, 0.07):
        t0_ = delta/np.sqrt(3)*(1 + delta)
        ang_u = [-t0_, np.pi + t0_/3, np.pi/3]; ang_v = [t0_, np.pi - t0_/3, -np.pi/3]
        vec = lambda th: np.array([np.cos(th), np.sin(th)])
        Em = np.array([[vec(a) @ vec(b) for b in ang_v] for a in ang_u])[None]
        Sg, Tm = sub_stats(Em, 3)
        eta_cfg = 4 - Tm[0]
        ub_ok &= (Sg[0] >= 2.5 + delta - 1e-12) and (eta_cfg <= (8/9)*delta**2*(1+delta)**2 + 1e-12)
    report("A.6.3 명시적 배치: S_glob >= 5/2+delta 이고 eta <= (8/9)delta^2(1+delta)^2", ub_ok)

def main():
    a = sys.argv[1:]; seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 12345
    modes = [x for x in a if x.startswith("--") and x != "--seed"] or ["--exact", "--curve", "--perturb", "--adv", "--thm8", "--prop7", "--sec7"]
    rng = np.random.default_rng(seed)
    if "--exact" in modes: run_exact(rng)
    if "--curve" in modes: run_curve(seed)
    if "--perturb" in modes: run_perturb(seed)
    if "--adv" in modes: run_adv(seed)
    if "--thm8" in modes: run_thm8(seed)
    if "--prop7" in modes: run_prop7()
    if "--sec7" in modes: run_sec7(seed)
    print("\n=== 요약 ===", "PASS" if all(OK_ALL) else "FAIL", f"({sum(OK_ALL)}/{len(OK_ALL)})")

if __name__ == "__main__":
    main()
