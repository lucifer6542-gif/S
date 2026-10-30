#!/usr/bin/env python3
"""
논문 B 통합 검증 스크립트 (3자 이진 NS 상자와 결정론적 피드백, Ω=0)
자체 완결형: 다른 verify_*.py 에 의존하지 않는다. 필요: numpy, scipy (cvxpy/cdd 불필요).

사용:
  python3 verify_B_all.py                 # 전부 실행 (--dims --selftest --research --gynin)
  python3 verify_B_all.py --dims          # 차원: NS 상자 다면체(26) vs process 다면체
  python3 verify_B_all.py --selftest      # 순수 패리티 16개, 기대값 대비 PASS/FAIL
  python3 verify_B_all.py --prop2         # 명제 2 (4096개 g, LP) 재계산
  python3 verify_B_all.py --research      # Ω=0 꼭짓점 전수 열거(이중 기술법) + 분류 + 궤도 + NPA
  python3 verify_B_all.py --gynin         # GYNIN: 항등식, NS 한계, process function 전수 8*p 최댓값
  python3 verify_B_all.py --boxes f.json  # 36개 상자 JSON ({"boxes":[{"P_o_given_i...":8x8}]}) 검사
  옵션: --out research_boxes.json (열거한 꼭짓점 궤도 대표 저장)

NPA: 단항식 19개(1, A_x,B_y,C_z, 쌍곱 12개) 모멘트 행렬의 실수 대칭 PSD 가능성을
교대 사영(Dykstra)으로 판정한다. 부동소수점 결과이다. 정확한(유리수) 검증은 verify_npa_rational.py 가 한다.
"""
import sys, json, itertools, time, collections
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9
O = list(itertools.product([0, 1], repeat=3))
idx = lambda t: 4 * t[0] + 2 * t[1] + t[2]

# ---------------- 피드백 g, 전략 f ----------------
tabs = list(itertools.product([0, 1], repeat=4))
allg = [(a, b, c) for a in tabs for b in tabs for c in tabs]
GI = np.zeros((4096, 8), dtype=int)            # GI[g,o] = 입력 인덱스 g(o)
for n, (a, b, c) in enumerate(allg):
    for o in O:
        GI[n, idx(o)] = idx((a[2*o[1]+o[2]], b[2*o[0]+o[2]], c[2*o[0]+o[1]]))
GKEY = {GI[n].tobytes(): n for n in range(4096)}
ROWS = np.arange(8)[None, :]
W_all = lambda P: P[ROWS, GI].sum(1)

fs = list(itertools.product([0, 1], repeat=2))
strat = list(itertools.product(fs, fs, fs))
ROWF = np.zeros((64, 8), dtype=int)            # ROWF[f,i] = 출력 인덱스 f(i)
for n, (fa, fb, fc) in enumerate(strat):
    for i in O:
        ROWF[n, idx(i)] = idx((fa[i[0]], fb[i[1]], fc[i[2]]))
A_LOC = np.zeros((64, 64))
for n in range(64):
    for i in range(8):
        A_LOC[ROWF[n, i]*8 + i, n] = 1          # 열 n = 결정론 상자 D_f 를 평탄화한 것
FP = (ROWF[:, GI] == np.arange(8)).sum(-1)      # FP[f,g] = f∘g 의 고정점 수
PF = [g for g in range(4096) if (FP[:, g] == 1).all()]   # process function (정확히 1개)

def local_residual(P):
    c = np.concatenate([np.zeros(64), np.ones(128)])
    Aeq = np.hstack([A_LOC, np.eye(64), -np.eye(64)])
    return linprog(c, A_eq=Aeq, b_eq=P.flatten(), bounds=(0, None), method="highs").fun

# ---------------- NS 조건 ----------------
def ns_matrix():
    rows = []
    for K in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
        for ok in itertools.product([0, 1], repeat=len(K)):
            ref = {}
            def vec(ii):
                v = np.zeros(64)
                for o in O:
                    if tuple(o[j] for j in K) == ok:
                        v[idx(o)*8 + idx(ii)] = 1
                return v
            for i in O:
                kk = tuple(i[j] for j in K)
                if kk not in ref: ref[kk] = vec(i)
                else: rows.append(vec(i) - ref[kk])
    n_ns = len(rows)
    for i in O:
        v = np.zeros(64)
        for o in O: v[idx(o)*8 + idx(i)] = 1
        rows.append(v)
    return np.array(rows), n_ns
EQ, N_NS = ns_matrix()
BEQ = np.zeros(EQ.shape[0]); BEQ[N_NS:] = 1
EQ_RANK = np.linalg.matrix_rank(EQ)
ns_violation = lambda P: np.abs(EQ[:N_NS] @ P.flatten()).max()

# ---------------- 이변 주변, CHSH ----------------
def marginal(P, X, Y):
    Z = ({0, 1, 2} - {X, Y}).pop()
    M = np.zeros((4, 4))
    for o in O:
        for ix in (0, 1):
            for iy in (0, 1):
                i = [0, 0, 0]; i[X] = ix; i[Y] = iy; i[Z] = 0
                M[2*o[X]+o[Y], 2*ix+iy] += P[idx(o), idx(tuple(i))]
    return M
LOC2 = []
for fx in itertools.product([0, 1], repeat=2):
    for fy in itertools.product([0, 1], repeat=2):
        D = np.zeros((4, 4))
        for ix in (0, 1):
            for iy in (0, 1): D[2*fx[ix]+fy[iy], 2*ix+iy] = 1
        LOC2.append(D.flatten())
LOC2 = np.array(LOC2).T
local2 = lambda M: linprog(np.zeros(16), A_eq=LOC2, b_eq=M.flatten(), bounds=(0, None), method="highs").status == 0
def chsh_max(M):
    E = np.zeros((2, 2))
    for ix in (0, 1):
        for iy in (0, 1):
            for ox in (0, 1):
                for oy in (0, 1): E[ix, iy] += (-1)**(ox+oy) * M[2*ox+oy, 2*ix+iy]
    S = E.sum(); return max(abs(S - 2*E[x, y]) for x in (0, 1) for y in (0, 1))

# ---------------- 위수 384 군 ----------------
def group_maps():
    maps = []
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([0, 1], repeat=3):
            for t in itertools.product([0, 1], repeat=3):
                rm = np.zeros(8, dtype=int); cm = np.zeros(8, dtype=int)
                for o in O: rm[idx(o)] = idx(tuple(o[perm[k]] ^ t[k] for k in range(3)))
                for i in O: cm[idx(i)] = idx(tuple(i[perm[k]] ^ s[k] for k in range(3)))
                maps.append((rm, cm))
    return maps
GROUP = group_maps()
def act(P, rm, cm):
    Q = np.zeros_like(P); Q[np.ix_(rm, cm)] = P; return Q
key = lambda P: tuple(np.round(P * 1e6).astype(np.int64).flatten())
def act_g(g, rm, cm):
    """W(act(P);act_g(g)) = W(P;g): g'(rm[o]) = cm[g(o)]"""
    row = np.zeros(8, dtype=int); row[rm] = cm[GI[g]]
    return GKEY[row.tobytes()]

# ---------------- 공통 속성 판정 ----------------
def analyze(P):
    r = {}
    Wv = W_all(P); G0 = np.where(np.abs(Wv) < TOL)[0]
    r["ok"] = bool(np.abs(P.sum(0)-1).max() < TOL and P.min() > -TOL and ns_violation(P) < TOL)
    r["omega0"] = bool(len(G0) > 0 and Wv.min() > -TOL); r["nG0"] = len(G0)
    r["resid"] = float(local_residual(P))
    r["supp_nl"] = not any(all(P[ROWF[f, i], i] > TOL for i in range(8)) for f in range(64))
    r["prof"] = bool(len(G0) > 0 and not any((FP[f, G0] == 0).all() for f in range(64)))
    ms = [marginal(P, X, Y) for X, Y in [(0, 1), (0, 2), (1, 2)]]
    r["bip_local"] = all(local2(M) for M in ms); r["bip_chsh"] = float(max(chsh_max(M) for M in ms))
    act_ = [j for j in range(64) if P.flatten()[j] < TOL]
    Z = np.zeros((len(act_), 64)); Z[np.arange(len(act_)), act_] = 1
    r["vertex"] = bool(np.linalg.matrix_rank(np.vstack([EQ, Z])) == 64)
    return r

# ================= NPA (19 단항식) =================
def _words():
    mons = [()]                                  # 단항식: 문자 (party, input) 튜플
    for p in range(3):
        for x in (0, 1): mons.append(((p, x),))
    for p, q in [(0, 1), (0, 2), (1, 2)]:
        for x in (0, 1):
            for y in (0, 1): mons.append(((p, x), (q, y)))
    return mons
MONS = _words()
def _canon(word):
    per = [[], [], []]
    for p, x in word:
        if per[p] and per[p][-1] == x: per[p].pop()        # A_x A_x = 1
        else: per[p].append(x)
    t = tuple(tuple(s) for s in per); r = tuple(tuple(reversed(s)) for s in per)
    return min(t, r)
def _build_npa():
    n = len(MONS); table = {}; pos = collections.defaultdict(list)
    for a in range(n):
        for b in range(n):
            w = _canon(tuple(reversed(MONS[a])) + MONS[b])
            pos[w].append((a, b))
    return pos
NPA_POS = _build_npa()
def _correlators(P):
    """P(o|i) -> dict canon word -> 기댓값 (1,2,3자 상관만)"""
    vals = {}
    for w in NPA_POS:
        parties = [p for p in range(3) if len(w[p]) > 0]
        if any(len(w[p]) > 1 for p in range(3)): continue   # 같은 당사자 곱 포함: 자유변수
        if not parties: vals[w] = 1.0; continue
        ins = [0, 0, 0]
        for p in parties: ins[p] = w[p][0]
        vals[w] = float(sum((-1)**sum(o[p] for p in parties) * P[idx(o), idx(tuple(ins))] for o in O))
    return vals
def npa_distance(P, iters=6000, tol=1e-10):
    """실수 대칭 모멘트 행렬의 PSD + 고정 항목 제약 사이 거리 (0이면 가능, 양수이면 허용 불가)"""
    n = len(MONS); fixed = _correlators(P); words = list(NPA_POS)
    def proj_aff(G):
        H = G.copy()
        for w in words:
            ps = NPA_POS[w]
            v = fixed[w] if w in fixed else np.mean([G[a, b] for a, b in ps])
            for a, b in ps: H[a, b] = v
        return H
    def proj_psd(G):
        e, U = np.linalg.eigh((G+G.T)/2); return (U*np.maximum(e, 0)) @ U.T
    X = np.zeros((n, n)); Pp = np.zeros((n, n)); Q = np.zeros((n, n)); dist = None
    for k in range(iters):                       # Dykstra
        Y = proj_psd(X + Pp); Pp = X + Pp - Y
        Xn = proj_aff(Y + Q); Q = Y + Q - Xn
        d = np.linalg.norm(Xn - Y); X = Xn
        if d < tol: break
        dist = d
    return float(np.linalg.norm(proj_aff(X) - proj_psd(proj_aff(X)))) if False else float(np.linalg.norm(X - proj_psd(X)))

def random_quantum_box(rng):
    """양성 대조군: 무작위 3큐빗 상태와 ±1 관측량으로 만든 양자 상자."""
    psi = rng.normal(size=8) + 1j*rng.normal(size=8); psi /= np.linalg.norm(psi)
    def obs():
        M = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)); H = M + M.conj().T
        w, U = np.linalg.eigh(H); return [U[:, k:k+1] @ U[:, k:k+1].conj().T for k in (0, 1)]
    meas = [[obs() for _ in range(2)] for _ in range(3)]
    P = np.zeros((8, 8))
    for i in O:
        for o in O:
            op = np.kron(np.kron(meas[0][i[0]][o[0]], meas[1][i[1]][o[1]]), meas[2][i[2]][o[2]])
            P[idx(o), idx(i)] = float(np.real(psi.conj() @ op @ psi))
    return P

# ================= 면 위 꼭짓점 전수 열거 (이중 기술법) =================
def _popcount(a): return np.bitwise_count(a)
def enumerate_face_vertices(zero_coords, verbose=False, tlimit=600):
    """F = {x: EQ x = BEQ, x>=0, x[zero_coords]=0} 의 꼭짓점 전수 (이중 기술법, 경계 있는 다면체)."""
    zc = list(zero_coords)
    E2 = np.vstack([EQ] + [np.eye(64)[[j]] for j in zc]); b2 = np.concatenate([BEQ, np.zeros(len(zc))])
    free = [j for j in range(64) if j not in zc]
    # 암묵적 등식: 면 위에서 항상 0인 좌표 제거
    implicit = []
    for j in free:
        c = np.zeros(64); c[j] = -1
        r = linprog(c, A_eq=E2, b_eq=b2, bounds=(0, None), method="highs")
        if r.status == 0 and -r.fun < 1e-9: implicit.append(j)
    zc2 = zc + implicit
    E3 = np.vstack([EQ] + [np.eye(64)[[j]] for j in zc2]); b3 = np.concatenate([BEQ, np.zeros(len(zc2))])
    xc = np.linalg.lstsq(E3, b3, rcond=None)[0]
    u, s, vt = np.linalg.svd(E3); r_ = int((s > 1e-9).sum()); N = vt[r_:].T   # 64 x d
    d = N.shape[1]; real = [j for j in range(64) if j not in zc2]
    if d == 0: return [xc], d
    m = len(real); nart = d + 1; nb = m + nart
    assert nb <= 128
    R = N[real, :]; off = xc[real]
    # 인공 단체: y_i >= -3 (i<d), sum y <= 20  (면 전체를 포함)
    Y = []; 
    for k in range(d + 1):
        y = np.full(d, -3.0)
        if k < d: y[k] = 20 + 3*(d-1)
        Y.append(y)
    Y = np.array(Y)
    def slack_art(Yv): return np.hstack([Yv + 3, (20 - Yv.sum(1))[:, None]])
    S = np.hstack([slack_art(Y), off + Y @ R.T])        # 열: 인공(d+1) | 실제(m)
    def mask(Sv, upto):
        z = np.abs(Sv[:, :upto]) < 1e-9
        lo = np.zeros(len(Sv), dtype=np.uint64); hi = np.zeros(len(Sv), dtype=np.uint64)
        for c in range(upto):
            bit = np.uint64(1) << np.uint64(c % 64)
            if c < 64: lo |= np.where(z[:, c], bit, np.uint64(0))
            else: hi |= np.where(z[:, c], bit, np.uint64(0))
        return lo, hi
    lo, hi = mask(S, nart)
    t0 = time.time()
    remaining = list(range(m))
    step = 0
    while remaining:
        # 동적 순서: 현재 꼭짓점에서 pos*neg 가 가장 작은 제약을 먼저 처리
        best = None
        for jj_ in remaining:
            c_ = S[:, nart + jj_]; npos = int((c_ > 1e-9).sum()); nneg = int((c_ < -1e-9).sum())
            sc = (0 if nneg == 0 else 1, npos * nneg, nneg)
            if best is None or sc < best[0]: best = (sc, jj_)
        jj = best[1]; remaining.remove(jj); step += 1
        col = nart + jj
        sj = S[:, col]
        pos = np.where(sj > 1e-9)[0]; neg = np.where(sj < -1e-9)[0]; zer = np.where(np.abs(sj) <= 1e-9)[0]
        newS = []; newlo = []; newhi = []
        if len(neg):
            for p in pos:
                cl = lo[p] & lo[neg]; ch = hi[p] & hi[neg]
                pc = _popcount(cl) + _popcount(ch)
                sel = pc >= d - 1
                cand = neg[sel]; cl = cl[sel]; ch = ch[sel]
                for s0 in range(0, len(cand), 256):
                    c_ = cand[s0:s0+256]; a_ = cl[s0:s0+256]; b_ = ch[s0:s0+256]
                    sup = ((lo[None, :] & a_[:, None]) == a_[:, None]) & ((hi[None, :] & b_[:, None]) == b_[:, None])
                    okc = sup.sum(1) <= 2
                    for n_, aa, bb in zip(c_[okc], a_[okc], b_[okc]):
                        sp, sn = sj[p], sj[n_]
                        v = (sp * S[n_] - sn * S[p]) / (sp - sn); v[col] = 0.0
                        newS.append(v); newlo.append(aa); newhi.append(bb)
        # 갱신
        keep = np.concatenate([pos, zer]); S = S[keep]; lo = lo[keep]; hi = hi[keep]
        zmask = np.concatenate([np.zeros(len(pos), bool), np.ones(len(zer), bool)])
        bit = np.uint64(1) << np.uint64(col % 64)
        if col < 64: lo = np.where(zmask, lo | bit, lo)
        else: hi = np.where(zmask, hi | bit, hi)
        if newS:
            S = np.vstack([S, np.array(newS)])
            nl = np.array(newlo, dtype=np.uint64); nh = np.array(newhi, dtype=np.uint64)
            if col < 64: nl = nl | bit
            else: nh = nh | bit
            lo = np.concatenate([lo, nl]); hi = np.concatenate([hi, nh])
        if verbose and step % 3 == 0: print(f"    step {step}/{m} V={len(S)} t={time.time()-t0:.1f}s", flush=True)
        if time.time() - t0 > tlimit: raise TimeoutError(f"DD time limit (V={len(S)} at step {step}/{m})")
    verts = []
    for sv in S:
        x = np.zeros(64); x[real] = sv[nart:]; verts.append(x)
    return verts, d


# ================= 실행 단위 =================
def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return bool(ok)

def run_dims():
    print("\n=== DIMS: 차원 구분 (NS 상자 다면체 vs process 다면체) ===")
    d_ns = 64 - EQ_RANK
    # process 다면체: 744개 결정론 process function 을 δ_{i,g(o)} (64-벡터)로 보고 아핀 포락 차원
    vecs = []
    for g in PF:
        v = np.zeros(64)
        for o in range(8): v[o*8 + GI[g, o]] = 1
        vecs.append(v)
    V0 = np.array(vecs); d_proc = np.linalg.matrix_rank(V0[1:] - V0[0])
    print(f"  3-2-2 NS 상자 다면체(P(o|i), 64좌표)의 아핀 차원 = {d_ns}")
    print(f"  결정론적 process function {len(PF)}개의 볼록껍질(process 다면체) 아핀 차원 = {d_proc}")
    check("NS 상자 다면체 차원 26", d_ns == 26)
    check("process 다면체 차원 37 ([BW2])", d_proc == 37, f"계산값 {d_proc}")
    print("  (37차원은 process 다면체의 차원, NS 상자 다면체는 26차원)")

def run_selftest():
    print("\n=== SELFTEST: 순수 패리티 상자 16개 ===")
    boxes = []
    for c0, cx, cy, cz in itertools.product([0, 1], repeat=4):
        P = np.zeros((8, 8))
        for i in O:
            x, y, z = i
            f = (x*y) ^ (x*z) ^ (y*z) ^ c0 ^ (cx*x) ^ (cy*y) ^ (cz*z)
            for o in O:
                if (o[0] ^ o[1] ^ o[2]) == f: P[idx(o), idx(i)] = 0.25
        boxes.append(((c0, cx, cy, cz), P))
    R = [analyze(P) for _, P in boxes]
    ok = True
    ok &= check("전부 정규화·비음수·NS", all(r["ok"] for r in R))
    ok &= check("전부 NS 꼭짓점", all(r["vertex"] for r in R))
    ok &= check("전부 지지집합 비국소", all(r["supp_nl"] for r in R))
    ok &= check("국소 L1 잔차 = 8/3", all(abs(r["resid"] - 8/3) < 1e-7 for r in R))
    n0 = sum(r["omega0"] for r in R)
    ok &= check("Ω=0 인 상자 8/16", n0 == 8, f"{n0}/16")
    ok &= check("Ω=0 인 8개의 |G0| = 6", all(r["nG0"] == 6 for r in R if r["omega0"]))
    ok &= check("Ω=0 인 8개 모두 프로파일 내재적", all(r["prof"] for r in R if r["omega0"]))
    ok &= check("Ω>0 인 8개는 프로파일 내재적 아님", not any(r["prof"] for r in R if not r["omega0"]))
    lin_odd = [(c0, cx, cy, cz) for (c0, cx, cy, cz), _ in boxes if (cx + cy + cz) % 2 == 1]
    z0 = [t for (t, _), r in zip(boxes, R) if r["omega0"]]
    ok &= check("Ω=0 ⇔ 선형 계수 합이 홀수 (A.4)", sorted(z0) == sorted(lin_odd))
    ok &= check("이변 주변 모두 국소(CHSH 0)", all(r["bip_local"] and r["bip_chsh"] < 1e-9 for r in R))
    okeys = [min({key(act(P, rm, cm)) for rm, cm in GROUP}) for (_, P), r in zip(boxes, R) if r["omega0"]]
    ok &= check("Ω=0 인 8개가 군 아래 궤도 하나(크기 8)", len(set(okeys)) == 1 and
                len({key(act(boxes[[t for t, _ in boxes].index(z0[0])][1], rm, cm)) for rm, cm in GROUP}) == 8)
    ok &= check("대조군: process function 744개", len(PF) == 744)
    pf_dev = max(float(np.abs(W_all(P)[PF] - 1).max()) for _, P in boxes)
    ok &= check("process function 위 max|W-1| = 0", pf_dev < 1e-12, f"{pf_dev}")
    rng = np.random.default_rng(0)
    nloc = sum(local_residual((A_LOC @ rng.dirichlet(np.ones(64))).reshape(8, 8)) < 1e-7 for _ in range(20))
    ok &= check("무작위 국소 상자 20/20 국소 인식", nloc == 20)
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return ok

def run_prop2():
    print("\n=== PROP2: 4096개 g, NS 다면체 위 LP ===")
    C = np.zeros((4096, 64))
    for g in range(4096):
        for o in range(8): C[g, o*8 + GI[g, o]] = 1
    u, s, vt = np.linalg.svd(EQ); r = (s > 1e-9).sum(); Nn = vt[r:].T
    proj = C @ Nn; trivial = np.abs(proj).max(1) < 1e-9
    nz = [g for g in range(4096) if not trivial[g]]
    keys = {tuple(np.round(proj[g], 6)) for g in nz}
    mins, maxs = [], []
    for g in nz:
        mins.append(linprog(C[g], A_eq=EQ, b_eq=BEQ, bounds=(0, None), method="highs").fun)
        maxs.append(-linprog(-C[g], A_eq=EQ, b_eq=BEQ, bounds=(0, None), method="highs").fun)
    ok = check("W 상수인 g = process function = 744", trivial.sum() == 744 and set(np.where(trivial)[0]) == set(PF))
    ok &= check("나머지 3352개 NS 최솟값 모두 0", len(nz) == 3352 and np.allclose(mins, 0))
    cm = collections.Counter(np.round(maxs, 5))
    ok &= check("최댓값 분포 2:2992, 3:352, 4:8", cm == collections.Counter({2.0: 2992, 3.0: 352, 4.0: 8}), str(dict(cm)))
    ok &= check("서로 다른 w_g 346(+영 1 = 347)", len(keys) == 346)
    # 인과 / 비인과
    def gk(g, k, o):
        t = allg[g][k]; oth = [j for j in range(3) if j != k]
        return t[2*o[oth[0]] + o[oth[1]]]
    def causal(g):
        for k in range(3):
            if len(set(allg[g][k])) != 1: continue
            good = True
            for v in (0, 1):
                found = False
                for j in range(3):
                    if j == k: continue
                    l = 3 - j - k; vals = set()
                    for ol in (0, 1):
                        o = [0, 0, 0]; o[k] = v; o[l] = ol; vals.add(gk(g, j, tuple(o)))
                    if len(vals) == 1: found = True
                if not found: good = False
            if good: return True
        return False
    nc = sum(causal(g) for g in PF)
    ok &= check("process function 중 인과 680 / 비인과 64", nc == 680 and len(PF) - nc == 64, f"{nc}/{len(PF)-nc}")
    ok &= check("인과 g 는 모두 process function", sum(causal(g) for g in range(4096) if g not in set(PF)) == 0)
    return ok

def run_gynin():
    print("\n=== GYNIN: 항등식, NS 한계, process function 전수 8*p_gynin ===")
    def gidx(fn): return GKEY[np.array([idx(fn(o)) for o in O]).tobytes()]
    g1 = gidx(lambda o: (o[1], o[2], o[0]))
    g2 = gidx(lambda o: (1-o[1], 1-o[2], 1-o[0]))
    rng = np.random.default_rng(1); P = rng.dirichlet(np.ones(8), size=8).T
    p = 0.0
    for x in O:
        for a in O:
            w = (x[0] == a[2] and x[1] == a[0] and x[2] == a[1]) or \
                (x[0] == 1-a[2] and x[1] == 1-a[0] and x[2] == 1-a[1])
            p += P[idx(x), idx(a)] * w
    p /= 8
    lhs = (W_all(P)[g1] + W_all(P)[g2]) / 8
    ok = check("항등식 p_gynin = (W(g1)+W(g2))/8 (무작위 조건부분포)", abs(p - lhs) < 1e-12, f"{p:.6f} vs {lhs:.6f}")
    C = lambda g: np.bincount([o*8 + GI[g, o] for o in range(8)], minlength=64).astype(float)
    mx = lambda c: -linprog(-c, A_eq=EQ, b_eq=BEQ, bounds=(0, None), method="highs").fun
    w1, w2, w12 = mx(C(g1)), mx(C(g2)), mx(C(g1) + C(g2))
    ok &= check("NS 위 max W(g1)=max W(g2)=2, max(W1+W2)=2 → p_gynin ≤ 1/4", abs(w1-2)+abs(w2-2)+abs(w12-2) < 1e-7,
                f"{w1:.4f},{w2:.4f},{w12:.4f}")
    # process function 전수: 두 설정 a_k∈{0,1} 에 지역 전략 2개를 배정(4^3=... 각 당사자 4x4=16가지)
    fp_idx = np.zeros((64, 744), dtype=int)       # 고정점 (process function 이므로 유일)
    for c, g in enumerate(PF):
        fixed = (ROWF[:, GI[g]] == np.arange(8))
        assert (fixed.sum(1) == 1).all(); fp_idx[:, c] = fixed.argmax(1)
    tgt = np.array([idx((a[2], a[0], a[1])) for a in O]); tgtc = np.array([idx((1-a[2], 1-a[0], 1-a[1])) for a in O])
    best = np.zeros(744, dtype=int)
    pairs = list(itertools.product(range(4), repeat=2))             # 당사자 하나의 (a=0 전략, a=1 전략)
    for PA in pairs:
        for PB in pairs:
            for PC in pairs:
                sc = np.zeros(744, dtype=int)
                for ai, a in enumerate(O):
                    f = (PA[a[0]]*4 + PB[a[1]])*4 + PC[a[2]]
                    ox = fp_idx[f]; sc += ((ox == tgt[ai]) | (ox == tgtc[ai]))
                best = np.maximum(best, sc)
    mxv = best.max(); cnt_any = int((best == mxv).sum())
    ok &= check("[규약 무관] 지역 전략을 임의로 배정해도 process function 전수 8*p_gynin 최댓값 = 5 (5/8)", mxv == 5, f"최댓값 {mxv}")
    # 표준 규약: 설정 a_k 가 항등(a_k=0)/반전(a_k=1) 중 하나를 고름, 즉 출력 o_k = i_k XOR a_k (또는 두 순서)
    ID, NOT = 1, 2                                  # fs 인덱스: (f(0),f(1)) = (0,1) 항등, (1,0) 반전
    best_std = None
    for pr in [(ID, NOT), (NOT, ID)]:
        sc = np.zeros(744, dtype=int)
        for ai, a in enumerate(O):
            f = (pr[a[0]]*4 + pr[a[1]])*4 + pr[a[2]]
            ox = fp_idx[f]; sc += ((ox == tgt[ai]) | (ox == tgtc[ai]))
        best_std = sc if best_std is None else np.maximum(best_std, sc)
        cnt_pr = int((sc == 5).sum()); print(f"  규약 o_k=i_k XOR a_k{' (a 반전 순서)' if pr==(NOT,ID) else ''}: 최댓값 {sc.max()}, 개수 {cnt_pr}")
    cnt = int((best_std == 5).sum())
    ok &= check("[항등/반전 규약] 8*p_gynin 최댓값 5 를 주는 process function 수 = 16",
                best_std.max() == 5 and cnt == 16, f"계산값 {cnt}")
    print(f"  (참고) 지역 전략을 설정마다 임의 배정까지 허용하면 최댓값 5 를 주는 g 는 합집합 {cnt_any}개. "
          "'16개'는 항등/반전 규약에서의 값이다.")
    # AF/BW: I_A = (1-O_B) O_C, I_B = O_A (1-O_C), I_C = (1-O_A) O_B
    gaf = gidx(lambda o: ((1-o[1]) & o[2], o[0] & (1-o[2]), (1-o[0]) & o[1]))
    in_pf = gaf in set(PF); ci = PF.index(gaf) if in_pf else -1
    check("식 (13) AF/BW 가 process function 목록에 포함", in_pf)
    if in_pf: check("식 (13) AF/BW 의 8*p_gynin = 5 (항등/반전 규약)", best_std[ci] == 5, f"{best_std[ci]}")
    return ok

# ---------------- Ω=0 꼭짓점 전수 열거 ----------------
def faces_and_orbits():
    C = np.zeros((4096, 64))
    for g in range(4096):
        for o in range(8): C[g, o*8 + GI[g, o]] = 1
    u, s, vt = np.linalg.svd(EQ); r = (s > 1e-9).sum(); Nn = vt[r:].T
    proj = C @ Nn; pfs = set(PF)
    nz = [g for g in range(4096) if g not in pfs]
    keys, reps, fid = {}, [], {}
    for g in nz:
        k = tuple(np.round(proj[g], 6))
        if k not in keys: keys[k] = len(reps); reps.append(g)
        fid[g] = keys[k]
    seen, orb = set(), []
    for f0 in range(len(reps)):
        if f0 in seen: continue
        mem = {f0}
        for rm, cm in GROUP: mem.add(fid[act_g(reps[f0], rm, cm)])
        seen |= mem; orb.append((f0, sorted(mem)))
    return reps, orb

def run_research(out_json=None, tlimit=7200, cache=None, cache_file="face_cache.pkl"):
    print("\n=== RESEARCH: Ω=0 NS 꼭짓점 전수 열거 (이중 기술법; 면 F_g={W_g=0} 별) ===")
    reps, orb = faces_and_orbits()
    print(f"  서로 다른 W_g 함수 {len(reps)}개, 군 아래 면 궤도 {len(orb)}개 (크기 {[len(m) for _, m in orb]})")
    import os, pickle
    cache = dict(cache or {})
    if cache_file and os.path.exists(cache_file):
        try: cache.update(pickle.load(open(cache_file, "rb"))); print(f"  캐시 {cache_file} 에서 면 {sorted(cache)} 로드")
        except Exception: pass
    found = {}; face_info = []
    for f0, mem in orb:
        g = reps[f0]; zc = [o*8 + GI[g, o] for o in range(8)]
        if f0 in cache: vs, d = cache[f0], None
        else:
            t = time.time(); vs, d = enumerate_face_vertices(zc, tlimit=tlimit)
            print(f"  면 대표 g={g}: 차원 {d}, 꼭짓점 {len(vs)}개 ({time.time()-t:.1f}s)", flush=True)
            cache[f0] = vs
            if cache_file: pickle.dump(cache, open(cache_file, "wb"))
        face_info.append((g, zc, vs))
        for x in vs:
            P = np.round(x.reshape(8, 8) * 8) / 8
            P = np.where(np.abs(x.reshape(8, 8) - P) < 1e-7, P, x.reshape(8, 8))
            found[key(P)] = P
    print(f"  면 대표들에서 얻은 꼭짓점(중복 제거) {len(found)}개; 군(384)으로 확장 중 ...")
    allb = {}
    for P in found.values():
        for rm, cm in GROUP:
            Q = act(P, rm, cm); allb.setdefault(key(Q), Q)
    Ps = list(allb.values())
    print(f"  Ω=0 꼭짓점 전체(군 확장) = {len(Ps)}개")
    # 궤도 분해
    orbs = {}
    for P in Ps:
        k0 = min(key(act(P, rm, cm)) for rm, cm in GROUP); orbs.setdefault(k0, []).append(P)
    reps_b = [v[0] for v in orbs.values()]
    print(f"  궤도 {len(orbs)}개, 크기 {sorted(len(v) for v in orbs.values())}")
    rows = []
    for P in reps_b:
        r = analyze(P); r["size"] = len(orbs[min(key(act(P, rm, cm)) for rm, cm in GROUP)])
        rows.append((r, P))
    for r, P in rows:
        r["local"] = r["resid"] < 1e-7
        r["npa"] = None if r["local"] else npa_distance(P, iters=4000)
    ok = check("모든 열거 상자가 NS 꼭짓점이고 Ω=0", all(r["vertex"] and r["omega0"] and r["ok"] for r, _ in rows))
    # 면별 완전성 점검: 각 면에서 무작위 목적함수 LP 해(꼭짓점)가 모두 열거 집합에 있는가
    rng0 = np.random.default_rng(11); pmiss = 0; ptot = 0
    for g, zc, vs in face_info:
        lb = np.zeros(64); ub = np.full(64, None); bnds = [(0, 0) if j in zc else (0, None) for j in range(64)]
        have = {key(np.round(x.reshape(8, 8)*8)/8) for x in vs}
        for _ in range(60):
            res = linprog(rng0.normal(size=64), A_eq=EQ, b_eq=BEQ, bounds=bnds, method="highs-ds")
            ptot += 1
            if key(np.round(res.x.reshape(8, 8)*8)/8) not in have: pmiss += 1
    ok0 = check("면별 무작위 LP 꼭짓점이 모두 그 면의 열거 집합에 있음", pmiss == 0, f"{ptot}회 중 누락 {pmiss}")
    ok &= ok0
    # 완전성 점검: 무작위 LP 목적함수로 찾은 꼭짓점 중 Ω=0 인 것이 전부 열거 집합에 있는가
    rng = np.random.default_rng(7); miss = 0; hit = 0
    for _ in range(300):
        c = rng.normal(size=64); res = linprog(c, A_eq=EQ, b_eq=BEQ, bounds=(0, None), method="highs-ds")
        P = res.x.reshape(8, 8)
        if (W_all(P) < 1e-9).any():
            hit += 1
            if key(np.round(P*8)/8) not in allb and key(P) not in allb: miss += 1
    ok &= check("무작위 LP 꼭짓점 중 Ω=0 인 것이 모두 열거 집합에 있음 (완전성 점검)", miss == 0, f"Ω=0 관측 {hit}회, 누락 {miss}")
    tot_loc = sum(r["size"] for r, _ in rows if r["local"]); tot_nl = sum(r["size"] for r, _ in rows if not r["local"])
    gen3 = sum(r["size"] for r, _ in rows if not r["local"] and r["bip_local"])
    biNL = sum(r["size"] for r, _ in rows if not r["local"] and not r["bip_local"])
    print(f"  국소 {tot_loc}, 비국소 {tot_nl} (진짜 3자[이변 주변 모두 국소] {gen3}, 이변 주변 비국소 {biNL})")
    print("  궤도 | 크기 | 국소 | 지지집합비국소 | 프로파일내재 | 이변주변국소 | CHSH | |G0| | NPA거리")
    for r, P in sorted(rows, key=lambda t: (t[0]['size'], not t[0]['prof'])):
        print(f"   {r['size']:4d} | {int(r['local'])} | {int(r['supp_nl'])} | {int(r['prof'])} | {int(r['bip_local'])} | "
              f"{r['bip_chsh']:.2f} | {r['nG0']:3d} | " + ("-" if r['npa'] is None else f"{r['npa']:.4f}"))
    prof_sizes = sorted(r["size"] for r, _ in rows if r["prof"])
    print(f"  프로파일 내재적 궤도 크기: {prof_sizes}")
    # NPA 대조군
    rng = np.random.default_rng(3)
    qd = [npa_distance(random_quantum_box(rng), iters=20000, tol=1e-12) for _ in range(5)]
    print(f"  NPA 양성 대조군(무작위 양자 상자 5개) 거리 최대 {max(qd):.2e}")
    ok &= check("NPA 양성 대조군은 허용(거리 < 1e-3)", max(qd) < 1e-3)
    nl_npa = [r["npa"] for r, _ in rows if r["npa"] is not None]
    check("비국소 궤도 전부 NPA 19-단항식 수준에서 허용 불가(거리 > 1e-3)", all(x > 1e-3 for x in nl_npa),
          f"최소 거리 {min(nl_npa):.4f}" if nl_npa else "")
    if out_json:
        json.dump({"boxes": [{"P_o_given_i": P.tolist(), "orbit_size": r["size"], "profile_intrinsic": r["prof"]}
                             for r, P in rows]}, open(out_json, "w"))
        print("  궤도 대표 저장:", out_json)
    return ok

def run_boxes(path):
    print(f"\n=== BOXES: {path} ===")
    boxes = json.load(open(path))["boxes"]
    pk = [k for k in boxes[0] if k.startswith("P_o_given_i")][0]
    Ps = [np.array(b[pk], dtype=float) for b in boxes]
    R = [analyze(P) for P in Ps]
    orbs = {}
    for n, P in enumerate(Ps):
        orbs.setdefault(min(key(act(P, rm, cm)) for rm, cm in GROUP), []).append(n)
    print(f"  {len(Ps)}개 상자; 전부 Ω=0: {all(r['omega0'] for r in R)}; 전부 NS 꼭짓점: {all(r['vertex'] for r in R)}; "
          f"지지집합 비국소 {sum(r['supp_nl'] for r in R)}/{len(R)}")
    print("  프로파일 내재적:", [n for n, r in enumerate(R) if r['prof']])
    print("  궤도(크기 | 목록 내 원소 수 | 프로파일 내재):")
    rows_ = []
    for k0, mem in orbs.items():
        sz = len({key(act(Ps[mem[0]], rm, cm)) for rm, cm in GROUP})
        rows_.append((sz, len(mem), R[mem[0]]['prof'], mem))
    for sz, k, p, mem in sorted(rows_, key=lambda t: (t[0], not t[2])):
        print(f"   {sz:4d} | {k} | {p} | {mem if k <= 3 else '[...]'}")
    print("  이변 주변 모두 국소:", all(r['bip_local'] for r in R), "| 최대 이변 CHSH:", max(r['bip_chsh'] for r in R))
    for n, P in enumerate(Ps):
        d = npa_distance(P, iters=4000)
        if d < 1e-3: print(f"  경고: 상자 {n} 는 NPA 수준에서 허용 가능(거리 {d:.2e})")
    return True

def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1] if "--out" in a else "research_boxes.json"
    modes = [x for x in a if x.startswith("--") and x not in ("--out",)]
    if not modes: modes = ["--dims", "--selftest", "--prop2", "--research", "--gynin"]
    print(f"NS affine dim = {64-EQ_RANK}; |group| = {len(GROUP)}; #process functions = {len(PF)}")
    res = {}
    if "--dims" in modes: run_dims()
    if "--selftest" in modes: res["selftest"] = run_selftest()
    if "--prop2" in modes or "--prop10" in modes: res["prop2"] = run_prop2()  # --prop10: 옛 이름(호환용)
    if "--gynin" in modes: res["gynin"] = run_gynin()
    if "--research" in modes: res["research"] = run_research(out_json=out)
    if "--boxes" in modes: run_boxes(a[a.index("--boxes") + 1])
    print("\n=== 요약 ===", {k: ("PASS" if v else "FAIL") for k, v in res.items()})

if __name__ == "__main__":
    main()
