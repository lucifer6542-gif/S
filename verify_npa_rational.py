#!/usr/bin/env python3
"""
NPA(19 단항식) 허용 불가의 정확한(유리수) 검증. numpy/scipy 불필요, fractions 만 사용.
입력: npa_certificates.json (각 상자의 유리수 P 와 유리수 쌍대 증인 Z)
각 상자에 대해 정확 산술로 다음을 검사한다.
  (1) P 가 정규화·비음수·무신호 (모든 부분집합 주변분포 입력 독립)
  (2) Ω=0 (W(P;g)=0 인 g 존재)
  (3) NS 다면체 꼭짓점 (영좌표 + NS 등식계의 해가 유일: 정확한 랭크 64)
  (4) 증인 Z: 대칭, 양의 정부호 (정확한 LDL, 모든 피벗>0),
      자유 단어(같은 당사자 곱 포함)마다 Z 의 위치 합 = 0,
      <Z,G0> = sum_{고정 단어} v_w * (Z 의 위치 합) < 0.
  (4)가 성립하면 임의의 PSD 모멘트 행렬 G 에 대해 <Z,G> >= 0 인데, 고정 항목이 P 와 같고
  자유 단어가 위치별 동일값이면 <Z,G> = <Z,G0> < 0 이므로 모순 -> 양자 상관이 아니다.
"""
import json, sys, itertools, collections
from fractions import Fraction as Fr

O = list(itertools.product([0, 1], repeat=3))
idx = lambda t: 4*t[0] + 2*t[1] + t[2]

def words():
    mons = [()]
    for p in range(3):
        for x in (0, 1): mons.append(((p, x),))
    for p, q in [(0, 1), (0, 2), (1, 2)]:
        for x in (0, 1):
            for y in (0, 1): mons.append(((p, x), (q, y)))
    return mons
MONS = words()
def canon(word):
    per = [[], [], []]
    for p, x in word:
        if per[p] and per[p][-1] == x: per[p].pop()
        else: per[p].append(x)
    t = tuple(tuple(s) for s in per); r = tuple(tuple(reversed(s)) for s in per)
    return min(t, r)
POS = collections.defaultdict(list)
for a in range(len(MONS)):
    for b in range(len(MONS)):
        POS[canon(tuple(reversed(MONS[a])) + MONS[b])].append((a, b))

def correlators(P):
    vals = {}
    for w in POS:
        parties = [p for p in range(3) if len(w[p]) > 0]
        if any(len(w[p]) > 1 for p in range(3)): continue
        if not parties: vals[w] = Fr(1); continue
        ins = [0, 0, 0]
        for p in parties: ins[p] = w[p][0]
        vals[w] = sum((-1)**sum(o[p] for p in parties) * P[idx(o)][idx(tuple(ins))] for o in O)
    return vals

def check_box(P):
    ok = all(P[o][i] >= 0 for o in range(8) for i in range(8))
    ok &= all(sum(P[o][i] for o in range(8)) == 1 for i in range(8))
    for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
        rest = [p for p in range(3) if p not in S]
        base = {}
        for i in O:
            for oS in itertools.product([0, 1], repeat=len(S)):
                m = sum(P[idx(o)][idx(i)] for o in O if all(o[S[k]] == oS[k] for k in range(len(S))))
                key = (oS, tuple(i[p] for p in S))
                if key in base: ok &= (base[key] == m)
                else: base[key] = m
    return bool(ok)

def omega_zero(P):
    tabs = list(itertools.product([0, 1], repeat=4))
    for a in tabs:
        for b in tabs:
            for c in tabs:
                W = sum(P[idx(o)][idx((a[2*o[1]+o[2]], b[2*o[0]+o[2]], c[2*o[0]+o[1]]))] for o in O)
                if W == 0: return True
    return False

def vertex_rank(P):
    rows = []
    def unit(o, i): v = [Fr(0)]*64; v[idx(o)*8 + idx(i)] = Fr(1); return v
    for i in O:
        v = [Fr(0)]*64
        for o in O: v[idx(o)*8 + idx(i)] = Fr(1)
        rows.append(v)                                  # 정규화
    for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
        for i in O:
            i0 = tuple(0 if p not in S else i[p] for p in range(3))
            if i == i0: continue
            for oS in itertools.product([0, 1], repeat=len(S)):
                v = [Fr(0)]*64
                for o in O:
                    if all(o[S[k]] == oS[k] for k in range(len(S))):
                        v[idx(o)*8 + idx(i)] += 1; v[idx(o)*8 + idx(i0)] -= 1
                rows.append(v)
    for o in O:
        for i in O:
            if P[idx(o)][idx(i)] == 0: rows.append(unit(o, i))
    return rank(rows)

def rank(M):
    M = [r[:] for r in M]; rk = 0; ncol = len(M[0])
    for c in range(ncol):
        piv = next((r for r in range(rk, len(M)) if M[r][c] != 0), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        pv = M[rk][c]
        for r in range(len(M)):
            if r != rk and M[r][c] != 0:
                f = M[r][c] / pv
                M[r] = [x - f*y for x, y in zip(M[r], M[rk])]
        rk += 1
    return rk

def is_pd(Z):
    n = len(Z); A = [r[:] for r in Z]
    for k in range(n):
        if A[k][k] <= 0: return False
        for i in range(k+1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= f * A[k][j]
    return True

def check_cert(P, Z):
    n = len(MONS)
    sym = all(Z[i][j] == Z[j][i] for i in range(n) for j in range(n))
    fixed = correlators(P)
    free_ok = all(sum(Z[a][b] for a, b in ps) == 0 for w, ps in POS.items() if w not in fixed)
    val = sum(fixed[w] * sum(Z[a][b] for a, b in POS[w]) for w in fixed)
    return sym, free_ok, is_pd(Z), val

if __name__ == "__main__":
    certs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "npa_certificates.json"))
    allok = True
    for c in certs:
        P = [[Fr(x) for x in r] for r in c["P"]]; Z = [[Fr(x) for x in r] for r in c["Z"]]
        box = check_box(P); om = omega_zero(P); rk = vertex_rank(P)
        sym, fo, pd, val = check_cert(P, Z)
        ok = box and om and rk == 64 and sym and fo and pd and val < 0
        allok &= ok
        print(f"상자 {c['box']:2d} (궤도 {c['orbit']:3d}): NS {box} | Ω=0 {om} | 랭크 {rk}/64 | Z대칭 {sym} | 자유합=0 {fo} | Z>0 {pd} | <Z,G0>={float(val):+.4f} -> {'허용불가(정확)' if ok else '실패'}")
    print("전체:", "PASS" if allok else "FAIL")
