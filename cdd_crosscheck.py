#!/usr/bin/env python3
"""논문 B 명제 3 의 외부 도구(pycddlib, 정확한 유리수) 교차 확인.

사용:  pip install pycddlib-standalone   (또는 pycddlib + libgmp)
       python3 cdd_crosscheck.py --check-h          # cdd 없이 H-행렬만 점검
       python3 cdd_crosscheck.py                    # 면 궤도 대표 12개 -> 군 확장
       python3 cdd_crosscheck.py --all-faces        # 346개 면 전부 (군 불사용, 더 독립적)
verify_B_all.py 와 같은 폴더에서 실행. research_boxes.json 이 있으면 상자 단위로 대조.
cdd H-표현 규약: 행 [b, A] 는 b + A·x >= 0.  lin_set 의 행은 등식.
"""
import sys, json, time, itertools
from fractions import Fraction
import numpy as np
import verify_B_all as V          # GI, EQ, BEQ, GROUP, PF, act_g, GKEY ... (main 은 실행되지 않음)

# ---------- cdd 래퍼 (pycddlib 3.x 와 2.x 모두) ----------
def _vertices_of_H(rows, lin):
    """rows: 정수 리스트의 리스트 [b, A...]; lin: 등식 행 번호 집합. 꼭짓점 목록(Fraction 벡터) 반환."""
    try:                                   # pycddlib >= 3
        try:
            import cdd.gmp as cdd          # 정확 산술
        except ImportError:
            import cdd                     # 배정밀도 빌드: 아래에서 반환된 꼭짓점을 유리수로 환산해 정확히 재검사한다
            if not globals().get('_WARNED'):
                print("pycddlib: 배정밀도 빌드 (cdd.gmp 없음); 결과 상자는 정확 산술로 재검사한다", file=sys.stderr)
                globals()['_WARNED'] = True
        mat = cdd.matrix_from_array([[Fraction(x) for x in r] for r in rows],
                                    lin_set=set(lin), rep_type=cdd.RepType.INEQUALITY)
        poly = cdd.polyhedron_from_matrix(mat)
        gen = cdd.copy_generators(poly)
        garr, glin = gen.array, set(gen.lin_set)
    except (AttributeError, TypeError):    # pycddlib 2.x
        import cdd
        mat = cdd.Matrix([], number_type='fraction')
        for n, r in enumerate(rows): mat.extend([Fraction(x) for x in r], linear=(n in lin))
        mat.rep_type = cdd.RepType.INEQUALITY
        gen = cdd.Polyhedron(mat).get_generators()
        garr = [list(gen[i]) for i in range(gen.row_size)]; glin = set(gen.lin_set)
    verts = []
    for n, r in enumerate(garr):
        r = [Fraction(x) for x in r]
        if n in glin or r[0] == 0:
            raise RuntimeError("광선/직선 발생: 다면체가 유계가 아님 (H-행렬 오류)")
        verts.append(tuple(x / r[0] for x in r[1:]))
    return verts

# ---------- H-표현 구성 ----------
EQI = V.EQ.astype(int); BEQI = V.BEQ.astype(int)
assert (EQI == V.EQ).all() and (BEQI == V.BEQ).all()

def H_face(g):
    """NS 다면체 ∩ {W(P;g)=0}.  변수 P[o,i] 를 idx(o)*8+idx(i) 로 평탄화한 64개."""
    rows, lin = [], []
    for k in range(64):                                   # P_k >= 0
        r = [0]*65; r[k+1] = 1; rows.append(r)
    for e, b in zip(EQI, BEQI):                           # EQ·P = BEQ  ->  -b + e·P = 0
        lin.append(len(rows)); rows.append([-int(b)] + [int(x) for x in e])
    w = [0]*65                                            # W_g = sum_o P[o, g(o)] = 0
    for o in range(8): w[1 + o*8 + int(V.GI[g, o])] += 1
    lin.append(len(rows)); rows.append(w)
    return rows, lin

def W_exact(Pf, g):
    return sum(Pf[o*8 + int(V.GI[g, o])] for o in range(8))

# ---------- 면 분류 ----------
def face_classes():
    """W_g 가 NS 위에서 같은 함수인 g 끼리 묶는다 (계수차가 EQ 행공간에 속하면 동일)."""
    proj = np.eye(64) - np.linalg.pinv(V.EQ.astype(float), rcond=1e-9) @ V.EQ.astype(float)
    cls = {}
    for g in range(4096):
        if g in PFSET: continue                           # W≡1, 면이 비어 있음
        w = np.zeros(64)
        for o in range(8): w[o*8 + V.GI[g, o]] += 1
        k = (np.round(w @ proj, 6) + 0.0).tobytes()
        cls.setdefault(k, []).append(g)
    return list(cls.values())
PFSET = set(V.PF)

def group_expand(verts):
    """Fraction 정확 산술로 위수 384 군 작용 (Q[rm[o],cm[i]] = P[o,i])."""
    out = set()
    for v in verts:
        for rm, cm in V.GROUP:
            q = [None]*64
            for o in range(8):
                for i in range(8): q[int(rm[o])*8 + int(cm[i])] = v[o*8 + i]
            out.add(tuple(q))
    return out

def class_orbits(classes):
    cid = {g: n for n, c in enumerate(classes) for g in c}
    seen, reps = set(), []
    for n, c in enumerate(classes):
        if n in seen: continue
        reps.append(n)
        for rm, cm in V.GROUP: seen.add(cid[V.act_g(c[0], rm, cm)])
    return reps

def omega_zero_exact(v):
    """4096개 g 전수, 정확 산술: min_g W == 0 ?"""
    return min(W_exact(v, g) for g in range(4096)) == 0

def main():
    t0 = time.time()
    classes = face_classes()
    print(f"서로 다른 W_g 함수(면) {len(classes)}개  (기대: 346)")
    allf = "--all-faces" in sys.argv
    reps = list(range(len(classes))) if allf else class_orbits(classes)
    print(f"처리할 면 {len(reps)}개  (기대: {'346' if allf else '12'})")
    found = set()
    for n, r in enumerate(reps):
        g = classes[r][0]; rows, lin = H_face(g)
        vs = _vertices_of_H(rows, lin)
        for v in vs:
            assert all(x >= 0 for x in v) and W_exact(v, g) == 0
        found |= set(vs)
        print(f"  면 {n+1}/{len(reps)} (g={g}): 꼭짓점 {len(vs)}개, 누적 {len(found)}개  [{time.time()-t0:.0f}s]", flush=True)
    print(f"면 대표 꼭짓점(중복 제거): {len(found)}  (기대: 130, --all-faces 이면 1224)")
    allv = found if allf else group_expand(found)
    print(f"Ω=0 꼭짓점 총수: {len(allv)}   (논문 B 값: 1224)")
    bad = [v for v in allv if not omega_zero_exact(v)]
    print(f"정확 산술 Ω=0 재검사 실패: {len(bad)}")
    try:                                                   # 상자 단위 대조
        with open("research_boxes.json") as fh:
            R = json.load(fh)["boxes"]
        keyf = lambda v: tuple(int(round(float(x)*1e6)) for x in v)
        mine = {keyf(v) for v in allv}
        theirs = set()
        for b in R:
            P = np.array(b["P_o_given_i"], float)
            for rm, cm in V.GROUP: theirs.add(V.key(V.act(P, rm, cm)))
        print(f"research_boxes 군 확장 {len(theirs)}개 vs cdd {len(mine)}개: 일치 = {mine == theirs}",
              f"| cdd에만 {len(mine-theirs)}, 기존에만 {len(theirs-mine)}")
    except FileNotFoundError:
        print("research_boxes.json 없음: 상자 단위 대조 생략")
    with open("cdd_vertices_exact.json", "w") as fh:
        json.dump([[str(x) for x in v] for v in sorted(allv)], fh)
    print("저장: cdd_vertices_exact.json")

def check_h():
    """cdd 없이 H-행렬만 점검: 알려진 꼭짓점이 모든 행을 만족하고 W=0 인지(정수 산술)."""
    with open("research_boxes.json") as fh:
        R = json.load(fh)["boxes"]
    classes = face_classes(); print("면 클래스", len(classes), "대표 수", len(class_orbits(classes)))
    ok = 0
    for b in R:
        P = [Fraction(x).limit_denominator(1000) for x in np.array(b["P_o_given_i"]).flatten()]
        gs = [g for g in range(4096) if W_exact(P, g) == 0]
        g = gs[0]; rows, lin = H_face(g)
        for n, r in enumerate(rows):
            s = r[0] + sum(r[k+1]*P[k] for k in range(64))
            assert (s == 0) if n in lin else (s >= 0), (n, s)
        ok += 1
    print(f"{ok}/15 대표 상자가 자기 면의 H-행렬을 정확 산술로 만족")

if __name__ == "__main__":
    check_h() if "--check-h" in sys.argv else main()
