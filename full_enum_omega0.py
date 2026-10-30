#!/usr/bin/env python3
"""Paper B: full vertex enumeration of the tripartite binary NS polytope, and the count of Omega=0 vertices.

Stages
  1. Enumerate all vertices of the NS polytope (26 correlator coordinates, 64 inequalities
     P(o|x) >= 0) with pycddlib (double description).  Needs pycddlib; slow (about 10 minutes).
  2. Rationalize every vertex and verify it in exact integer arithmetic:
       - all 64 probabilities are >= 0,
       - the active inequalities (P(o|x) = 0) have rank 26 (computed modulo a prime, which can
         only under-estimate the rank over Q, so rank 26 proves the vertex property),
       - no vertex is listed twice.
  3. For every vertex and every one of the 4096 feedbacks g compute W(P;g) in exact integers
     and count the vertices with Omega = min_g W = 0.

Expected output: 53856 vertices, 64 of them deterministic (local), 1224 with Omega = 0, and 46 orbits
under the group of order 3072 (the vertex set is closed under that group).

Usage:  python3 full_enum_omega0.py [--cache verts.pkl]
        (--cache: reuse/save the raw output of stage 1)
Note: stage 2 and 3 verify the vertices that were produced; they do not prove that the list is
complete.  Completeness rests on the double-description enumeration (with an exact-arithmetic
build of pycddlib, cdd.gmp, the enumeration itself is exact), and on the agreement of the count
with the orbit structure checked in stage 4 (46 classes under the group of order 3072).
"""
import sys, os, itertools, time, pickle, math
from fractions import Fraction
import numpy as np

O = list(itertools.product((0, 1), repeat=3))
idx = lambda t: 4 * t[0] + 2 * t[1] + t[2]
COORDS = [(S, xs) for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
          for xs in itertools.product((0, 1), repeat=len(S))]
assert len(COORDS) == 26
# 8 P(o|x) = 1 + M e,   row index  idx(o)*8 + idx(x)
M = np.zeros((64, 26), dtype=np.int64)
for x in O:
    for o in O:
        for j, (S, xs) in enumerate(COORDS):
            if all(x[p] == xs[i] for i, p in enumerate(S)):
                M[idx(o) * 8 + idx(x), j] = (-1) ** sum(o[p] for p in S)

# feedbacks: GI[g,o] = idx of input g(o);  g = (g_A,g_B,g_C), g_k : other two outputs -> input
tabs = list(itertools.product((0, 1), repeat=4))
GI = np.zeros((4096, 8), dtype=np.int64)
n = 0
for a in tabs:
    for b in tabs:
        for c in tabs:
            for o in O:
                GI[n, idx(o)] = idx((a[2 * o[1] + o[2]], b[2 * o[0] + o[2]], c[2 * o[0] + o[1]]))
            n += 1


def stage1(cache):
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    try:
        import cdd.gmp as cdd
        print("pycddlib: exact (gmp) build")
    except ImportError:
        import cdd
        print("pycddlib: DOUBLE-precision build (cdd.gmp not available); stage 2 verifies exactly")
    rows = [[1] + [int(v) for v in M[r]] for r in range(64)]         # 1 + M e >= 0
    t = time.time()
    mat = cdd.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    gens = cdd.copy_generators(poly)
    assert not gens.lin_set
    V = []
    for g in gens.array:
        assert g[0] != 0
        V.append([x / g[0] for x in g[1:]])
    print("stage 1:", len(V), "vertices,", round(time.time() - t), "s")
    if cache:
        with open(cache, 'wb') as fh:
            pickle.dump(V, fh)
    return V


def rank_mod_p(A, p=2147483629):
    """rank of each integer matrix A[i] (shape (N,64,26)) modulo the prime p, vectorised."""
    A = A.copy() % p
    N, R, C = A.shape
    used = np.zeros((N, R), dtype=bool)
    rank = np.zeros(N, dtype=np.int64)
    ar = np.arange(N)
    for c in range(C):
        cand = (A[:, :, c] != 0) & ~used
        has = cand.any(1)
        piv = cand.argmax(1)
        used[ar[has], piv[has]] = True
        rank += has
        prow = A[ar, piv, :]                                  # (N,C)
        inv = np.ones(N, dtype=np.int64)
        base = prow[:, c] % p
        e = p - 2
        while e:                                              # modular inverse by Fermat
            if e & 1:
                inv = inv * base % p
            base = base * base % p
            e >>= 1
        prow = prow * inv[:, None] % p
        fac = A[:, :, c].copy()
        fac[ar, piv] = 0
        fac[~has] = 0
        A = (A - fac[:, :, None] * prow[:, None, :] % p) % p
    return rank


def orbits(P, D):
    """Orbits of the vertex set under the group of order 3072 (party permutations, input flips,
    input-dependent output flips).  The vertex list is a union of full orbits iff it is closed
    under the generators; we count the orbits (the literature has 46 classes: 1 local + 45)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(P)

    def perm(sigma, f, c):
        """index permutation pi with P'[pi[i]] = P[i];  i = idx(o)*8 + idx(x)"""
        pi = np.zeros(64, dtype=np.int64)
        for x in O:
            for o in O:
                xp = tuple(x[sigma[k]] ^ f[k] for k in range(3))
                op = tuple(o[sigma[k]] ^ c[k][xp[k]] for k in range(3))
                pi[idx(o) * 8 + idx(x)] = idx(op) * 8 + idx(xp)
        return pi
    ident = (0, 1, 2); z3 = (0, 0, 0); zc = [[0, 0]] * 3
    gens = [perm((1, 0, 2), z3, zc), perm((1, 2, 0), z3, zc), perm(ident, (1, 0, 0), zc),
            perm(ident, z3, [[1, 0], [0, 0], [0, 0]]), perm(ident, z3, [[0, 1], [0, 0], [0, 0]])]
    rng = np.random.default_rng(12345)
    w = rng.integers(1, 2**62, size=64, dtype=np.int64)
    key = (P.astype(np.int64) * w).sum(1)                       # wraps mod 2^64; collisions negligible
    order = np.argsort(key); sk = key[order]
    assert len(np.unique(sk)) == N
    ri, ci = [], []
    for pi in gens:
        Q = np.empty_like(P); Q[:, pi] = P                      # Q[:,pi[i]] = P[:,i]
        kq = (Q.astype(np.int64) * w).sum(1)
        pos = np.searchsorted(sk, kq)
        assert (pos < N).all() and (sk[pos] == kq).all(), "vertex set not closed under the group"
        ri.append(np.arange(N)); ci.append(order[pos])
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    ncomp, lab = connected_components(coo_matrix((np.ones(len(ri)), (ri, ci)), shape=(N, N)), directed=False)
    sizes = np.bincount(lab)
    print("orbits under the group of order 3072:", ncomp, "(literature: 46); sizes of the local orbit:",
          sorted(int(v) for v in set(sizes[lab[((P == 0) | (P == D)).all(1)]])))


def main():
    cache = sys.argv[sys.argv.index('--cache') + 1] if '--cache' in sys.argv else None
    V = stage1(cache)
    N = len(V)
    # ---- stage 2: rationalize and verify exactly
    t = time.time()
    Pn = np.zeros((N, 64), dtype=object)
    maxerr = 0.0
    for i, v in enumerate(V):
        e = [Fraction(x).limit_denominator(1000) for x in v]
        maxerr = max(maxerr, max(abs(float(a) - float(b)) for a, b in zip(e, v)))
        for r in range(64):
            Pn[i, r] = (1 + sum(int(M[r, j]) * e[j] for j in range(26) if M[r, j])) / 8
    D = 1
    for x in Pn.flat:
        D = D * x.denominator // math.gcd(D, x.denominator)
    P = np.array([[int(x * D) for x in row] for row in Pn], dtype=np.int64)    # P*D, integer
    print("rationalization: max |e - rational e| = %.2e, common denominator D = %d" % (maxerr, D))
    assert (P >= 0).all(), "negative probability"
    print("all %d vertices have 64 nonnegative probabilities" % N)
    assert len({row.tobytes() for row in P}) == N, "duplicate vertices"
    print("no duplicates")
    ranks = np.concatenate([rank_mod_p(((P[s:s + 4096] == 0)[:, :, None] * M[None]).astype(np.int64))
                            for s in range(0, N, 4096)])
    assert (ranks == 26).all(), "not a vertex: rank %s" % np.unique(ranks)
    print("every vertex has active-constraint rank 26 (mod p): all are vertices")
    nloc = int(((P == 0) | (P == D)).all(1).sum())
    print("deterministic (local) vertices:", nloc)
    # ---- stage 3: Omega exactly
    rows8 = np.arange(8)[None, :]
    Pm = P.reshape(N, 8, 8)                                   # [vertex, o, x]
    best = np.full(N, np.iinfo(np.int64).max)
    for g in range(4096):
        w = Pm[:, np.arange(8), GI[g]].sum(1)                 # sum_o P[o, g(o)]
        np.minimum(best, w, out=best)
    n0 = int((best == 0).sum())
    print("vertices with Omega = 0:", n0)
    print("stage 2-3: %.0f s" % (time.time() - t))
    assert N == 53856 and n0 == 1224
    orbits(P, D)
    np.savez_compressed('ns_vertices_omega0.npz', P=P.astype(np.int32), D=D, omega0=(best == 0))
    print("OK (saved ns_vertices_omega0.npz)")


if __name__ == '__main__':
    main()
