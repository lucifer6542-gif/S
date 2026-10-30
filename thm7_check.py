#!/usr/bin/env python3
"""thm7_check.py -- independent check of Proposition 2 (Paper B) via the
pairwise-exclusive-completeness criterion (Theorem 7 of [D2], arXiv:2609.39735).

For every feedback g=(g_A,g_B,g_C) on the tripartite binary scenario (4096 in all)
the event list is E_g = {(o | g(o))}. Two events (o|i),(o'|i') are exclusive iff some
party k has i_k = i'_k and o_k != o'_k. g is a process function iff E_g is
pairwise exclusive (the list is complete automatically, |E_g| = 8 = |O|).

The script also runs the fixed-point test (every product deterministic f gives
f o g exactly one fixed point) and checks that both tests agree for every g.
Expected output: 744 / 744 / True / 64 / 680 / 12.
Standard library only.
"""
import itertools

O = list(itertools.product((0, 1), repeat=3))
F4 = list(itertools.product((0, 1), repeat=4))   # g_k : 2 bits -> 1 bit


def build_g(gA, gB, gC):
    g = {}
    for o in O:
        a, b, c = o
        g[o] = (gA[2 * b + c], gB[2 * a + c], gC[2 * a + b])
    return g


def exclusive(e, f, n=3):
    (o, i), (p, j) = e, f
    return any(i[k] == j[k] and o[k] != p[k] for k in range(n))


def theorem7(g):
    ev = [(o, g[o]) for o in O]
    return all(exclusive(ev[m], ev[n]) for m in range(8) for n in range(m + 1, 8))


def fixed_point_test(g):
    for fa in itertools.product((0, 1), repeat=2):
        for fb in itertools.product((0, 1), repeat=2):
            for fc in itertools.product((0, 1), repeat=2):
                n = 0
                for o in O:
                    i = g[o]
                    if (fa[i[0]], fb[i[1]], fc[i[2]]) == o:
                        n += 1
                if n != 1:
                    return False
    return True


def tripartite():
    n7 = nfp = noglobal = 0
    agree = True
    for gA in F4:
        for gB in F4:
            for gC in F4:
                g = build_g(gA, gB, gC)
                t7, fp = theorem7(g), fixed_point_test(g)
                agree &= (t7 == fp)
                n7 += t7
                nfp += fp
                if t7 and not any(len(set(x)) == 1 for x in (gA, gB, gC)):
                    noglobal += 1
    return n7, nfp, agree, noglobal


def bipartite():
    c = 0
    for gA in itertools.product((0, 1), repeat=2):
        for gB in itertools.product((0, 1), repeat=2):
            ev = [((a, b), (gA[b], gB[a])) for a in (0, 1) for b in (0, 1)]
            c += all(exclusive(ev[m], ev[n], 2) for m in range(4) for n in range(m + 1, 4))
    return c


if __name__ == "__main__":
    n7, nfp, agree, noglobal = tripartite()
    print("Theorem 7 (pairwise exclusive) count      :", n7)
    print("fixed-point test count                    :", nfp)
    print("per-g agreement of the two tests          :", agree)
    print("process functions with no constant g_k    :", noglobal)
    print("process functions with a constant g_k     :", n7 - noglobal)
    print("bipartite process functions               :", bipartite())
