"""Check of Theorem 3 (Paper C): the GHZ_4 box with sigma_x / sigma_y settings.

  P(o|x) = (1 + (-1)^{|o|} cos(m*pi/2)) / 16,   m = |x|.

Prints: W of the explicit feedback g = (a, a, b, b), a = 1+o3+o4, b = o1+o2 (mod 2);
normalisation; Mermin value of the box (local bound 4, algebraic maximum 8);
and the number of feedbacks g (each g_k a function of the other three outputs)
with W = 0, counted by backtracking.  Standard library only.
"""
from itertools import product

O = list(product([0, 1], repeat=4))
COS = [1, 0, -1, 0, 1]


def P(o, x):
    return (1 + (-1) ** sum(o) * COS[sum(x)]) / 16


def g(o):
    a = 1 - (o[2] ^ o[3])
    b = o[0] ^ o[1]
    return (a, a, b, b)


# each g_k must not depend on o_k
for k in range(4):
    for o in O:
        o2 = list(o)
        o2[k] ^= 1
        assert g(o)[k] == g(tuple(o2))[k]

print("W =", sum(P(o, g(o)) for o in O))
print("normalised:", all(abs(sum(P(o, x) for o in O) - 1) < 1e-12 for x in O))


def E(x):
    return sum((-1) ** sum(o) * P(o, x) for o in O)


M = E((0, 0, 0, 0)) + E((1, 1, 1, 1)) - sum(E(x) for x in O if sum(x) == 2)
print("Mermin value of the box =", M)

best = -9
for a0 in product([1, -1], repeat=4):
    for a1 in product([1, -1], repeat=4):
        A = (a0, a1)

        def e(x):
            r = 1
            for k in range(4):
                r *= A[x[k]][k]
            return r

        v = e((0, 0, 0, 0)) + e((1, 1, 1, 1)) - sum(e(x) for x in O if sum(x) == 2)
        best = max(best, v)
print("local bound =", best)

# feedbacks with W = 0: P(o|x) = 0 forces |x| in {0,4} for odd |o|, |x| = 2 for even |o|
req = {o: [x for x in O if (sum(o) % 2 == 1 and sum(x) in (0, 4)) or (sum(o) % 2 == 0 and sum(x) == 2)] for o in O}


def key(o, k):
    return (k, tuple(o[j] for j in range(4) if j != k))


count = 0


def bt(i, tab):
    global count
    if i == 16:
        count += 1
        return
    o = O[i]
    for x in req[o]:
        new = {}
        ok = True
        for k in range(4):
            kk = key(o, k)
            if kk in tab:
                if tab[kk] != x[k]:
                    ok = False
                    break
            else:
                new[kk] = x[k]
        if ok:
            tab.update(new)
            bt(i + 1, tab)
            for kk in new:
                del tab[kk]


bt(0, {})
print("feedbacks with W = 0:", count)
