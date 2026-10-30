import itertools
from collections import Counter
O=list(itertools.product([0,1],repeat=3))
# g=(gA,gB,gC); gA:(oB,oC)->xA etc
def mk(tbl):  # tbl: 4 bits indexed by 2-bit
    return tbl
gs=[]
for bits in range(4096):
    t=[(bits>>k)&1 for k in range(12)]
    gA=t[0:4];gB=t[4:8];gC=t[8:12]
    gs.append((gA,gB,gC))
def gfun(g,o):
    a,b,c=o
    return (g[0][2*b+c],g[1][2*a+c],g[2][2*a+b])
Ss=[S for S in itertools.product([0,1],repeat=3) if any(S)]
def coef(g):
    d=Counter()
    for o in O:
        i=gfun(g,o)
        for S in Ss:
            key=(S,tuple(i[k] for k in range(3) if S[k]))
            sign=(-1)**sum(o[k] for k in range(3) if S[k])
            d[key]+=sign
    return tuple(sorted((k,v) for k,v in d.items() if v))
def fp(f,g):
    # f: product strategy f(i)=(fA(i0),fB(i1),fC(i2)) each bit
    n=0
    for o in O:
        i=gfun(g,o)
        if tuple(f[k][i[k]] for k in range(3))==o:n+=1
    return n
F=[((a0,a1),(b0,b1),(c0,c1)) for a0 in(0,1) for a1 in(0,1) for b0 in(0,1) for b1 in(0,1) for c0 in(0,1) for c1 in(0,1)]
const=0;vecs=set();zero_ok=0;proc=0
for g in gs:
    c=coef(g)
    if not c: const+=1
    else:
        vecs.add(c)
        if any(fp(f,g)==0 for f in F): zero_ok+=1
    if all(fp(f,g)==1 for f in F): proc+=1
print(const,len(vecs)+1,len(vecs),zero_ok,proc)
