/-! Finite verification of the combinatorial claims of Paper B (tripartite binary).
    Core Lean only. Feedback g: 12 bits; party k (0..2) receives input bit
    gk(o) = bit (4k + 2*oa + ob) of g, where (a,b) are the other two parties in increasing order. -/

def pc (s : Nat) : Nat := (s &&& 1) + ((s >>> 1) &&& 1) + ((s >>> 2) &&& 1)

def oth (k : Nat) : Nat × Nat :=
  match k with | 0 => (1,2) | 1 => (0,2) | _ => (0,1)

def bitN (x j : Nat) : Nat := (x >>> j) &&& 1

/-- input vector (3 bits, bit k = input of party k) produced by g at outcome o -/
def gin (g o : Nat) : Nat := Id.run do
  let mut r := 0
  for k in [0:3] do
    let (a,b) := oth k
    r := r ||| (bitN g (4*k + 2*(bitN o a) + bitN o b) <<< k)
  return r

/-- functional 8*(W-1) non-constant part: coefficients indexed S*8 + (i &&& S), S=1..7 -/
def wvec (g : Nat) : Array Int := Id.run do
  let mut a : Array Int := (List.replicate 64 (0 : Int)).toArray
  for o in [0:8] do
    let i := gin g o
    for S in [1:8] do
      let s : Int := if pc (o &&& S) % 2 == 0 then 1 else -1
      let j := S*8 + (i &&& S)
      a := a.set! j (a[j]! + s)
  return a

def isZero (a : Array Int) : Bool := a.all (· == 0)

def lexLt (a b : Array Int) : Bool := Id.run do
  for j in [0:64] do
    if a[j]! < b[j]! then return true
    if a[j]! > b[j]! then return false
  return false

def sortedDedup (xs : Array (Array Int)) : Array (Array Int) :=
  let s := xs.qsort lexLt
  s.foldl (fun acc x => if acc.size > 0 && acc.back! == x then acc else acc.push x) #[]

/-- symmetry (party perm index p, input flips r, output flips c) acting on a functional -/
def perms : Array (Nat × Nat × Nat) := #[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def pmap (p : Nat × Nat × Nat) (k : Nat) : Nat := if k == 0 then p.1 else if k == 1 then p.2.1 else p.2.2

def actF (p : Nat × Nat × Nat) (r c : Nat) (a : Array Int) : Array Int := Id.run do
  let mut out : Array Int := (List.replicate 64 (0 : Int)).toArray
  for S in [1:8] do
    for i in [0:8] do
      if i &&& S == i then
        let v := a[S*8+i]!
        if v != 0 then
          let mut S' := 0
          let mut i' := 0
          for k in [0:3] do
            if bitN S k == 1 then
              let pk := pmap p k
              S' := S' ||| (1 <<< pk)
              i' := i' ||| (((bitN i k) ^^^ (bitN r pk)) <<< pk)
          let s : Int := if pc (c &&& S') % 2 == 0 then 1 else -1
          let j := S'*8 + i'
          out := out.set! j (out[j]! + s*v)
  return out


def memb (s : Array (Array Int)) (x : Array Int) : Bool :=
  s.any (· == x)

/-- number of fixed points of f∘g ; f encoded by 6 bits: f_k(i_k) = bit (2k + i_k) -/
def fixpts (f g : Nat) : Nat := Id.run do
  let mut n := 0
  for o in [0:8] do
    let i := gin g o
    let mut o' := 0
    for k in [0:3] do
      o' := o' ||| (bitN f (2*k + bitN i k) <<< k)
    if o' == o then n := n + 1
  return n

def isProc (g : Nat) : Bool := (List.range 64).all (fun f => fixpts f g == 1)

def hasConst (g : Nat) : Bool :=
  (List.range 3).any (fun k =>
    let t := (g >>> (4*k)) &&& 15
    t == 0 || t == 15)

def result : Nat × Nat × Bool × Bool × Nat × Nat × Nat × Nat × Bool × Nat × List Nat × Nat := Id.run do
  let ws : Array (Array Int) := (Array.range 4096).map wvec
  let nproc := ((List.range 4096).filter isProc).length
  let nzero := (ws.filter isZero).size
  let agree := (List.range 4096).all (fun g => isProc g == isZero ws[g]!)
  let nonproc0 := (List.range 4096).all (fun g => isProc g || (List.range 64).any (fun f => fixpts f g == 0))
  let ncausal := ((List.range 4096).filter (fun g => isProc g && hasConst g)).length
  let nnon := ((List.range 4096).filter (fun g => isProc g && !hasConst g)).length
  let distinct := sortedDedup ws
  let nz := distinct.filter (fun a => !isZero a)
  let mut syms : Array (Nat × Nat × Nat × Nat × Nat) := #[]
  for pi in [0:6] do
    for r in [0:8] do
      for c in [0:8] do
        let p := perms[pi]!
        syms := syms.push (p.1, p.2.1, p.2.2, r, c)
  let mut closed := true
  let mut reps : Array (Array Int) := #[]
  for v in nz do
    let mut best := v
    for s in syms do
      let w := actF (s.1, s.2.1, s.2.2.1) s.2.2.2.1 s.2.2.2.2 v
      if !(memb nz w) then closed := false
      if lexLt w best then best := w
    reps := reps.push best
  let rr := sortedDedup reps
  let sizes := ((rr.map (fun r => (reps.filter (· == r)).size)).qsort (· > ·)).toList
  let nloc := ((List.range 64).filter (fun f => (List.range 4096).any (fun g => fixpts f g == 0))).length
  return (nproc, nzero, agree, nonproc0, ncausal, nnon, distinct.size, syms.size, closed, rr.size, sizes, nloc)

theorem result_ok : (result == (744, 744, true, true, 680, 64, 347, 384, true, 12, [96,64,48,48,24,24,12,12,6,6,4,2], 32)) = true := by native_decide

#print axioms result_ok
