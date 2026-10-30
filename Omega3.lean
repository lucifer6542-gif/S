/-! Finite verification for Lemmas 2 and 3 of "3-party Ω=0 ⇒ local".
    Polynomials in ℤ[t0,t1,t2] stored densely (exponent < 9 each). Core Lean only. -/

abbrev Poly := Array Int
def zeroP : Poly := (List.replicate 729 (0 : Int)).toArray
def idx (a b c : Nat) : Nat := a*81 + b*9 + c

/-- one factor of an eigenvector of A_x (x=input, o=outcome, c=component): (sign, exponent of t) -/
def fac (x o c : Nat) : Int × Nat :=
  match x, o, c with
  | 0,0,0 => (1,0) | 0,0,1 => (0,0) | 0,1,0 => (0,0) | 0,1,1 => (1,0)
  | 1,0,0 => (1,0) | 1,0,1 => (1,1) | 1,1,0 => (-1,1) | 1,1,1 => (1,0)
  | _,_,_ => (0,0)

/-- a monomial entry: sign and exponent triple -/
abbrev Ent := Int × Nat × Nat × Nat

def popc (s : Nat) : Nat := Id.run do
  let mut c := 0
  let mut t := s
  for _ in [0:12] do
    c := c + t % 2
    t := t / 2
  return c

/-- determinant of an n×n matrix of monomials, by subset DP over columns -/
def detM (n : Nat) (ent : Nat → Nat → Ent) : Poly := Id.run do
  let sz := 2^n
  let mut dp : Array Poly := (List.replicate sz zeroP).toArray
  dp := dp.set! 0 (zeroP.set! 0 1)
  for S in [0:sz] do
    let r := popc S
    if r < n then
      let p := dp[S]!
      for c in [0:n] do
        if (S / 2^c) % 2 == 0 then
          let (s, e0, e1, e2) := ent r c
          if s != 0 then
            let sg : Int := if (popc (S / 2^(c+1))) % 2 == 0 then 1 else -1
            let sh := idx e0 e1 e2
            let S' := S + 2^c
            let mut q := dp[S']!
            for i in [0:729] do
              let v := p[i]!
              if v != 0 then
                q := q.set! (i+sh) (q[i+sh]! + sg * s * v)
            dp := dp.set! S' q
  return dp[sz-1]!

/-- input chosen by party k at outputs o (bit encoding of g: bit 4k+2a+b) -/
def xin (g k : Nat) (o0 o1 o2 : Nat) : Nat :=
  let (a,b) := match k with
    | 0 => (o1,o2) | 1 => (o0,o2) | _ => (o0,o1)
  (g / 2^(4*k+2*a+b)) % 2

def ent8 (g : Nat) (r c : Nat) : Ent :=
  let o0 := r/4%2; let o1 := r/2%2; let o2 := r%2
  let (s0,e0) := fac (xin g 0 o0 o1 o2) o0 (c/4%2)
  let (s1,e1) := fac (xin g 1 o0 o1 o2) o1 (c/2%2)
  let (s2,e2) := fac (xin g 2 o0 o1 o2) o2 (c%2)
  (s0*s1*s2, e0, e1, e2)

def D8 (g : Nat) : Poly := detM 8 (ent8 g)

def nonneg (p : Poly) : Bool := p.all (· ≥ 0)
def nonpos (p : Poly) : Bool := p.all (· ≤ 0)
def nonzero (p : Poly) : Bool := p.any (· != 0)
def neg (p : Poly) : Poly := p.map (fun v => -v)

def mulP (a b : Poly) : Poly := Id.run do
  let mut r := zeroP
  for i in [0:729] do
    if a[i]! != 0 then
      for j in [0:729] do
        if b[j]! != 0 then
          -- exponents add digitwise; indices add since no digit overflow occurs for our uses
          r := r.set! (i+j) (r[i+j]! + a[i]! * b[j]!)
  return r

def mono (c : Int) (e0 e1 e2 : Nat) : Poly := zeroP.set! (idx e0 e1 e2) c
def addP (a b : Poly) : Poly := Id.run do
  let mut r := a
  for i in [0:729] do
    r := r.set! i (r[i]! + b[i]!)
  return r

/-- P_{m,±} = t_m²(t_p t_q ∓ 1)² + (t_p²+1)(t_q²+1) ; variables given as exponent vectors -/
def ev (v e : Nat) : Nat × Nat × Nat := match v with
  | 0 => (e,0,0) | 1 => (0,e,0) | _ => (0,0,e)
def addE (a b : Nat × Nat × Nat) : Nat × Nat × Nat := (a.1+b.1, a.2.1+b.2.1, a.2.2+b.2.2)
def mn (c : Int) (l : List (Nat × Nat)) : Poly :=   -- list of (var, exponent)
  let e := l.foldl (fun acc (v,k) => addE acc (ev v k)) (0,0,0)
  mono c e.1 e.2.1 e.2.2

def Pm (m p q : Nat) (sgn : Int) : Poly :=
  [mn 1 [(m,2),(p,2),(q,2)], mn (-2*sgn) [(m,2),(p,1),(q,1)], mn 1 [(m,2)],
   mn 1 [(p,2),(q,2)], mn 1 [(p,2)], mn 1 [(q,2)], mn 1 []].foldl addP zeroP

def cert (m : Nat) : Poly :=
  let (p,q) := match m with | 0 => (1,2) | 1 => (0,2) | _ => (0,1)
  mulP (Pm m p q 1) (Pm m p q (-1))

def certs : List Poly := [cert 0, cert 1, cert 2]

/-- classification of one g: 0 = all coefficients ≥0 and nonzero; 1 = matches ±cert; 2 = bad -/
def classify8 (g : Nat) : Nat :=
  let d := D8 g
  if nonzero d && (nonneg d || nonpos d) then 0
  else if certs.any (fun c => d == c || d == neg c) then 1
  else 2

def lemma2 : Nat × Nat × Nat :=
  (List.range 4096).foldl (fun (a,b,c) g =>
    match classify8 g with | 0 => (a+1,b,c) | 1 => (a,b+1,c) | _ => (a,b,c+1)) (0,0,0)

/-! Lemma 3: one classical party k with a_k = (a0,a1). 4×4 determinant in (ti,tj) = (t0,t1). -/
def ent4 (g k a0 a1 : Nat) (r c : Nat) : Ent :=
  let (i,j) := match k with | 0 => (1,2) | 1 => (0,2) | _ => (0,1)
  let oi := r/2%2; let oj := r%2
  let o : Nat → Nat := fun m => if m == i then oi else if m == j then oj else 0
  let xk := xin g k (o 0) (o 1) (o 2)
  let ok := if xk == 0 then a0 else a1
  let o' : Nat → Nat := fun m => if m == k then ok else o m
  let xi := xin g i (o' 0) (o' 1) (o' 2)
  let xj := xin g j (o' 0) (o' 1) (o' 2)
  let (s0,e0) := fac xi oi (c/2%2)
  let (s1,e1) := fac xj oj (c%2)
  (s0*s1, e0, e1, 0)

def D4 (g k a0 a1 : Nat) : Poly := detM 4 (ent4 g k a0 a1)

/-- the special polynomial 1 - ti² tj² -/
def spec : Poly := addP (mono 1 0 0 0) (mono (-1) 2 2 0)

/-- table (xi,xj) for (oi,oj) is (oi^oj^ci, oi^oj^cj) for some ci,cj -/
def tableOK (g k a0 a1 : Nat) : Bool :=
  let (i,j) := match k with | 0 => (1,2) | 1 => (0,2) | _ => (0,1)
  let tab := (List.range 4).map fun r =>
    let oi := r/2%2; let oj := r%2
    let o : Nat → Nat := fun m => if m == i then oi else if m == j then oj else 0
    let xk := xin g k (o 0) (o 1) (o 2)
    let ok := if xk == 0 then a0 else a1
    let o' : Nat → Nat := fun m => if m == k then ok else o m
    (xin g i (o' 0) (o' 1) (o' 2), xin g j (o' 0) (o' 1) (o' 2), (oi + oj) % 2)
  -- xi xor s constant, xj xor s constant over all four rows
  match tab with
  | [] => false
  | (x0,y0,s0) :: _ =>
    let ci := (x0 + s0) % 2; let cj := (y0 + s0) % 2
    tab.all fun (x,y,s) => (x + s) % 2 == ci && (y + s) % 2 == cj

/-- (positive-type count, special count, special with correct table, bad count) -/
def lemma3 : Nat × Nat × Nat × Nat := Id.run do
  let mut pos := 0; let mut sp := 0; let mut spt := 0; let mut bad := 0
  for g in [0:4096] do
    for k in [0:3] do
      for a0 in [0:2] do
        for a1 in [0:2] do
          let d := D4 g k a0 a1
          if nonzero d && (nonneg d || nonpos d) then pos := pos + 1
          else if d == spec || d == neg spec then
            sp := sp + 1
            if tableOK g k a0 a1 then spt := spt + 1
          else bad := bad + 1
  return (pos, sp, spt, bad)

def showP (p : Poly) : String := Id.run do
  let mut out := ""
  for i in [0:729] do
    if p[i]! != 0 then out := out ++ s!"{p[i]!}*t0^{i/81}t1^{i/9%9}t2^{i%9} "
  return out
def hist4 : IO Unit := do
  let mut seen : List (String × Nat) := []
  for g in [0:4096] do
    for k in [0:3] do
      for a0 in [0:2] do
        for a1 in [0:2] do
          let s := showP (D4 g k a0 a1)
          match seen.find? (·.1 == s) with
          | some _ => seen := seen.map fun (t,c) => if t == s then (t,c+1) else (t,c)
          | none => seen := seen ++ [(s,1)]
  for (t,c) in seen do IO.println s!"{c}  {t}"

def main : IO Unit := do
  IO.println "--- 4x4 histogram ---"
  hist4
  IO.println s!"lemma2 (positive-type, cert-type, bad) = {lemma2}"
  IO.println s!"lemma3 (positive, special, special-with-table, bad) = {lemma3}"
  IO.println s!"lemma2 (positive-type, cert-type, bad) = {lemma2}"
  IO.println s!"lemma3 (positive, special, special-with-table, bad) = {lemma3}"

/-- Lemma 2 (finite part): of the 4096 feedbacks, 4072 have a determinant polynomial with all
    coefficients ≥ 0 and nonzero; the other 24 equal ±P₋P₊ (sum-of-squares type); none is bad. -/
theorem lemma2_ok : lemma2 = (4072, 24, 0) := by native_decide

/-- Lemma 3 (finite part): over the 49152 tuples (g,k,a), 49104 have same-sign nonzero determinant,
    exactly 48 have ±(1 - ti² tj²), all 48 with the table T = (oi⊕oj⊕ci, oi⊕oj⊕cj); none is bad. -/
theorem lemma3_ok : lemma3 = (49104, 48, 48, 0) := by native_decide

#print axioms lemma2_ok
#print axioms lemma3_ok
