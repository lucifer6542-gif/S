# No-Signaling Boxes and Deterministic Feedback: Verification Code and Data

Code and data that reproduce the computations of Paper A (two parties: exact reduction and the quantum curve) Paper B (tripartite binary scenario: boxes with Ω=0, computer-assisted), and Paper C (Ω=0 ⇒ local for three-party commuting-operator quantum boxes, and the four-party GHZ counterexample).
Apart from the Lean check below, the results are computations, not proofs; their confidence levels follow the "scope of verification" statements in each paper.

## Paper B (tripartite binary)

| File | Contents |
|---|---|
| `verify_B_all.py` | Integrated script. Reproduces the dimensions, self-tests, Proposition 2, GYNIN, vertex enumeration, and the NPA verdicts. |
| `cdd_crosscheck.py` | Cross-checks the enumeration of Ω=0 vertices with pycddlib; the returned vertices are converted to rationals and rechecked in exact arithmetic. Imports `verify_B_all.py`. |
| `full_enum_omega0.py` | Enumerates all 53,856 vertices of the NS polytope with pycddlib, verifies in exact integer arithmetic that each is a vertex (64 deterministic), tests every vertex against all 4096 feedbacks (exactly 1224 have Ω=0), and counts the orbits under the group of order 3072 (46). Needs pycddlib, numpy, scipy; about 12 minutes. |
| `exact_prop2.py` | Integer-arithmetic check of Proposition 2 (744 feedbacks g with w_g≡0, 347 distinct w_g, minimum 0 for the remaining 3352, 744 process functions). Standard library only. |
| `thm7_check.py` | Independent reproduction of Proposition 2 via the pairwise-exclusivity check of Theorem 7 of [8] (744 of the 4096 g pass, 680/64 split, 12 for two parties, agreement with the fixed-point check). |
| `verify_npa_rational.py` | Verifies the rational dual witnesses at the 19-monomial NPA level in exact arithmetic. |
| `npa_certificates.json` | Input to the script above: rational boxes and witnesses for the 13 nonlocal orbit representatives. |
| `research_boxes.json` | The 15 orbit representatives of Ω=0 vertices (output of `verify_B_all.py --research`). |
| `research_run.txt` | Log of the `--research` run (the basis of the paper's values 130, 1224, 15 orbits, and the NPA distances). |
| `OmegaBthm.lean` | Lean 4.15.0 (core only) check of the combinatorial claims of Proposition 2 and Appendix A.2. Uses `native_decide`; the vertex enumeration is not covered. |
| `tripartite_nonlocal_omega0_boxes.json` | A list of 36 boxes (checked with `verify_B_all.py --boxes`). |

## Paper C (commuting-operator quantum boxes)

| File | Contents |
|---|---|
| `Omega3.lean` | Lean 4.15.0 (core only) recomputation of the finite parts of Lemmas 2 and 3: all 4096 feedbacks (result `lemma2 = (4072, 24, 0)`) and all 49152 tuples (result `lemma3 = (49104, 48, 48, 0)`). Uses `native_decide`. The C*-algebra part and the trigonometric steps are not formalised. |
| `ghz4_check.py` | Checks Theorem 3: the GHZ₄ box has W = 0 for the explicit feedback, Mermin value 8 (local bound 4), and exactly 6 feedbacks with W = 0. Standard library only. |

## Paper A (two-party binary)

| File | Contents |
|---|---|
| `verify_A_all.py` | Numerical cross-checks of Theorems 1–5 and 8, Proposition 7, the table of Section 7, and the near-threshold behavior of Theorem 9. Requires numpy and scipy. |
| `verify_m3_curve.py` | Check of the closed form of Theorem 9 (m=3 quantum curve): enumerates the 72 variants and their 7 orbits, verifies the exact rational dual certificates for six orbits (exact LDLᵀ), and the analytic certificate of the main orbit (exact identities, positive trace); the two comparisons with the transcendental closed form (τ < f(2√2) and bound < f(11/4)) use rigorous interval arithmetic (`mpmath.iv`). `--sdp` compares with an SDP solve of all 72 variants (needs cvxpy). Requires sympy; reads `m3_certificates.json`. |
| `m3_certificates.json` | Rational certificates (λ, d) used by the script above. |

Items based on random sampling (the sample size and minimum margin in Theorem 4, the maximum in Theorem 8) vary with the sampling method and `--seed`. Only the deterministic items (the errors in Theorems 1, 2, 3, and 5, the attaining configuration of Theorem 8, Proposition 7, the table of Section 7, and the bounds of Theorem 9) are compared directly with the values in the paper.

## Requirements

- Python 3 (tested with 3.12 and 3.15.0b4), numpy ≥ 2.0 (`np.bitwise_count`), scipy (`verify_B_all.py`, `verify_A_all.py`, `full_enum_omega0.py`)
- sympy (`verify_m3_curve.py`; optional cvxpy for `--sdp`)
- Standard library only: `verify_npa_rational.py` (fractions), `exact_prop2.py`, `thm7_check.py`
- Lean (core only; tested with 4.15.0, 4.24.0, 4.35.0 and 4.36.0-rc2): only for `OmegaBthm.lean` and `Omega3.lean`
- Standard library only: `ghz4_check.py`
- pycddlib ≥ 3 (tested with pycddlib-standalone 3.0.0): only for `cdd_crosscheck.py` and `full_enum_omega0.py` (`pip install pycddlib-standalone`; the wheel used here is the double-precision build, and the scripts verify the returned vertices in exact arithmetic)

Versions of the packages used in the last full run: numpy 2.5.4, scipy 1.18.1, sympy 1.14.0.

## Usage

Run all scripts from the same folder.

```bash
python3 verify_B_all.py --dims          # NS polytope dimension 26, process polytope dimension 37
python3 verify_B_all.py --selftest      # self-test on the pure parity boxes (14 items)
python3 verify_B_all.py --prop2         # Proposition 2: 744, 3352, 346, etc.
python3 verify_B_all.py --gynin         # GYNIN identity and the NS bound
python3 verify_B_all.py --research      # vertex enumeration, orbit decomposition, NPA verdicts (about 3 minutes)
python3 verify_B_all.py --boxes tripartite_nonlocal_omega0_boxes.json
python3 exact_prop2.py                  # integer-arithmetic check of Proposition 2 (expected: 744 347 346 3352 744)
python3 thm7_check.py                   # check of Theorem 7 of [8] (expected: 744 744 True 64 680 12)
python3 verify_npa_rational.py npa_certificates.json
lean OmegaBthm.lean                     # Lean check (1-2 minutes; prints the axioms used, and an error message appears only on failure)
python3 full_enum_omega0.py --cache raw.pkl   # all 53,856 NS vertices; expected: 64 local, 1224 with Omega=0, 46 orbits (about 12 minutes)
lean Omega3.lean                        # Lean check of Lemmas 2 and 3 of Paper C (10-15 minutes; prints the axioms used). It needs a local Lean installation; the web editor is too slow for it
python3 ghz4_check.py                   # GHZ4 box: W=0, Mermin value 8, local bound 4, 6 feedbacks
python3 verify_m3_curve.py              # Theorem 9 of Paper A (about 10 s; add --sdp for the SDP comparison)
python3 verify_A_all.py                 # all of Paper A (about 3 minutes; includes the --adv and --sec7 parts)
python3 verify_A_all.py --exact         # Theorems 1, 2, 3 only
python3 cdd_crosscheck.py --check-h     # check only the H-matrix, without cdd
python3 cdd_crosscheck.py               # 12 face-orbit representatives -> group expansion (expected: 130 -> 1224; about 25 minutes)
python3 cdd_crosscheck.py --all-faces   # all 346 faces, no group used; very slow (about 2 minutes per face, roughly 11 hours in total) and not part of the final check; `full_enum_omega0.py` is the group-free check that was run
```

`--research` generates `face_cache.pkl`, and `cdd_crosscheck.py` generates `cdd_vertices_exact.json` (neither is included in the repository).

## Notes

- The vertex enumeration (1224 vertices, 15 orbits) was cross-checked with pycddlib and by a full enumeration of all 53,856 NS vertices (`full_enum_omega0.py`: exactly 1224 have Ω=0), and the NPA inadmissibility verdicts were verified exactly with rational witnesses.
- The counts in Proposition 2 (744, minimum 0, 347 distinct w_g) were verified exactly by the integer arithmetic of `exact_prop2.py`. The distribution of maxima in Proposition 2 and the GYNIN numbers are floating-point LP results.
- The combinatorial counts of Proposition 2 and Appendix A.2 are also proved in Lean (`OmegaBthm.lean`). `native_decide` trusts the compiled execution (`Lean.ofReduceBool`).
- The NPA verdicts are valid only at the 19-monomial level.
- Paper C: the determinant tables (36 orbits, 13 types) are symbolic computations; `Omega3.lean` recomputes the sign patterns for all 4096 feedbacks and 49152 tuples without using the symmetry. The relay-stratum statistics in Paper C are numerical only and are not included here.
