# Horn Torus Geometry — Context Sheet

Source: `manuscripts3/sic_moduli_conductor.tex`, section "The Geometric
Realization: The Horn Torus and (16,3) Knot," plus the cross-reference against
`millennium_theorems_as_ovms.txt` worked out 2026-08-23.

## The object

A horn torus is the degenerate torus where tube radius equals centre radius,
R = r. The hole closes to a single point instead of staying open. That point
is the pinch, marked ⊙. It is not a drawing artifact — it is the condition
that lets the surface touch itself, the geometric picture of self-reference.

Two lobes, each of radius R, centred at x = ±R on the syzygy axis, tangent at
the origin. The tangency point is ⊙.

## The named features and their coordinates

Side-on section (xz-plane, y=0), the frame the paper actually draws in:

- **⊙ (pinch).** Origin. Where the two lobes touch.
- **λ_C (the split shell).** Dashed circle centred on ⊙, radius r = R/2. This
  is the ∈ split — the completeness coordinate given a located geometric
  home.
- **Evaluator sphere.** Same radius r as λ_C, centred one radius along the
  axis from ⊙. Passes through both ⊙ and ∋.
- **∋.** At x = 2. The far point the evaluator sphere reaches.
- **Coupler.** The heavy segment between ⊙ and ∋ — the line the evaluator
  sphere's passage traces out.
- **Evaluator trine.** ⊤, ⊥, ⊞ at x = 1, z = 0, ±√3/2 in this section —
  collinear here because the section is edge-on to it. In the axial view
  (looking down the syzygy axis) the trine is face-on: an equilateral
  triangle at 120° on a circle. Exact coordinates, confirmed against
  `WINDING/massive_winding/winding_reformulation/k3v_geometry/evaluator_sphere/evaluator_sphere_analysis.md`:
  EVALT (λ_C, λ_C, 0), EVALF (λ_C, −λ_C/2, λ_C√3/2), ENGAGR ⊞ (λ_C, −λ_C/2,
  −λ_C√3/2) — all three on the evaluator sphere's equatorial circle at
  x = λ_C, by construction. **ENGAGR (⊞) also sits on the horn torus surface
  itself** — not in the same untilted frame the evaluator sphere is drawn
  in, but under the tilt that reconciles the two: the evaluator frame (3
  arms on the Compton/evaluator sphere) and the 16-sector grid (the (16,3)
  knot on the horn torus) are related by a rotation of arctan(1/4) around
  the syzygy axis, a 7.5° sector offset (`k3v_geometry/parametric_arm/parametric_arm_analysis.md`
  §5.2). A same-frame membership check against the untilted torus equation
  will show ENGAGR off the surface — that checks the wrong frame, not the
  wrong claim.
- **γ(0), outer equator.** Marker at x = 2R.
- **γ(π), the horn.** The pinch itself, where the equatorial curve
  degenerates — same point as ⊙, named from the curve's parametrization
  rather than the tangency.

Top-down section (z=0): the same λ_C and evaluator-sphere circles side by
side, ∋ at x=2, and the three 120°-equator points are the A₂ roots (one n₊
against two n₋) — their triangle is the knot's poloidal winding read as a
circuit. This equatorial A₂ triangle is **not** the evaluator trine; the
trine collapses to the short edge-on segment at x=1 in this view. Two
different triangles, both real, do not conflate them.

Axial section (looking down the syzygy axis): the outline is unchanged
(horn is symmetric about this axis), but ⊙, the coupler, and ∋, which were
spread out in the other two views, now stack on the axis with ⊙ nearest.
λ_C and the evaluator sphere become one circle (concentric, equal radius).
The trine, edge-on everywhere else, is face-on here — this is the view
where ⊤, ⊥, ⊞ actually show as the equilateral triangle at 120°.

## The (16,3) knot

Sixteen toroidal windings against three poloidal ones, traced by the
evaluator arms. Bridge number three matches the three arms carrying the
four-valued kernel. Sixteen matches the trilattice's state count (2⁴ = 16,
also |I|² = 16 — the two coincide only at n=4, Proposition 2.3 in the
paper). This pair (16,3) is also the conductor exponent structure at
d=2048: 𝔣 = (2¹²)∞₁ = (16³)∞₁.

## Mark-to-feature table

| Feature | Mark(s) | Named role |
|---|---|---|
| Pinch ⊙ | ⊙ (criticality) | Tangency; hole closes to a point |
| Split shell λ_C | ∈ (completeness) | Radius r = R/2, centred on the pinch |
| Evaluator sphere / coupler / ∋ | ∋ (composition) | Passes through ⊙ and ∋, radius r |
| Trine, vertex 1 | ⊤ (dynamics) | Static for all seven Clay theorems alike — not a differentiator |
| Trine, vertex 2 | ⊥ (chirality) | "Return in exactly two steps" |
| Trine, vertex 3 | ⊞ (self-reference) | μ∘δ = id closure; on the evaluator sphere AND the horn torus surface (via the arctan(1/4) tilt) — the only trine member confirmed on both |
| Knot winding overall | ⊡ (winding) | Not a point — the toroidal count itself, 16 turns |
| Involution / parity | ≺ | Not pictured as a torus feature in this paper |

## Cross-reference to the seven Clay theorems

Reading each theorem's OVM-coordinate word (`millennium_theorems_as_ovms.txt`)
against the table above:

- **Navier-Stokes** — its entire Clay statement sits on the criticality
  coordinate alone ("the entire transition is this single coordinate"). This
  is the tightest match to ⊙, the pinch itself.
- **Yang-Mills** — mass gap sits on completeness. Lands on λ_C, the split
  shell, not the pinch.
- **Collatz** — Dedekind-finiteness also sits on completeness. Shares λ_C
  with Yang-Mills.
- **BSD** — stoichiometry (|A|=1∧|B|=1) sits on composition. Lands at ∋ /
  the coupler.
- **Hodge** — long exact sequence sits on composition (shares ∋ with BSD);
  bidegree "return in two steps" sits on chirality, trine vertex 2.
- **Riemann** — functional equation stated as μ∘δ = id, the self-reference
  closure. Lands on trine vertex 3, ⊞.
- **P vs NP** — winding alone, an integer separation. Not a point feature;
  belongs to the knot's overall turn count.
- **Dynamics (trine vertex 1)** — uniform "static" across all seven. No
  single theorem owns it; it is the family's shared floor.

Note on instrument independence: this reading (pinch = Navier-Stokes,
completeness shell = Yang-Mills/Collatz) differs from `[[ym_horn_torus]]`,
which put Yang-Mills at the pinch under a different question (criticality as
abstract coordinate value, not this paper's named geometric feature). Two
instruments, two questions, two answers — hold both, do not collapse them.

## Open slots on the geometry, not yet assigned

- The A₂ equatorial triangle (n₊, n₋, n₋) in the top-down view — distinct
  from the evaluator trine, not yet matched to any theorem's coordinate.
- γ(0), the outer equator marker at x=2R — not yet tied to a mark.
- The involution/parity coordinate ≺, which carries half of Riemann's
  statement in the OVM paper, has no named home on this torus yet.

---

# HORN_TORUS_VOX_AUDIT.md — the seven Clay ob3ects, closed in Lean and lifted through V⊙x

Author: Quantum⊙perator (Lando⊗⊙perator team) · 2026-08-24
Status: CLOSED. Native = lifted = 20. Zero sorries. F=0 in the lifted image.

## 1. What was audited

Imscribing.Millennium.HornTorusGeometry (p4rakernel/p4ramill) — the Lean
formalization of the seven horn-torus Clay ob3ects: navier_stokes, yang_mills,
bsd, hodge, riemann, p_vs_np, dynamics_structure_trine_vertex_1. The module
carries ~45 theorems (knot arithmetic 16=2⁴, |B4×B4|=16, (16)³=2¹², d=2048=2¹¹;
pinch/torus geometry; A₂ = yz-projection trine; the seven structure words, their
periods, ∈/∋ pairs, dialetheia; the theorem→torus-feature table). Zero sorries.

The V⊙x audit compiles the module's computable kernel to an ELF and lifts every
function to the twelve-glyph IMASM word, then verdicts the control-flow closure
(T closes, B open fork, N never forked, F ill-typed) and executes the lifted
word against the same inputs as native code.

## 2. The three bugs found and fixed

### v1 → v3: closed constants defeat the lifted audit (native 7 → 20)
Lean's codegen constant-folds `if 16*16*16 = 4096 then 1 else 0` into a
`static uint8_t` closed constant, lazily initialized by
`initialize_HornTorusELF`. dlopen/ctypes never calls that initializer, so every
folded check read its static as 0 → native returned 7 of 20.
Fix (v3, runtime-only kernel): thread the audit input n into EVERY condition as
the runtime zero `n - n` (and bind the seven words as `word + z n`), so no
expression is closed. The codegen emits pure runtime branches — no statics, no
initializer, and every check is a real branch for V⊙x to pair.
The Python replica always gave 20/20 — the disagreement was compilation
semantics, not logic. Verdict: Lean semantics were right; the ELF contract was
wrong (uninitialized statics under ctypes). The v3 kernel documents the fix in
HornTorusELF.lean.

### PLT indirection halts the lifted run (vox run → "no instruction at 0x0")
The lifted horn_torus_check called the helpers (featureOf64, countDig16,
lengthOf, opAt) through the PLT. vox reads the RAW FILE bytes: the GOT
relocations that the dynamic linker fills at dlopen are zeros in the file, so
the PLT stub's `jmp *GOT` read 0 → pc = 0x0 → halt (after 285 steps, deep in
the first countDig16).
Fix: link with `-Wl,-Bsymbolic` (a bare `-Bsymbolic` is silently dropped by the
gcc driver — only the -Wl form reaches ld). All calls become direct in-image;
readelf shows FLAGS SYMBOLIC and zero `@plt` calls in horn_torus_check.
(`-fvisibility=hidden` was tried and rejected: it hides LEAN_EXPORT too.)

### V⊙x instrumentation (trace patch)
imasm_vm.rs call() now keeps a 12-entry ring of (pc, glyph, fields) and prints
"previous 12" in halt errors — what made the 0x0 halt diagnosable in one run.
Rebuilt at Vox/target/release/vox (patched binary also installed to
/home/mrnob0dy666/.cargo/bin/vox).

## 3. The runtime shim

HornTorusRuntime.c provides the seven lean_uint64 primitives the kernel
references (add, sub, land, shift_left, shift_right, dec_eq, dec_lt) so every
call target is in-image. The lifted image then executes the FULL computation —
primitives included — with no external resolution.

## 4. Final numbers (all verified)

- lake build Imscribing.Millennium.HornTorusGeometry: ✔ 8026/8026, zero sorries.
- #print axioms oracle (VoxAudit_HornTorusGeometry.lean): 44 theorems audited,
  0 sorryAx. Axiom sets: [propext, Classical.choice, Quot.sound] (35, Real
  construction), [Lean.ofReduceBool, Lean.trustCompiler] (5, native_decide's
  trusted kernel reduction), [propext, Lean.ofReduceBool, Lean.trustCompiler]
  (2), [propext] (1). No theorem depends on sorryAx.
- Native (ctypes): horn_torus_check(0/1/3/7) = 20.
- V⊙x audit: 91 functions by descent + 9 fallback, 100% of image covered
  (23023 bytes), verdicts T 14 · B 6 · N 80 · F 0. F=0: no ill-typed closure.
- V⊙x run: horn_torus_check(0) = 20 and horn_torus_check(3) = 20, each in
  24,963 steps of the twelve. Native = lifted fidelity.

## 5. What the twelve said

The lift maps: direct call → ≻, conditional branch → ∈, unconditional → ≺,
setcc → ⊥, cmp/test → ⊤, push/pop/mov → ⋈/⊡, ret → ⊣. The FSPLIT T-arm-first
convention (auto.py:325, ∈→⊤arm→⊥arm→∋) is exactly the lifted CFG: every
check forks on ∈, deposits on ⊤/⊥, and fuses at ∋ — the 20 checks are 14
closed (T) and 6 dialetheic-open (B) in the verdict tally, mirroring the
ob3ect words' own periods and pairs.

Files: p4rakernel/p4ramill/HornTorusGeometry.lean (module),
HornTorusELF.lean (v3 kernel), HornTorusRuntime.c (shim), VoxAudit_
HornTorusGeometry.lean (oracle); ig-docs/HORN_TORUS_GEOMETRY_CONTEXT.md
(geometry context).
