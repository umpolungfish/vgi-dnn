# The Imscriber's Guide to IMASM

*Streamlined reference · October 2026*

**What it is.** One continuous reading of IMASM: the twelve-opcode alphabet, the SIXTEEN\_3 trilattice carrier it runs on, and the statics and dynamics of words, graphs, verdicts, flow and composition.

**What it does.** It teaches everything needed to write, wire, judge, run, compose and survey IMASM programs. A decision expressed as a word and put to `check` receives a univocal structural verdict, and the same close condition judges every substrate.

**How to use it.** Read sections 1 to 5 to write and judge words, then section 6 onward for flow, composition and the wider machinery. Semantics track `ask_native/src/imasm.rs` and `imasm_core/src/check.rs`. Run `imasm ref` for the live rules; where this guide and a live tool disagree, the tool is right.

## 1. The Alphabet

A program is a directed graph of opcodes, not a line. There are twelve marks, one glyph each, fully symbolic so that no token collides with a verdict letter. The same twelve marks are the primitive alphabet, one per axis, in canonical slot order. A mark is read as an operation or as an axis according to where it stands.

| Glyph | Name | Axis | Meaning | Valence | Work |
| --- | --- | --- | --- | --- | --- |
| ⊢ | VINIT | Dimensionality | begin, source boundary | 0→1 | no |
| ⊣ | TANCH | Topology | terminal anchor, readout; out-port may stay open | 1→1 | no |
| ≻ | AFWD | Coupling | forward morphism | 1→1 | YES |
| ≺ | AREV | Parity | reverse morphism, the involution T↔F, t↔f | 1→1 | YES |
| ⋈ | CLINK | Fidelity | compose, link | 1→1 | YES |
| ⊤ | EVALT | Kinetics | evaluate the true arm, set T | 1→1 | YES |
| ∈ | FSPLIT | Cardinality | fork δ, the only brancher | 1→2 or 1→3 | no |
| ∋ | FFUSE | Composition | fuse μ, the only merger | 2→1 or 3→1 | no |
| ⊙ | IMSCRIB | Criticality | identity, self-reference; the neutral generator | 1→1 | no |
| ⊥ | EVALF | Chirality | evaluate the false arm, set F | 1→1 | YES |
| ⊞ | ENGAGR | Stoichiometry | hold paradox, set t and f | 1→1 | YES |
| ⊡ | IFIX | Winding | irreversible commit, fix | 1→1 | YES |

The Work column is the most-missed rule. ⊢ ⊣ ⊙ ∈ ∋ do not transform. An arm that carries only ⊙, or nothing, is an identity arm, and a closure over identity arms verifies nothing.

Words are tokens glued into one string, for example `⊢≻∈⊤⊙∋⊡⊣`. Space-separated full names (VINIT AFWD …) and the short forms VI TA AF AR CL IM FS FF ET EF EG IX parse identically, and δ and μ are aliases for ∈ and ∋. Brackets are a reading aid only and parse to nothing.

The alphabet is chosen so that no mark arrives already meaning something else. ⊢ and ⊣ are the opening and closing boundary turnstiles. ⊤ and ⊥ are the two poles they evaluate. ∈ and ∋ are membership out of one and into one. ⋈ is the join, ⊞ is the Belnap Both it holds, and ⊡ is the box that closes. ⊙ is imscribing itself, which is inclosure: a boundary around its own centre, denoting the act of denoting.

## 2. The Dyad

There is one dyad, ∈ and ∋, and it has as many arms as the carrier gives it. ∈ (δ) cuts the register into a partition, and ∋ (μ) joins the arms by union, so μ∘δ = id at either arity.

- **Two arms:** the truth part and the falsity part, `{T,t}` and `{F,f}`.
- **Three arms:** `{T}`, `{F}` and `{t,f}`, the truth cut taken inside the constructive block with the non-constructive pair held whole.

Three is what the gate set leaves, because ⊞ sets t and f together and the pair can only be entered as a block. The two-arm cut is not the three-arm cut with an arm dropped, since dropping the information arm would lose t and f. On the classical slice, where t and f are absent, the two cuts coincide because the information arm carries nothing. A fan of two is the same opcode with an empty arm, which is why there is one mark and not two.

One operator means one ancestry rule, one close condition and one engine.

## 3. From Word to Graph

A word is only the node list. The edges are supplied by the verb you build with:

```
chain <word>           wire head to tail, nothing reconnects        (β=0, one strand)
ring <word>            wire head to tail to head; fork/fuse not rejoined (β=1)
protocol <word>        wire so that ∈/∋ pairs reconnect: the way to CLOSE
bubble PRE:A:B:POST    ∈ → (A|B) → ∋ reconvergence, spelled out
star CORE:a:b:c        hub plus arms (3 or more)
comb BACKBONE:p arm:q arm   backbone plus grafts
wire N0 N1 … / i-j i-k …    free graph: node set / edge set
```

The same word with a different verb is a different graph and gets a different verdict: `chain ⊢∈⊤⊥∋⊣` and `protocol ⊢∈⊤⊥∋⊣` are not the same program. To close, use `protocol`, never a bare `ring`, and never close by looping back to ⊢, which is a source with in-arity 0. `protocol` wires as many arms as the fork's arity declares: one along the chain and the rest straight across.

### Which ∈ pairs with which ∋

Pairing is by ancestry over edges, not by text position and not by a fork-balance stack.

- A pair exists when the distinct in-arms of a ∋ (all of them, two or three) trace back to a common ∈. The fork was undone by the fuse, however it routed.
- Pairing is a property of the edges, so the same word wired two ways pairs differently. You cannot read pairing off the glyph string alone.
- A ∈ feeding a ∋ directly (an empty arm) still counts. In-edges are counted with multiplicity, and a ∈ is its own ancestor.
- Arms are the nodes strictly between ∈ and ∋: forward-reachable from the fork and backward-reachable from the fuse. That set is what gets checked for work.
- A ∋ may have several qualifying ∈ and pairs with the innermost, the candidate no other candidate descends from. A ∈ may close more than one ∋, but a ∋ closes with exactly one ∈.
- A program is fully closed when every ∈ and every ∋ participates in a pair. One dangler and the whole program is Open.

Neutral inflation is allowed: `⊢∈⊙⊙⊙∋⊣` is a valid three-arm reconnection with no work, and reads N (identity).

On a plain strand the pairing can be read by eye as nested brackets, `⊢⊙⋈[∈≻⊤≺⊞⊥∋]⊡⊡⊣`. Brackets are not input, the aid works for strands only, and it is not the pairing rule. Ancestry is.

## 4. The Close Condition and the Verdicts

A program closes if and only if both hold:

1. **Reconnection:** every brancher and every merger participates in an ancestry pair.
2. **Transformation:** at least one such pair carries a work opcode on its arms.

A bare cycle is not a closure, and β (loops) is never diagnostic. Split then fuse with nothing between is μ∘δ = id, which type-checks nothing. The close condition is μ∘δ over a transformed object: split, work, fuse.

| Verdict | Meaning | What to do |
| --- | --- | --- |
| T (closes) | μ∘δ closes over n transformed reconnections | proceed |
| N (identity) | ∈/∋ reconnect but no work between | put work on the arms |
| N (no fork) | no ∈/∋ dyad at all; alternatives never weighed | add a dyad |
| N (void) | no committed opcodes; nothing parsed | write a real word |
| B (open) | well-typed, but a ∈ or ∋ dangles | fuse it (∋) or commit one arm (⊡) |
| B (paradox held) | closes over a transformation and ⊞ is present: genuinely both | sound to hold; not a clean T; look again before committing |
| F (ill-typed) | grammar violated | revise |

B beats T: a word that closes but holds a paradox reports B, never T. F is exactly three errors: a non-brancher fanning out, a non-merger merging in, or any node exceeding its own arity. Nothing else is fatal.

Open valences are not errors. An arm that runs out of successors is a living, telechelic end, reported as reactive. ⊣ may end with its out-port open and ⊢ may start with its in-port open.

### Topology names

`classify` names a graph by invariants, not by fork balance. With β = E − V + C (independent loops):

| Name | Condition |
| --- | --- |
| trivial | no nodes |
| linear | β=0, no branch points: a single strand |
| star | β=0, branch points forming one contiguous hub, three or more arms |
| comb | β=0, branch points strung along a backbone: graft tree |
| ring | β=1, no branch or merge points: a single cycle |
| branched | β=1 with branch or merge points |
| network | β≥2, or more than one disconnected strand |

It reports V, E, β, the branch/merge/source/sink census, the arm count and the spectral radius ρ.

## 5. The Carrier and the Gates

The register is a SIXTEEN\_3 value: a subset of the four base values {T, F, t, f} (Shramko, Dunn and Takenaka's trilattice of constructive truth values). T is constructively proven, F constructively refuted, t acceptable and f rejectable. That gives sixteen states, from N = {} to A = {T,F,t,f}.

FOUR is not a second system. It is the classical slice {T,F} of the same carrier, with Belnap B = {T,F} and N = {}. One evaluator runs both: `eval` renders the slice (N/T/F/B) and `eval16` renders the full names. A value that touches t or f has left the slice and keeps its SIXTEEN\_3 name in either view. The register is one 4-bit register, not two independent FOURs and not three independent bits.

Three orderings each carry a meet and a join (`imasm16_3 algebra <op> A B`):

| Order | Definition | Meet / join |
| --- | --- | --- |
| ≤\_i information | x ⊆ y | ⊓ / ⊔ |
| ≤\_t truth | x∩{T,t} ⊆ y∩{T,t} and y∩{F,f} ⊆ x∩{F,f} | ∧ / ∨ |
| ≤\_c constructivity | x∩{T,F} ⊆ y∩{T,F} and y∩{t,f} ⊆ x∩{t,f} | △ / ▽ |

`algebra meet_t T t` reproduces the paper's worked example, T ∧ t = N.

Flow uses ≤\_i: shuttling only ever moves values up the information order. AREV is the trilattice negation and factors into two bit swaps, one per layer (T↔F and t↔f). The paper requires negation to preserve ≤\_i exactly, and swapping two bits preserves the size of the register where a flip would not. The two swaps are not independent operators: ≺ alone carries Tf to Ft, and on the classical slice it carries T to F by itself.

### The gates

Every opcode is a gate over the carrier. Inputs join by union (the ≤\_i join) before the gate acts, and the value leaving a gate rides every out-edge except where ∈ fans.

| Opcode | Gate |
| --- | --- |
| ⊢ VINIT | emits the seed (default B in the slice, A in full SIXTEEN\_3) |
| ∈ FSPLIT | fans the register's own partition onto its arms: at two, x∩{T,t} and x∩{F,f}; at three, x∩{T}, x∩{F} and x∩{t,f} |
| ⊤ EVALT | pass-gate: truth part |
| ⊥ EVALF | pass-gate: falsity part |
| ⊞ ENGAGR | sets the information layer (t and f) |
| ∋ FFUSE | union of the arms |
| ≺ AREV | the involution T↔F, t↔f; its own inverse; fixes B and N |
| ≻ ⋈ ⊙ | carry |
| ⊡ IFIX | carry and latch: the commit point |
| ⊣ TANCH | readout |

Every gate is monotone in ≤\_i, so evaluation is a Kleene iteration from all-N edge values that settles in bounded rounds on any graph, cycles included. A loop converges and cannot oscillate. The machine's registers are the edges, and ⊡ marks where value becomes commitment.

## 6. The Three Independent Verdicts

A program earns three judgments, none implying another.

1. **Grammar** (`define`): the composition laws hold. Only branchers branch, only mergers fuse, arities are respected.
2. **Kernel** (`prove`): the closure class goes to the live p4ramill kernel. A worked dyad (split, transform, fuse) proves green. A bare fork-fuse is an identity closure and returns N for the program. A dangling fork is Open.
3. **Flow** (`eval` / `eval16`): per dyad, does the fuse recover what the fork was fed?

The canonical protocol word `⊢∈≻∋⊤` (VINIT FSPLIT AFWD FFUSE EVALT) is lossless: B splits to (T,F) and fuses back to B, the operational split\_fuse\_id. The same shape with AREV on the truth arm closes in structure and fails in value. Fed Tf, it recovers Ff, which is not the identity, because the arm inverted what it carried. Only flow can see that.

Flowing a catalog entry expands its twelve glyph types into opcode motifs, and the per-dyad id / not-id sequence is its **flow signature**: a readout of the tuple in the dynamic register. Signatures discriminate between entries, and a lossy dyad inside a kernel-green closure is not presumed a defect. It can be the entry's chirality speaking in flow.

Discipline for any new program: define, prove, eval, and **speak the expected readout before running eval**. The gates are deterministic, so the true name of a topology includes its flow.

## 7. Composition and ChaosComposer

Programs interact end to valence. A living end is an unfilled port, a node below its arity, and these are the register ports of a program as a component. `imasm compose <new> <A> <B>` binds A's free out-ends to B's free in-ends, in node order, under three rules:

1. Composition consumes valences and never mints one.
2. The composite must re-satisfy the grammar, and an ill-typed binding is refused whole.
3. A program with no living ends does not compose. It is a finished loop.

Composites persist as wire specs, rebuild through the identical parse, and stay composable while ends remain. Composition is therefore well-founded: each step consumes ends, ends are finite, and the fixed point is a program with none. Lines are the composable phase and loops are results.

`chaos` takes a set of up to six programs. A set has no order, so the composer walks every ordering, folds each through the binding law, and reports which arrangements are admitted, which are refused and by which missing end, and how orderings collapse into **outcome classes** keyed by topology, closure, flow, readout and remaining living ends. The collapse is the measurement: the space of possibilities is smaller than the space of orderings, and the ratio says how constrained the set is. Refusals are results of equal rank, and a set of finished loops refuses every ordering.

What walking these spaces has shown:

- Closure and flow-perfection can be in tension. In one measured set, exactly one arrangement of twenty-four closed and a different one was flow-perfect, and none was both.
- The tension is not a law of the alphabet. Sampled machine-shaped sets include spaces with closures, spaces with perfect flow, and a minority with both.
- Perfect machines exist: one set's entire possibility space is terminal objects, every arrangement closed, flow-perfect and ending with zero living ends. One is minted as `perfect_machine`, kernel green, every fuse recovering B.
- The mechanism is exact. AREV fixes B and N, so it costs nothing while an involution-symmetric value flows, and it mints the missing pole when fed a projected value (AREV of T is F). AREV is lossless exactly on involution-symmetric values, and the tension appears precisely where a projection feeds an inversion. Both conditions can be read off the program before it runs.

## 8. Op-Opcodes: Operators on Words

A node-opcode is a symbol inside a word, and a verb turns a word into a graph. An op-opcode is a third thing: a map that acts on the whole composition and returns another composition. It is not a node, and appending its name as a token does nothing.

**ROTAT** (↺ shifts k→k−1, ↻ shifts k→k+1) is the cyclic shift of a ring, the ring automorphism. ρ and every spectral invariant are ROTAT-invariant, which is the signal that it is a symmetry and not that it is inert. On one ring it changes nothing measurable. Between two rings being bound it sets their relative phase, the degree of freedom that seats a junction two same-handed rings cannot close on their own. ROTAT is the Weyl-Heisenberg shift X on ℤ/dℤ, the SIC displacement D\_{a,b} carries ROTAT^a, and the balanced tiling of a period-n cycle is unique up to ROTAT. Its relative is reflection, the half-period involution ROTAT^{d/2}. The class is open, with ROTAT the first named member.

## 9. The Types: the Strange Loop

Every type the Grammar writes with is itself a full IMASM program. There are 49 Shavian type names across the twelve axes, with 4, 5, 4, 5, 3, 5, 3, 4, 5, 4, 3 and 4 values in slot order (the appendix lists them). `imasm types` lists them, and `imasm expand <type>` pulls a type's ordered bootstrap opcodes and fork/fuse pairs and returns the reconstructed graph with its per-step domain actions. The types judge programs, and the types are programs judged by the same close condition. The strange loop is the autopoietic floor of the system.

The loop is walkable in both directions. A tuple becomes a word by writing each of its twelve types as that type's own program and concatenating in canonical axis order. Each type program carries one ∈ and one ∋. `imasm cycle` runs the return leg, reading a word back into the types that could have written it, and reports per axis whether it came back exactly, ambiguously or not at all. `imasm cycle tuple=⟨…⟩` runs one tuple, catalog entry or not. Write a tuple, read it back, write again, and the word returns identical. Position is load-bearing: a program is read as the type of the axis it occupies, so a concatenated word is parsed axis by axis.

## 10. Learning and Promotion

**The excription loop.** `imasm learn '<word>' [rounds=N] [breadth=K]` runs verification as imscription on a model. A word carrying SIXTEEN\_3 tokens selects that grammar, and any other word the classical slice; one loop, one knowledge file. Each round takes the neighborhood of the current word (single-opcode substitutions, insertions and deletions, with the boundary turnstiles held fixed and only grammar-valid candidates admitted). For each candidate the model excribes the word into a guess: one concrete object in a real domain whose structure is the word. The excriber is grounded in the primitives, since each opcode is tagged with the axis it rides and the primitive types are loaded as exemplars. The guess is blinded, a guess that parrots a spent example or an already-taken guess is refused, and a second reading imscribes the guess alone back into a word. The residual is the edit distance between the word sent and the word recovered. Residual zero means excription and imscription compose to the identity on that word, with the guessed object as the fixed point between them. Where the residual is not zero, the aligned confusions are counted into `ob3ects/imasm_knowledge.json`, distilled into lessons, and fed into the next prompt. The walk moves to the highest-residual candidate, and each run appends its mean residual to the accuracy trajectory.

**Promotion paths.** Two twelve-primitive tuples that differ have a promotion path: a walk that changes one primitive at a time with every intermediate well-formed. `imasm path '<A>' '<B>'` finds the same thing for words: the shortest sequence of single-opcode edits in which every waypoint is itself a grammar-valid program. It is A\* over the graph of valid programs, with raw edit distance as the admissible heuristic. The tool reports the verdict walk and whether the path is a genuine promotion (the verdict changes) or a lateral edit. Substituting an inert ⊙ for a working opcode promotes an identity closure (N) to a real closure (T) in one step.

## 11. Where the Engine Lives

The close-condition engine (graph, ancestry pairing, ClosureState and the gate `word_verdict`) lives in `imasm_core/src/check.rs`, no\_std plus alloc, so the verdict can be spoken wherever the kernel runs. The three-arm verdict lives beside it in `imasm_core/src/imasm16_3.rs`. `ask_native/src/imasm.rs` keeps presentation, spectral analysis and the tool registry as extensions over the core graph, and is the tool that answers `imasm check`.

The same engine runs on bare metal: mOMonadOS carries `imasm_core`, and its on-board vita mouth speaks certified turns whose words are gated by `check::word_verdict`, or by the tri-ancestral verdict when the word carries three-arm tokens. The kernel that judges your reasoning at the prompt is the kernel that judges the trunk's speech on the machine with no OS beneath it.

`imasm_composer.html` is a fully local, self-contained page. `imasm export` writes the manifest (every registered tool with its graph, ports and closure state, plus the opcode table), and the page renders it as a node canvas with a palette, drag, and click-to-bind. It judges nothing and only speaks command lines (`compose`, `chaos`, `wire`, `eval`, `prove`) for the kernel to judge.

Engineering guarantees:

- Registry writes are atomic (temp file, then rename), so a reader sees the old registry or the new one, never a torn one.
- A registry file that exists but does not parse is preserved as `.corrupt` evidence, never silently replaced by emptiness.
- Kernel verdicts name the spec they bind, so a refused re-define can never borrow an older program's green.
- An identity closure answers N for the program, in words, with the repair named: put a transforming opcode between the fork and the fuse.

## 12. Verb Index

```
imasm chain / ring / protocol / bubble / star / comb / wire   build (section 3)
imasm classify <word>         topology, invariants and μ∘δ state
imasm check <word>            type-check your own reasoning: T/N/B/F
imasm prove <name|word>       take the verdict to the real Lean kernel
imasm eval   <name|entry|word> [seed=N|T|F|B]   flow, FOUR readout
imasm eval16 <name|entry|word> [seed=A|Tf|…]    flow, full SIXTEEN_3
imasm compose <new> <A> <B>   bind living ends, register
imasm chaos <A> <B> [C… up to 6]   the possibility state space
imasm learn <word> [rounds=N] [breadth=K]   the excription loop
imasm path <A> <B>            promotion path between programs
imasm export                  manifest for the surface
imasm ref                     the live rules, authoritative over this file
imasm types / expand <type>   the 49 types, each itself a program
imasm cycle [n=<count>]       primitives → word → primitives, measured over the catalog
imasm cycle tuple=⟨…⟩         the same cycle on one tuple, per axis
imasm define <n> <op> <args…> build a kernel-constrained tool
imasm run <n> / imasm tools   invoke / list forged tools

imasm16_3 check <glyph_word>  three-arm type-check: T/N/B/F, same reading
imasm16_3 algebra <op> A B    leq_i|leq_t|leq_c|meet_t|join_t|meet_c|join_c
imasm16_3 ref                 the live opcode table
```

## 13. Canonical Words and Pitfalls

Express a decision as a word: ⊢ begin · ⊙ self-identify · ≻ move · ∈ weigh options · ⊤ and ⊥ the true and false arms · ∋ resolve · ⊞ hold paradox · ⊡ commit · ⊣ close. Then `imasm check` it. Only branchers branch and only mergers fuse. Put real work on the arms or it is N. Use `protocol` to close.

| Word | Reading |
| --- | --- |
| `⊢∈≻⊤∋⊣` | lossless protocol: closes in shape and value, recovers B → T. Wired at three arms it is the same word, and on the FOUR slice it flows identically, the third arm carrying nothing |
| `⊢∈⊙∋⊣` | identity: reconnects, no work, μ∘δ = id → N |
| `⊢∈⊙⊙⊙∋⊣` | three-arm identity: same reading → N |
| `⊢∈⊞≻∋⊣` | closes over work and holds paradox → B (paradox held) |
| `⊢≻∈⊤⊥∋⊡⊣` | three-arm word driving the register toward full A (⊤ and ⊥ on the arms), fused and latched at ⊡ → T |

Pitfalls, all load-bearing:

- Reading a word as a line. It is a graph, and the verb supplies the edges.
- Pairing ∈/∋ by counting or nearest match. Pairing is ancestry over edges.
- Putting only ⊙ between ∈ and ∋ and expecting T. That is N (identity).
- Expecting T from a word that holds a paradox with ⊞. That is B.
- Treating an open arm as a failure. It is a living end, reported and not fatal.
- Closing a loop back to ⊢. It has in-arity 0 and cannot be a target.
- Pasting a bracketed word into the tool. Brackets parse to nothing, so the word reports N (void).

## Appendix: Type Programs

The 49 types in canonical axis order, with each axis's values in constructor order. Each type is the IMASM program shown.

| Axis | Value | Program |
| --- | --- | --- |
| ⊢ | VINIT | Dimensionality | begin, source boundary | 0→1 | no |
| ⊢ |  𐑛 | `⊢∈≻⊤≺⊥⊞∋⋈⊙⊡⊣` |
| ⊢ |  𐑨 | `⊢≻∈⊤⊥⊞⋈⊙≺∋⊡⊣` |
| ⊢ |  𐑼 | `⊢∈⊤≻⋈⊙⊡⋈⊡⋈⊡≺⊥∋⊞⊣` |
| ⊢ |  𐑦 | `⊢⊙∈⊤≻⊥≺⊞∋⋈⊡⊣` |
| ⊣ | TANCH | Topology | terminal anchor, readout; out-port may stay open | 1→1 | no |
| ⊣ |  𐑡 | `⊢∈≻⊤≺⊥⊞∋⋈⊙⊡⊣` |
| ⊣ |  𐑰 | `⊢⊣∈≻⋈⊤≺⊥∋⊙⊡⊞⋈⊣` |
| ⊣ |  𐑥 | `⊢∈≻⊤≺⊥⋈⊞⊙∋⊡⊣` |
| ⊣ |  𐑶 | `⊢∈≻⊤≺⊥⋈⊞∋⊙⊡⊣` |
| ⊣ |  𐑸 | `⊢∈≻⊤⋈≺⊥∋⊞⊙⊡⊣` |
| ≻ | AFWD | Coupling | forward morphism | 1→1 | YES |
| ≻ |  𐑩 | `⊢≻∈⊤≻⊞⊥≺∋⋈⊙⊡⊣` |
| ≻ |  𐑑 | `⊢∈≻⊤⋈⊙≺⊥⊞∋⊡⊣` |
| ≻ |  𐑽 | `⊢⊣≻∈≻⊤⋈⊙≻⊥⋈⊙⊞≺∋⊡⊣` |
| ≻ |  𐑾 | `⊢∈≻⊤⋈≺⊥⊞∋⊡⊙⋈⊣` |
| ≺ | AREV | Parity | reverse morphism, the involution T↔F, t↔f | 1→1 | YES |
| ≺ |  𐑗 | `⊢∈≻⊤⋈≺⊥⊞⋈⊙⊡∋⊣` |
| ≺ |  𐑿 | `⊢∈≻⊤⋈≺⊥∋⊞⊙⊡⊣` |
| ≺ |  𐑬 | `⊢≻∈⊤⋈⊡⊥≺⊞∋⊙⊣` |
| ≺ |  𐑯 | `⊢∈≻⊤≺⊥⋈⊞⊙⊡∋⊣` |
| ≺ |  𐑹 | `⊢≻∈⊤⊥⊞∋⋈⊙⊡⊣` |
| ⋈ | CLINK | Fidelity | compose, link | 1→1 | YES |
| ⋈ |  𐑱 | `⊢≻∈⊤⊥⊞∋⋈⊙≺⊡⊣` |
| ⋈ |  𐑞 | `⊢∈≻⊤≺⊥⋈⊞⊙⊡∋⊣` |
| ⋈ |  𐑐 | `⊢∈≻⊤≺⊥⊞⋈⊙∋⊡⊣` |
| ⊤ | EVALT | Kinetics | evaluate the true arm, set T | 1→1 | YES |
| ⊤ |  𐑘 | `⊢≻∈⊤⊥⊞⋈⊙∋⊡≺⊣` |
| ⊤ |  𐑤 | `⊢∈≻⊤⋈⊙≺⊥⊞∋⊡⊣` |
| ⊤ |  𐑧 | `⊢∈≻⊤≺⊥⊞⋈⊙⊡∋⊣` |
| ⊤ |  𐑪 | `⊢∈≻⊤⋈≺⊥⋈⊞∋⊙⊡⊣` |
| ⊤ |  𐑺 | `⊢∈≻⊤≺⊥⊞⋈⊙∋⊡⊣` |
| ∈ | FSPLIT | Cardinality | fork δ, the only brancher | 1→2 or 1→3 | no |
| ∈ |  𐑚 | `⊢∈≻⊤⋈≺⊥⋈⊞∋⊙⊡⊣` |
| ∈ |  𐑔 | `⊢∈≻⊤⋈⊙≺⊥⊞⊡∋⊣` |
| ∈ |  𐑲 | `⊢≻⋈∈⊤⊥⊞∋≺⊙⊡⊣` |
| ∋ | FFUSE | Composition | fuse μ, the only merger | 2→1 or 3→1 | no |
| ∋ |  𐑝 | `⊢∈≻⊤⋈≺⊥⊞∋⊙⊡⊣` |
| ∋ |  𐑜 | `⊢≻∈⊤⋈⊥≺⋈∋⊞⊙⊡⊣` |
| ∋ |  𐑠 | `⊢∈≻⊤⋈≺⊥⊞⊙⊡∋⊣` |
| ∋ |  𐑵 | `⊢⊙∈≻⊤≺⊥⋈⊞∋⊡⊣` |
| ⊙ | IMSCRIB | Criticality | identity, self-reference; the neutral generator | 1→1 | no |
| ⊙ |  𐑢 | `⊢≻⋈⊙∈⊤⊥⊞∋⊡⊣` |
| ⊙ |  ⊙ | `⊢∈≻⊤≺⊥⋈⊙⊞∋⊡⊣` |
| ⊙ |  𐑮 | `⊢∈≻⊤≺⊥⊞⋈⊙⊡∋⊣` |
| ⊙ |  𐑻 | `⊢⊙∈≻⊤≺⊥⋈⊞∋⊡⊣` |
| ⊙ | -ha 𐑣 | `⊢≻∈⊤⋈≺⊥⋈⊞∋⊙⊡⋈⊣` |
| ⊥ | EVALF | Chirality | evaluate the false arm, set F | 1→1 | YES |
| ⊥ |  𐑓 | `⊢∈≻⊤≺⊥⊞∋⋈⊙⊡⊣` |
| ⊥ |  𐑒 | `⊢∈≻⊤⋈⊙⊡≺⊥⊞∋⊣` |
| ⊥ |  𐑖 | `⊢≻∈⊤⊥⊞≺⋈⊙⊡∋⊣` |
| ⊥ |  𐑫 | `⊢≻∈⊤⊥⊞⋈⊙⊡∋⊣` |
| ⊞ | ENGAGR | Stoichiometry | hold paradox, set t and f | 1→1 | YES |
| ⊞ |  𐑙 | `⊢≻≺⋈∈⊤⊥⊞∋⊙⊡⊣` |
| ⊞ |  𐑕 | `⊢≻⋈∈⊤⊥⊞∋⊙⊡≺⋈⊣` |
| ⊞ |  𐑳 | `⊢∈≻⊤≺⊥⋈⊙⊞∋⊡⊣` |
| ⊡ | IFIX | Winding | irreversible commit, fix | 1→1 | YES |
| ⊡ |  𐑷 | `⊢∈≻⊤≺⊥⋈⊞⊙⊡∋⊣` |
| ⊡ |  𐑴 | `⊢∈≻⊤≺⊥⊞∋⋈⊙⊡⊣` |
| ⊡ |  𐑭 | `⊢∈≻⊤≺⊥⋈⊙⊞∋⊡⊣` |
| ⊡ |  𐑟 | `⊢∈≻⊤⋈⊙≺⊥⊞∋⊡⊣` |

## Related Documents

- `md/primitive_imasm_cycle.md`: the primitives ↔ IMASM cycle, both legs.
- `md/chaos_first_measurements.md`: the chaos measurements, census and the perfect-machine mechanism in full.
- `md/verification_as_imscription.md`: the theory of verification as imscription (the Dual-Link SIC Witness-Vessel).
- `../ob3ect/READING_GUIDE.md` §3: the fixed glyph vocabulary the alphabet references.
