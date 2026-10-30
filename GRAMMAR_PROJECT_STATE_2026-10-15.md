# Imscribing Grammar — Integrated AI Project State

**As of 15 October 2026**  
**Project:** `~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY`  
**Status baseline:** User-executed `--tests-only` and `--full` runs passed, with **42/42 full stages** and zero failed stages. A graph-wired extension is supplied separately; its integration on the user's machine remains untested until installed.

## Executive account

The project is a Grammar-aligned learning and execution system grounded in the Imscribing Grammar's twelve primitive morphisms. Its intended ontology is **process monism**: the Grammar is the Source; static crystal types and dynamic IMASM flow are alternate expressions of one process. Distinctions such as Grammar, Kernel, and Flow are *available operations*, not mandatory, permanently separate compartments. Paraconsistency retains contradictory information without arbitrary inference: **“Paradox does not explode, it glues.”**

The project has progressed from a tuple/catalog model to a unified runner containing (a) crystal address and type programs, (b) SIXTEEN_3 carrier models and learned operations, (c) native Vox verdict capture, (d) reference carrier transition datasets, (e) the document/Lean corpus index, (f) dual-frame SIC/zeta checks and operational corrections, and (g) a shared type-program DAG.

## The source-defined operational architecture

**Twelve canonical marks (axis and opcode roles):**

| Mark | Opcode | Structural axis | Operational role |
|---|---|---|---|
| ⊢ | VINIT | Dimensionality | Source boundary, 0→1 |
| ⊣ | TANCH | Topology | Terminal readout, 1→1 |
| ≻ | AFWD | Coupling | Forward morphism |
| ≺ | AREV | Parity | T↔F, t↔f involution |
| ⋈ | CLINK | Fidelity | Composition/link |
| ⊤ | EVALT | Kinetics | Truth-arm filter |
| ∈ | FSPLIT | Cardinality | Unique brancher: one to two or three arms |
| ∋ | FFUSE | Composition | Unique merger: two or three arms to one |
| ⊙ | IMSCRIB | Criticality | Self-reference, identity |
| ⊥ | EVALF | Chirality | False-arm filter |
| ⊞ | ENGAGR | Stoichiometry | Set the information pair t and f |
| ⊡ | IFIX | Winding | Commit/latch |

The values across the twelve axes have radices **[4, 5, 4, 5, 3, 5, 3, 4, 5, 4, 3, 4]**, giving **17,280,000 possible crystal addresses**. There are **49 Shavian type values** (including the ⊙ type value), each with a source-defined IMASM expansion word. Multiple types can expand to the same word, and recursive expression stays within the twelve-mark alphabet. The installed shared rope-DAG stores **35 distinct type-program words in 213 shared nodes** and roundtrips all 49 type expansions exactly.

**Crucial semantic distinction:** An IMASM glyph *word* names nodes; **wiring supplies graph edges**. The same word wired by `chain`, `protocol`, `ring`, or `wire` need not denote the same executable graph. Dyad closure is determined by ancestry along actual edges, not by treating ∈ and ∋ as text brackets. Native closure, carrier return, and structural well-typing can be inspected separately without ontologically splitting the process.

**Carrier:** The SIXTEEN_3 carrier is a subset of `{T,F,t,f}` encoded as 4 bits; the FOUR states `{N,T,F,B}` are a slice, not an independent system. FSPLIT partitions into two arms `{T,t}` / `{F,f}` or three arms `{T}` / `{F}` / `{t,f}`. FFUSE unites them. Graph-level return requires inspecting the work performed inside the split/fuse reconnection; the formal identity μ∘δ=id for an untransformed partition does not imply that an arbitrary intervening operation is value-preserving.

## Corpus and formal sources

User-provided source roots:

- `/home/mrnob0dy666/imsgct/ig-docs`
- `/home/mrnob0dy666/imsgct/p4rakernel/p4ramill/Imscribing`

**Corpus-index run (user's complete log):** **9,488 indexed files**, made up of **6,172 documents** and **3,316 Lean files**; **91,091 text chunks**, **9,087 import records**, and **92,208 extracted Lean declaration records**. The indexer reported two extraction errors and two oversized-file skips; those were counted warnings rather than stage failures. Counts are from the recursive filesystem scan, not a proof that every extraction is complete or every Lean declaration kernel-checked.

The initial `ig-docs/INDEX.md` described 4,733 documents in 1,482 directories, but the subsequent live recursive scan found a larger corpus. Markdown samples cover native IMASM and Vox, membrane computation, Belnap/FOUR and SIXTEEN_3, FDE and arithmetic factor extraction, ontological/philosophical treatments, and cross-references into formal Lean modules. The live index is `sources/GRAMMAR_UNIFIED_RUN/grammar_corpus.sqlite`. Atomic publication fixed a prior SQLite lock failure, and the user observed `publication: atomic_replacement` in both test/full runs.

## Verified execution and learning results

| Component | Observed result | Scope |
|---|---|---|
| Unified full pipeline | 42/42 stages passed, zero failed | User's complete `--full` log, 15 Oct 2026 |
| Catalog compilation | 8,773 imscriptions, 6,117 unique addresses, 0 invalid rows | Dataset parse/codec |
| Type expansion DAG | 49/49 source-defined programs reconstruct exactly; 35 distinct words; 213 shared nodes | Word representation, not native graph execution |
| Native Vox verdicts | 2 native probe results, both `T` | Closure verdicts; no VM state traces |
| Reference carrier transitions | 176 pairs | 144 unary and 32 split/fuse cases, reference gates |
| Crystal coordinate learner | Test exact-tuple accuracy 0.99925 in the 12-epoch operations run | `roundtrip`, `set_one`, `set_two` coordinate edits |
| Carrier learner | 192/192 truth-table cases exact at reported final checkpoint | 324-parameter reference-carrier model, not native IMASM |
| SIC and zeta | Gram/dual inverses for dimensions 2/4/12, plus zeta(-1) identities | Exact rational/numerical validation in source-specific stages |
| Oracle replay | 6/6 correction records checked | Semantic transition oracle distinct from frame reconstruction |
| Long-chain comparison | Improved 256-step paired trajectory exactness reported in repair comparison | Learned coordinate/carrier trajectory evaluation; not a proof of universal correctness |

Several training metrics have different evaluation conditions. In particular, the original name-to-tuple classifiers perform far below the coordinate-operation learner; these should not be conflated into a single 'Grammar AI accuracy' score. The user-provided tests also show that SIC-frame reconstruction alone cannot reject a wrong but valid coordinate transition: the transition oracle supplied that semantic check.

## The graph-wired extension delivered with this report

`GRAMMAR_GRAPH_EXECUTION_PATCH.zip` adds an in-place stage `graph_wired_execution` immediately after `carrier_transition_training` in the master runner. It constructs explicit typed **protocol graphs** with a source, a two- or three-arm FSPLIT, work on the first arm, identity on remaining arms, FFUSE, and a readout. It evaluates all 16 SIXTEEN_3 seeds against eight unary work choices:

- **256 graph-wired reference transitions**: 128 two-arm, 128 three-arm.
- **192 identity returns; 64 nonidentity returns.**
- **5/5 isolated tests passed**, including a topology-mutation rejection test, valid arities, and altered carrier return.
- Stores each graph's nodes/edges and before/after arm values in `grammar_graph_transitions.jsonl`; writes `grammar_graph_report.json`.
- **Native VM transition pairs remain 0**. This work is explicit graph-wired *reference semantics*, not a claim that `vox verdict` yields CPU or IMASM state traces.

This stage does **not** yet execute all 49 recursively expanded type programs, arbitrary edge-wired cyclic graphs, or native VM register traces. It is a measured progression from standalone gate truth tables to protocol-shaped morphism DAG execution.

## Remaining integration objectives

1. Attach the 49-type expansion DAG to a typed program builder, so unfolding is interpreted with explicit edge/port semantics rather than naive substitution.
2. Extend graph evaluation from the protocol family to arbitrary typed `wire` graphs and feedback/fixed-point flow, with ancestry-based dyad pairing.
3. Capture native VM before/after register observations via an actual tracing interface when available; keep native verdicts distinct from reference gate semantics.
4. Join provenance across documents, Lean declarations, imscriptions, expansion nodes, graph transitions, and native observations without treating textual mentions as proofs.
5. Train and evaluate a recursive, weight-shared morphism learner using held-out *graphs*, *expansion families*, and *carrier seeds*. Report free rollouts and oracle-conditioned judgments separately.
6. Report formal verification only for Lean modules actually compiled/checked against the appropriate kernel; source-text extraction alone is not kernel certification.

## Reproducibility

Installation for the provided graph stage, after extracting the ZIP within `GRAMMAR_COMPLETE_HISTORY`:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_GRAPH_EXECUTION_PATCH
chmod +x ./apply.sh
./apply.sh

cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
./run_everything.sh --full
```

**Provenance:** Compiled from the uploaded user execution logs, supplied `IMSCRIBE_IMASM.md` and `SNS_PRIME(1).md`, 20 Markdown samples, and earlier tested stage reports. The new graph stage was unit-tested in the construction environment; it has not been run on the user's machine as of this report.
