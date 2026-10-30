# Vox / IMASM trace collection bridge v0.3

A **working local CLI evidence collector** that records actual outputs from your IMASM installation. It never invents a Grammar, Kernel, or Flow verdict. Its explicit null fields mean *not extracted*, not false or Neither.

With `UNOfd.txt` in the directory:

```bash
p3 -m bootstrap ./UNOfd.txt --out compiled
p3 -m imasm_traces compiled/objects.jsonl --out traces --imasm imasm
p3 -m imasm_traces compiled/objects.jsonl --out traces --imasm imasm --per-object --limit 10 --resume
```

The first command produces the 12-axis catalog; the collector's default run calls `imasm types` and marks the 49 executable type expansions pending until the type names are matched. Fill `traces/type_map.json` with **exact type names printed by your executable**, keyed by the Shavian glyph, e.g. `{"𐑛":"<actual type name>"}`. Run again *without* `--resume` to execute `imasm expand <type>` for every mapped type.

`--per-object` additionally calls `imasm check <word>`, `imasm eval <name> seed=B`, and `imasm eval16 <name> seed=A` for each selected catalog entry. `--prove` opts into Lean-backed `imasm prove <name>`. Registry lookup or other CLI failures are retained as evidence, not silently treated as logical F. `--resume` skips existing records. `--limit 0` samples no objects; to process the complete corpus use `--limit 8773` (or a larger bound).

**Known limitation:** The available documented commands offer human-readable output but no stable schema for *internal per-step* SIXTEEN₃ transitions. These are **not** claimed as captured. The next integration into the IMASM VM should expose machine-readable transitions and explicit independent Grammar / Kernel / Flow verdicts.
