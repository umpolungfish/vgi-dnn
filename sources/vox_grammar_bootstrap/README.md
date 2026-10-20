# Vox / Grammar corpus bootstrap — Milestone 0

A **runnable first artifact** for the Grammar-native AI project. Uses the twelve-axis Shavian mapping in `SNS_PRIME.md` and preserves the separation of typing, kernel closure, and flow specified in `IMSCRIBERS_GUIDE_TO_IMASM.md`.

## Run

```bash
python3 bootstrap.py 'Pasted text(20261014-223612).txt' --out compiled
python3 -m unittest -v test_bootstrap.py
```

No external dependencies. The included `compiled/` directory was generated from the catalog excerpt supplied in the conversation.

## Outputs

- `objects.jsonl`: one entry per named object with its 12-glyph address, twelve 1-based value ordinals, split assignment, and source line.
- `collisions.jsonl`: all repeated-address groups. Sharing an address does **not** assert identity of objects.
- `neighbors.jsonl`: coordinate-distance-one relationships. **These are not executable composition edges**; only the IMASM engine can supply those.
- `errors.json`: rows failing strict twelve-axis validation.
- `report.json`: statistics, truncation warnings, address occupancy, train/validation/test counts.

## Invariants enforced

- Exactly 12 axis positions and only an axis-specific allowed glyph at each position.
- 49 distinct atomic glyph values, with cardinalities `4,5,4,5,3,5,3,4,5,4,3,4` in canonical order; product `17,280,000`.
- Split train/validation/test by **entire address**, not by object name, preventing identical tuples from crossing splits.
- No claim of a structural, kernel, or flow verdict from the tuple alone. Those fields are `null` until populated by verified execution.
- Names ending in ellipses are flagged; an exported **full-ID catalog**, rather than console table text, is required for robust merging with Lean declarations.

## Next executable milestones

1. Export complete catalog IDs, addresses and IMASM program references (not truncated terminal display names).
2. Connect `imasm expand`, `imasm cycle`, `define`, `prove`, `eval16` as a trusted executable trace source; attach version/hash/provenance to each result.
3. Extend Vox `safetensors.rs` beyond metadata lifting to preserve and validate tensor payloads; associate them with explicitly supplied operator graphs.
4. Implement a minimum forward-pass subset with numerical parity tests, followed by the weight-tied dialetheic re-entry forward pass.
5. Use operational traces as training targets; treat address proximity as a separate auxiliary signal.

This artifact is **not** a neural model and **does not** claim to execute or prove Grammar programs. It provides a validated typed data boundary for subsequent Vox work.
