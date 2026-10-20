# Vox Grammar Bootstrap — full tree integration

This edition reads `grammartree.txt` directly (hierarchical tree plus complete catalog table), avoiding ChatGPT temporary paste filenames and most terminal-truncated names.

## Run

```bash
python3 -m tree_catalog grammartree.txt --out compiled
python3 -m unittest -v test_bootstrap.py
```

The compiler uses the **full table** for canonical twelve-glyph addresses and full object IDs, and attaches the matching tree path and named fidelity/parity fields as `tree_occurrences` metadata. It retains the Induxion mixed-radix codec in `bootstrap.py`, and exports `objects.jsonl`, `neighbors.jsonl`, `collisions.jsonl`, `errors.json`, `schema.json`, and `report.json`.

**Do not interpret coordinate neighbors as validated IMASM composition.** Kernel verdicts, execution traces and flow values remain unpopulated pending linkage to executable IMASM.

### Data quality

The catalog title reports 8777 entries. The typed compiler admits 8744 and quarantines 33, preserving the exact source errors in `compiled/errors.json`. Two of those don't match the table format; other failures include noncanonical axis-position glyphs. Nothing is silently repaired. Thirteen imported full-table names do not match an exact tree-name entry.
