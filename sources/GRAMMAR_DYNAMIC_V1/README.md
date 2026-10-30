# Grammar Dynamic Bridge v1

This add-on has no dependencies and does not alter your existing DNN or checkpoint.

From the extracted folder:

```bash
./run.sh
```

Creates `dynamic_training/train.jsonl`, `validation.jsonl`, `test.jsonl`, and `report.json`.
Contains all 16 FOUR/SIXTEEN₃ carrier subsets, two-arm and three-arm partitioning,
AREV involution, and union fusion (192 exact examples). Split is by input carrier state.
This is a **specified carrier algebra oracle**, NOT an IMASM VM trace and NOT
an independent native proof of the full Grammar.

Optional real IMASM evidence, only if you have the separate `imasm` executable:

```bash
./run.sh --native-probes 8 --imasm /actual/path/to/imasm
```

Raw native CLI outputs are written to `dynamic_training/native_probes.jsonl`
without invented Grammar/Kernel/Flow judgments. The `vox` executable is
not treated as interchangeable with `imasm`. The documented command `imasm eval16`
accepts named seed notation, but the exact installed CLI protocol must be checked
for parity before using native traces as learning labels.

## Training objective boundary

These examples can supervise a learned carrier transition head. They are **not**
training on native IMASM graph transformations yet. Preserve `grammar_judgment`,
`kernel_judgment`, `native_flow_judgment` as separate nullable columns.
