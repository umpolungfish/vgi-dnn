# GRAMMAR_UNIFIED_RUN

Single **self-contained source folder** consolidating the tested SIC/zeta exact rational Gram oracle, qubit SIC tests, neural crystal operation, carrier kernel, oracle-gated mixed execution, correction attribution and paired original-versus-repaired rollouts. No more module-by-module ZIPs or sibling Python imports.

## Run

Extract this folder into `~/imsgct/vox_grammar_induxion/` alongside your already-trained folders and run:

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_UNIFIED_RUN
chmod +x ./run.sh
./run.sh
```

Outputs: **unified_report.json** and **oracle_corrections.jsonl** (every correction independently replay-verified before export), with all tests, SIC/zeta exact-frame certificates, 256-step mixed execution, replayable correction events and side-by-side paired trajectory comparisons.

For a quick local sanity run: `./run.sh --quick`.

## Checkpoints

The launcher searches the neighboring folders first, then searches `~/imsgct` for the checkpoint files. If the trained files are missing, packaged baseline weights let the run finish, with `CHECKPOINT_FALLBACK` printed and recorded (baseline results will be weaker). Nothing overwrites trained weights. Preferred local paths:

- `../GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt`
- `../IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt` (for A/B)
- `../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt`

Override with `--ops-checkpoint`, `--original-checkpoint`, `--carrier-checkpoint`. Missing local weights are reported explicitly before packaged fallback is used. Use the overrides to select your strong checkpoints. `--skip-compare` skips the original-checkpoint requirement.

Run a smaller or larger evaluation with `--cases N --steps N`.

## Components and validation

The SIC and zeta Gram inverses are exact rational matrix identities for d=2,4,12, and the qubit numerical SIC certificate is tested. Frame retraction **does not** decide transition correctness: the exact crystal and carrier operational oracles do. Gated trajectory perfection is oracle enforcement, not the network's own accuracy. The Lean zeta theorem, banked weight claims and native IMASM execution have **not** been independently compiled or executed by this package.

## Oracle replay

The single run now exports full failing pre-states, proposed states, exact oracle targets and operations to `oracle_corrections.jsonl`. Every exported record is checked by executing its operation again, and the reread file is verified. Use these examples for subsequent counterexample-guided training without losing their provenance. Nothing overwrites trained checkpoints.
