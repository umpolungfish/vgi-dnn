# GRAMMAR_INTEGRATED_V1 — mixed-operation stress harness

Runs **actual learned inference** through two previously trained weight-tied components: `CrystalReentry` (12-axis typed coordinate updates) and `CarrierKernel` (FOUR/SIXTEEN₃ arm transformations), interleaved. Tests 8–256 operations per trajectory against separately computed exact oracles. Logs final correctness, entire-trajectory correctness, first-failure position, and recoveries.

This is an integrated **inference pipeline**, not joint retraining and not native IMASM execution. `FSPLIT` and `FFUSE` remain exact; carrier arm transformation and crystal edits use neural checkpoint inference. It does not fabricate Grammar/Kernel/Flow verdicts.

## Run from the existing workspace

Extract the zip alongside `IMASM_GRAMMAR_DNN_OPS` and `GRAMMAR_CARRIER_LEARNER`. From the new folder:

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_INTEGRATED_V1
./run.sh
```

Defaults load *your* latest trained checkpoints at `../IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt` and `../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt`. Use `--ops-checkpoint` / `--carrier-checkpoint` to override.

```bash
./run.sh --cases 512 --lengths 64 128 256 512
```

`integrated_report.json` records results. This package is intentionally small and does not contain stale checkpoints or copied catalog files.
