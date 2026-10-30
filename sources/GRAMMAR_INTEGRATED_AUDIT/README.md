# Integrated Grammar Failure Audit

Place this folder alongside `GRAMMAR_INTEGRATED_V1`, `IMASM_GRAMMAR_DNN_OPS`, and `GRAMMAR_CARRIER_LEARNER` in `~/imsgct/vox_grammar_induxion`.

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_INTEGRATED_AUDIT
./run.sh
```

This uses the **latest checkpoints in your existing folders**, not the demo ones, and writes `audit_report.json`.

The audit isolates three categories: independent one-step coordinate errors by operation type, rollout divergence and recovery for long coordinate-only chains, and exhaustive 16-state carrier truth-table agreement. Examples include exact model inputs and predictions for reproducible diagnosis. It is **not** a native IMASM execution test, and the inference components are not jointly trained. No checkpoints are modified.

Options: `./run.sh --samples 24000 --cases 512 --steps 512`.
