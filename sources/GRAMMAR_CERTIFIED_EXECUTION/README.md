# Grammar-certified mixed execution (SIC + operational checks)

A single folder that contains all executable source. Loads **existing** local checkpoints (no retraining, no checkpoint modifications).

Run from `~/imsgct/vox_grammar_induxion/GRAMMAR_CERTIFIED_EXECUTION`:

```bash
chmod +x ./run.sh
./run.sh
```

Default checkpoint locations:
- `../GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt`
- `../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt`

Optional: `./run.sh --cases 256 --steps 256 --ops-checkpoint /path/to/checkpoint.pt`.

Reports both **raw neural** and **gated** results for identical sequences; all attempted repairs are counted and sample failures retained. Each gated state is validated through the **exact rational** SIC frame reconstruction for d=12 crystal embedding and d=2 carrier embedding. The independent coordinate and carrier oracles certify transition correctness; **SIC retraction alone cannot validate a transition**. A 100% gated accuracy result is expected by construction and **is not a neural performance claim**.

**Not** a Lean compilation, native IMASM execution, general WH-SIC existence result, or full continuum correspondence. The zeta(-1) value and Gram coefficients follow supplied source. These finite-dimensional checks do not assert a universal perfect oracle.
