# Paired mixed-rollout comparison

Extract this folder beside `IMASM_GRAMMAR_DNN_OPS`, `GRAMMAR_COORDINATE_REPAIR`, `GRAMMAR_CARRIER_LEARNER`, and `GRAMMAR_INTEGRATED_V1` inside `~/imsgct/vox_grammar_induxion`.

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_REPAIR_AB
chmod +x ./run.sh
./run.sh
```

Uses your actual original and newly saved candidate checkpoints, with the same carrier kernel and identical random seeds for both runs. Compares coordinate step, mixed intermediate, entire trajectory and final-state exactness for 8–256 operations. Results written to `repair_ab_report.json`. Does not modify checkpoints. This is a deterministic operation-oracle test, not native IMASM verification.
