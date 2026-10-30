# IMASM GRAMMAR DNN — corrected v1

**Fixed:** the default model dimension is **32**, not 64, so the default
model has **18,673 parameters** (with the bundled training vocabulary).
Earlier archives incorrectly trained a 53,681-parameter model.

Extract the ZIP: it contains **one** `IMASM_GRAMMAR_DNN_FIXED/` folder.
From inside that folder, run:

```bash
./run.sh --epochs 12
python3 grammar_dnn.py predict quantum_gravity --checkpoint checkpoints/best.pt
```

No other folders, source files or executable path assumptions are required.
`run.sh` validates the 8,773-entry catalog, generates the three splits,
runs eight tests including the exact model-size assertion, and trains.
The packaged checkpoint is generated using the corrected default configuration.

**Honest scope:** this is a reproducible name-to-twelve-axis neural baseline
with weight-tied re-entry, not the previously tested 34,540-parameter DT
and not yet IMASM/SIXTEEN₃ dynamic execution. A successful run is not proof
of exact-tuple prediction quality.
