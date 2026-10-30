# IMASM Grammar DNN — first trained baseline

This is a **trainable** small neural model, not merely a catalog exporter. Its **12 independently typed categorical outputs** use radices `[4,5,4,5,3,5,3,4,5,4,3,4]`. Twelve learned axis query vectors jointly attend to the object's Unicode name through a **single transformer block re-entered with tied weights** (`--passes`). No ordinary last-layer output filtering is used. Training is supervised by the catalog's existing attestations, not fabricated logical verdicts.

**Limitations:** This is *not* the previously tested 34,540-parameter Dialetheic Transformer implementation (which has not been supplied here), and there are no SIXTEEN₃ transitions, Grammar/Kernel/Flow execution proofs, or SIC-POVM frame operations in this implementation. It is a deliberately honest, runnable first neural baseline to be replaced/extended when those executable operations are available. Some catalog coordinates were previously repaired provisionally; see `UNOfd_repair_audit.json`.

## Install / run

Extract archive *into your chosen directory*; archive has one top-level `IMASM_GRAMMAR_DNN_V1` folder.

```bash
cd IMASM_GRAMMAR_DNN_V1
./run.sh --epochs 12
```

Requires Python 3.10+ and `torch`; if missing, install in your active environment with `python3 -m pip install torch` (or `uv pip install torch`). If your `p3` alias is the environment Python and `python3` is different, replace `python3` with `p3` in `run.sh`.

### Predict

```bash
python3 grammar_dnn.py predict 'quantum_gravity' --checkpoint checkpoints/best.pt
```

Training prints metrics each epoch. `checkpoints/best.pt` includes weights + name-character vocabulary + axis alphabets. `checkpoints/metrics.json` includes validation curves and held-out test scores. Addresses are isolated across splits; a split tests generalization to unseen **addresses**, not verified logical validity.

Try `./run.sh --epochs 2 --batch 64` first for a smoke test. No IMASM executable is required for this baseline. Each epoch can take time on CPU.

## Included trained checkpoint

A tested checkpoint is included at `checkpoints/best.pt` (32-dimensional model, two tied re-entry passes, 18,673 parameters). A training run completed **3 epochs** before the execution session ended; the best available validation checkpoint was retained. On 890 held-out records it scored **41.09% per-axis accuracy** and **0.00% exact-tuple accuracy**. These results establish that the training/inference loop works, **not** that the model already performs reliable imscription or operational Grammar inference. The `run.sh` command starts a fresh training run (12 epochs by default); using `predict` requires no training.

To inspect a prediction immediately after extracting the archive:

```bash
python3 grammar_dnn.py predict quantum_gravity --checkpoint checkpoints/best.pt
```
