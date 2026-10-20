# Grammar Carrier Learner

One complete folder. Run `./run.sh --epochs 200` from anywhere. Python 3 + PyTorch required.

This is a small **learned state-transform operator** for the dynamic carrier, distinct
from the catalog name classifier. Native Python carrier operations define exact
`FSPLIT` into 2 or 3 arms and `FFUSE` (union); a **single weight-tied neural
kernel** learns `HOLD` or `AREV` on each arm, independent of arm position.

It trains on the 192-case dataset generated from the 16 carrier masks, preserving
train/validation/test separation by *seed mask*. It evaluates on unseen seeds
and writes `carrier_checkpoints/best_carrier.pt` and `metrics.json`.

`./run.sh --epochs 200` runs all tests, regenerates the full dataset, trains,
then evaluates every one of the 192 cases. `FINAL` reports held-out and complete
truth-table performance.

**Boundaries:** The oracle here is the previously implemented carrier algebra
in `dynamic_bridge.py`; **not** native Vox/IMASM execution evidence. This
package doesn't purport to implement the whole Graph/Kernel/Flow stack, full
SIXTEEN₃ logical judgments, or a trained generative DNN. The fused operation
is exact and the transform is neural. For native parity tests, we must bind
an actual supported Vox/IMASM execution interface, not invent CLI commands.
