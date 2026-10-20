# Coordinate error repair (add-on)

Extract alongside `IMASM_GRAMMAR_DNN_OPS` and `GRAMMAR_INTEGRATED_V1` within `~/imsgct/vox_grammar_induxion`.

Run: `./run.sh` (or `./run.sh --pool 30000 --holdout 12000 --epochs 8`).

Uses the existing 12-epoch coordinate checkpoint, mines difficult coordinate transitions, fine-tunes with mixed hard/ordinary examples, compares before and after on disjoint generated examples, and **only writes a candidate** if exact accuracy improves. Does not overwrite your trained checkpoint. Uses CPU and PyTorch already installed. Does not claim native IMASM execution.

Inspect `repair_report.json` before selecting a candidate; separate validation of any candidate is required. No carrier checkpoint is needed.
