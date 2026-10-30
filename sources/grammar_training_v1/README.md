# Grammar training corpus v1

This project uses the **real catalog**, not `vox verdict` on 12-address tuples.

From `~/imsgct/vox_grammar_induxion`:

```bash
cp /path/to/unpacked/grammar_training_v1/prepare_training.py .
p3 prepare_training.py ./UNOfd.txt --out grammar_training
```

If `bootstrap.py` isn't alongside the script, copy the bundled `bootstrap.py` too.
Or simply run directly from the unpacked package:

```bash
cd /path/to/unpacked/grammar_training_v1
p3 prepare_training.py ~/imsgct/vox_grammar_induxion/UNOfd.txt --out grammar_training
```

Output: `train.jsonl`, `validation.jsonl`, `test.jsonl`, `schema.json`, `errors.json`, and `report.json`.
Every training object includes the **original full object name**, **12 axis-aware glyph values**, **numeric crystal index**, **split grouped by complete address**, and **explicitly unset VM/proof/flow judgments**. It can support typed imscription prediction and masked-coordinate learning, but is not an IMASM execution trace set.
