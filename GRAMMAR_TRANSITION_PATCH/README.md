# Grammar carrier-transition training patch

Fixes the hard-coded `transition_pairs: 0` *as an integration omission* by adding actual input/output pairs generated from the supplied SIXTEEN_3 carrier gate laws, not by reinterpreting the native `vox verdict` closure evidence as VM traces.

The stage runs after `native_verdict_join` and produces `grammar_transition_training.jsonl` plus `grammar_transition_report.json`. Each transition has `before_bits`, `after_bits`, gate, and source provenance. Includes all 16 states across the nine unary gate primitives plus 16 states for each 2/3-arm split-fuse roundtrip, for **176** transitions. TANCH, AFWD, AREV, CLINK, EVALT, IMSCRIB, EVALF, ENGAGR, IFIX are unary. VINIT is an inputless source whose state depends on a seed; split/fuse require graph arity.

Native Vox verdict records remain intact and distinct. The report shows `transition_pairs: 176`, `reference_transition_pairs: 176`, `native_vm_transition_pairs: 0`. More detailed *native* VM transitions require a native execution/trace command, not `vox verdict --tsv`. That distinction is explicit in every row and report.

Extract into the master directory and run `chmod +x ./apply.sh && ./apply.sh` from this patch folder. Then `chmod +x ./run_everything.sh && ./run_everything.sh --tests-only` at the master root. Installation is idempotent.
