# Vox-native evidence stage

Adds real `vox verdict --tsv` capture to the **existing** GRAMMAR_COMPLETE_HISTORY project, without replacing any model or historical archive. The initial probes are `sourceWord` and `strandedWord` from the supplied ZetaSICDualFrame Lean module. Additional words may be supplied with `--words FILE.tsv`.

This stage records the actual CLI input, exit status, standard output, standard error, and SHA-256 input checksum in `native_vox_evidence.json`. It does not derive IMASM executions from 12-axis crystal addresses or infer verdicts from exit codes.

Apply to the user's existing location:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_NEXT_NATIVE_PATCH
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
```

To repeat the full pipeline: `./run_everything.sh --full`.
