# GRAMMAR COMPLETE HISTORY — original scripts and all stages

This archive contains **every recoverable historical ZIP** from our conversation, extracted source trees, original catalogs and research notes, and a chronological test/training runner. Each original archive is retained byte-for-byte in `original_archives/`; unpacked stage folders are in `sources/`.

## Run everything from the beginning

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --full
```

This runs historical tests from the first Vox/bootstrap stages forward, then builds/trains and runs all later stages. All logs are written to `logs/`. It can take a long time, particularly the 1,024-epoch carrier training. It continues after a failed stage and reports a nonzero exit code if anything fails.

For a quick environment/smoke run: `./run_everything.sh --quick`. For all historical unit tests only: `./run_everything.sh --tests-only`.

## Package layout

- `original_archives/` includes the stage archives as originally supplied, including obsolete and later corrected versions.
- `sources/GRAMMAR_UNIFIED_RUN/` is the latest **fixed** edition; the earlier unified edition remains in `original_archives/GRAMMAR_UNIFIED_RUN.zip`.
- Historical stage scripts are deliberately not silently rewritten. Any original defects are preserved and will appear in the corresponding log.
- All training checkpoints included in original ZIPs are included. Checkpoints created **only on your machine** cannot be reconstructed from this conversation; the `--full` runner retrains stages in sequence to recreate them.
- The Python runner executes the packaged Python modules. Lean sources and native IMASM tools are separate runtime components.
- Original provided reference materials are in `historical_notes/`.
