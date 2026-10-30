# Native evidence join patch

Installs in the existing `GRAMMAR_COMPLETE_HISTORY` project. Converts the real `vox verdict --tsv` capture into named, hashed, replayable JSONL training evidence. Rejects missing output, inconsistent glyph lengths, duplicate labels and failed native invocations. Does not infer IMASM transition pairs from two independent word verdicts.

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_NATIVE_JOIN_PATCH
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
```

Outputs: `sources/GRAMMAR_UNIFIED_RUN/native_verdict_training.jsonl` and `native_join_report.json`.
