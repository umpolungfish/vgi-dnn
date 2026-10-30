# Primitive morphism DAG patch

Adds a native-to-source typed registry to `GRAMMAR_COMPLETE_HISTORY/sources/GRAMMAR_UNIFIED_RUN/primitive_registry/` and a new first stage in the existing `run_everything.sh` runner. All 49 Shavian value programs are extracted from the bundled SNS_PRIME source. They are assembled into a shared, lossless word DAG. The 12-primitive opcode alphabet is preserved. Duplicate type programs share identical root nodes.

The stage runs tests, verifies exact source-word reconstruction, expands any well-typed 12-axis tuple to IMASM program text and writes `primitive_dag_report.json`. This is a *word-level* DAG; IMASM graph edges are determined separately by the wiring verb. It does not claim that recursively replacing every opcode with a fixed word has already been established—this table supplies **type value → opcode word** expansions. The engine is the first executable layer toward the full recursive grammar graph.

## Install

Extract ZIP into `~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY`. Run:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_PRIMITIVE_DAG_PATCH
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
```

All source folders and checkpoints stay in place. Patch installation is idempotent.
