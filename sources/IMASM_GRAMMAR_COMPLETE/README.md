# Complete Imscribing Grammar dataset package

**One folder. No external input files. No Vox or standalone `imasm` executable required for this data-preparation step.**

## Install (download ZIP and save it anywhere)

Extract the archive into `~/imsgct` using your file manager, or replace `/path/to/downloaded.zip` with the actual path to the ZIP you downloaded:

```bash
mkdir -p ~/imsgct
unzip -oq /path/to/downloaded.zip -d ~/imsgct
cd ~/imsgct/IMASM_GRAMMAR_COMPLETE
./run.sh
```

The ZIP has exactly one top-level folder, `IMASM_GRAMMAR_COMPLETE/`; it contains everything required. Do **not** create an extra folder before extracting.

## Included

- `UNOfd.txt`: repaired catalog, 8,773 structurally valid imscriptions
- `UNOfd_repair_audit.json`: provenance of the 29 **provisional**, inferred glyph repairs (not independently verified derivations)
- `bootstrap.py`: validates the catalog and builds compiled objects, collisions, neighbor candidates, schema
- `prepare_training.py`: builds typed reconstruction records with no fabricated execution verdicts
- `test_bootstrap.py`: structural tests
- `INDUXION.pdf`: Induxion specification
- `run.sh`: runs compiler, exporter, and tests from this folder
- `compiled/`: compiler outputs generated and already included
- `grammar_training/`: generated train/validation/test data already included

## Expectations

- `parsed_objects`: **8773**
- `invalid_rows`: **0**
- `potential_name_truncations`: **0**
- `unique_addresses`: **6117**
- Training **6943**, validation **940**, test **890**.

The 29 tuple substitutions in the repair audit are provisional. The data exporter does **not** pretend crystal addresses are IMASM executable traces. Actual step-by-step VM traces require the separate Grammar/IMASM runtime.
