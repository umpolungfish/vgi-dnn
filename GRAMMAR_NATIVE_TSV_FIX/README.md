# Native Vox TSV fix

`vox verdict --tsv` requires a **filename**. The collector now writes a temporary input file and invokes `vox verdict --tsv /path/to/probes.tsv`. Native stdout, stderr, and exit status are kept in `native_vox_evidence.json`.

Extract this patch folder under `~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY`, then:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_NATIVE_TSV_FIX
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
```
