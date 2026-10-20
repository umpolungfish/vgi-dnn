# Grammar corpus ingestion patch

Installs into the existing `GRAMMAR_COMPLETE_HISTORY/sources/GRAMMAR_UNIFIED_RUN` project.

Recursively indexes the two existing source trees **without moving them**:

- `/home/mrnob0dy666/imsgct/ig-docs`
- `/home/mrnob0dy666/imsgct/p4rakernel/p4ramill/Imscribing`

Extract this patch ZIP in `GRAMMAR_COMPLETE_HISTORY` and run:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_CORPUS_PATCH
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
```

`--tests-only` runs corpus indexer unit tests. `--quick` and `--full` additionally **index all files recursively**. The indexer saves `grammar_corpus.sqlite` and `grammar_corpus_report.json` in `sources/GRAMMAR_UNIFIED_RUN/`.

Supported: `.lean`, `.md`, `.txt`, `.rst`, `.org`, `.tex`, `.json`, `.yaml`, `.yml`, `.toml`, `.csv`, `.html`, `.htm`, `.adoc`, `.typ`, `.docx`, `.odt`, `.pdf`. PDF text requires `pymupdf` or `pypdf` installed; unsupported/unextractable files are catalogued with extraction error. No OCR. Includes FTS5 keyword retrieval, file SHA256, Lean declarations, imports, and source line references. Index files are regenerated on every run.

Search an indexed corpus:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/sources/GRAMMAR_UNIFIED_RUN
python3 grammar_corpus.py --db grammar_corpus.sqlite --search 'dual frame'
```

The patch is idempotent; applying it twice does not duplicate stages.
