# Grammar corpus SQLite lock fix

Extends the existing `GRAMMAR_COMPLETE_HISTORY` corpus stage, not a second project.

The original indexer deleted live `chunk_search` and table rows before rebuilding, which failed with `sqlite3.OperationalError: database is locked`. This version builds a complete index in a temporary file in the same directory and publishes it with atomic replacement only after `integrity_check`, and after detaching the previous database's WAL state. SQLite waits up to 60 seconds for database locks. If it cannot detach the old journal safely, it errors and retains the prior index. Source files are not changed.

Extract the folder inside `~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY` and run:

```bash
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY/GRAMMAR_CORPUS_LOCK_FIX
chmod +x ./apply.sh
./apply.sh
cd ~/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY
chmod +x ./run_everything.sh
./run_everything.sh --tests-only
./run_everything.sh --full
```

The source document index was successfully built in an earlier full run (9,488 files). The failure here is at the SQLite rebuild. No need to delete `grammar_corpus.sqlite`, its WAL, or SHM files by hand.
