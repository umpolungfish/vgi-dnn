#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
TARGET="${1:-$HOME/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY}"
DEST="$TARGET/sources/GRAMMAR_UNIFIED_RUN"
[[ -f "$TARGET/run_everything.sh" && -d "$DEST" ]] || { echo "Missing master project: $TARGET" >&2; exit 2; }
cp grammar_corpus.py test_grammar_corpus.py "$DEST/"
python3 - "$TARGET/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
test='run_in corpus_indexer_tests GRAMMAR_UNIFIED_RUN python3 -m unittest -q test_grammar_corpus.py\n'
stage='run_in grammar_corpus_index GRAMMAR_UNIFIED_RUN python3 grammar_corpus.py --db grammar_corpus.sqlite --report grammar_corpus_report.json\n'
if test not in s:
 anchor='run_in unified GRAMMAR_UNIFIED_RUN python3 -m unittest discover -p \'test_*.py\'\n'
 if anchor not in s:raise SystemExit('Cannot find unified test stage')
 s=s.replace(anchor,anchor+test)
if stage not in s:
 anchor='if [[ "$MODE" != \'--tests-only\' ]]; then\n'
 if anchor not in s:raise SystemExit('Cannot find full pipeline branch')
 s=s.replace(anchor,stage+anchor)
p.write_text(s)
print('Installed Grammar document and Lean indexer into the master project')
PY
(cd "$DEST" && python3 -m unittest -q test_grammar_corpus.py)
