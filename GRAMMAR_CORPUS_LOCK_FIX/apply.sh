#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
TARGET="${1:-$HOME/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY}"
DEST="$TARGET/sources/GRAMMAR_UNIFIED_RUN"
[[ -f "$TARGET/run_everything.sh" && -f "$DEST/grammar_corpus.py" ]] || { echo 'Missing installed master corpus stage' >&2; exit 2; }
cp grammar_corpus.py "$DEST/grammar_corpus.py"
cp test_grammar_corpus.py "$DEST/test_grammar_corpus.py"
(cd "$DEST" && python3 -m unittest -q test_grammar_corpus.py)
echo 'Installed atomic corpus rebuild fix; existing index preserved until a replacement succeeds.'
