#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
TARGET="${1:-$HOME/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY}"
if [[ ! -f "$TARGET/run_everything.sh" || ! -d "$TARGET/sources/GRAMMAR_UNIFIED_RUN" ]]; then
  echo "Expected GRAMMAR_COMPLETE_HISTORY directory: $TARGET" >&2; exit 2
fi
cp vox_native_evidence.py test_vox_native_evidence.py "$TARGET/sources/GRAMMAR_UNIFIED_RUN/"
# Idempotent insertion after the unified unit-test stage.
python3 - "$TARGET/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
line="run_in native_vox_evidence GRAMMAR_UNIFIED_RUN python3 vox_native_evidence.py --output native_vox_evidence.json\n"
anchor="run_in unified GRAMMAR_UNIFIED_RUN python3 -m unittest discover -p 'test_*.py'\n"
if line not in s:
    if anchor not in s: raise SystemExit('Cannot find unified test stage anchor')
    s=s.replace(anchor, anchor+line)
    p.write_text(s)
PY
printf 'Installed Vox-native evidence stage into %s\n' "$TARGET"
