#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
TARGET="${1:-$HOME/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY}"
DEST="$TARGET/sources/GRAMMAR_UNIFIED_RUN"
if [[ ! -d "$DEST" || ! -f "$TARGET/run_everything.sh" ]]; then
  echo "Missing historical project: $TARGET" >&2
  exit 2
fi
cp vox_native_evidence.py test_vox_native_evidence.py "$DEST/"
python3 - "$TARGET/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text()
needle = 'run_in native_vox_evidence GRAMMAR_UNIFIED_RUN python3 vox_native_evidence.py --output native_vox_evidence.json\n'
anchor = "run_in unified GRAMMAR_UNIFIED_RUN python3 -m unittest discover -p 'test_*.py'\n"
if needle not in s:
    if anchor not in s:
        raise SystemExit('Missing unified test anchor')
    p.write_text(s.replace(anchor, anchor + needle))
print('Vox TSV file-argument fix installed')
PY
