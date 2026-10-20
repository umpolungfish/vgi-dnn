#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
TARGET="${1:-$HOME/imsgct/VGI/GRAMMAR_COMPLETE_HISTORY}"
DEST="$TARGET/sources/GRAMMAR_UNIFIED_RUN"
[[ -f "$TARGET/run_everything.sh" && -d "$DEST" ]] || { echo "Missing master project: $TARGET" >&2; exit 2; }
cp vox_evidence_join.py test_vox_evidence_join.py "$DEST/"
python3 - "$TARGET/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
anchor='run_in native_vox_evidence GRAMMAR_UNIFIED_RUN python3 vox_native_evidence.py --output native_vox_evidence.json\n'
new='run_in native_verdict_join GRAMMAR_UNIFIED_RUN python3 vox_evidence_join.py --evidence native_vox_evidence.json --output native_verdict_training.jsonl\n'
if new not in s:
    if anchor not in s: raise SystemExit('Expected native Vox evidence stage missing; apply GRAMMAR_NATIVE_TSV_FIX first')
    s=s.replace(anchor,anchor+new)
    p.write_text(s)
print('Installed native Vox evidence join into the master runner')
PY
(cd "$DEST" && python3 -m unittest -q test_vox_evidence_join.py)
