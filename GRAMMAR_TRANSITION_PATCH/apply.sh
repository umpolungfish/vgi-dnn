#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/sources/GRAMMAR_UNIFIED_RUN"
[[ -d "$DEST" && -f "$ROOT/run_everything.sh" ]] || { echo 'Missing master project' >&2; exit 2; }
cp -f "$HERE/carrier_transitions.py" "$HERE/test_carrier_transitions.py" "$DEST/"
python3 - "$ROOT/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
anchor='run_in native_verdict_join GRAMMAR_UNIFIED_RUN python3 vox_evidence_join.py --evidence native_vox_evidence.json --output native_verdict_training.jsonl'
line='run_in carrier_transition_training GRAMMAR_UNIFIED_RUN bash -c \'python3 -m unittest -q test_carrier_transitions.py && python3 carrier_transitions.py --native native_verdict_training.jsonl --output grammar_transition_training.jsonl --report grammar_transition_report.json\''
if line not in s:
    if anchor not in s:raise SystemExit('Missing native_verdict_join: apply GRAMMAR_NATIVE_JOIN_PATCH first')
    s=s.replace(anchor,anchor+'\n'+line,1)
    p.write_text(s)
print('INSTALLED carrier_transition_training stage')
PY
(cd "$DEST" && python3 -m unittest -q test_carrier_transitions.py)
