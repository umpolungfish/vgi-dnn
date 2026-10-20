#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/sources/GRAMMAR_UNIFIED_RUN/primitive_registry"
mkdir -p "$DEST"
cp -f "$HERE/payload/primitive_dag.py" "$HERE/payload/test_primitive_dag.py" "$HERE/payload/SNS_PRIME.md" "$HERE/payload/IMSCRIBE_IMASM.md" "$DEST/"
python3 - "$ROOT/run_everything.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); text=p.read_text()
line="run_in primitive_morphism_dag GRAMMAR_UNIFIED_RUN/primitive_registry bash -c 'python3 -m unittest -q test_primitive_dag.py && python3 primitive_dag.py --report primitive_dag_report.json'"
if line not in text:
    anchor='# Every historical Python unit test suite, in chronological stage order.'
    if anchor not in text: raise SystemExit('Cannot safely find historical runner insertion point')
    text=text.replace(anchor, line+'\n'+anchor,1)
    p.write_text(text)
print('INSTALLED primitive_morphism_dag stage')
PY
(cd "$DEST" && python3 -m unittest -q test_primitive_dag.py && python3 primitive_dag.py --report primitive_dag_report.json)
