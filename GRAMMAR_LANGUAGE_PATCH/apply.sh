#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$HERE")" == "GRAMMAR_LANGUAGE_PATCH" ]]; then
  TARGET="$(dirname "$HERE")"
else
  echo "Extract GRAMMAR_LANGUAGE_PATCH inside GRAMMAR_COMPLETE_HISTORY" >&2
  exit 2
fi
[[ -f "$TARGET/run_everything.sh" && -d "$TARGET/sources" ]] || { echo 'Expected GRAMMAR_COMPLETE_HISTORY root' >&2; exit 2; }
cp -a "$HERE/overlay/." "$TARGET/"
echo "Applied language-only patch to $TARGET"
echo "Original archives and historical logs remain unchanged."
