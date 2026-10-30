#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest -v test_dynamic_bridge.py
python3 dynamic_bridge.py "$@"
