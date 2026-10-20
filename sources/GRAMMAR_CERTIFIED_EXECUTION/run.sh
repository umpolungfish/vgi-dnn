#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -v test_certified_execution.py
python3 certified_execution.py "$@"
