#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -q test_repair.py
python3 repair.py "$@"
