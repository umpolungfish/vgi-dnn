#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -q test_operations.py
python3 grammar_ops.py "$@"
