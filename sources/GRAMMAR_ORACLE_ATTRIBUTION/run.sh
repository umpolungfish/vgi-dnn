#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -v test_oracle_attribution.py
python3 oracle_attribution.py "$@"
