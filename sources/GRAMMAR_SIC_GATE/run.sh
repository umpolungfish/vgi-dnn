#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -v test_sic_gate.py
python3 sic_gate.py --out sic_gate_report.json
