#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -m unittest -v test_zeta_oracle.py test_sic_gate.py
python3 zeta_oracle.py
