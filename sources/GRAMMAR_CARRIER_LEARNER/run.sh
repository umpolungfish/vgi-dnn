#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest -v test_dynamic_bridge.py test_carrier_net.py
python3 dynamic_bridge.py --out dynamic_training
python3 carrier_net.py "$@"
