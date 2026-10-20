#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 -c 'import torch; print("PyTorch",torch.__version__)'
python3 -m unittest -q test_bootstrap.py test_model.py
# Inputs are distributed with this package; rerun reproducibly on every training run.
python3 bootstrap.py UNOfd.txt --out compiled > compiled_report.json
python3 prepare_training.py UNOfd.txt --out grammar_training > training_report.json
python3 grammar_dnn.py train --data grammar_training --out checkpoints "$@"
