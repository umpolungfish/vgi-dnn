#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
python3 bootstrap.py ./UNOfd.txt --out compiled
python3 prepare_training.py ./UNOfd.txt --out grammar_training
python3 -m unittest -v test_bootstrap.py
printf '\nSUCCESS: outputs in %s/compiled and %s/grammar_training\n' "$PWD" "$PWD"
