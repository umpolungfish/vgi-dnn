#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="$PWD"; S="$ROOT/sources"; LOG="$ROOT/logs"
mkdir -p "$LOG"
MODE="${1:---full}"
if [[ "$MODE" != "--full" && "$MODE" != "--tests-only" && "$MODE" != "--quick" ]]; then
 echo "Usage: ./run_everything.sh [--full|--tests-only|--quick]" >&2; exit 2
fi
failures=0; runs=0
run() {
 local name="$1"; shift
 runs=$((runs+1))
 printf '\n========== [%02d] %s ==========\n' "$runs" "$name"
 local logfile="$LOG/$(printf '%02d' "$runs")_${name//[^a-zA-Z0-9_-]/_}.log"
 timeout --signal=TERM 1800 "$@" 2>&1 | tee "$logfile"
 local status=${PIPESTATUS[0]}
 if (( status != 0 )); then
   echo "FAILED ($status): $name — $logfile"
   failures=$((failures+1))
 else echo "PASSED: $name"; fi
}
run_in() { local label="$1"; local dir="$2"; shift 2; run "$label" bash -c 'cd "$1"; shift; "$@"' _ "$S/$dir" "$@"; }
run_in primitive_morphism_dag GRAMMAR_UNIFIED_RUN/primitive_registry bash -c 'python3 -m unittest -q test_primitive_dag.py && python3 primitive_dag.py --report primitive_dag_report.json'
# Every historical Python unit test suite, in chronological stage order.
run_in bootstrap_v01 vox_grammar_induxion_v01 python3 -m unittest discover -p 'test_*.py'
run_in bootstrap vox_grammar_bootstrap python3 -m unittest discover -p 'test_*.py'
run_in grammar_tree vox_grammar_tree_v02 python3 -m unittest discover -p 'test_*.py'
run_in imasm_traces vox_imasm_traces_v03 python3 -m unittest discover -p 'test_*.py'
run_in training_export grammar_training_v1 python3 prepare_training.py --help
run_in complete_bootstrap IMASM_GRAMMAR_COMPLETE python3 -m unittest discover -p 'test_*.py'
run_in original_dnn IMASM_GRAMMAR_DNN_V1 python3 -m unittest discover -p 'test_*.py'
run_in fixed_dnn IMASM_GRAMMAR_DNN_FIXED python3 -m unittest discover -p 'test_*.py'
run_in ops_dnn IMASM_GRAMMAR_DNN_OPS python3 -m unittest discover -p 'test_*.py'
run_in dynamic GRAMMAR_DYNAMIC_V1 python3 -m unittest discover -p 'test_*.py'
run_in carrier GRAMMAR_CARRIER_LEARNER python3 -m unittest discover -p 'test_*.py'
run_in integrated GRAMMAR_INTEGRATED_V1 python3 -m unittest discover -p 'test_*.py'
run_in integrated_audit GRAMMAR_INTEGRATED_AUDIT python3 -m unittest discover -p 'test_*.py'
run_in repair GRAMMAR_COORDINATE_REPAIR python3 -m unittest discover -p 'test_*.py'
run_in repair_ab GRAMMAR_REPAIR_AB python3 -m unittest discover -p 'test_*.py'
run_in sic GRAMMAR_SIC_GATE python3 -m unittest discover -p 'test_*.py'
run_in zeta GRAMMAR_ZETA_SIC_ORACLE python3 -m unittest discover -p 'test_*.py'
run_in certified GRAMMAR_CERTIFIED_EXECUTION python3 -m unittest discover -p 'test_*.py'
run_in attribution GRAMMAR_ORACLE_ATTRIBUTION python3 -m unittest discover -p 'test_*.py'
run_in unified GRAMMAR_UNIFIED_RUN python3 -m unittest discover -p 'test_*.py'
run_in corpus_indexer_tests GRAMMAR_UNIFIED_RUN python3 -m unittest -q test_grammar_corpus.py
run_in native_vox_evidence GRAMMAR_UNIFIED_RUN python3 vox_native_evidence.py --output native_vox_evidence.json
run_in native_verdict_join GRAMMAR_UNIFIED_RUN python3 vox_evidence_join.py --evidence native_vox_evidence.json --output native_verdict_training.jsonl
run_in carrier_transition_training GRAMMAR_UNIFIED_RUN bash -c 'python3 -m unittest -q test_carrier_transitions.py && python3 carrier_transitions.py --native native_verdict_training.jsonl --output grammar_transition_training.jsonl --report grammar_transition_report.json'
run_in grammar_corpus_index GRAMMAR_UNIFIED_RUN python3 grammar_corpus.py --db grammar_corpus.sqlite --report grammar_corpus_report.json
if [[ "$MODE" != '--tests-only' ]]; then
  ep=12; carrier_ep=1024; pool=30000; held=12000; cases=128; steps=256
  if [[ "$MODE" == '--quick' ]]; then ep=1; carrier_ep=50; pool=2000; held=1000; cases=8; steps=16; fi
  # Rebuild training artifacts from the original repaired catalog.
  run_in catalog_compilation IMASM_GRAMMAR_COMPLETE bash ./run.sh
  run_in original_dnn_training IMASM_GRAMMAR_DNN_V1 bash ./run.sh --epochs "$ep"
  run_in fixed_dnn_training IMASM_GRAMMAR_DNN_FIXED bash ./run.sh --epochs "$ep"
  run_in ops_training IMASM_GRAMMAR_DNN_OPS bash ./run_operations.sh --epochs "$ep"
  run_in ops_long_chains IMASM_GRAMMAR_DNN_OPS bash ./run_chain_test.sh
  run_in dynamic_truth_table GRAMMAR_DYNAMIC_V1 bash ./run.sh
  run_in carrier_training GRAMMAR_CARRIER_LEARNER bash ./run.sh --epochs "$carrier_ep"
  run_in integrated_rollout GRAMMAR_INTEGRATED_V1 bash ./run.sh
  run_in error_audit GRAMMAR_INTEGRATED_AUDIT bash ./run.sh
  run_in hard_case_repair GRAMMAR_COORDINATE_REPAIR bash ./run.sh --pool "$pool" --holdout "$held"
  run_in paired_comparison GRAMMAR_REPAIR_AB bash ./run.sh
  run_in sic_frame GRAMMAR_SIC_GATE bash ./run.sh
  run_in zeta_exact GRAMMAR_ZETA_SIC_ORACLE bash ./run.sh
  run_in certified_execution GRAMMAR_CERTIFIED_EXECUTION bash ./run.sh
  run_in oracle_attribution GRAMMAR_ORACLE_ATTRIBUTION bash ./run.sh
  # Pass resolved checkpoint paths explicitly to eliminate directory-search surprises.
  ops="$S/GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt"
  original="$S/IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt"
  carrier="$S/GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt"
  if [[ ! -f "$ops" ]]; then echo "Repair candidate absent; using original trained ops checkpoint"; ops="$original"; fi
  run_in unified_full GRAMMAR_UNIFIED_RUN bash ./run.sh --ops-checkpoint "$ops" --original-checkpoint "$original" --carrier-checkpoint "$carrier" --cases "$cases" --steps "$steps"
fi
printf '\nCOMPLETE: %d stages; %d failed. Logs: %s\n' "$runs" "$failures" "$LOG"
(( failures == 0 ))
