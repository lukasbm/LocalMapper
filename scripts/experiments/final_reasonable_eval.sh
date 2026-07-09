#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Final practical evaluation launcher.
#
# This intentionally does not run the full matrix for every seed. The default is:
# 1. one full comparison matrix with seed 0;
# 2. targeted repeat seeds only for the expensive/important subsets where
#    stochasticity matters most.
#
# Use PLAN_ONLY=1 to print the exact work without running it.

RUN_ID="${RUN_ID:-final_eval_$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
PLAN_ONLY="${PLAN_ONLY:-0}"
FULL_SEEDS="${FULL_SEEDS:-0}"
REPEAT_SEEDS="${REPEAT_SEEDS:-1 2}"
COMPARISON_NUM_EPOCHS="${COMPARISON_NUM_EPOCHS:-100}"

# Full seed-0 matrix: all main datasets, both trainable modes, low + standard budget.
FULL_DATASETS="${FULL_DATASETS:-USPTO_50K Golden ringreactions metAMDB}"
FULL_MODES="${FULL_MODES:-scratch finetune}"
FULL_BUDGET_PROFILES="${FULL_BUDGET_PROFILES:-low:50:3 standard:200:5}"
FULL_RINGREACTIONS_BUDGET_PROFILES="${FULL_RINGREACTIONS_BUDGET_PROFILES:-low:5:5 standard:10:10}"

# Targeted repeat matrix: defaults to standard budget only for the two datasets
# where source train/test splits are now preserved and conclusions are most
# likely to matter.
REPEAT_DATASETS="${REPEAT_DATASETS:-ringreactions metAMDB}"
REPEAT_MODES="${REPEAT_MODES:-scratch finetune}"
REPEAT_BUDGET_PROFILES="${REPEAT_BUDGET_PROFILES:-standard:200:5}"
REPEAT_RINGREACTIONS_BUDGET_PROFILES="${REPEAT_RINGREACTIONS_BUDGET_PROFILES:-standard:10:10}"

export RUN_ID
export PLAN_ONLY
export COMPARISON_NUM_EPOCHS

run_sweep() {
  local phase="$1"
  local datasets="$2"
  local modes="$3"
  local seeds="$4"
  local budget_profiles="$5"
  local ring_budget_profiles="$6"

  echo
  echo "### ${phase} ###"
  echo "RUN_ID=${RUN_ID}"
  echo "DATASETS=${datasets}"
  echo "MODES=${modes}"
  echo "SEEDS=${seeds}"
  echo "BUDGET_PROFILES=${budget_profiles}"
  echo "RINGREACTIONS_BUDGET_PROFILES=${ring_budget_profiles}"
  echo "COMPARISON_NUM_EPOCHS=${COMPARISON_NUM_EPOCHS}"
  echo "PLAN_ONLY=${PLAN_ONLY}"

  DATASETS="$datasets" \
  MODES="$modes" \
  SEEDS="$seeds" \
  BUDGET_PROFILES="$budget_profiles" \
  RINGREACTIONS_BUDGET_PROFILES="$ring_budget_profiles" \
  RUN_ID="$RUN_ID" \
  COMPARISON_NUM_EPOCHS="$COMPARISON_NUM_EPOCHS" \
  PLAN_ONLY="$PLAN_ONLY" \
    "$SCRIPT_DIR/comparison_sweep.sh"
}

run_sweep \
  "Full comparison matrix" \
  "$FULL_DATASETS" \
  "$FULL_MODES" \
  "$FULL_SEEDS" \
  "$FULL_BUDGET_PROFILES" \
  "$FULL_RINGREACTIONS_BUDGET_PROFILES"

if [[ -n "${REPEAT_SEEDS// }" && -n "${REPEAT_DATASETS// }" ]]; then
  run_sweep \
    "Targeted repeat seeds" \
    "$REPEAT_DATASETS" \
    "$REPEAT_MODES" \
    "$REPEAT_SEEDS" \
    "$REPEAT_BUDGET_PROFILES" \
    "$REPEAT_RINGREACTIONS_BUDGET_PROFILES"
fi

echo
echo "Final evaluation launcher complete."
echo "Analyze with: MPLCONFIGDIR=/tmp/mpl python outputs/analyze_metrics_summaries.py --run-id ${RUN_ID}"
