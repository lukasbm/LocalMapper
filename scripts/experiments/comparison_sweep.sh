#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="${DATASETS:-USPTO_50K Golden ringreactions metAMDB}"
MODES="${MODES:-scratch pretrained}"
SEEDS="${SEEDS:-0}"
RUN_ID="${RUN_ID:-comparison_$(date +%Y%m%d_%H%M%S)_${RANDOM}}"
COMPARISON_NUM_EPOCHS="${COMPARISON_NUM_EPOCHS:-100}"
PLAN_ONLY="${PLAN_ONLY:-0}"

# Format: label:sample_limit:iterations. The default grid varies annotation
# budget, while keeping epochs fixed so the comparison isolates active learning.
BUDGET_PROFILES="${BUDGET_PROFILES:-low:50:3 standard:200:5}"
RINGREACTIONS_BUDGET_PROFILES="${RINGREACTIONS_BUDGET_PROFILES:-low:5:5 standard:10:10}"

export RUN_ID

budget_profiles_for_dataset() {
  case "$1" in
    ringreactions) echo "$RINGREACTIONS_BUDGET_PROFILES" ;;
    *) echo "$BUDGET_PROFILES" ;;
  esac
}

num_epochs_for_dataset() {
  case "$1" in
    metAMDB) echo "${METAMDB_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    ringreactions) echo "${RINGREACTIONS_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    USPTO_50K) echo "${USPTO50K_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    Golden) echo "${GOLDEN_NUM_EPOCHS:-$COMPARISON_NUM_EPOCHS}" ;;
    *) echo "$COMPARISON_NUM_EPOCHS" ;;
  esac
}

RUN_COUNT=0
ITERATION_COUNT=0

print_plan_header() {
  echo "### Comparison sweep plan ###"
  echo "RUN_ID=${RUN_ID}"
  echo "DATASETS=${DATASETS}"
  echo "MODES=${MODES}"
  echo "SEEDS=${SEEDS}"
  echo "COMPARISON_NUM_EPOCHS=${COMPARISON_NUM_EPOCHS}"
  echo "PLAN_ONLY=${PLAN_ONLY}"
}

print_run_line() {
  echo "RUN dataset=$1 mode=$2 budget=$3 sample_limit=$4 iterations=$5 seed=$6 num_epochs=$7 model=$8"
}

print_totals() {
  echo "TOTAL_RUNS=${RUN_COUNT}"
  echo "TOTAL_ACTIVE_LEARNING_ITERATIONS=${ITERATION_COUNT}"
}

print_plan_header

for DATASET_NAME in $DATASETS; do
  for PROFILE in $(budget_profiles_for_dataset "$DATASET_NAME"); do
    IFS=: read -r BUDGET_LABEL_VALUE SAMPLE_LIMIT_VALUE ITERATIONS_VALUE <<< "$PROFILE"
    if [[ -z "${BUDGET_LABEL_VALUE:-}" || -z "${SAMPLE_LIMIT_VALUE:-}" || -z "${ITERATIONS_VALUE:-}" ]]; then
      echo "Invalid budget profile '${PROFILE}'. Expected label:sample_limit:iterations." >&2
      exit 2
    fi
    for MODE in $MODES; do
      for SEED_VALUE in $SEEDS; do
        export DATASET="$DATASET_NAME"
        export INIT_MODE="$MODE"
        export SEED="$SEED_VALUE"
        export SAMPLE_LIMIT="$SAMPLE_LIMIT_VALUE"
        export ITERATIONS="$ITERATIONS_VALUE"
        export BUDGET_LABEL="$BUDGET_LABEL_VALUE"
        export NUM_EPOCHS="$(num_epochs_for_dataset "$DATASET_NAME")"
        export MODEL="LocalMapper_${MODE}_${DATASET_NAME}_${BUDGET_LABEL_VALUE}_${RUN_ID}"
        RUN_COUNT=$((RUN_COUNT + 1))
        ITERATION_COUNT=$((ITERATION_COUNT + ITERATIONS_VALUE))
        print_run_line \
          "$DATASET" \
          "$INIT_MODE" \
          "$BUDGET_LABEL" \
          "$SAMPLE_LIMIT" \
          "$ITERATIONS" \
          "$SEED" \
          "$NUM_EPOCHS" \
          "$MODEL"
        if [[ "$PLAN_ONLY" != "1" ]]; then
          "$SCRIPT_DIR/run_active_learning.sh"
        fi
      done
    done
  done
done

print_totals
